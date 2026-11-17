"""按账号查询历史生成单岗位理由；JSON状态租约避免重复调用与过期写回。"""

import copy
import json
import logging
import time
import uuid

from langchain_core.runnables import RunnableLambda
from langsmith import tracing_context
from sqlalchemy import select, update

from .conditions import SearchRequest
from .personalized_ranking import model_cost, scrub_contacts
from .storage import history, jobs

LOGGER = logging.getLogger("jobcompass.explanations")


class ExplanationConflict(ValueError):
    pass


def member(payload, key):
    if not payload.get("agent"):
        raise LookupError("推荐记录不存在")
    row = next((r for r in payload["results"] if r["job"]["job_key"] == key), None)
    rec = next((r for r in payload.get("recommendations", []) if r["job_key"] == key), None)
    if row is None or rec is None:
        raise LookupError("岗位不在本次推荐中")
    return row, rec


class ExplanationRepository:
    def __init__(self, store):
        self.store = store

    def _load(self, connection, owner, query_id, *, locked=False):
        statement = select(history.c.payload).where(history.c.query_id == query_id, history.c.owner == owner)
        if locked:
            statement = statement.with_for_update()
        payload = connection.execute(statement).scalar_one_or_none()
        if payload is None:
            raise LookupError("推荐记录不存在")
        return copy.deepcopy(payload)

    def load(self, owner, query_id):
        with self.store.engine.connect() as connection:
            return self._load(connection, owner, query_id)

    def _save(self, connection, owner, query_id, payload):
        connection.execute(
            update(history)
            .where(history.c.query_id == query_id, history.c.owner == owner)
            .values(payload=payload)
        )

    def claim(self, owner, query_id, key, lease_seconds=90):
        with self.store.engine.begin() as connection:
            payload = self._load(connection, owner, query_id, locked=True)
            member(payload, key)
            states = payload.setdefault("explanations", {})
            old = states.get(key, {})
            if old.get("status") == "completed":
                return dict(status="completed", value=old["value"])
            if old.get("status") == "running" and old.get("lease_until", 0) > time.time():
                return dict(status="running")
            token = str(uuid.uuid4())
            states[key] = dict(
                status="running",
                token=token,
                lease_until=time.time() + lease_seconds,
                attempts=old.get("attempts", 0) + 1,
            )
            self._save(connection, owner, query_id, payload)
            return dict(status="claimed", token=token, payload=payload)

    def finish(self, owner, query_id, key, token, value):
        with self.store.engine.begin() as connection:
            payload = self._load(connection, owner, query_id, locked=True)
            state = payload.get("explanations", {}).get(key, {})
            if state.get("status") != "running" or state.get("token") != token:
                raise ExplanationConflict("生成状态已变化，请重新读取")
            row, _ = member(payload, key)
            current = (
                connection.execute(select(jobs).where(jobs.c.job_key == key).with_for_update())
                .mappings()
                .first()
            )
            snapshot = row["job"]
            if (
                not current
                or not current["active"]
                or current["version"] != snapshot["version"]
                or current["content_hash"] != snapshot["content_hash"]
            ):
                raise ExplanationConflict("岗位已更新或下架，请重新推荐")
            state.update(status="completed", value=value, lease_until=0)
            state.pop("token", None)
            if value.get("recommendation"):
                payload["recommendations"] = [
                    value["recommendation"] if r["job_key"] == key else r for r in payload["recommendations"]
                ]
            self._save(connection, owner, query_id, payload)

    def fail(self, owner, query_id, key, token, error_type):
        with self.store.engine.begin() as connection:
            payload = self._load(connection, owner, query_id, locked=True)
            state = payload.get("explanations", {}).get(key, {})
            if state.get("status") == "running" and state.get("token") == token:
                state.update(status="failed", error_type=error_type, lease_until=0)
                state.pop("token", None)
                self._save(connection, owner, query_id, payload)


class ExplanationService:
    def __init__(self, service, *, repository=None):
        self.service = service
        self.repository = repository or ExplanationRepository(service.store)

    def _current(self, payload, key):
        row, rec = member(payload, key)
        current = self.service.store.current([key]).get(key)
        old = row["job"]
        if (
            not current
            or current["version"] != old["version"]
            or current["content_hash"] != old["content_hash"]
        ):
            raise ExplanationConflict("岗位已更新或下架，请重新推荐")
        return row, rec

    def get(self, owner, query_id, key):
        payload = self.repository.load(owner, query_id)
        _, rec = self._current(payload, key)
        state = payload.get("explanations", {}).get(key, {})
        if state.get("status") == "completed":
            return dict(state["value"], status="completed", cache="hit", calls_this_request=0)
        # v1历史中的现成解释仍按原记录读取，不补发模型。
        if not payload["agent"].get("explanations_on_demand") and rec.get("citations"):
            return dict(
                status="completed",
                recommendation=rec,
                generation=payload.get("generation", {}),
                query_id=query_id,
                job_key=key,
                cache="legacy_history",
                calls_this_request=0,
            )
        running = state.get("status") == "running" and state.get("lease_until", 0) > time.time()
        return dict(
            status="running" if running else "not_requested",
            query_id=query_id,
            job_key=key,
            cache="miss",
            calls_this_request=0,
            retry_after=2 if running else None,
        )

    def generate(self, owner, query_id, key, request_id=None):
        cached = self.get(owner, query_id, key)
        if cached["status"] in {"completed", "running"}:
            return cached
        svc = self.service
        if not svc.limit.acquire(timeout=2):
            raise RuntimeError("当前请求较多，请稍后重试")
        claimed = None
        try:
            claimed = self.repository.claim(owner, query_id, key)
            if claimed["status"] != "claimed":
                return self.get(owner, query_id, key)
            payload = claimed["payload"]
            row, original = self._current(payload, key)
            saved = payload["request"]
            request = SearchRequest.model_validate(
                {k: v for k, v in saved.items() if k in SearchRequest.model_fields}
                | dict(query=scrub_contacts(saved["query"]), profile="", generate=True, top_k=1)
            )
            retrieval = dict(
                results=[dict(job=row["job"], hard_condition_checks=row["hard_condition_checks"])],
                conditions=payload.get("conditions", {}),
                unresolved_conditions=[],
                degradations=[],
            )
            work = RunnableLambda(lambda data: svc.generator.answer(data, request, timeout_seconds=15))
            # 历史画像/偏好不进入LangSmith，模型仅选择公开JD工作内容证据。
            with tracing_context(enabled=False):
                answer = work.invoke(retrieval, config={"callbacks": []})
            rec = next((r for r in answer["recommendations"] if r["job_key"] == key), None)
            if rec:
                for field in (
                    "background_checks",
                    "personal_signals",
                    "profile_mentions",
                    "profile_evidence",
                ):
                    if field in original:
                        rec[field] = original[field]
                rec["skills_without_profile_evidence"] = [
                    s
                    for s in rec.get("skills_without_profile_evidence", [])
                    if s not in rec.get("profile_mentions", [])
                ]
                rec["explanation_status"] = (
                    "fallback" if answer["generation"]["mode"] == "extractive_fallback" else "ready"
                )
            generation = answer["generation"]
            generation["cost_estimate"] = model_cost(
                generation.get("usage", {}),
                generation.get("model") or svc.settings.api_config.get("MODEL"),
                generation.get("calls", 0),
            )
            value = dict(
                query_id=query_id,
                job_key=key,
                recommendation=rec,
                generation=generation,
                message=answer["answer"],
                based_on="推荐时的查询与已保存画像依据",
                request_id=request_id,
            )
            self._current(payload, key)
            self.repository.finish(owner, query_id, key, claimed["token"], value)
            LOGGER.info(
                json.dumps(
                    dict(
                        event="recommendation_explanation",
                        request_id=request_id,
                        query_id=query_id,
                        mode=generation["mode"],
                        calls=generation.get("calls", 0),
                    )
                )
            )
            return dict(
                value, status="completed", cache="miss", calls_this_request=generation.get("calls", 0)
            )
        except Exception as error:
            if claimed and claimed["status"] == "claimed":
                try:
                    self.repository.fail(owner, query_id, key, claimed["token"], type(error).__name__)
                except Exception as state_error:
                    LOGGER.warning(
                        json.dumps(
                            dict(
                                event="explanation_state_write_failed",
                                request_id=request_id,
                                error_type=type(state_error).__name__,
                            )
                        )
                    )
            raise
        finally:
            svc.limit.release()
