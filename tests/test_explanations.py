"""按需解释API：归属、版本、重复/并发请求、失败降级与历史保存。"""

import copy
from concurrent.futures import ThreadPoolExecutor
import threading
import time
from types import SimpleNamespace
import uuid

from fastapi.testclient import TestClient
import httpx
import pytest

from jobcompass.explanations import ExplanationConflict, ExplanationService
from jobcompass.generation import Generator
from jobcompass.settings import Settings
from jobcompass.web import app, current_user, service


class MemoryRepository:
    def __init__(self, payload):
        self.payload = payload
        self.lock = threading.Lock()

    def load(self, owner, query_id):
        if owner != "owner" or query_id != self.payload["id"]:
            raise LookupError()
        return copy.deepcopy(self.payload)

    def claim(self, owner, query_id, key):
        with self.lock:
            self.load(owner, query_id)
            states = self.payload.setdefault("explanations", {})
            old = states.get(key, {})
            if old.get("status") == "completed":
                return dict(status="completed", value=old["value"])
            if old.get("status") == "running" and old.get("lease_until", 0) > time.time():
                return dict(status="running")
            token = str(uuid.uuid4())
            states[key] = dict(status="running", token=token, lease_until=time.time() + 90)
            return dict(status="claimed", token=token, payload=copy.deepcopy(self.payload))

    def finish(self, owner, query_id, key, token, value):
        with self.lock:
            state = self.payload["explanations"][key]
            if state.get("token") != token:
                raise ExplanationConflict()
            state.update(status="completed", value=value)
            if value["recommendation"]:
                self.payload["recommendations"] = [
                    value["recommendation"] if r["job_key"] == key else r
                    for r in self.payload["recommendations"]
                ]

    def fail(self, owner, query_id, key, token, error_type):
        with self.lock:
            state = self.payload["explanations"][key]
            if state.get("token") == token:
                state.update(status="failed", lease_until=0, error_type=error_type)


@pytest.fixture
def api_fixture(monkeypatch, job_factory):
    import jobcompass.web as web

    job = job_factory(
        "reason", jd="负责Python业务接口开发\n维护MySQL数据库", education_text="本科", experience_text="不限"
    )
    qid = str(uuid.uuid4())
    key = job["job_key"]
    payload = dict(
        id=qid,
        request=dict(query="Python业务接口", generate=False, ranking_strategy="llm"),
        results=[dict(job=copy.deepcopy(job), hard_condition_checks=[])],
        recommendations=[
            dict(
                job_key=key,
                citations=[],
                explanation_status="not_requested",
                profile_mentions=["Python"],
                profile_evidence="PRIVATE_PROFILE",
            )
        ],
        agent=dict(explanations_on_demand=True),
        conditions={},
    )
    repo = MemoryRepository(payload)
    active = {key: job}
    generated = []

    class FakeGenerator:
        def answer(self, retrieval, request, **kwargs):
            assert "PRIVATE_PROFILE" not in str(retrieval) and request.profile == "" and request.generate
            generated.append(key)
            return dict(
                answer="有原文证据",
                recommendations=[
                    dict(
                        job_key=key,
                        title=job["title"],
                        match_reason="接口工作有原文支持",
                        citations=[dict(line=1, quote=job["jd_text"].split("\n")[0])],
                        version=1,
                        content_hash=job["content_hash"],
                    )
                ],
                generation=dict(mode="llm_evidence_selection", calls=1, usage={}),
            )

    svc = SimpleNamespace(
        store=SimpleNamespace(current=lambda keys: {k: active[k] for k in keys if k in active}),
        generator=FakeGenerator(),
        limit=threading.BoundedSemaphore(2),
        settings=Settings("test"),
    )
    svc.explanations = ExplanationService(svc, repository=repo)
    app.dependency_overrides[service] = lambda: svc
    owner = {"id": "owner"}
    app.dependency_overrides[current_user] = lambda: owner
    monkeypatch.setattr(web, "rate_limit", lambda *args: None)
    client = TestClient(app)
    yield SimpleNamespace(
        client=client,
        svc=svc,
        repo=repo,
        active=active,
        job=job,
        key=key,
        qid=qid,
        owner=owner,
        generated=generated,
        url=f"/api/recommendations/{qid}/explanation",
    )
    client.close()
    app.dependency_overrides.clear()


def test_lazy_explanation_and_duplicate_requests_use_owned_history_cache(api_fixture):
    f = api_fixture
    initial = f.client.get(f.url, params={"job_key": f.key})
    assert initial.json()["status"] == "not_requested" and not f.generated
    first = f.client.post(f.url, json={"job_key": f.key})
    assert first.status_code == 200 and first.json()["calls_this_request"] == 1
    rec = first.json()["recommendation"]
    assert rec["citations"][0]["quote"] in f.job["jd_text"] and rec["profile_evidence"] == "PRIVATE_PROFILE"
    assert f.repo.payload["recommendations"][0]["citations"] == rec["citations"]
    for response in (
        f.client.post(f.url, json={"job_key": f.key}),
        f.client.get(f.url, params={"job_key": f.key}),
    ):
        assert response.json()["cache"] == "hit" and response.json()["calls_this_request"] == 0
    assert f.generated == [f.key]


def test_foreign_query_nonmember_or_changed_job_never_calls_model(api_fixture):
    f = api_fixture
    f.owner["id"] = "other"
    assert f.client.post(f.url, json={"job_key": f.key}).status_code == 404
    f.owner["id"] = "owner"
    assert f.client.post(f.url, json={"job_key": "not-in-result"}).status_code == 404
    assert f.client.post(f.url, json={"job_key": f.key, "owner": "other"}).status_code == 422
    f.active.clear()
    assert f.client.post(f.url, json={"job_key": f.key}).status_code == 409
    assert not f.generated


def test_authenticated_and_parameter_boundaries(api_fixture, monkeypatch):
    import jobcompass.web as web

    f = api_fixture
    assert f.client.post(f.url, json={"job_key": "x" * 193}).status_code == 422
    app.dependency_overrides.pop(current_user)
    monkeypatch.setattr(web, "Identity", lambda store: SimpleNamespace(current=lambda token: None))
    assert f.client.post(f.url, json={"job_key": f.key}).status_code == 401
    assert not f.generated


def test_concurrent_clicks_return_running_then_cache_without_duplicate_generation(api_fixture):
    f = api_fixture
    entered, release = threading.Event(), threading.Event()
    old = f.svc.generator.answer

    def waiting(*args, **kwargs):
        entered.set()
        assert release.wait(3)
        return old(*args, **kwargs)

    f.svc.generator.answer = waiting
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(f.client.post, f.url, json={"job_key": f.key})
        assert entered.wait(2)
        duplicate = f.client.post(f.url, json={"job_key": f.key})
        assert duplicate.status_code == 202 and duplicate.json()["status"] == "running"
        assert f.client.get(f.url, params={"job_key": f.key}).json()["status"] == "running"
        release.set()
        assert future.result().status_code == 200
    assert f.generated == [f.key]


def test_job_update_during_generation_rejected_and_not_saved_as_ready(api_fixture):
    f = api_fixture
    old = f.svc.generator.answer

    def changed(*args, **kwargs):
        result = old(*args, **kwargs)
        f.active[f.key] = dict(f.active[f.key], version=2)
        return result

    f.svc.generator.answer = changed
    assert f.client.post(f.url, json={"job_key": f.key}).status_code == 409
    assert f.repo.payload["recommendations"][0]["citations"] == []
    assert f.repo.payload["explanations"][f.key]["status"] == "failed"


def test_cached_explanation_is_rejected_after_job_changes(api_fixture):
    f = api_fixture
    assert f.client.post(f.url, json={"job_key": f.key}).status_code == 200
    f.active[f.key] = dict(f.active[f.key], content_hash="changed")
    assert f.client.get(f.url, params={"job_key": f.key}).status_code == 409
    assert len(f.generated) == 1


@pytest.mark.parametrize("invalid", [False, True])
def test_generation_timeout_or_invalid_quote_falls_back_with_visible_original(
    api_fixture, monkeypatch, invalid
):
    f = api_fixture
    gen = Generator(Settings("test"))
    f.svc.generator = gen

    def selected(packs, request, **kwargs):
        gen.context.calls += 1
        if not invalid:
            raise httpx.ReadTimeout("fixed")
        return (
            dict(items=[dict(job_key=f.key, citations=[dict(line=1, quote="fabricated")])]),
            dict(input_tokens=10, output_tokens=5),
            "Pro/deepseek-ai/DeepSeek-V3.2",
        )

    monkeypatch.setattr(gen, "select", selected)
    result = f.client.post(f.url, json={"job_key": f.key})
    assert result.status_code == 200
    value = result.json()
    assert value["generation"]["mode"] == "extractive_fallback" and value["calls_this_request"] == 1
    assert value["recommendation"]["explanation_status"] == "fallback"
    assert all(c["quote"] in f.job["jd_text"] for c in value["recommendation"]["citations"])
    assert "fabricated" not in str(value)
    assert f.client.post(f.url, json={"job_key": f.key}).json()["calls_this_request"] == 0


@pytest.mark.integration
def test_mysql_claim_concurrency_expired_lease_and_stale_completion(real_store, job_factory):
    from sqlalchemy import select, update
    from jobcompass.explanations import ExplanationRepository
    from jobcompass.storage import history, jobs

    s = real_store
    job = job_factory(s.test_prefix)
    with s.engine.begin() as c:
        c.execute(
            jobs.insert().values(
                job_key=job["job_key"],
                source=job["source"],
                source_job_id=job["source_job_id"],
                version=1,
                content_hash=job["content_hash"],
                active=True,
                owner=s.test_owner,
                payload=job,
            )
        )
    payload = dict(
        agent=dict(explanations_on_demand=True),
        request=dict(query="Python"),
        results=[dict(job=job)],
        recommendations=[dict(job_key=job["job_key"], citations=[])],
    )
    qid = s.save_history(s.test_owner, payload)
    repo = ExplanationRepository(s)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: repo.claim(s.test_owner, qid, job["job_key"]), range(4)))
    assert sum(x["status"] == "claimed" for x in results) == 1
    first = next(x for x in results if x["status"] == "claimed")
    assert s.engine.pool.checkedout() == 0
    with s.engine.begin() as c:
        data = c.execute(select(history.c.payload).where(history.c.query_id == qid)).scalar_one()
        data["explanations"][job["job_key"]]["lease_until"] = 0
        c.execute(update(history).where(history.c.query_id == qid).values(payload=data))
    retry = repo.claim(s.test_owner, qid, job["job_key"])
    assert retry["token"] != first["token"]
    with pytest.raises(ExplanationConflict):
        repo.finish(s.test_owner, qid, job["job_key"], first["token"], {})
    value = dict(recommendation=dict(job_key=job["job_key"], citations=[dict(line=1, quote="原文")]))
    repo.finish(s.test_owner, qid, job["job_key"], retry["token"], value)
    assert repo.claim(s.test_owner, qid, job["job_key"])["status"] == "completed"
    with pytest.raises(LookupError):
        repo.load("another-owner", qid)
    other = s.save_history(s.test_owner, payload)
    claim = repo.claim(s.test_owner, other, job["job_key"])
    with s.engine.begin() as c:
        c.execute(update(jobs).where(jobs.c.job_key == job["job_key"]).values(active=False))
    with pytest.raises(ExplanationConflict):
        repo.finish(s.test_owner, other, job["job_key"], claim["token"], value)


def test_unavailable_history_returns_clear_service_error_without_generation(api_fixture, monkeypatch):
    f = api_fixture

    def unavailable(*args):
        raise RuntimeError("PRIVATE_DATABASE_DETAIL")

    monkeypatch.setattr(f.repo, "load", unavailable)
    result = f.client.get(f.url, params={"job_key": f.key})
    assert result.status_code == 503 and "PRIVATE_DATABASE_DETAIL" not in result.text
    assert not f.generated


@pytest.mark.integration
def test_mysql_agent_to_lazy_reason_history_and_user_isolation(real_store, job_factory, monkeypatch):
    from sqlalchemy import select, update
    import hashlib
    import json
    from jobcompass.career import Background, Career
    from jobcompass.identity import Identity
    from jobcompass.personalized_ranking import PersonalizedRanker
    from jobcompass.service import RagService
    from jobcompass.storage import history, jobs

    s = real_store
    identity = Identity(s)
    uid = identity.register(s.test_prefix + "a", "fixed-test-password")
    identity.register(s.test_prefix + "b", "fixed-test-password")
    source = [
        job_factory(
            s.test_prefix + str(i),
            title="Python业务接口开发" + str(i),
            jd="负责Python业务接口开发\n维护MySQL数据事务",
            education_text="本科",
            experience_text="不限",
        )
        for i in range(2)
    ]
    with s.engine.begin() as c:
        for j in source:
            c.execute(
                jobs.insert().values(
                    job_key=j["job_key"],
                    source=j["source"],
                    source_job_id=j["source_job_id"],
                    version=1,
                    content_hash=j["content_hash"],
                    active=True,
                    owner=s.test_owner,
                    payload=j,
                )
            )
    Career(s).background(
        uid,
        Background(
            education="本科", experience_years=1, skills=["Python"], projects="接口项目，联系13812345678"
        ),
    )
    identity.profile(uid, "RAW_PRIVATE_RESUME")
    svc = RagService(s.settings)
    svc.settings.prefix += "rank_feature:" + s.test_prefix + ":"

    class Retriever:
        def search(self, request, eligible_keys):
            return dict(
                results=[
                    dict(job=j, hard_condition_checks=[]) for j in source if j["job_key"] in eligible_keys
                ],
                conditions={},
                degradations=[],
                timings_ms={},
                unresolved_conditions=[],
            )

    svc.get_retriever = lambda: Retriever()
    rank_settings = copy.copy(s.settings)
    rank_settings.values = s.settings.values | dict(
        API_BASE="https://example.invalid/v1",
        API_TYPE="openai_chat_completions",
        API_KEY_ENV="RANK_UNIT_KEY",
        RANK_UNIT_KEY="fixed-test-only",
    )
    desired = [hashlib.sha256(j["job_key"].encode()).hexdigest()[:12] for j in reversed(source)]
    sent = []

    def ranked(request):
        sent.append(request.content.decode())
        return httpx.Response(
            200,
            json=dict(
                choices=[dict(finish_reason="stop", message=dict(content=json.dumps(dict(ranking=desired))))],
                usage=dict(prompt_tokens=100, completion_tokens=30, prompt_cache_hit_tokens=0),
            ),
        )

    svc.ranker = PersonalizedRanker(rank_settings, transport=httpx.MockTransport(ranked))
    selects = []

    def selected(packs, request, **kwargs):
        svc.generator.context.calls += 1
        selects.append(packs)
        assert request.profile == "" and "RAW_PRIVATE_RESUME" not in str(packs)
        return (
            dict(
                items=[
                    dict(
                        job_key=p["job_key"],
                        citations=[dict(line=p["lines"][0]["line"], quote=p["lines"][0]["text"])],
                    )
                    for p in packs
                ]
            ),
            dict(input_tokens=30, output_tokens=20),
            "Pro/deepseek-ai/DeepSeek-V3.2",
        )

    monkeypatch.setattr(svc.generator, "select", selected)
    app.dependency_overrides[service] = lambda: svc
    client = TestClient(app)
    try:
        client.cookies.set("jc_auth", identity.login(s.test_prefix + "a", "fixed-test-password"))
        result = client.post(
            "/api/recommendations", json=dict(query="Python业务接口开发", generate=True)
        ).json()
        assert result["ranking"]["mode"] == "llm" and result["generation"]["mode"] == "deferred"
        assert result["results"][0]["job"]["job_key"] == source[1]["job_key"] and not selects
        assert "RAW_PRIVATE_RESUME" not in sent[0] and "13812345678" not in sent[0]
        qid = result["query_id"]
        key = source[1]["job_key"]
        url = f"/api/recommendations/{qid}/explanation"
        generated = client.post(url, json=dict(job_key=key))
        assert generated.status_code == 200 and generated.json()["calls_this_request"] == 1
        assert all(
            c["quote"] in source[1]["jd_text"] for c in generated.json()["recommendation"]["citations"]
        )
        assert (
            client.post(url, json=dict(job_key=key)).json()["calls_this_request"] == 0 and len(selects) == 1
        )
        with s.engine.connect() as c:
            saved = c.execute(select(history.c.payload).where(history.c.query_id == qid)).scalar_one()
            assert saved["explanations"][key]["status"] == "completed"
            assert saved["recommendations"][0]["explanation_status"] == "ready"
        assert svc.store.engine.pool.checkedout() == 0
        client.cookies.set("jc_auth", identity.login(s.test_prefix + "b", "fixed-test-password"))
        assert client.post(url, json=dict(job_key=key)).status_code == 404
        client.cookies.set("jc_auth", identity.login(s.test_prefix + "a", "fixed-test-password"))
        with s.engine.begin() as c:
            c.execute(update(jobs).where(jobs.c.job_key == key).values(active=False))
        assert client.get(url, params=dict(job_key=key)).status_code == 409 and len(selects) == 1
    finally:
        client.close()
        app.dependency_overrides.clear()
        keys = list(svc.redis.scan_iter(match=svc.settings.prefix + "*"))
        if keys:
            svc.redis.delete(*keys)
        svc.close()
