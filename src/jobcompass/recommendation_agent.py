"""LangGraph有界推荐：本地资格、完整JD模型排序/规则回退、理由按需生成。"""

import json
import uuid
from time import perf_counter
from typing import Literal, TypedDict

from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langsmith import tracing_context
from pydantic import Field

from .career import Background, Career, applications, background_checks, preferences, soft_signals
from .conditions import SearchRequest
from .evidence import apply_evidence_result, direction_query
from .personalized_ranking import rule_order
from .settings import ROOT


class RecommendationRequest(SearchRequest):
    generate: bool = False
    ranking_strategy: Literal["llm", "rules"] = "llm"
    allow_unknown_background: bool = False
    include_closed: bool = False
    top_k: int = Field(default=5, ge=1, le=10)


class AgentState(TypedDict, total=False):
    eligible: list[str]
    checks: dict
    preference_map: dict
    application_map: dict
    retrieval: dict
    answer: dict
    trace: list[dict]


class RecommendationAgent:
    version = "recommendation.v2"

    def __init__(self, service):
        self.service = service

    def run(self, request, owner, request_id=None, persist=True):
        svc, started = self.service, perf_counter()
        request_id = request_id or str(uuid.uuid4())
        if not svc.limit.acquire(timeout=2):
            raise RuntimeError("当前查询较多，请稍后重试")
        try:
            career = Career(svc.store)
            background = Background.model_validate(career.background(owner))
            if request.education_max is not None:
                background = background.model_copy(update={"education": request.education_max})
            public = SearchRequest.model_validate(
                {k: v for k, v in request.model_dump().items() if k in SearchRequest.model_fields}
                | dict(profile="", top_k=20, generate=False)
            )
            prepared = {}

            def prepare(state):
                t = perf_counter()
                pm = {r["job_key"]: r for r in career.records(owner, preferences)}
                am = {r["job_key"]: r for r in career.records(owner, applications)}
                checks, eligible = {}, []
                for job in svc.store.all_jobs():
                    key = job["job_key"]
                    check = background_checks(job, background)
                    checks[key] = check
                    if pm.get(key, {}).get("preference") == "dislike":
                        continue
                    if not request.include_closed and am.get(key, {}).get("status") in {"offer", "rejected"}:
                        continue
                    if any(
                        c["status"] == "fail"
                        or (c["status"] == "unknown" and not request.allow_unknown_background)
                        for c in check
                    ):
                        continue
                    eligible.append(key)
                prepared.update(
                    eligible=eligible,
                    checks=checks,
                    preference_map=pm,
                    application_map=am,
                    trace=[dict(node="prepare", ms=(perf_counter() - t) * 1000, eligible=len(eligible))],
                )
                return prepared

            @tool
            def retrieve_jobs(query: str) -> dict:
                """按本轮条件和本地核验范围检索公开岗位。"""
                retriever = svc.get_retriever()
                with svc.start_lock:
                    return retriever.search(
                        public.model_copy(update={"query": query}), eligible_keys=prepared["eligible"]
                    )

            def retrieve(state):
                t = perf_counter()
                data = retrieve_jobs.invoke({"query": public.query}, config={"callbacks": []})
                return dict(
                    retrieval=data,
                    trace=state["trace"] + [dict(node="retrieve", ms=(perf_counter() - t) * 1000)],
                )

            def personalize(state):
                t = perf_counter()
                rows = []
                for rank, row in enumerate(state["retrieval"]["results"]):
                    key = row["job"]["job_key"]
                    signals = soft_signals(
                        row["job"],
                        background,
                        state["preference_map"].get(key),
                        state["application_map"].get(key),
                    )
                    rows.append(
                        dict(
                            row,
                            background_checks=state["checks"][key],
                            personal_signals=signals,
                            rule_score=1 / (rank + 1) + signals["score"],
                        )
                    )
                ranker = getattr(svc, "ranker", None)
                if ranker is not None and direction_query(public):
                    rows, ranking = ranker.rank(
                        rows,
                        background,
                        public.query,
                        strategy=request.ranking_strategy,
                        request_id=request_id,
                    )
                else:
                    rows = rule_order(rows)
                    ranking = dict(
                        requested=request.ranking_strategy,
                        effective="rules",
                        calls=0,
                        usage={},
                        mode="rules"
                        if request.ranking_strategy == "rules" or not direction_query(public)
                        else "fallback_rules",
                        fallback_reason="not_available" if direction_query(public) else None,
                        cost=dict(cny=0, basis="no_call"),
                        seconds=0,
                    )
                rows = [dict(row, ranking_position=i) for i, row in enumerate(rows, 1)]
                data = dict(state["retrieval"], results=rows[: request.top_k], ranking=ranking)
                if ranking["mode"] == "fallback_rules":
                    data["degradations"] = list(data.get("degradations", [])) + [
                        dict(
                            stage="personalized_ranking", fallback="rules", reason=ranking["fallback_reason"]
                        )
                    ]
                if not data["results"] and not data.get("unresolved_conditions"):
                    data["no_result_reason"] = (
                        "没有可确认符合当前条件与背景的岗位；可补全学历/经验，或允许展示背景待确认的候选。"
                    )
                return dict(
                    retrieval=data,
                    trace=state["trace"]
                    + [dict(node="personalize", mode=ranking["mode"], ms=(perf_counter() - t) * 1000)],
                )

            # 首次只做本地职责/原文校验；generate即便由旧客户端传true也不在此调用外部解释。
            evidence = RunnableLambda(
                lambda data: svc.generator.answer(
                    dict(
                        data,
                        results=[
                            dict(job=r["job"], hard_condition_checks=r["hard_condition_checks"])
                            for r in data["results"]
                        ],
                    ),
                    public,
                )
            )

            def explain(state):
                t = perf_counter()
                answer = evidence.invoke(state["retrieval"], config={"callbacks": []})
                return dict(
                    answer=answer,
                    trace=state["trace"]
                    + [dict(node="evidence", local_only=True, ms=(perf_counter() - t) * 1000)],
                )

            def empty(state):
                return dict(
                    answer=svc.generator.answer(state["retrieval"], public),
                    trace=state["trace"] + [dict(node="no_results", ms=0)],
                )

            graph = StateGraph(AgentState)
            for name, fn in (
                ("prepare", prepare),
                ("retrieve", retrieve),
                ("personalize", personalize),
                ("evidence", explain),
                ("no_results", empty),
            ):
                graph.add_node(name, fn)
            graph.add_edge(START, "prepare")
            graph.add_edge("prepare", "retrieve")
            graph.add_edge("retrieve", "personalize")
            graph.add_conditional_edges(
                "personalize", lambda state: "evidence" if state["retrieval"]["results"] else "no_results"
            )
            graph.add_edge("evidence", END)
            graph.add_edge("no_results", END)
            with tracing_context(enabled=False):
                output = graph.compile().invoke({}, config={"callbacks": [], "recursion_limit": 10})
            retrieval, answer = output["retrieval"], output["answer"]
            retrieval = apply_evidence_result(retrieval, answer)
            latest = svc.store.current([r["job"]["job_key"] for r in retrieval["results"]])
            valid = {
                r["job"]["job_key"]
                for r in retrieval["results"]
                if latest.get(r["job"]["job_key"], {}).get("content_hash") == r["job"]["content_hash"]
                and latest.get(r["job"]["job_key"], {}).get("version") == r["job"]["version"]
            }
            retrieval["results"] = [r for r in retrieval["results"] if r["job"]["job_key"] in valid]
            answer["recommendations"] = [r for r in answer["recommendations"] if r["job_key"] in valid]
            for rec in answer["recommendations"]:
                row = next(r for r in retrieval["results"] if r["job"]["job_key"] == rec["job_key"])
                mentioned = sorted(
                    set(row["personal_signals"]["skills"] + row["personal_signals"]["project_skills"])
                )
                rec.update(
                    background_checks=row["background_checks"],
                    personal_signals=row["personal_signals"],
                    profile_mentions=mentioned,
                    profile_evidence=(
                        "自述技能：" + "、".join(background.skills) + "；项目原文：" + background.projects
                    )
                    if mentioned
                    else None,
                    version=row["job"]["version"],
                    content_hash=row["job"]["content_hash"],
                    match_reason=None,
                    citations=[],
                    explanation_status="not_requested",
                )
                rec["skills_without_profile_evidence"] = [
                    s for s in rec.get("skills_without_profile_evidence", []) if s not in mentioned
                ]
            if answer["recommendations"]:
                answer["answer"] = "已按当前条件与画像推荐岗位；推荐理由可在详情中按需生成。"
                answer["local_evidence"] = answer["generation"]
                answer["generation"] = dict(
                    mode="deferred",
                    calls=0,
                    usage={},
                    cost=0,
                    note="首次仅本地职责校验；理由由单独接口按需生成",
                )
            if not valid:
                answer["answer"] = retrieval.get("no_result_reason") or "岗位已更新或下架，请重新推荐。"
            ranking = retrieval["ranking"]
            result = dict(
                request_id=request_id,
                request=request.model_dump(),
                **retrieval,
                **answer,
                cache="disabled_for_personalization",
                agent=dict(
                    version=self.version,
                    nodes=output["trace"],
                    explanations_on_demand=True,
                    background_snapshot=background.model_dump(),
                    background_policy="allow_unknown"
                    if request.allow_unknown_background
                    else "confirmed_only",
                    background_egress=bool(ranking["calls"]),
                    privacy="排序发送已确认背景字段（过滤常见联系信息）和公开JD；原始简历及备注不外发"
                    if ranking["calls"]
                    else "本次背景仅在本地处理；按需理由仅发送查询和公开JD证据",
                    limitations=[
                        "专业/学校等复杂资格未自动判定",
                        "背景为用户自述，非胜任认证",
                        "投递记录不代表外部投递",
                        "按需解释是证据展示，不是模型内部推理过程",
                    ],
                ),
                total_ms=(perf_counter() - started) * 1000,
            )
            if persist:
                result["query_id"] = svc.store.save_history(owner, result)
                directory = ROOT / "artifacts/runs" / svc.settings.stage
                directory.mkdir(parents=True, exist_ok=True)
                (directory / f"{result['query_id']}.json").write_text(
                    json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                )
            return result
        finally:
            svc.limit.release()
