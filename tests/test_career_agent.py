"""固定模型测试有界图、背景未知、个性化隔离与MySQL状态。"""

import threading
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from jobcompass.career import (
    Application,
    Background,
    Career,
    ConflictError,
    background_checks,
    soft_signals,
)
from jobcompass.generation import Generator
from jobcompass.recommendation_agent import RecommendationAgent, RecommendationRequest


def test_background_is_conservative(job_factory):
    bg = Background(education="本科", experience_years=0)
    job = job_factory(education_text="本科及以上", experience_text="1-3年")
    assert [r["status"] for r in background_checks(job, bg)] == ["pass", "fail"]
    job.update(experience_text="应届毕业生")
    assert all(r["status"] == "pass" for r in background_checks(job, bg))
    assert background_checks(job_factory(), bg)[0]["status"] == "unknown"
    assert background_checks(job_factory(education_text="本科或同等经验"), bg)[0]["status"] == "unknown"
    assert background_checks(job_factory(education_text="本科"), Background())[0]["status"] == "unknown"


def test_soft_signals_use_actual_self_report(job_factory):
    bg = Background(
        skills=["Python", "Java"], projects="使用 FastAPI 完成项目；不会 SQL", preferred_cities=["深圳"]
    )
    job = job_factory(jd="使用Python、FastAPI开发接口\n熟悉SQL")
    signals = soft_signals(job, bg)
    assert signals["skills"] == ["Python"]
    assert signals["project_skills"] == ["FastAPI"]
    assert signals["project_evidence"] == bg.projects


def fake_service(monkeypatch, job_factory, tmp_path, *, empty=False, during_generation=None):
    import jobcompass.recommendation_agent as module

    rows = [
        job_factory("one", education_text="本科", experience_text="不限"),
        job_factory("liked", education_text="本科", experience_text="不限"),
        job_factory("closed", education_text="本科", experience_text="不限"),
        job_factory("disliked", education_text="本科", experience_text="不限"),
        job_factory("unknown"),
    ]

    class FakeCareer:
        def __init__(self, store):
            pass

        def background(self, owner):
            return Background(
                education="本科", experience_years=0, projects="PRIVATE_BACKGROUND", skills=["Python"]
            ).model_dump()

        def records(self, owner, table):
            if table is module.preferences:
                return [
                    {"job_key": "fixture:liked", "preference": "like"},
                    {"job_key": "fixture:disliked", "preference": "dislike"},
                ]
            return [{"job_key": "fixture:closed", "status": "rejected"}]

    monkeypatch.setattr(module, "Career", FakeCareer)
    monkeypatch.setattr(module, "ROOT", tmp_path)

    class FakeRetriever:
        def search(self, request, *, eligible_keys):
            assert request.profile == ""
            assert request.query == "Python"
            return dict(
                results=[]
                if empty
                else [dict(job=j, hard_condition_checks=[]) for j in rows if j["job_key"] in eligible_keys],
                degradations=[],
                timings_ms={},
                conditions={},
                unresolved_conditions=[],
            )

    class FakeGenerator:
        def answer(self, retrieval, request):
            assert request.profile == "" and "PRIVATE_BACKGROUND" not in str(retrieval)
            if during_generation:
                during_generation()
            return dict(
                answer="固定原文",
                recommendations=[
                    dict(job_key=r["job"]["job_key"], citations=[]) for r in retrieval["results"]
                ],
                generation={"calls": 0},
            )

    valid = {r["job_key"]: r for r in rows}
    store = SimpleNamespace(
        all_jobs=lambda: rows,
        current=lambda keys: {k: valid[k] for k in keys if k in valid},
        save_history=lambda owner, result: "fixed-history",
    )
    svc = SimpleNamespace(
        store=store,
        limit=threading.BoundedSemaphore(1),
        start_lock=threading.Lock(),
        get_retriever=lambda: FakeRetriever(),
        generator=FakeGenerator(),
        settings=SimpleNamespace(stage="test"),
    )
    return svc, valid


def test_graph_prefilters_and_personalizes(monkeypatch, job_factory, tmp_path):
    svc, _ = fake_service(monkeypatch, job_factory, tmp_path)
    result = RecommendationAgent(svc).run(
        RecommendationRequest(query="Python", generate=False), "owner", persist=False
    )
    keys = [r["job"]["job_key"] for r in result["results"]]
    assert keys == ["fixture:liked", "fixture:one"]
    assert [n["node"] for n in result["agent"]["nodes"]] == ["prepare", "retrieve", "personalize", "evidence"]
    assert result["recommendations"][0]["personal_signals"]["skills"] == ["Python"]
    assert result["recommendations"][0]["profile_mentions"] == ["Python"]
    assert svc.limit.acquire(blocking=False)
    svc.limit.release()


def test_graph_unknown_opt_in_empty_and_final_version_check(monkeypatch, job_factory, tmp_path):
    svc, valid = fake_service(monkeypatch, job_factory, tmp_path)
    req = RecommendationRequest(
        query="Python", generate=False, allow_unknown_background=True, include_closed=True
    )
    result = RecommendationAgent(svc).run(req, "owner", persist=False)
    keys = {r["job"]["job_key"] for r in result["results"]}
    assert "fixture:unknown" in keys and "fixture:closed" in keys and "fixture:disliked" not in keys
    svc, _ = fake_service(monkeypatch, job_factory, tmp_path, empty=True)
    result = RecommendationAgent(svc).run(req, "owner", persist=False)
    assert result["agent"]["nodes"][-1]["node"] == "no_results"
    assert result["recommendations"] == []
    svc, valid = fake_service(monkeypatch, job_factory, tmp_path, during_generation=lambda: valid.clear())
    result = RecommendationAgent(svc).run(req, "owner", persist=False)
    assert not result["results"] and not result["recommendations"]


def test_initial_recommendation_defers_model_explanation(monkeypatch, job_factory, tmp_path):
    svc, _ = fake_service(monkeypatch, job_factory, tmp_path)
    from jobcompass.settings import Settings

    svc.generator = Generator(Settings("test"))

    def timeout(*args):
        raise TimeoutError("fixed")

    monkeypatch.setattr(svc.generator, "select", timeout)
    result = RecommendationAgent(svc).run(
        RecommendationRequest(query="Python", generate=True), "owner", persist=False
    )
    assert result["generation"]["mode"] == "deferred"
    assert result["generation"]["calls"] == 0
    assert result["recommendations"][0]["citations"] == []
    assert result["recommendations"][0]["explanation_status"] == "not_requested"


@pytest.mark.integration
def test_career_mysql_api_isolation_retry_conflict(real_store, job_factory):
    from jobcompass.identity import Identity
    from jobcompass.service import RagService
    from jobcompass.storage import jobs
    from jobcompass.web import app, service

    s = real_store
    identity = Identity(s)
    uid1 = identity.register(s.test_prefix + "a", "test-only-password")
    uid2 = identity.register(s.test_prefix + "b", "test-only-password")
    row = job_factory(s.test_prefix, education_text="本科", experience_text="不限")
    with s.engine.begin() as c:
        c.execute(
            jobs.insert().values(
                job_key=row["job_key"],
                source=row["source"],
                source_job_id=row["source_job_id"],
                version=1,
                content_hash=row["content_hash"],
                active=True,
                owner=s.test_owner,
                payload=row,
            )
        )
    career = Career(s)
    career.background(uid1, Background(education="本科", experience_years=0, projects="项目原文"))
    identity.profile(uid1, "通用画像")
    assert career.background(uid1)["projects"] == "项目原文"
    assert career.background(uid2)["education"] is None
    with __import__("concurrent.futures").futures.ThreadPoolExecutor(max_workers=4) as pool:
        records = list(
            pool.map(
                lambda _: career.put(uid1, row["job_key"], Application(status="applied", expected_version=0)),
                range(4),
            )
        )
    assert {r["version"] for r in records} == {1}
    with pytest.raises(ConflictError):
        career.put(uid1, row["job_key"], Application(status="offer", expected_version=0))
    svc = RagService(s.settings)
    app.dependency_overrides[service] = lambda: svc
    try:
        with TestClient(app, raise_server_exceptions=True) as client:
            assert client.get("/api/applications").status_code == 401
            client.cookies.set("jc_auth", identity.login(s.test_prefix + "a", "test-only-password"))
            assert client.get("/api/applications").json()[0]["status"] == "applied"
            assert (
                client.put("/api/job-preferences/" + row["job_key"], json={"preference": "like"}).status_code
                == 200
            )
            assert (
                client.put("/api/applications/" + row["job_key"], json={"status": "invalid"}).status_code
                == 422
            )
            assert (
                client.put(
                    "/api/applications/" + row["job_key"], json={"status": "offer", "expected_version": 0}
                ).status_code
                == 409
            )
            client.cookies.set("jc_auth", identity.login(s.test_prefix + "b", "test-only-password"))
            assert client.get("/api/applications").json() == []
            assert client.get("/api/job-preferences").json() == []
            assert (
                client.put(
                    "/api/applications/" + row["job_key"], json={"status": "rejected", "owner": uid1}
                ).status_code
                == 422
            )
            assert client.put("/api/applications/missing", json={"status": "applied"}).status_code == 404
    finally:
        app.dependency_overrides.clear()
        svc.close()


def test_model_order_adopted_only_after_background_and_feedback_filters(monkeypatch, job_factory, tmp_path):
    svc, _ = fake_service(monkeypatch, job_factory, tmp_path)
    observed = []

    class Model:
        def rank(self, rows, background, query, **kwargs):
            observed.extend(r["job"]["job_key"] for r in rows)
            return rows, dict(mode="llm", effective="llm", calls=1)

    svc.ranker = Model()
    result = RecommendationAgent(svc).run(
        RecommendationRequest(query="Python", generate=True), "owner", persist=False
    )
    assert observed == ["fixture:one", "fixture:liked"]
    assert [r["job"]["job_key"] for r in result["results"]] == observed
    assert result["ranking"]["mode"] == "llm" and result["generation"]["calls"] == 0
    assert all(
        not r["citations"] and r["explanation_status"] == "not_requested" for r in result["recommendations"]
    )


def test_update_during_model_ranking_does_not_escape_final_version_check(monkeypatch, job_factory, tmp_path):
    svc, current = fake_service(monkeypatch, job_factory, tmp_path)

    class Model:
        def rank(self, rows, *args, **kwargs):
            current.clear()
            return rows, dict(mode="llm", effective="llm", calls=1)

    svc.ranker = Model()
    result = RecommendationAgent(svc).run(RecommendationRequest(query="Python"), "owner", persist=False)
    assert not result["results"] and not result["recommendations"]
