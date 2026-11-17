import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
import redis

from jobcompass.conditions import SearchRequest
from jobcompass.generation import Generator
from jobcompass.preprocessing import digest
from jobcompass.retrieval import Retriever
from jobcompass.service import RagService
from jobcompass.settings import Settings


class MemoryStore:
    def __init__(self, job):
        self.job = job
        self.active = True

    def all_jobs(self):
        return [self.job] if self.active else []

    def current(self, keys):
        return {self.job["job_key"]: self.job} if self.active and self.job["job_key"] in keys else {}


def test_vector_and_reranker_outage_fall_back_to_latest_bm25(job_factory, tmp_path, monkeypatch):
    from jobcompass import retrieval

    job = job_factory()
    settings = Settings()
    manifest = dict(
        snapshot=digest([[job["job_key"], job["content_hash"], job["version"]]]),
        collection="fixture",
        embedding=settings.config["embedding"],
        retrieval=settings.config["retrieval"],
    )
    target = tmp_path / "artifacts/indexes/dev/current.json"
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps(manifest))
    monkeypatch.setattr(retrieval, "ROOT", tmp_path)

    def unavailable(*a):
        raise TimeoutError("fixed Milvus outage")

    monkeypatch.setattr(retrieval, "milvus", unavailable)

    class BrokenRanker:
        def encode(self, *a, **kw):
            return np.array([[1.0, 0.0]])

        def rerank(self, *a):
            raise TimeoutError("fixed reranker timeout")

    retriever = Retriever(settings, MemoryStore(job), BrokenRanker())
    result = retriever.search(SearchRequest(query="Python", generate=False))
    assert result["results"][0]["job"]["job_key"] == job["job_key"]
    assert [d["stage"] for d in result["degradations"]] == ["dense", "reranker"]
    retriever.store.active = False
    assert retriever.search(SearchRequest(query="Python", generate=False))["results"] == []


def test_generator_payload_excludes_profile_and_rejects_fabrication(job_factory, monkeypatch):
    settings = Settings()
    generator = Generator(settings)
    monkeypatch.setenv("UNIT_MODEL_KEY", "fixed-test-key")
    monkeypatch.setattr(
        generator,
        "_config",
        lambda: dict(
            API_BASE="https://unit.invalid/v1",
            MODEL="fixed",
            API_KEY_ENV="UNIT_MODEL_KEY",
            API_TYPE="openai_responses",
            REASONING_EFFORT="",
        ),
    )
    captured = []

    def handle(request):
        captured.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "output_text": json.dumps(
                    {"items": [{"job_key": "outside", "citations": [{"line": 1, "quote": "编造的原文"}]}]}
                )
            },
        )

    original = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kw: original(transport=httpx.MockTransport(handle), **kw))
    job = job_factory()
    result = generator.answer(
        {"results": [{"job": job, "hard_condition_checks": []}]},
        SearchRequest(query="Python", profile="PERSONAL_PROFILE_DO_NOT_SEND"),
    )
    assert "PERSONAL_PROFILE_DO_NOT_SEND" not in json.dumps(captured)
    assert result["generation"]["mode"] == "extractive_fallback"
    assert result["generation"]["calls"] == 1
    assert result["recommendations"][0]["job_key"] == job["job_key"]


def fake_service(job):
    service = object.__new__(RagService)
    service.settings = SimpleNamespace(prefix="unit:", config={})
    service.store = MemoryStore(job)
    service.limit = threading.BoundedSemaphore(4)
    service.start_lock = threading.Lock()

    class UnavailableCache:
        def get(self, *a):
            raise redis.ConnectionError("fixed outage")

    service.redis = UnavailableCache()

    class FixedRetriever:
        def search(self, request):
            return dict(
                results=[{"job": job, "hard_condition_checks": []}],
                conditions={},
                unresolved_conditions=[],
                degradations=[],
                timings_ms={},
                no_result_reason=None,
            )

    service.retriever = FixedRetriever()
    return service


def test_bounded_concurrency_and_cache_outage(job_factory):
    service = fake_service(job_factory())
    lock = threading.Lock()
    active = peak = 0

    class FixedGenerator:
        def answer(self, *a):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.02)
            with lock:
                active -= 1
            return dict(answer="固定证据", recommendations=[], generation={"mode": "fixed"})

    service.generator = FixedGenerator()
    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(
            pool.map(lambda _: service.query(SearchRequest(query="Python"), persist=False), range(16))
        )
    assert len(results) == 16 and 1 < peak <= 4
    assert all(r["cache"] == "unavailable" for r in results)


def test_job_taken_down_during_generation_is_removed(job_factory):
    service = fake_service(job_factory())

    class FixedGenerator:
        def answer(self, *a):
            service.store.active = False
            return dict(
                answer="原候选",
                recommendations=[{"job_key": service.store.job["job_key"]}],
                generation={"mode": "fixed"},
            )

    service.generator = FixedGenerator()
    result = service.query(SearchRequest(query="Python"), persist=False)
    assert result["results"] == result["recommendations"] == []
    assert "下架" in result["answer"]


@pytest.mark.parametrize("query", ["必须Python并支持双休", "只接受北京工资30K", "必须Python或Java"])
def test_unknown_combined_hard_conditions_are_not_silently_dropped(query):
    from jobcompass.conditions import parse_conditions

    assert parse_conditions(SearchRequest(query=query))[1]
