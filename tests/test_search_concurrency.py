"""检索快照、并发、缓存边界和BM25等价性回归；固定模型。"""

import math
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import redis
from fastapi.testclient import TestClient

from jobcompass import retrieval
from jobcompass.conditions import SearchRequest
from jobcompass.preprocessing import digest
from jobcompass.service import RagService
from jobcompass.settings import Settings


class MemoryStore:
    def __init__(self, rows):
        self.rows = rows
        self.loads = 0
        self.read_sizes = []

    def corpus_revision(self):
        return digest([[r["job_key"], r["content_hash"], r["version"]] for r in self.rows])

    def all_jobs(self):
        self.loads += 1
        return list(self.rows)

    def current(self, keys):
        self.read_sizes.append(len(keys))
        return {r["job_key"]: r for r in self.rows if r["job_key"] in keys}


def test_bm25_postings_match_full_document_formula(job_factory):
    rng = np.random.default_rng(37)
    words = ["Python", "Java", "MySQL", "Redis", "开发", "测试", "接口"]
    rows = [job_factory(str(i), jd=" ".join(rng.choice(words, size=i % 13 + 1))) for i in range(45)]
    model = retrieval.BM25(rows)
    counts = [Counter(retrieval.tokenize(j["title"] + "\n" + j["title"] + "\n" + j["jd_text"])) for j in rows]
    for query in ("Python MySQL", "Redis测试", "Java开发", "完全未知 xxxxx", ""):
        expected = np.zeros(len(rows))
        for term in set(retrieval.tokenize(query)):
            tf = np.asarray([c[term] for c in counts])
            df = sum(term in c for c in counts)
            idf = math.log(1 + (len(rows) - df + 0.5) / (df + 0.5))
            expected += idf * tf * 2.5 / (tf + 1.5 * (0.25 + 0.75 * model.lengths / model.average))
        actual = model.scores(query)
        np.testing.assert_allclose(actual, expected, atol=1e-12)
        assert np.argsort(-actual).tolist() == np.argsort(-expected).tolist()


def test_snapshot_reuses_corpus_and_rechecks_only_candidates(job_factory):
    store = MemoryStore([job_factory(str(i)) for i in range(40)])
    retriever = retrieval.Retriever(Settings(), store, object())
    request = SearchRequest(query="Python", mode="bm25", rerank=False, generate=False, cities=["深圳"])
    with ThreadPoolExecutor(max_workers=8) as pool:
        answers = list(pool.map(lambda _: retriever.search(request), range(16)))
    assert store.loads == 1 and all(len(r["results"]) == 5 for r in answers)
    assert max(store.read_sizes) == 5
    answers[0]["results"][0]["hard_condition_checks"][0]["satisfied"] = False
    answers[0]["results"][0]["job"]["jd_text"] = "caller must not change the shared corpus"
    assert retriever.search(request)["results"][0]["hard_condition_checks"][0]["satisfied"]
    assert (
        retriever.search(request)["results"][0]["job"]["jd_text"]
        != answers[0]["results"][0]["job"]["jd_text"]
    )


def test_condition_cache_respects_revision_conditions_and_personal_scope(job_factory):
    original = job_factory("one")
    store = MemoryStore([original])
    retriever = retrieval.Retriever(Settings(), store, object())
    request = SearchRequest(query="Python", mode="bm25", rerank=False, generate=False, skills_all=["Java"])
    assert retriever.search(request)["results"] == []
    store.rows = [job_factory("one", jd="Python Java接口开发", version=2)]
    result = retriever.search(request)
    assert result["results"][0]["job"]["version"] == 2 and store.loads == 2
    assert retriever.search(request, eligible_keys=[])["results"] == []
    assert retriever.search(request.model_copy(update={"cities": ["广州"]}))["results"] == []
    store.rows = []
    assert retriever.search(request)["results"] == []


def test_old_inflight_snapshot_cannot_expose_updated_or_removed_job(job_factory):
    store = MemoryStore([job_factory("old")])
    retriever = retrieval.Retriever(Settings(), store, object())
    reached, resume = threading.Event(), threading.Event()
    original_current = store.current

    def current(keys):
        if keys == ["fixture:old"]:
            reached.set()
            assert resume.wait(3)
        return original_current(keys)

    store.current = current
    request = SearchRequest(query="Python", mode="bm25", rerank=False, generate=False)
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = pool.submit(retriever.search, request)
        assert reached.wait(3)
        store.rows = [job_factory("new")]
        new = retriever.search(request)
        resume.set()
        old = pending.result(timeout=3)
    assert old["results"] == []
    assert new["results"][0]["job"]["job_key"] == "fixture:new"


def test_service_cold_searches_overlap_and_share_one_milvus_client(job_factory, monkeypatch):
    job = job_factory("one")
    store = MemoryStore([job])
    config = Settings()
    barrier = threading.Barrier(4)
    creations = []

    class Client:
        def search(self, **kwargs):
            barrier.wait(timeout=3)
            return [[dict(entity={k: job[k] for k in ("job_key", "content_hash", "version")}, distance=1.0)]]

    def create(config):
        creations.append(1)
        return Client()

    monkeypatch.setattr(retrieval, "milvus", create)
    model = SimpleNamespace(encode=lambda *a, **kw: np.array([[1.0, 0.0]]))
    retriever = retrieval.Retriever(
        config,
        store,
        model,
        index_manifest=dict(
            snapshot=store.corpus_revision(),
            collection="fixed",
            embedding=config.config["embedding"],
            retrieval=config.config["retrieval"],
        ),
    )
    service = object.__new__(RagService)
    service.settings, service.store, service.retriever = config, store, retriever
    service.limit, service.start_lock = threading.BoundedSemaphore(4), threading.Lock()
    service.redis = SimpleNamespace(get=lambda *a: (_ for _ in ()).throw(redis.ConnectionError()))
    service.generator = SimpleNamespace(
        answer=lambda *a: dict(answer="fixed", recommendations=[], generation={"mode": "fixed"})
    )
    request = SearchRequest(query="Python", rerank=False, generate=False)
    with ThreadPoolExecutor(max_workers=4) as pool:
        answers = list(pool.map(lambda _: service.query(request, persist=False), range(4)))
    assert creations == [1] and all(not r["degradations"] for r in answers)


def test_capacity_rejection_remains_503_with_retry_hint():
    from jobcompass import web

    service = object.__new__(RagService)
    service.limit = SimpleNamespace(acquire=lambda **kw: False)
    service.settings = SimpleNamespace(prefix="unit:")
    service.redis = SimpleNamespace(eval=lambda *a: 1)
    previous = dict(web.app.dependency_overrides)
    web.app.dependency_overrides[web.service] = lambda: service
    web.app.dependency_overrides[web.current_user] = lambda: {"id": "unit", "profile": {"text": ""}}
    try:
        response = TestClient(web.app).post("/api/search", json={"query": "Python"})
        assert response.status_code == 503 and response.headers["retry-after"] == "2"
        assert response.headers["x-request-id"]
    finally:
        web.app.dependency_overrides.clear()
        web.app.dependency_overrides.update(previous)


def test_corpus_change_between_cache_key_and_retrieval_does_not_fill_wrong_key(job_factory):
    store = MemoryStore([job_factory("old")])
    config = Settings()
    retriever = retrieval.Retriever(config, store, object())
    original_revision = store.corpus_revision
    calls = []

    def revision():
        value = original_revision()
        if not calls:
            store.rows = [job_factory("new")]
        calls.append(value)
        return value

    store.corpus_revision = revision
    writes = []
    service = object.__new__(RagService)
    service.settings, service.store, service.retriever = config, store, retriever
    service.limit, service.start_lock = threading.BoundedSemaphore(4), threading.Lock()
    service.redis = SimpleNamespace(get=lambda *a: None, set=lambda *a, **kw: writes.append(a))
    service.generator = SimpleNamespace(
        answer=lambda *a: dict(answer="fixed", recommendations=[], generation={"mode": "fixed"})
    )
    response = service.query(
        SearchRequest(query="Python", mode="bm25", rerank=False, generate=False), persist=False
    )
    assert response["results"][0]["job"]["job_key"] == "fixture:new"
    assert response["corpus_revision"] != calls[0] and writes == []


def test_filter_cache_has_fixed_capacity(job_factory):
    store = MemoryStore([job_factory()])
    retriever = retrieval.Retriever(Settings(), store, object())
    for i in range(50):
        retriever.eligibility(
            retriever.revision,
            retriever.by_key,
            dict(cities=["fixture" + str(i)], skills_all=[], employment_types=[]),
        )
    assert len(retriever.filter_cache) == 32
