import math

import pytest

from jobcompass.conditions import SearchRequest
from jobcompass.evaluation import ranking_metrics
from jobcompass.retrieval import Retriever
from jobcompass.settings import Settings
from jobcompass.validation import summarize


def test_recall_ceiling_and_precision_explain_broad_query():
    labels = {str(n): 2 for n in range(100)}
    result = ranking_metrics([str(n) for n in range(10)], labels)
    assert result["recall"] == result["recall_ceiling"] == 0.1
    assert result["precision"] == result["ndcg"] == 1
    assert ranking_metrics(["0"], labels)["precision"] == 0.1


def test_unknown_labels_are_not_silently_irrelevant():
    result = ranking_metrics(["a", "unknown"], {"a": 2}, 2)
    assert result["unjudged"] == 1
    assert result["recall"] is result["precision"] is result["ndcg"] is None
    assert ranking_metrics([], {"a": 0})["recall"] is None
    with pytest.raises(ValueError, match="重复"):
        ranking_metrics(["a", "a"], {"a": 2})
    with pytest.raises(ValueError):
        ranking_metrics(["a"], {"a": 3})


def test_discounted_gain_uses_graded_relevance():
    result = ranking_metrics(["b", "a"], {"a": 2, "b": 1}, 2)
    expected = (1 + 3 / math.log2(3)) / (3 + 1 / math.log2(3))
    assert result["ndcg"] == pytest.approx(expected)
    assert result["precision"] == 0.5 and result["recall"] == 1


def test_trace_exposes_candidates_before_top_k_without_public_leak(job_factory):
    rows = [job_factory(str(n)) for n in range(12)]

    class Store:
        def all_jobs(self):
            return rows

        def current(self, keys):
            return {j["job_key"]: j for j in rows if j["job_key"] in keys}

    retriever = Retriever(Settings(), Store(), models=object())
    request = SearchRequest(query="Python", mode="bm25", rerank=False, generate=False, top_k=5)
    public = retriever.search(request)
    traced = retriever.search(request, include_trace=True)
    assert "evaluation_trace" not in public
    assert traced["results"] == public["results"]
    assert len(traced["evaluation_trace"]["before_rerank"]) == 12
    assert len(traced["results"]) == 5


def test_run_namespace_cannot_target_development_or_escape_prefix(monkeypatch):
    monkeypatch.setenv("JOBCOMPASS_RUN_NAMESPACE", "check_a")
    assert Settings("test").prefix == "jobcompass:test:rag:v1:check_a:"
    with pytest.raises(ValueError):
        Settings("dev")
    monkeypatch.setenv("JOBCOMPASS_RUN_NAMESPACE", "../../dev:*")
    with pytest.raises(ValueError):
        Settings("test")


def test_load_summary_separates_rejection_and_business_failure():
    result = summarize(
        [
            {"status": 200, "http_ms": 10, "cache": "hit"},
            {"status": 429, "http_ms": 1},
            {"status": 200, "http_ms": 30, "cache": "miss", "error": "bad_citation"},
        ]
    )
    assert result["status_counts"] == {"200": 2, "429": 1}
    assert result["validation_failures"] == 1
    assert result["p50_http_ms"] == 20
