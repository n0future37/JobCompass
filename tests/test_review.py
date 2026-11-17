import json
from datetime import UTC, datetime

import pytest

from jobcompass import review
from jobcompass.conditions import SearchRequest, mention_lines, parse_conditions, profile_mentions
from jobcompass.preprocessing import digest


@pytest.mark.parametrize("operator", ["不要", "不得", "不能", "不接受"])
def test_negative_hard_clause_is_never_reversed(operator):
    conditions, unresolved = parse_conditions(SearchRequest(query=operator + "Java"))
    assert unresolved == [operator + "Java"]
    assert conditions["skills_all"] == []


def test_negation_stays_within_clause():
    job = {"jd_text": "不要求Python，但要求MySQL"}
    assert mention_lines(job, "Python") == []
    assert mention_lines(job, "MySQL") == [1]
    assert profile_mentions("不熟悉Java，熟悉Python") == ["Python"]


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    rows = [
        {"job_key": "a", "content_hash": "ha", "title": "Python开发", "jd_text": "使用Python开发接口"},
        {"job_key": "b", "content_hash": "hb", "title": "Java开发", "jd_text": "使用Java"},
    ]

    class Store:
        def all_jobs(self):
            return rows

        def close(self):
            pass

    monkeypatch.setattr(review, "Store", lambda settings: Store())
    source = tmp_path / "evaluation"
    source.mkdir()
    review.write(
        source / "summary.json", {"corpus_hash": digest([[r["job_key"], r["content_hash"]] for r in rows])}
    )
    review.write(source / "queries.json", [{"query_id": "D01", "query": "Python后端"}])
    review.write(
        source / "per_query.json",
        [
            {
                "query_id": "D01",
                "split": "dev",
                "method": "bm25",
                "ranking": ["a", "b"],
                "recommendations": [
                    {"job_key": "a", "citations": [{"line": 1, "quote": rows[0]["jd_text"]}]}
                ],
            },
            {
                "query_id": "D01",
                "split": "dev",
                "method": "dense",
                "ranking": ["b", "a"],
                "recommendations": [],
            },
        ],
    )
    output = tmp_path / "review"
    result = review.export_bundle(source, output, settings=object())
    assert result["relevance"] == 2 and result["citations"] == 1
    return output


def save_rows(bundle, rows):
    (bundle / "annotations.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )


def test_empty_annotations_cannot_be_counted_as_gold(bundle):
    result = review.review_report(bundle)
    assert result["completed"] == 0 and result["pending"] == 3
    assert not result["human_gold_declared_complete"]
    assert result["semantic_support_rate_on_reviewed"] is None
    assert all(row["precision_at_10"] is None for row in result["measures"])
    assert all(
        "method" not in r and "ranking" not in r for r in review.read_rows(bundle / "annotations.jsonl")
    )


def test_machine_annotations_stay_machine_and_pool_scoped(bundle):
    rows = review.read_rows(bundle / "annotations.jsonl")
    for row in rows:
        row.update(
            label=(2 if row["job_key"] == "a" else 0) if row["kind"] == "relevance" else "supported",
            reason="固定测试标签",
            annotator="pytest_fixture",
            annotation_kind="model",
            reviewed_at=datetime.now(UTC).isoformat(),
        )
    save_rows(bundle, rows)
    result = review.review_report(bundle)
    assert result["pending"] == 0 and result["annotation_kinds"] == ["model"]
    assert result["human_gold_declared_complete"] is False
    assert result["measures"][0]["precision_at_10"] == 0.1
    assert result["measures"][0]["pooled_recall_at_10"] == 1
    assert result["unsupported_statement_rate"] is None


def test_frozen_jd_cannot_be_changed_during_annotation(bundle):
    rows = review.read_rows(bundle / "annotations.jsonl")
    rows[0]["jd_text"] = "改写的岗位要求"
    save_rows(bundle, rows)
    with pytest.raises(ValueError, match="冻结内容"):
        review.review_report(bundle)


def test_missing_rows_cannot_pass_as_complete(bundle):
    rows = review.read_rows(bundle / "annotations.jsonl")
    save_rows(bundle, rows[:1])
    with pytest.raises(ValueError, match="缺项"):
        review.review_report(bundle)
