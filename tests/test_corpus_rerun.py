import json
from types import SimpleNamespace

import pytest

from jobcompass.corpus_rerun import compare, seed_labels
from jobcompass.preprocessing import digest


def fixture(directory, jobs, labels, query_count=1):
    directory.mkdir()
    queries = [
        dict(
            query_id=f"q{n}",
            query="接口开发",
            split="test",
            group=f"g{n}",
            kind="assistant_scenario",
            request={},
            anchor_job=None,
            hard_constraints_reference=[],
            soft_preferences_reference=[],
        )
        for n in range(query_count)
    ]
    (directory / "queries.json").write_text(json.dumps(queries), encoding="utf-8")
    rows = [
        dict(job_key=k, content_hash=digest(k.encode()), jd_text="原文证据", title="接口开发") for k in jobs
    ]
    (directory / "corpus.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    ranking = [
        dict(
            query_id=q["query_id"],
            split="test",
            variant="line700_o1__original__hybrid_rerank",
            ranking=jobs,
            eligible_jobs=len(jobs),
            total_ms=1,
            unresolved_conditions=[],
        )
        for q in queries
    ]
    (directory / "per_query.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in ranking), encoding="utf-8"
    )
    annotations = [
        dict(
            pair_id=digest([q["query_id"], k, digest(k.encode())])[:24],
            query_id=q["query_id"],
            job_key=k,
            content_hash=digest(k.encode()),
            direction_grade=2,
            grade=None,
            strict_usable=False,
            hard_status="unknown",
            reason="方向有据，严格条件未标注",
            lines=[1],
            evidence=[dict(line=1, text="原文证据")],
            annotation_kind="model",
        )
        for q in queries
        for k in labels
    ]
    (directory / "judge").mkdir()
    (directory / "judge/direction_annotations.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in annotations), encoding="utf-8"
    )
    (directory / "summary.json").write_text(
        json.dumps(dict(status="complete", corpus_jobs=len(jobs), transform_calls=[], transform_tokens=0)),
        encoding="utf-8",
    )
    (directory / "city_field_audit.json").write_text('{"items":[]}', encoding="utf-8")
    return annotations


def test_seed_reuses_only_exact_query_and_valid_jd_evidence(tmp_path, monkeypatch):
    from jobcompass import corpus_rerun

    monkeypatch.setattr(corpus_rerun, "Settings", lambda: SimpleNamespace(api_config={"MODEL": "fixed"}))
    old, new = tmp_path / "old", tmp_path / "new"
    fixture(old, ["a"], ["a"])
    fixture(new, ["a", "b"], [])
    result = seed_labels(new, old)
    assert result["seeded"] == 1 and result["new_pairs"] == 1
    seeded = json.loads((new / "judge/seeded_annotations.jsonl").read_text(encoding="utf-8"))
    assert seeded["annotation_kind"] == "model" and seeded["grade"] is None
    assert seed_labels(new, old) == result
    (new / "queries.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="查询不同"):
        seed_labels(new, old)


def test_comparison_uses_same_positive_pool_for_old_and_new(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    fixture(old, ["a"], ["a"], 60)
    fixture(new, ["a", "b"], ["a", "b"], 60)
    (new / "judge/checkpoint.json").write_text('{"calls":[],"reported_tokens":0}', encoding="utf-8")
    (new / "judge/reuse.json").write_text('{"seeded":60}', encoding="utf-8")
    result = compare(new, old)
    a, b = result["measures"]
    assert result["shared_label_pool_pairs"] == 120 and result["common_positive_queries"] == 60
    assert a["pooled_recall10"] == 0.5 and b["pooled_recall10"] == 1
    assert a["direction_p10"] == 0.1 and b["direction_p10"] == 0.2
    assert result["direction_complete"] and not result["strict_metrics_validated"]


def test_seeded_unknown_strict_grade_stays_unknown_in_recovery(tmp_path):
    from jobcompass.scenario_label_cache import recover

    fixture(tmp_path / "data", ["a"], ["a"])
    directory = tmp_path / "data"
    row = json.loads((directory / "judge/direction_annotations.jsonl").read_text(encoding="utf-8"))
    (directory / "judge/annotations.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    (directory / "judge/checkpoint.json").write_text(
        '{"status":"complete","model":"fixed","calls":[]}', encoding="utf-8"
    )
    result = recover(directory)
    assert result["strict_usable"] == 0 and result["direction_labeled"] == 1
