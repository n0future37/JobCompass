import json

from jobcompass.intent import HardConditions
from jobcompass.scenario_judging import make_batches
from jobcompass.stage_ablation import injected_intent, reference_check, summarize


def conditions(**values):
    return HardConditions(**values).model_dump()


def test_reference_checker_handles_unknown_and_permitted_city():
    c = conditions(cities=["北京", "上海"], excluded_cities=["北京"], employment_types=["全职"])
    assert reference_check(dict(cities=["北京市", "上海市"], employment_type_text="全职"), c) == "satisfied"
    assert reference_check(dict(cities=["北京"], employment_type_text="全职"), c) == "violated"
    assert reference_check(dict(cities=None, employment_type_text="全职"), c) == "unknown"
    assert reference_check(dict(cities=["上海"], employment_type_text="面议"), c) == "unknown"


def test_reference_checker_jd_mentions_are_bounded_and_no_private_skills():
    c = conditions(skills_all=["Python"])
    assert reference_check(dict(jd_text="负责Python接口开发"), c) == "satisfied"
    assert reference_check(dict(jd_text="无需Python经验", title="Python开发"), c) == "unknown"
    assert reference_check(dict(jd_text="MyPythonTool开发"), c) == "unknown"


def test_frozen_parser_diagnostic_keeps_refusal_and_old_output():
    row = dict(expected_conditions=conditions(cities=["北京"]), expected_status="conflict")
    old = dict(conditions={"cities": ["上海"]}, unresolved=[])
    assert injected_intent(row, old, "reference").status == "conflict"
    assert injected_intent(row, old, "legacy").hard.cities == ["上海"]


def test_judging_skips_directionless_and_unsupported_diagnostics(tmp_path):
    (tmp_path / "queries.json").write_text(json.dumps([{"query_id": "q"}]))
    (tmp_path / "corpus.jsonl").write_text(json.dumps({"job_key": "j", "content_hash": "h"}))
    rows = [
        dict(query_id="q", split="test", ranking=["j"], quality_evaluation=False),
        dict(query_id="q", split="dev", ranking=["j"], quality_evaluation=True),
    ]
    (tmp_path / "per_query.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    assert make_batches(tmp_path) == []


def test_unjudged_results_stay_unknown_in_stage_report(tmp_path):
    q = dict(query_id="q", expected_conditions=conditions())
    (tmp_path / "summary.json").write_text('{"status":"complete"}')
    (tmp_path / "queries.json").write_text(json.dumps([q]))
    (tmp_path / "corpus.jsonl").write_text(json.dumps({"job_key": "j", "content_hash": "h"}))
    row = dict(
        query_id="q",
        split="test",
        variant="hybrid",
        category="finance",
        ranking=["j"],
        quality_evaluation=True,
        trace={"candidate40": ["j"], "rerank_input15": ["j"]},
        reference_checks={"j": "satisfied"},
        cold_start=False,
        returned=1,
        total_ms=1,
        false_rejection=False,
        missed_rejection=False,
    )
    (tmp_path / "per_query.jsonl").write_text(json.dumps(row))
    result = summarize(tmp_path)
    hybrid = next(r for r in result["measures"] if r["variant"] == "hybrid")
    assert hybrid["constrained_p10"] is None
    assert hybrid["p10_lower"] == 0 and hybrid["p10_upper"] == 0.1
    assert hybrid["pool_recall10"] is None
