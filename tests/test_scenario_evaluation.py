import asyncio
import json
from types import SimpleNamespace

import pytest

from jobcompass.experiments import load_scenario_queries
from jobcompass.preprocessing import digest
from jobcompass.scenario_judging import payload_items, validate


def freeze(directory, records):
    data = json.dumps(records, ensure_ascii=False).encode()
    (directory / "queries.json").write_bytes(data)
    (directory / "manifest.json").write_text(
        json.dumps({"queries": len(records), "hashes": {"queries.json": digest(data)}}), encoding="utf-8"
    )


def scenario(qid="q1", split="test", group="one"):
    return dict(
        query_id=qid,
        split=split,
        group=group,
        kind="assistant_scenario",
        anchor_job=None,
        query="Python开发，不要纯销售",
        request={},
        hard_constraints_reference=["排除纯销售"],
        soft_preferences_reference=["Python"],
    )


def test_frozen_scenarios_preserve_input_and_reject_tampering_or_split_leak(tmp_path):
    records = [scenario("dev", "dev", "a"), scenario("test", "test", "b")]
    freeze(tmp_path, records)
    loaded, _ = load_scenario_queries(tmp_path)
    assert loaded == records and loaded[0]["request"] == {}
    # 作者参考不能自动变成生产过滤器；这会掩盖条件解析缺陷。
    (tmp_path / "queries.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="指纹"):
        load_scenario_queries(tmp_path)
    records[1]["group"] = "a"
    freeze(tmp_path, records)
    with pytest.raises(ValueError, match="跨开发测试"):
        load_scenario_queries(tmp_path)


def candidate():
    return dict(
        pair_id="p1",
        query_id="q1",
        query=scenario(),
        job=dict(
            job_key="j1",
            content_hash="hash",
            title="开发",
            jd_text="岗位职责\n负责Python接口开发",
            profile="PRIVATE_SENTINEL",
        ),
        variant="HIDDEN_METHOD",
        ranking=1,
    )


def test_judge_payload_hides_methods_and_private_fields_and_validates_evidence():
    batch = [candidate()]
    payload = json.dumps(payload_items(batch), ensure_ascii=False)
    assert "PRIVATE_SENTINEL" not in payload and "HIDDEN_METHOD" not in payload
    item = dict(
        pair_id="p1",
        direction_grade=2,
        hard_status="satisfied",
        grade=2,
        reason="明确Python接口开发职责",
        lines=[2],
    )
    assert validate({"items": [item]}, batch) == [item]
    for change in ({"lines": [99]}, {"grade": 2, "hard_status": "violated"}, {"pair_id": "invented"}):
        with pytest.raises(ValueError):
            validate({"items": [item | change]}, batch)


def test_failed_model_judge_stays_unjudged_and_cache_does_not_rebill(tmp_path, monkeypatch):
    from jobcompass import scenario_judging

    (tmp_path / "summary.json").write_text('{"status":"complete"}', encoding="utf-8")
    (tmp_path / "queries.json").write_text(json.dumps([scenario()]), encoding="utf-8")
    job = candidate()["job"]
    (tmp_path / "corpus.jsonl").write_text(json.dumps(job) + "\n", encoding="utf-8")
    (tmp_path / "per_query.jsonl").write_text(
        json.dumps({"query_id": "q1", "split": "test", "ranking": ["j1"]}) + "\n", encoding="utf-8"
    )
    monkeypatch.setenv("SCENARIO_UNIT_KEY", "fake-unit-key")
    monkeypatch.setattr(
        scenario_judging,
        "Settings",
        lambda: SimpleNamespace(
            api_config={
                "API_KEY_ENV": "SCENARIO_UNIT_KEY",
                "API_BASE": "https://unit.invalid/v1",
                "MODEL": "fixed",
                "REASONING_EFFORT": "",
            }
        ),
    )
    calls = []

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def post(self, *args, **kwargs):
            calls.append(kwargs["json"])
            return SimpleNamespace(
                raise_for_status=lambda: None,
                json=lambda: {"output_text": '{"items":[]}', "usage": {"total_tokens": 10}},
            )

    monkeypatch.setattr(scenario_judging.httpx, "AsyncClient", FakeClient)
    result = asyncio.run(scenario_judging.judge(tmp_path))
    assert result["status"] == "partial_unjudged" and result["labeled"] == 0
    assert json.loads((tmp_path / "judge/checkpoint.json").read_bytes())["unjudged_pairs"] == 1
    assert (tmp_path / "judge/annotations.jsonl").read_text(encoding="utf-8") == ""
    assert asyncio.run(scenario_judging.judge(tmp_path))["labeled"] == 0 and len(calls) == 1


def test_partial_judge_preserves_valid_items_without_accepting_bad_lines():
    from jobcompass.scenario_judging import validate_partial

    first = candidate()
    second = candidate() | {"pair_id": "p2"}
    good = dict(
        pair_id="p1",
        direction_grade=2,
        hard_status="satisfied",
        grade=2,
        reason="明确Python接口开发职责",
        lines=[2],
    )
    bad = good | {"pair_id": "p2", "lines": [99]}
    valid, errors = validate_partial({"items": [good, bad]}, [first, second])
    assert valid == [good] and errors == {"p2": "证据行号不合法"}
    valid, errors = validate_partial({"items": [good, good]}, [first, second])
    assert valid == [] and errors["p1"] == "重复ID" and errors["p2"] == "遗漏ID"


def test_offline_recovery_keeps_direction_but_not_inconsistent_strict_grade(tmp_path):
    from jobcompass.scenario_judging import make_batches
    from jobcompass.scenario_label_cache import recover

    (tmp_path / "queries.json").write_text(json.dumps([scenario()]), encoding="utf-8")
    jobs = [
        candidate()["job"] | {"job_key": key, "jd_text": "职责\nPython接口\nSQL\n测试"}
        for key in ["j1", "j2"]
    ]
    (tmp_path / "corpus.jsonl").write_text("".join(json.dumps(j) + "\n" for j in jobs), encoding="utf-8")
    (tmp_path / "per_query.jsonl").write_text(
        json.dumps({"query_id": "q1", "split": "test", "ranking": ["j1", "j2"]}) + "\n", encoding="utf-8"
    )
    pairs = {p["job"]["job_key"]: p for b in make_batches(tmp_path) for p in b}
    first = dict(
        pair_id=pairs["j1"]["pair_id"],
        direction_grade=2,
        hard_status="unknown",
        grade=2,
        reason="正文方向相符，但严格匹配字段矛盾",
        lines=[1, 2, 3, 4],
    )
    second = first | {"pair_id": pairs["j2"]["pair_id"], "lines": [99]}
    out = tmp_path / "judge"
    out.mkdir()
    (out / "annotations.jsonl").write_text("", encoding="utf-8")
    (out / "checkpoint.json").write_text(
        json.dumps(
            {
                "status": "partial_unjudged",
                "model": "fixed",
                "calls": [{"batch_id": "batch", "raw_output": json.dumps({"items": [first, second]})}],
            }
        ),
        encoding="utf-8",
    )
    result = recover(tmp_path)
    rows = [
        json.loads(s) for s in (out / "direction_annotations.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert (
        result["direction_labeled"] == 1
        and result["direction_unjudged"] == 1
        and result["external_calls"] == 0
    )
    assert rows[0]["direction_grade"] == 2 and rows[0]["grade"] is None and not rows[0]["strict_usable"]
    assert rows[0]["evidence"][3] == {"line": 4, "text": "测试"}
    assert (out / "annotations.jsonl").read_text(encoding="utf-8") == ""
