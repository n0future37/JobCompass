"""新协议的原文独立性、等级分歧、硬字段和防混入机制。"""

import json

import pytest

from jobcompass import rag_comparison as comparison
from jobcompass import query_policy_review as review


def test_independent_judging_windows_retain_qualification_and_tail_without_production_guard(job_factory):
    job = job_factory(jd="任职要求：本科学历\n岗位职责：负责业务系统联调\n负责对能源行业AI智能体进行开发落地")
    fragments = comparison.independent_fragments(job, "AI智能体开发")
    assert any("智能体" in f["text"] for f in fragments)
    for f in fragments:
        assert len(f["text"]) <= 240
        assert (
            job["jd_text"].split("\n")[f["line"] - 1][f["start"] : f["start"] + len(f["text"])] == f["text"]
        )


def test_reference_field_check_does_not_reuse_production_qualification_function(job_factory, monkeypatch):
    import jobcompass.qualifications as qualifications

    monkeypatch.setattr(qualifications, "education_requirement", lambda *_: pytest.fail("circular evaluator"))
    j = job_factory(education_text=None, jd="本科及以上学历，负责测试平台开发")
    assert comparison.reference_status(j, dict(education_max="大专")) == "fail"
    assert comparison.reference_status(job_factory(), dict(education_max="大专")) == "unknown"


def test_three_grade_validation_never_accepts_invented_or_missing_evidence():
    pair = dict(pair_id="x", fragments=[dict(id="f1", line=1, text="负责测试平台开发")])
    item = dict(pair_id="x", grade=1, reason="部分工作方向相符", evidence=["f1"])
    assert comparison.validate_grades(dict(items=[item]), [pair]) == [item]
    for change in [dict(grade=True), dict(evidence=["f9"]), dict(evidence=[])]:
        with pytest.raises(ValueError):
            comparison.validate_grades(dict(items=[item | change]), [pair])


def test_old_review_tool_now_requires_zero_vs_one_adjudication(tmp_path, monkeypatch):
    monkeypatch.setattr(review, "freeze", lambda _: None)
    for name, items in {
        "pool_dev.jsonl": [dict(pair_id="x", query_id="q", job_key="j")],
        "runs.jsonl": [dict(dataset="main", split="dev", query_id="q", ranking=["j"])],
        "labels_dev.jsonl": [dict(pair_id="x", grade=1, source="model", model="a")],
        "second_labels_dev.jsonl": [dict(pair_id="x", grade=0, reason="无工作依据")],
    }.items():
        (tmp_path / name).write_text("".join(json.dumps(i) + "\n" for i in items), encoding="utf-8")
    with pytest.raises(ValueError):
        review.merge(tmp_path, "dev")
    assert len(review.disagreements(tmp_path, "dev")) == 1


def test_new_merge_all_grade_disagreements_need_explicit_reconsideration(tmp_path, monkeypatch):
    pool = [dict(pair_id="x", query_id="q", job_key="j")]
    monkeypatch.setattr(comparison, "make_pool", lambda *args: pool)
    for role, grade in [("primary", 1), ("secondary", 0)]:
        comparison.append(
            tmp_path / f"{role}_dev.jsonl",
            dict(pair_id="x", grade=grade, reason="不同方向判断", evidence=["f1"]),
        )
    with pytest.raises(ValueError):
        comparison.merge_grades(tmp_path, "dev")
    comparison.append(
        tmp_path / "adjudicator_dev.jsonl", dict(pair_id="x", grade=2, reason="原文直接开发", evidence=["f1"])
    )
    summary = comparison.merge_grades(tmp_path, "dev")
    label = comparison.rows(tmp_path / "labels_dev.jsonl")[0]
    assert summary["weak_0_1_disagreements"] == 1
    assert label["grade"] == 2 and label["source"] == "secondary_model_explicit_reconsideration"


def test_metrics_keep_pending_separate_and_freeze_dev_selection_before_test(
    tmp_path, monkeypatch, job_factory
):
    # 一条完整核验、一条只允许未知展示，不能共享同一个已核验精度。
    monkeypatch.setattr(comparison, "frozen", lambda _: dict(selection=dict(baseline="original")))
    queries = [
        dict(query_id=f"q{i}", category="testing", reference=dict(education_max="大专")) for i in range(32)
    ]
    (tmp_path / "queries.json").write_text(json.dumps(queries), encoding="utf-8")
    job = job_factory(education_text=None)
    comparison.append(tmp_path / "corpus.jsonl", job)
    pool = []
    for q in queries:
        pid = q["query_id"]
        pool.append(dict(pair_id=pid, query_id=pid, job_key=job["job_key"]))
        comparison.append(
            tmp_path / "labels_dev.jsonl", dict(pair_id=pid, query_id=pid, job_key=job["job_key"], grade=2)
        )
        for variant in comparison.VARIANTS:
            comparison.append(
                tmp_path / "runs.jsonl",
                dict(
                    query_id=pid,
                    category="testing",
                    split="dev",
                    variant=variant["name"],
                    intent=dict(status="ready"),
                    raw_ranking=[job["job_key"]],
                    ranking=[job["job_key"]],
                    recommendations=[],
                    timings=dict(transform_seconds=0, retrieval_seconds=0.1, guard_seconds=0.1),
                ),
            )
    monkeypatch.setattr(comparison, "make_pool", lambda *_: pool)
    (tmp_path / "label_summary_dev.json").write_text(json.dumps(dict(pairs=32)), encoding="utf-8")
    result = comparison.summarize(tmp_path, "dev")
    m = result["metrics"]["original"]
    assert m["final_direction_precision"] == pytest.approx(0.1)
    assert m["final_verified_precision"] == 0
    assert m["unknown_qualification_positions"] == 32 and m["known_hard_violations"] == 0
    assert json.loads((tmp_path / "dev_selection.json").read_bytes())["selected"] == "original"


def test_partial_response_validates_existing_items_without_inventing_missing_grade():
    batch = [dict(pair_id=p, fragments=[dict(id="f1", text="负责接口开发")]) for p in ["a", "b"]]
    value = dict(items=[dict(pair_id="a", grade=2, reason="直接负责接口", evidence=["f1"])])
    assert comparison.validate_partial_grades(value, batch) == value["items"]
    with pytest.raises(ValueError):
        comparison.validate_grades(value, batch)
    with pytest.raises(ValueError):
        comparison.validate_partial_grades(dict(items=value["items"] * 2), batch)


def test_grading_retries_only_missing_items_and_does_not_start_next_role_incomplete(tmp_path, monkeypatch):
    import asyncio
    from types import SimpleNamespace
    import httpx

    plan = dict(budget=dict(max_recorded_api_calls=30, max_known_cny=20, max_reported_tokens=100000))
    pool = [
        dict(
            pair_id=p,
            query_id="q",
            job_key=p,
            query="接口开发",
            title="后端",
            fragments=[dict(id="f1", text="负责接口开发")],
        )
        for p in ["a", "b"]
    ]
    monkeypatch.setattr(comparison, "frozen", lambda _: plan)
    monkeypatch.setattr(comparison, "make_pool", lambda *_: pool)
    config = dict(
        API_BASE="https://unit.invalid/v1",
        API_TYPE="openai_chat_completions",
        API_KEY_ENV="RAG_TEST_KEY",
        MODEL=comparison.JUDGES["primary"],
    )
    monkeypatch.setattr(
        comparison,
        "Settings",
        lambda: SimpleNamespace(api_config=config, values=dict(RAG_TEST_KEY="unit-key")),
    )
    original_client = httpx.AsyncClient
    seen = []

    def handle(request):
        body = json.loads(request.content)
        pairs = json.loads(body["messages"][1]["content"])["pairs"]
        seen.append((body["model"], [p["pair_id"] for p in pairs]))
        selected = pairs[:1] if len(seen) == 1 else pairs
        content = json.dumps(
            dict(
                items=[
                    dict(pair_id=p["pair_id"], grade=2, reason="直接负责接口开发", evidence=["f1"])
                    for p in selected
                ]
            )
        )
        return httpx.Response(
            200,
            json=dict(
                choices=[dict(finish_reason="stop", message=dict(content=content))],
                usage=dict(prompt_tokens=100, completion_tokens=30),
            ),
        )

    monkeypatch.setattr(
        comparison.httpx,
        "AsyncClient",
        lambda **kwargs: original_client(transport=httpx.MockTransport(handle), **kwargs),
    )
    summary = asyncio.run(comparison.judge_async(tmp_path, "dev"))
    assert [ids for _, ids in seen] == [["a", "b"], ["b"], ["a", "b"]]
    assert summary["pairs"] == 2 and summary["disagreements"] == 0
    assert [r["status"] for r in comparison.rows(tmp_path / "judge_calls.jsonl")] == ["partial", "ok", "ok"]


def test_explicit_zero_without_citations_is_only_a_format_normalization():
    batch = [dict(pair_id=p, fragments=[dict(id="f1", text="负责接口开发")]) for p in ["a", "b"]]
    result = comparison.validate_partial_grades(
        dict(
            items=[
                dict(pair_id="a", grade=0, reason="岗位职责方向不同"),
                dict(pair_id="b", grade=2, reason="疑似匹配但伪造引用", evidence=["f4"]),
            ]
        ),
        batch,
    )
    assert result == [dict(pair_id="a", grade=0, reason="岗位职责方向不同", evidence=[])]
    with pytest.raises(ValueError):
        comparison.validate_partial_grades(dict(items=[dict(pair_id="a", reason="缺少分数不能填零")]), batch)
