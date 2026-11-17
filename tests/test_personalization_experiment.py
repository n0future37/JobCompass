"""实验的身份、证据、缺项与费用账本约束，所有模型响应固定。"""

import asyncio
import copy

import httpx
import pytest

from jobcompass import personalization_experiment as pe


@pytest.fixture
def case():
    return dict(
        case_id="example",
        candidate_ids=["a", "b"],
        profile_facts=[
            dict(id="p1", kind="project", text="开发API"),
            dict(id="p2", kind="education", text="本科"),
        ],
        candidates=[
            dict(job_id="a", lines=[dict(id="L1", text="开发业务接口")]),
            dict(job_id="b", lines=[dict(id="L2", text="维护数据库")]),
        ],
    )


def ranked():
    return dict(
        ranking=["a", "b"],
        explanations=[
            dict(job_id="a", reason="有接口项目", profile_evidence=["p1"], jd_evidence=["L1"]),
            dict(job_id="b", reason="经验待确认", profile_evidence=[], jd_evidence=["L2"]),
        ],
    )


def graded(key="a"):
    return dict(
        job_id=key,
        direction=2,
        fit=2,
        reason="实际接口任务与项目相符",
        jd_evidence=["L1"],
        profile_evidence=["p1"],
    )


def test_rank_identity_and_empty_evidence_are_distinct(case):
    assert pe.valid_rank(ranked(), case)["ranking"] == ["a", "b"]


@pytest.mark.parametrize("ranking", [["a", "a"], ["a"], ["a", "outside"], ["a", ["b"]]])
def test_rank_rejects_duplicates_missing_and_fabricated_ids(case, ranking):
    value = ranked()
    value["ranking"] = ranking
    with pytest.raises(ValueError):
        pe.valid_rank(value, case)


@pytest.mark.parametrize("field,value", [("profile_evidence", ["invented"]), ("jd_evidence", ["L2"])])
def test_rank_requires_evidence_from_same_job_and_profile(case, field, value):
    output = ranked()
    output["explanations"][0][field] = value
    with pytest.raises(ValueError):
        pe.valid_rank(output, case)


def test_partial_judgment_keeps_valid_sibling_but_never_defaults_missing_grade(case):
    bad = graded("b")
    bad.pop("fit")
    result = pe.partial_grades(dict(items=[graded(), bad]), case, case["candidates"])
    assert [r["job_id"] for r in result] == ["a"]
    assert "b" not in {r["job_id"] for r in result}


def test_duplicate_judgments_rejected_even_if_one_is_valid(case):
    duplicate = copy.deepcopy(graded())
    assert pe.partial_grades(dict(items=[graded(), duplicate]), case, case["candidates"]) == []


def test_degree_alone_cannot_prove_personal_fit(case):
    item = graded()
    item["profile_evidence"] = ["p2"]
    assert not pe.valid_grade(item, case, case["candidates"])


def test_direction_mismatch_cannot_receive_high_personal_fit(case):
    item = graded()
    item["direction"] = 0
    assert not pe.valid_grade(item, case, case["candidates"])


def test_missing_explicit_zero_evidence_can_normalize_without_inventing_grade(case):
    item = dict(job_id="a", direction=0, fit=0, reason="工作方向不同")
    accepted = pe.partial_grades(dict(items=[item]), case, case["candidates"])
    assert len(accepted) == 1 and accepted[0]["jd_evidence"] == []
    assert pe.partial_grades(dict(items=[dict(job_id="b", reason="未评分")]), case, case["candidates"]) == []


def test_started_request_is_unknown_cost_not_free(tmp_path):
    pe.append(tmp_path / "requests_started.jsonl", dict(call_id="interrupted", phase="rank:a:0"))
    total = pe.totals(pe.all_calls(tmp_path))
    assert total["calls"] == 1 and total["unknown_cost_calls"] == 1
    pe.append(
        tmp_path / "calls.jsonl",
        dict(
            call_id="interrupted",
            phase="rank:a:0",
            status="ok",
            usage=dict(input_tokens=5, output_tokens=2),
            cost=dict(cny=0.01),
        ),
    )
    total = pe.totals(pe.all_calls(tmp_path))
    assert total["calls"] == 1 and total["unknown_cost_calls"] == 0 and total["reported_tokens"] == 7


@pytest.mark.asyncio
async def test_timeout_is_checkpointed_as_unknown_usage(tmp_path, monkeypatch):
    class FakeSettings:
        api_config = dict(
            API_BASE="https://example.invalid/v1",
            API_TYPE="openai_chat_completions",
            API_KEY_ENV="FAKE_TEST_KEY",
        )
        values = {}

    monkeypatch.setattr(pe, "Settings", FakeSettings)
    monkeypatch.setattr(pe, "api_key", lambda *args: "fixed-test-value")

    def handler(request):
        raise httpx.ReadTimeout("fixed failure")

    plan = dict(budget=dict(max_started_calls=10, max_known_cny=1, max_reported_tokens=100))
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        value, ident = await pe.call(client, tmp_path, plan, "rank:a:0", "fake", {}, "test", 20)
    assert value is None
    records = pe.all_calls(tmp_path)
    assert len(records) == 1 and records[0]["call_id"] == ident
    assert records[0]["status"] == "error"
    assert pe.totals(records)["unknown_cost_calls"] == 1


@pytest.mark.asyncio
async def test_cancellation_leaves_completed_audit_record(tmp_path, monkeypatch):
    class FakeSettings:
        api_config = dict(
            API_BASE="https://example.invalid/v1",
            API_TYPE="openai_chat_completions",
            API_KEY_ENV="FAKE_TEST_KEY",
        )
        values = {}

    monkeypatch.setattr(pe, "Settings", FakeSettings)
    monkeypatch.setattr(pe, "api_key", lambda *args: "fixed-test-value")

    async def handler(request):
        raise asyncio.CancelledError

    plan = dict(budget=dict(max_started_calls=10, max_known_cny=1, max_reported_tokens=100))
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(asyncio.CancelledError):
            await pe.call(client, tmp_path, plan, "rank:a:0", "fake", {}, "test", 20)
    record = pe.all_calls(tmp_path)[0]
    assert record["status"] == "cancelled" and record["cost"]["cny"] is None


def test_frozen_inputs_reject_change(tmp_path, monkeypatch):
    monkeypatch.setattr(pe, "code_hashes", lambda: {"fixed": "sha"})
    pe.save(tmp_path / "cases.json", [dict(query="before")])
    pe.save(
        tmp_path / "plan.json",
        dict(
            code_hashes={"fixed": "sha"},
            inputs={"cases.json": pe.sha((tmp_path / "cases.json").read_bytes())},
        ),
    )
    assert pe.frozen(tmp_path)["code_hashes"] == {"fixed": "sha"}
    pe.save(tmp_path / "cases.json", [dict(query="changed")])
    with pytest.raises(ValueError, match="冻结输入"):
        pe.frozen(tmp_path)
