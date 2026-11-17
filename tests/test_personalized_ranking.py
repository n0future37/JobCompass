"""正式排序协议：只接受同池全排列，测试不调用真实模型。"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import logging
import threading
from types import SimpleNamespace

import httpx
import pytest

from jobcompass.career import Background
from jobcompass.personalized_ranking import PersonalizedRanker


@pytest.fixture
def settings():
    return SimpleNamespace(
        config={"personalization": {}},
        values={"RANK_UNIT_KEY": "fixed-test-value"},
        api_config=dict(
            API_BASE="https://example.invalid/v1",
            API_TYPE="openai_chat_completions",
            API_KEY_ENV="RANK_UNIT_KEY",
            MODEL="Pro/deepseek-ai/DeepSeek-V3.2",
        ),
    )


@pytest.fixture
def rows(job_factory):
    return [
        dict(
            job=job_factory(str(i), title=f"岗位{i}"),
            rule_score=3 - i,
            personal_signals=dict(
                preference="like" if i == 1 else "neutral",
                application_status=None,
                note="PRIVATE_NOTE",
                owner="PRIVATE_OWNER",
            ),
        )
        for i in range(3)
    ]


def response(value):
    return httpx.Response(
        200,
        json=dict(
            choices=[dict(finish_reason="stop", message=dict(content=json.dumps(value)))],
            usage=dict(prompt_tokens=100, completion_tokens=30, prompt_cache_hit_tokens=0),
        ),
    )


def identifiers(rows):
    return [hashlib.sha256(r["job"]["job_key"].encode()).hexdigest()[:12] for r in rows]


def test_success_reads_full_jd_only_safe_profile_fields_and_no_reasons(settings, rows, caplog):
    rows[0]["job"]["jd_text"] = (
        "完整原文；" * 100 + "最后一个条件不得丢失。邮箱 me@example.org 电话13812345678"
    )
    rows[0]["job"]["provenance"] = [{"raw_file": "PRIVATE_PATH"}]
    captured = {}

    def handler(req):
        captured.update(json.loads(req.content))
        return response(dict(ranking=list(reversed(identifiers(rows)))))

    ranker = PersonalizedRanker(settings, transport=httpx.MockTransport(handler))
    with caplog.at_level(logging.INFO, logger="jobcompass.ranking"):
        ordered, meta = ranker.rank(
            rows,
            Background(skills=["Python"], projects="项目 me@example.org 13812345678"),
            "找开发",
            request_id="r1",
        )
    assert [r["job"]["job_key"] for r in ordered] == list(reversed([r["job"]["job_key"] for r in rows]))
    assert meta["mode"] == "llm" and meta["calls"] == 1 and meta["cost"]["cny"] > 0
    payload = json.dumps(captured, ensure_ascii=False)
    assert "最后一个条件不得丢失" in payload and "完整原文" in payload
    assert all(
        s not in payload
        for s in ("PRIVATE_NOTE", "PRIVATE_OWNER", "PRIVATE_PATH", "13812345678", "me@example.org")
    )
    assert captured["enable_thinking"] is False and captured["temperature"] == 0
    assert (
        "完整原文" not in caplog.text and "项目" not in caplog.text and "fixed-test-value" not in caplog.text
    )


@pytest.mark.parametrize("bad", ["duplicate", "missing", "outside", "type", "extra", "malformed"])
def test_invalid_output_falls_back_exactly_to_rule_order(settings, rows, bad):
    ids = identifiers(rows)
    value = {"ranking": ids}
    if bad == "duplicate":
        value["ranking"] = [ids[0]] * 3
    if bad == "missing":
        value["ranking"] = ids[:2]
    if bad == "outside":
        value["ranking"] = ids[:2] + ["not_a_candidate"]
    if bad == "type":
        value["ranking"] = ids[:2] + [True]
    if bad == "extra":
        value["reasons"] = ["不接受多余字段"]
    if bad == "malformed":
        value = "not-an-object"
    ranker = PersonalizedRanker(settings, transport=httpx.MockTransport(lambda req: response(value)))
    ordered, meta = ranker.rank(rows, Background(), "岗位")
    assert (
        ordered == rows and meta["mode"] == "fallback_rules" and meta["fallback_reason"] == "invalid_output"
    )
    assert meta["usage"]["input_tokens"] == 100 and meta["cost"]["cny"] is not None


def test_total_timeout_releases_slot_and_unknown_usage_is_not_free(settings, rows):
    settings.config["personalization"]["timeout_seconds"] = 0.01

    async def handler(req):
        await asyncio.sleep(2)
        return response(dict(ranking=identifiers(rows)))

    ranker = PersonalizedRanker(settings, transport=httpx.MockTransport(handler))
    ordered, meta = ranker.rank(rows, Background(), "岗位")
    assert ordered == rows and meta["fallback_reason"] == "timeout" and meta["calls"] == 1
    assert meta["cost"]["cny"] is None
    assert ranker.limit.acquire(blocking=False)
    ranker.limit.release()


@pytest.mark.parametrize("reason", ["rules", "oversized", "single", "empty", "not_configured"])
def test_skips_paid_call_when_not_needed_or_not_safe(settings, rows, reason):
    def handler(req):
        raise AssertionError("不应请求模型")

    strategy = "rules" if reason == "rules" else "llm"
    if reason == "oversized":
        settings.config["personalization"]["max_input_chars"] = 2
    if reason == "single":
        rows = rows[:1]
    if reason == "empty":
        rows = []
    if reason == "not_configured":
        settings.api_config["API_KEY_ENV"] = ""
    ranker = PersonalizedRanker(settings, transport=httpx.MockTransport(handler))
    _, meta = ranker.rank(rows, Background(), "岗位", strategy=strategy)
    assert meta["calls"] == 0 and meta["cost"]["cny"] == 0


def test_busy_model_returns_rule_result_without_another_request(settings, rows):
    settings.config["personalization"]["max_concurrency"] = 1
    entered, release = threading.Event(), threading.Event()

    async def handler(req):
        entered.set()
        await asyncio.to_thread(release.wait, 2)
        return response(dict(ranking=identifiers(rows)))

    ranker = PersonalizedRanker(settings, transport=httpx.MockTransport(handler))
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(ranker.rank, rows, Background(), "岗位")
        assert entered.wait(2)
        _, meta = ranker.rank(rows, Background(), "岗位")
        assert meta["fallback_reason"] == "busy" and meta["calls"] == 0
        release.set()
        assert future.result()[1]["mode"] == "llm"
