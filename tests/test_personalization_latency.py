"""同候选身份、原文压缩与流式计时/费用账本；模型全部固定响应。"""

import asyncio
import copy
import json

import httpx
import pytest

from jobcompass import personalization_latency as ex


@pytest.fixture
def job():
    return dict(
        job_id="a",
        title="业务开发",
        company="示例",
        lines=[
            dict(id="L1", text="企业介绍与团建福利。" * 55),
            dict(id="L2", text="负责Python业务接口开发，维护MySQL数据库事务。"),
            dict(id="L3", text="本科及以上学历，必须具备三年经验，不接受远程。"),
            dict(id="L4", text="员工福利和下午茶。" * 20),
        ],
    )


def test_compression_keeps_task_and_qualification_without_rewriting(job):
    original = copy.deepcopy(job)
    facts = [dict(id="p1", kind="project", text="Python接口与数据库事务")]
    value = ex.compress_job(job, "Python后端", facts)
    ex.check_spans(job, value)
    texts = "".join(x["text"] for x in value["lines"])
    assert "Python业务接口" in texts and "不接受远程" in texts
    assert sum(len(x["text"]) for x in value["lines"]) <= ex.MAX_CHARS
    assert value == ex.compress_job(job, "Python后端", facts)
    assert job == original


def test_short_source_is_retained_whole_and_unicode_offsets_roundtrip():
    job = dict(job_id="a", lines=[dict(id="L1", text="😀护理实习，不接受无证人员。")])
    assert ex.compress_job(job, "护理", []) == job
    job["lines"][0]["text"] *= 40
    compressed = ex.compress_job(job, "护理", [])
    ex.check_spans(job, compressed)


@pytest.mark.parametrize("kind", ["rewrite", "out_of_range", "outside_line", "changed_metadata"])
def test_source_mapping_rejects_invalid_spans(job, kind):
    compressed = ex.compress_job(job, "Python", [])
    if kind == "rewrite":
        compressed["lines"][0]["text"] += "虚构"
    elif kind == "out_of_range":
        compressed["lines"][0]["id"] = "L2:0-99999"
    elif kind == "outside_line":
        compressed["lines"][0]["id"] = "L999:0-1"
    else:
        compressed["job_id"] = "invented"
    with pytest.raises(ValueError):
        ex.check_spans(job, compressed)


@pytest.mark.parametrize(
    "value",
    [
        {"ranking": ["a", "a"]},
        {"ranking": ["a"]},
        {"ranking": ["a", "outside"]},
        {"ranking": ["a", 2]},
        {"ranking": ["a", "b"], "explanations": []},
    ],
)
def test_rank_only_rejects_changed_candidate_set_or_extra_fields(value):
    with pytest.raises(ValueError):
        ex.valid_output(value, dict(candidate_ids=["a", "b"]), "full_rank")


def test_rank_only_accepts_full_permutation():
    value = dict(ranking=["b", "a"])
    assert ex.valid_output(value, dict(candidate_ids=["a", "b"]), "compact_rank") == value


def test_sse_first_content_and_usage_are_distinct_from_first_empty_frame():
    result = ex.StreamResult()
    result.feed(json.dumps(dict(id="fixed", choices=[dict(delta=dict(role="assistant", content=""))])), 0.1)
    result.feed(json.dumps(dict(choices=[dict(delta=dict(content='{"ranking":'))])), 0.3)
    result.feed(json.dumps(dict(choices=[dict(delta=dict(content='["a"]}'), finish_reason="stop")])), 0.7)
    result.feed(json.dumps(dict(choices=[], usage=dict(prompt_tokens=10, completion_tokens=8))), 0.8)
    result.feed("[DONE]", 0.9)
    assert result.value() == dict(ranking=["a"])
    assert result.first_content_seconds == 0.3 and result.last_content_seconds == 0.7
    assert result.usage["input_tokens"] == 10 and result.usage["output_tokens"] == 8


@pytest.mark.parametrize("done,reason", [(False, "stop"), (True, "length"), (True, None)])
def test_sse_incomplete_or_truncated_output_rejected(done, reason):
    result = ex.StreamResult()
    result.content = ['{"ranking":["a"]}']
    result.done = done
    result.finish_reason = reason
    with pytest.raises(ValueError):
        result.value()


@pytest.fixture
def settings(tmp_path, monkeypatch):
    class FakeSettings:
        api_config = dict(
            API_BASE="https://example.invalid/v1",
            API_TYPE="openai_chat_completions",
            API_KEY_ENV="FIXED_TEST_ONLY",
        )
        values = {}

    monkeypatch.setattr(ex, "Settings", FakeSettings)
    monkeypatch.setattr(ex, "api_key", lambda *args: "fixed-test-only")
    ex.base.save(tmp_path / "prices.json", dict(models={ex.base.MODELS["ranker"]: dict(input=4, output=6)}))
    return tmp_path


@pytest.mark.asyncio
async def test_stream_request_records_timing_usage_and_payload(settings):
    observed = {}

    def handler(request):
        observed.update(json.loads(request.content))
        frames = [
            dict(id="fixed", choices=[dict(delta=dict(content='{"ranking":["a"]}'), finish_reason="stop")]),
            dict(choices=[], usage=dict(prompt_tokens=20, completion_tokens=10)),
        ]
        body = "".join("data: " + json.dumps(x) + "\n\n" for x in frames) + "data: [DONE]\n\n"
        return httpx.Response(200, content=body.encode(), headers={"content-type": "text/event-stream"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        value, ident = await ex.request(
            client, settings, "rank:test:full_rank", "test", {}, dict(budget=ex.budget())
        )
    record = ex.base.all_calls(settings)[0]
    assert value == dict(ranking=["a"]) and record["call_id"] == ident
    assert record["cost"]["cny"] == pytest.approx(0.00014)
    assert record["headers_seconds"] <= record["first_content_seconds"] <= record["seconds"]
    assert observed["stream"] and observed["stream_options"]["include_usage"]
    assert not observed["enable_thinking"] and observed["temperature"] == 0
    assert "fixed-test-only" not in json.dumps(record)


class BrokenStream(httpx.AsyncByteStream):
    def __init__(self, cancelled=False):
        self.cancelled = cancelled

    async def __aiter__(self):
        yield b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'
        if self.cancelled:
            raise asyncio.CancelledError
        raise httpx.ReadTimeout("fixed timeout")


@pytest.mark.asyncio
@pytest.mark.parametrize("cancelled", [False, True])
async def test_partial_timeout_or_cancellation_preserves_unknown_cost(settings, cancelled):
    def handler(request):
        return httpx.Response(200, stream=BrokenStream(cancelled))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        if cancelled:
            with pytest.raises(asyncio.CancelledError):
                await ex.request(
                    client, settings, "rank:test:full_rank", "test", {}, dict(budget=ex.budget())
                )
        else:
            value, _ = await ex.request(
                client, settings, "rank:test:full_rank", "test", {}, dict(budget=ex.budget())
            )
            assert value is None
    record = ex.base.all_calls(settings)[0]
    assert record["raw_text"] == "partial" and record["first_content_seconds"] is not None
    assert record["cost"]["cny"] is None
    assert record["status"] == ("cancelled" if cancelled else "error")


def test_freeze_rejects_changed_labels_before_paid_run(tmp_path, monkeypatch):
    monkeypatch.setattr(ex, "code_hashes", lambda: {"code": "fixed"})
    ex.base.save(tmp_path / "labels.json", [dict(fit=2)])
    ex.base.save(
        tmp_path / "plan.json",
        dict(
            code_hashes={"code": "fixed"},
            source_hashes={},
            inputs={"labels.json": ex.base.sha((tmp_path / "labels.json").read_bytes())},
        ),
    )
    assert ex.frozen(tmp_path)
    ex.base.save(tmp_path / "labels.json", [dict(fit=0)])
    with pytest.raises(ValueError, match="输入改变"):
        ex.frozen(tmp_path)
