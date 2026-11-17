"""硅基Chat协议的实际调用入口；固定模型响应，不访问外部服务。"""

import json
from types import SimpleNamespace

import httpx
import pytest

from jobcompass import generation, query_transform, scenario_judging
from jobcompass.conditions import SearchRequest
from jobcompass.evidence import EvidenceGuard
from jobcompass.generation import Generator
from jobcompass.model_api import usage_details


def chat_config():
    return dict(
        API_BASE="https://model.unit.invalid/v1",
        API_KEY_ENV="UNIT_SILICON_KEY",
        MODEL="unit-fixed",
        API_TYPE="openai_chat_completions",
        ENABLE_THINKING="false",
        TEMPERATURE="0",
        REASONING_EFFORT="low",
    )


def check_request(request):
    assert request.url.path == "/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer unit-only-key"
    payload = json.loads(request.content)
    assert payload["enable_thinking"] is False and payload["stream"] is False
    assert payload["temperature"] == 0 and payload["response_format"] == {"type": "json_object"}
    assert not {"input", "max_output_tokens", "reasoning", "reasoning_effort", "store"} & payload.keys()
    assert "PRIVATE_PROFILE_DO_NOT_SEND" not in request.content.decode()
    return json.loads(payload["messages"][1]["content"])


def reply(value, finish="stop"):
    return httpx.Response(
        200,
        json={
            "model": "unit-returned-model",
            "choices": [{"finish_reason": finish, "message": {"content": json.dumps(value)}}],
            "usage": {"prompt_tokens": 15, "completion_tokens": 5, "total_tokens": 20},
        },
    )


@pytest.mark.parametrize("case", ["valid", "truncated", "fabricated"])
def test_chat_evidence_is_verified_or_safely_falls_back(case, job_factory, monkeypatch):
    monkeypatch.setenv("UNIT_SILICON_KEY", "unit-only-key")
    generator = Generator(guard=EvidenceGuard(scorer=lambda q, p: [8] * len(p)))
    monkeypatch.setattr(generator, "_config", chat_config)
    calls = []

    def handle(request):
        body = check_request(request)
        pack = body["candidates"][0]
        citation = pack["lines"][0]
        quote = citation["text"] + ("不存在的职责" if case == "fabricated" else "")
        calls.append(body)
        return reply(
            {
                "items": [
                    {"job_key": pack["job_key"], "citations": [{"line": citation["line"], "quote": quote}]}
                ]
            },
            finish="length" if case == "truncated" else "stop",
        )

    original = httpx.Client
    monkeypatch.setattr(
        generation.httpx, "Client", lambda **kw: original(transport=httpx.MockTransport(handle), **kw)
    )
    job = job_factory(jd="岗位职责\n负责Python后端接口开发与维护。")
    answer = generator.answer(
        {"results": [{"job": job, "hard_condition_checks": []}]},
        SearchRequest(query="Python后端", profile="PRIVATE_PROFILE_DO_NOT_SEND", generate=True),
    )
    assert len(calls) == 1 and answer["generation"]["calls"] == 1
    assert answer["generation"]["mode"] == (
        "llm_evidence_selection" if case == "valid" else "extractive_fallback"
    )
    citation = answer["recommendations"][0]["citations"][0]
    assert citation["quote"] in job["jd_text"].split("\n")[citation["line"] - 1]
    if case == "valid":
        assert answer["generation"]["model"] == "unit-returned-model"
        assert answer["generation"]["usage"]["input_tokens"] == 15
        assert answer["generation"]["usage"]["output_tokens"] == 5


def test_chat_rewrite_hyde_and_cached_results(tmp_path, monkeypatch):
    monkeypatch.setenv("UNIT_SILICON_KEY", "unit-only-key")
    settings = SimpleNamespace(api_config=chat_config())
    calls = []

    def handle(request):
        items = check_request(request)
        calls.append(items)
        return reply({"items": [{"query_id": items[0]["query_id"], "text": "Python后端接口开发岗位"}]})

    original = httpx.Client
    monkeypatch.setattr(
        query_transform.httpx, "Client", lambda **kw: original(transport=httpx.MockTransport(handle), **kw)
    )
    queries = [
        {
            "query_id": "q1",
            "query": "Python后端",
            "kind": "scenario",
            "profile": "PRIVATE_PROFILE_DO_NOT_SEND",
        }
    ]
    result = query_transform.transform_queries(settings, queries, tmp_path)
    assert all(result["texts"][kind]["q1"] for kind in ("rewrite", "hyde"))
    assert result["reported_total_tokens"] == 40 and len(calls) == 2
    assert query_transform.transform_queries(settings, queries, tmp_path) == result
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_async_chat_judge_saves_labels_and_reuses_checkpoint(tmp_path, job_factory, monkeypatch):
    monkeypatch.setenv("UNIT_SILICON_KEY", "unit-only-key")
    monkeypatch.setattr(scenario_judging, "Settings", lambda: SimpleNamespace(api_config=chat_config()))
    job = job_factory()
    query = dict(
        query_id="q1", query="Python后端", hard_constraints_reference={}, soft_preferences_reference={}
    )
    for name, value in (("summary.json", {"status": "complete"}), ("queries.json", [query])):
        (tmp_path / name).write_text(json.dumps(value), encoding="utf-8")
    (tmp_path / "corpus.jsonl").write_text(json.dumps(job) + "\n", encoding="utf-8")
    (tmp_path / "per_query.jsonl").write_text(
        json.dumps(dict(query_id="q1", split="test", ranking=[job["job_key"]])) + "\n", encoding="utf-8"
    )
    calls = []

    def handle(request):
        body = check_request(request)
        calls.append(body)
        item = body["items"][0]
        return reply(
            {
                "items": [
                    dict(
                        pair_id=item["pair_id"],
                        direction_grade=2,
                        hard_status="not_applicable",
                        grade=2,
                        reason="核心工作明确为Python接口开发",
                        lines=[1],
                    )
                ]
            }
        )

    original = httpx.AsyncClient
    monkeypatch.setattr(
        scenario_judging.httpx,
        "AsyncClient",
        lambda **kw: original(transport=httpx.MockTransport(handle), **kw),
    )
    result = await scenario_judging.judge(tmp_path, max_calls=1, max_tokens=100)
    assert result["labeled"] == 1 and len(calls) == 1
    saved = json.loads((tmp_path / "judge/annotations.jsonl").read_text(encoding="utf-8"))
    assert saved["job_key"] == job["job_key"] and saved["direction_grade"] == 2
    assert (await scenario_judging.judge(tmp_path, max_calls=1, max_tokens=100))["labeled"] == 1
    assert len(calls) == 1


def test_missing_usage_remains_unknown():
    assert usage_details({"usage": None}) == {}
    assert usage_details({}) == {}
    assert usage_details({"usage": {"prompt_tokens": 15}}) == {"prompt_tokens": 15, "input_tokens": 15}
