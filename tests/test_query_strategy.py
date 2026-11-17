"""查询策略保护及独立成本计算；固定响应，不访问付费服务。"""

import json
from types import SimpleNamespace

import httpx
import pytest

from jobcompass.conditions import SearchRequest
from jobcompass.query_strategy import estimate_cost, plan_query
from jobcompass.retrieval import Retriever


def settings():
    return SimpleNamespace(
        api_config=dict(
            API_BASE="https://unit.invalid/v1",
            API_KEY_ENV="QUERY_UNIT_KEY",
            MODEL="Pro/deepseek-ai/DeepSeek-V3.2",
            API_TYPE="openai_chat_completions",
            ENABLE_THINKING="false",
            TEMPERATURE="0",
            REASONING_EFFORT="",
        )
    )


def response(text):
    return httpx.Response(
        200,
        json=dict(
            choices=[dict(finish_reason="stop", message=dict(content=json.dumps(dict(text=text))))],
            usage=dict(prompt_tokens=100, completion_tokens=10, prompt_cache_hit_tokens=80),
        ),
    )


@pytest.mark.parametrize("strategy", ["rewrite", "hyde"])
def test_valid_transform_and_profile_not_sent(strategy, monkeypatch):
    monkeypatch.setenv("QUERY_UNIT_KEY", "unit-key")
    seen = []

    def handle(request):
        body = json.loads(request.content)
        assert "PRIVATE_PROFILE" not in request.content.decode()
        seen.append(body)
        return response("找深圳全职Python服务端接口开发岗位")

    request = SearchRequest(query="找深圳全职Python后端", profile="PRIVATE_PROFILE")
    result = plan_query(request, strategy, settings(), transport=httpx.MockTransport(handle))
    assert len(seen) == 1
    assert result["metadata"]["effective"] == strategy
    assert result["metadata"]["preserved"] is True
    assert result["metadata"]["cost"]["cny"] == pytest.approx(0.000172)
    assert result["dense_is_query"] is (strategy == "rewrite")
    assert result["lexical_query"] == (result["dense_query"] if strategy == "rewrite" else request.query)


@pytest.mark.parametrize(
    "query,transformed",
    [
        ("找深圳全职Python后端", "找上海全职Python后端"),
        ("找深圳全职Python后端", "找Python后端"),
        ("找Python后端，不接受兼职", "找Python后端，接受兼职"),
        ("找Python后端，最好深圳", "只考虑深圳Python后端"),
    ],
)
def test_raw_constraint_drift_is_visible_and_falls_back(query, transformed, monkeypatch):
    monkeypatch.setenv("QUERY_UNIT_KEY", "unit-key")
    result = plan_query(
        SearchRequest(query=query),
        "rewrite",
        settings(),
        transport=httpx.MockTransport(lambda r: response(transformed)),
    )
    assert result["metadata"]["status"] == "fallback_constraint_drift"
    assert result["metadata"]["raw_text"] == transformed
    assert result["metadata"]["preserved"] is False
    assert result["metadata"]["effective"] == "original"
    assert result["lexical_query"] == query and result["dense_query"] == query


def test_timeout_unknown_cost_is_not_zero(monkeypatch):
    monkeypatch.setenv("QUERY_UNIT_KEY", "unit-key")

    def fail(request):
        raise httpx.ReadTimeout("fixed timeout", request=request)

    result = plan_query(
        SearchRequest(query="Python后端"), "hyde", settings(), transport=httpx.MockTransport(fail)
    )
    assert result["metadata"]["status"] == "fallback_error"
    assert result["metadata"]["calls"] == 1
    assert result["metadata"]["cost"]["cny"] is None
    assert result["metadata"]["error_type"] == "ReadTimeout"


def test_original_or_unsupported_never_calls_provider():
    def fail(request):
        pytest.fail("不得访问服务")

    for query, strategy in [("Python后端", "original"), ("找月薪至少3万的Python后端", "rewrite")]:
        result = plan_query(
            SearchRequest(query=query), strategy, settings(), transport=httpx.MockTransport(fail)
        )
        assert result["metadata"]["calls"] == 0
        assert result["metadata"]["effective"] == "original"


def test_cost_missing_usage_and_unknown_cache():
    assert estimate_cost({}, "Pro/deepseek-ai/DeepSeek-V3.2")["cny"] is None
    result = estimate_cost(
        dict(prompt_tokens=100, completion_tokens=10, prompt_tokens_details=None),
        "Pro/deepseek-ai/DeepSeek-V3.2",
    )
    assert result["cny"] == pytest.approx(0.00046)
    assert result["basis"] == "upper_bound_cache_unknown"


def test_retrieval_filters_original_even_with_conflicting_transform(job_factory):
    job = job_factory()
    job["cities"] = ["深圳"]

    class Store:
        def all_jobs(self):
            return [job]

        def current(self, keys):
            return {job["job_key"]: job}

    retriever = Retriever(settings=SimpleNamespace(), store=Store(), models=object())
    result = retriever.search(
        SearchRequest(query="只考虑上海Python后端"),
        query_plan=dict(lexical_query="深圳", dense_query="深圳", dense_is_query=True),
    )
    assert result["results"] == []
    assert result["conditions"]["cities"] == ["上海"]


def test_judge_cannot_invent_evidence_or_change_eligible_pool():
    from jobcompass.query_policy_evaluation import validate_labels

    batch = [dict(pair_id="x", fragments=[dict(line=2, text="负责Python接口开发")])]
    item = dict(pair_id="x", grade=2, reason="明确开发工作", lines=[2])
    assert validate_labels(dict(items=[item]), batch) == [item]
    for changed in (item | dict(lines=[99]), item | dict(pair_id="other"), item | dict(grade=True)):
        with pytest.raises(ValueError):
            validate_labels(dict(items=[changed]), batch)
    with pytest.raises(ValueError):
        validate_labels(dict(items=[item, item]), batch)


def test_author_field_mapping_does_not_silently_drop_unassessed_constraints():
    from jobcompass.query_policy_evaluation import author_fields

    conditions, unknown = author_fields(
        dict(hard_constraints_reference=["城市深圳或广州", "全职", "月薪下限15K"], request={})
    )
    assert conditions["cities"] == ["深圳", "广州"] and conditions["employment_types"] == ["全职"]
    assert unknown == ["月薪下限15K"]


def test_hyde_document_requirements_do_not_become_user_hard_conditions(monkeypatch):
    monkeypatch.setenv("QUERY_UNIT_KEY", "unit-key")
    text = "招聘深圳全职Python后端工程师，负责业务API开发。技术要求熟悉Python及Django/Flask，具备数据库开发经验。"
    result = plan_query(
        SearchRequest(query="找深圳全职Python后端"),
        "hyde",
        settings(),
        transport=httpx.MockTransport(lambda r: response(text)),
    )
    assert result["metadata"]["effective"] == "hyde"
    assert result["metadata"]["preserved"] is True
    assert result["metadata"]["transformed_intent"]["unresolved"]
    assert result["lexical_query"] == "找深圳全职Python后端"


def test_hyde_must_not_change_location_or_drop_required_skill(monkeypatch):
    monkeypatch.setenv("QUERY_UNIT_KEY", "unit-key")
    request = SearchRequest(query="找深圳全职后端，必须提及Python")
    for text in ("招聘上海全职Python后端，负责业务API开发", "招聘深圳全职Java后端，负责业务API开发"):
        result = plan_query(
            request, "hyde", settings(), transport=httpx.MockTransport(lambda r: response(text))
        )
        assert result["metadata"]["status"] == "fallback_constraint_drift"


def test_judge_zero_can_have_no_matching_evidence_but_positive_must_cite():
    from jobcompass.query_policy_evaluation import validate_labels

    batch = [dict(pair_id="x", fragments=[dict(line=2, text="负责财务核算")])]
    item = dict(pair_id="x", grade=0, reason="不承担后端开发", lines=[])
    assert validate_labels(dict(items=[item]), batch) == [item]
    with pytest.raises(ValueError):
        validate_labels(dict(items=[item | dict(grade=2)]), batch)


def test_judge_fragments_include_relevant_chinese_duty(job_factory):
    from jobcompass.query_policy_evaluation import judge_fragments

    job = job_factory(
        jd="岗位职责\n负责日常统计和报表维护。\n负责数据埋点配置。\n维护数据口径。\n分析用户流失、漏斗转化和渠道效果。\n管理其他分析工具。"
    )
    selected = judge_fragments(job, "喜欢分析用户为什么流失、哪些渠道效果好")
    assert len(selected) <= 3 and any("渠道效果" in x["text"] for x in selected)


def test_partial_judging_saves_legal_item_without_fabricating_missing_one():
    from jobcompass.query_policy_evaluation import validate_partial_labels

    batch = [dict(pair_id=x, fragments=[dict(line=2, text="负责Python接口开发")]) for x in ["a", "b", "c"]]
    item = dict(pair_id="a", grade=2, reason="直接负责接口开发", lines=[2])
    clean, errors = validate_partial_labels(dict(items=[item, item | dict(pair_id="b", lines=[99])]), batch)
    assert clean == [item] and set(errors) == {"b", "c"}
