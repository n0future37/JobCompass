"""查询变换只影响召回表达，原始请求始终承担条件解析和重排。"""

import json
import re
from time import perf_counter

import httpx

from .conditions import term_pattern
from .intent import parse_intent
from .model_api import api_key, build_request, extract_output, usage_details
from .query_transform import PROMPTS
from .settings import ROOT, Settings

VERSION = "query-policy.v2"
STRATEGIES = ("original", "rewrite", "hyde")


def conditions_equal(left, right):
    def normalized(value):
        return {k: sorted(set(v)) if isinstance(v, list) else v for k, v in value.items()}

    return normalized(left) == normalized(right)


def document_conditions_preserved(expected, parsed, text):
    """HyDE是文档；只核对城市/类型及原始必需技能，不把文档资格变为用户意图。"""
    fields = (
        "cities",
        "employment_types",
        "excluded_cities",
        "excluded_employment_types",
        "education_max",
        "include_unknown_education",
    )
    if not conditions_equal({k: expected[k] for k in fields}, {k: parsed[k] for k in fields}):
        return False
    for skill in expected["skills_all"]:
        pattern = term_pattern(skill)
        if not re.search(pattern, text, re.I) or re.search(
            r"(?:无需|不要求|不需要|不涉及|不使用)[^，。；;]{0,12}" + pattern, text, re.I
        ):
            return False
    return True


def estimate_cost(usage, model, card=None):
    card = card or json.loads((ROOT / "configs/model-prices.json").read_bytes())
    rate = card["models"].get(model)
    incoming = usage.get("input_tokens", usage.get("prompt_tokens"))
    outgoing = usage.get("output_tokens", usage.get("completion_tokens"))
    if not rate or not all(type(x) is int and x >= 0 for x in (incoming, outgoing)):
        return dict(cny=None, basis="missing_usage_or_price")
    if rate.get("max_input_tokens") and incoming >= rate["max_input_tokens"]:
        return dict(cny=None, basis="outside_verified_price_tier")
    details = usage.get("prompt_tokens_details") or {}
    cached = usage.get("prompt_cache_hit_tokens", details.get("cached_tokens"))
    known = type(cached) is int and 0 <= cached <= incoming
    cached = cached if known else 0
    cache_rate = rate.get("cached_input")
    value = (
        (incoming - cached) * rate["input"]
        + cached * (cache_rate if cache_rate is not None else rate["input"])
        + outgoing * rate["output"]
    ) / 1_000_000
    return dict(
        cny=value,
        basis="published_rate_reported_usage" if known or cache_rate is None else "upper_bound_cache_unknown",
    )


def original_plan(request, strategy="original", status="original"):
    return dict(
        lexical_query=request.query,
        dense_query=request.query,
        dense_is_query=True,
        metadata=dict(
            version=VERSION,
            requested=strategy,
            effective="original",
            status=status,
            calls=0,
            usage={},
            cost=dict(cny=0.0, basis="no_call"),
            seconds=0.0,
            raw_text=None,
            preserved=None,
        ),
    )


def plan_query(request, strategy="original", settings=None, timeout=10, transport=None):
    if strategy not in STRATEGIES:
        raise ValueError("不支持的查询策略")
    plan = original_plan(request, strategy)
    if strategy == "original":
        return plan
    before = parse_intent(request)
    if before.status != "ready":
        plan["metadata"]["status"] = "skipped_unexecutable_original"
        return plan
    settings = settings or Settings()
    config = settings.api_config
    meta = plan["metadata"]
    started = perf_counter()
    try:
        key = api_key(config, getattr(settings, "values", None))
        if not key:
            raise RuntimeError("查询变换密钥未配置")
        system = (
            PROMPTS[strategy]
            + '\n保持输入hard_conditions中的全部正负条件，不把soft_preferences变成硬条件。只返回JSON对象：{"text":"改写查询或假想JD"}。'
        )
        content = json.dumps(
            dict(
                query=request.query,
                hard_conditions=before.conditions(),
                soft_preferences=before.soft_preferences,
            ),
            ensure_ascii=False,
        )
        url, payload = build_request(config, system, content, 512)
        meta.update(calls=1, cost=dict(cny=None, basis="response_usage_unknown"))
        with httpx.Client(
            timeout=httpx.Timeout(timeout, connect=min(3, timeout)),
            follow_redirects=False,
            transport=transport,
        ) as client:
            response = client.post(url, headers={"Authorization": "Bearer " + key}, json=payload)
            response.raise_for_status()
            data = response.json()
        meta["usage"] = usage_details(data)
        meta["cost"] = estimate_cost(meta["usage"], config["MODEL"])
        meta["response_model"] = data.get("model")
        meta["response_id"] = data.get("id")
        value = json.loads(extract_output(data))
        if (
            not isinstance(value, dict)
            or set(value) != {"text"}
            or not isinstance(value["text"], str)
            or not 3 <= len(value["text"].strip()) <= (400 if strategy == "rewrite" else 1200)
        ):
            raise ValueError("查询变换结构或长度不合法")
        text = value["text"].strip()
        after = parse_intent(request.model_copy(update={"query": text}))
        meta.update(
            raw_text=text,
            transformed_intent=after.public(),
            preserved=(after.status == "ready" and conditions_equal(before.conditions(), after.conditions()))
            if strategy == "rewrite"
            else document_conditions_preserved(before.conditions(), after.conditions(), text),
        )
        if not meta["preserved"]:
            meta["status"] = "fallback_constraint_drift"
            return plan
        plan["dense_query"] = text
        plan["dense_is_query"] = strategy != "hyde"
        if strategy == "rewrite":
            plan["lexical_query"] = text
        meta.update(effective=strategy, status="ok")
        return plan
    except Exception as error:
        meta.update(status="fallback_error", error_type=type(error).__name__)
        return plan
    finally:
        meta["seconds"] = perf_counter() - started
