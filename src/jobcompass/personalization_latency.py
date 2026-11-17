"""同池2×2排序延迟实验：原文/原文压缩 × 有解释/仅名次；不修改生产链路。"""

import argparse
import asyncio
from collections import Counter
import copy
import json
import random
import re
from statistics import mean
from time import perf_counter
import uuid
import zipfile

import httpx
import numpy as np

from . import personalization_experiment as base
from .evaluation import ranking_metrics
from .model_api import api_key, build_request, usage_details
from .query_strategy import estimate_cost
from .settings import ROOT, Settings

SOURCE = base.DEFAULT
DEFAULT = ROOT / "artifacts/experiments/personalization_latency_20261117_v1"
VARIANTS = ("full_explain", "full_rank", "compact_explain", "compact_rank")
RANK_ONLY_PROMPT = base.RANK_PROMPT.split("输出严格JSON：")[0] + (
    '输出严格JSON：{"ranking":["全部候选job_id，各一次"]}。'
    "全部候选必须且仅出现一次，不输出理由、解释或其他字段。"
)
MAX_CHARS = 360
UNIT_CHARS = 180
TASK = re.compile(
    r"负责|参与|承担|开发|设计|实现|维护|搭建|分析|处理|执行|管理|授课|备课|护理|诊疗|"
    r"审核|编制|谈判|签约|采购|调度|施工|试验|实验|检测|招聘|培训|运营|销售|"
    r"responsib|develop|design|implement|maintain|deliver|manage|analy",
    re.I,
)
QUALIFICATION = re.compile(
    r"学历|本科|硕士|博士|大专|高中|中专|毕业|经验|年以上|年及以上|资格|证书|执业|"
    r"必须|至少|不得|不接受|不支持|不限|优先|熟悉|掌握|要求|"
    r"bachelor|master|degree|experience|require|minimum|must|prefer",
    re.I,
)
PROMOTION = re.compile(r"福利|薪酬|待遇|团建|下午茶|生日|公司介绍|公司简介|企业简介|包吃|五险|年终奖")


def text_terms(text):
    """不依赖外部词典/模型；中英文专业词与连续汉字二元组。"""
    terms = {m.lower() for m in re.findall(r"[A-Za-z][A-Za-z0-9_+#.-]*", text)}
    for part in re.findall(r"[\u4e00-\u9fff]+", text):
        terms.update(part[i : i + 2] for i in range(len(part) - 1))
    return terms


def source_units(lines):
    units = []
    for line in lines:
        text = line["text"]
        ends = [m.end() for m in re.finditer(r"[。；;!?！？]|[.](?=\s+[A-Z])", text)]
        ends.append(len(text))
        start = 0
        for end in sorted(set(ends)):
            while start < end:
                stop = min(start + UNIT_CHARS, end)
                if stop < end:
                    cuts = [m.end() for m in re.finditer(r"[，,:：、 ]", text[start:stop])]
                    near = [n for n in cuts if n >= UNIT_CHARS // 2]
                    if near:
                        stop = start + near[-1]
                if text[start:stop].strip():
                    units.append(dict(source_line=line["id"], start=start, end=stop, text=text[start:stop]))
                start = stop
    return units


def compress_job(job, query, facts):
    """360字符预算内选原文；片段包含零基字符范围，绝不抽象改写。"""
    result = copy.deepcopy(job)
    if sum(len(x["text"]) for x in job["lines"]) <= MAX_CHARS:
        return result
    units = source_units(job["lines"])
    qterms = text_terms(query)
    soft = " ".join(p["text"] for p in facts if p["kind"] not in {"education", "experience"})
    pterms = text_terms(soft)
    scored = []
    for i, unit in enumerate(units):
        text = unit["text"]
        terms = text_terms(text)
        task, qualification = bool(TASK.search(text)), bool(QUALIFICATION.search(text))
        score = 3 * min(len(terms & qterms), 12) + 2 * min(len(terms & pterms), 12)
        score += 2 * task + 1.5 * qualification
        score -= 5 * bool(PROMOTION.search(text) and not (task or qualification))
        scored.append((float(score), i, task, qualification))
    priority = []
    for field in (2, 3):
        choices = [r for r in scored if r[field]]
        if choices:
            priority.append(max(choices, key=lambda x: (x[0], -x[1]))[1])
    priority.extend(r[1] for r in sorted(scored, key=lambda x: (-x[0], x[1])))
    selected, chars = set(), 0
    for i in priority:
        if i not in selected and chars + len(units[i]["text"]) <= MAX_CHARS:
            selected.add(i)
            chars += len(units[i]["text"])
    if not selected:
        raise ValueError("压缩后为空")
    result["lines"] = [
        dict(id=f"{u['source_line']}:{u['start']}-{u['end']}", text=u["text"])
        for i, u in enumerate(units)
        if i in selected
    ]
    return result


def check_spans(original, compressed):
    before = {x["id"]: x["text"] for x in original["lines"]}
    if not compressed["lines"]:
        raise ValueError("原文片段为空")
    for line in compressed["lines"]:
        if line["id"] in before:
            expected = before[line["id"]]
        else:
            match = re.fullmatch(r"(L\d+):(\d+)-(\d+)", line["id"])
            if not match or match[1] not in before:
                raise ValueError("压缩来源不存在")
            start, end = int(match[2]), int(match[3])
            if not 0 <= start < end <= len(before[match[1]]):
                raise ValueError("压缩范围越界")
            expected = before[match[1]][start:end]
        if line["text"] != expected:
            raise ValueError("压缩片段不是逐字原文")
    if {k: v for k, v in original.items() if k != "lines"} != {
        k: v for k, v in compressed.items() if k != "lines"
    }:
        raise ValueError("压缩修改了岗位字段")


def view(case, variant):
    result = dict(case)
    if variant.startswith("compact"):
        result["candidates"] = case["compressed_candidates"]
    return result


def valid_output(value, case, variant):
    if variant.endswith("explain"):
        return base.valid_rank(value, case)
    if not isinstance(value, dict) or set(value) != {"ranking"}:
        raise ValueError("纯排序只接受ranking")
    order = value["ranking"]
    if not isinstance(order, list) or any(not isinstance(k, str) for k in order):
        raise ValueError("名次类型错误")
    if len(order) != len(case["candidate_ids"]) or len(set(order)) != len(order):
        raise ValueError("候选缺项或重复")
    if set(order) != set(case["candidate_ids"]):
        raise ValueError("候选身份变化")
    return value


def source_hashes():
    names = [p.name for p in SOURCE.iterdir() if p.is_file()]
    return {n: base.sha((SOURCE / n).read_bytes()) for n in sorted(names)}


def code_hashes():
    names = base.code_hashes() | {
        "src/jobcompass/personalization_latency.py": base.sha(
            (ROOT / "src/jobcompass/personalization_latency.py").read_bytes()
        ),
        "tests/test_personalization_latency.py": base.sha(
            (ROOT / "tests/test_personalization_latency.py").read_bytes()
        ),
    }
    return names


def frozen(directory):
    plan = json.loads((directory / "plan.json").read_bytes())
    if code_hashes() != plan["code_hashes"]:
        raise ValueError("实验源码改变")
    for n, expected in plan["inputs"].items():
        if base.sha((directory / n).read_bytes()) != expected:
            raise ValueError("实验输入改变：" + n)
    for n, expected in plan["source_hashes"].items():
        if base.sha((SOURCE / n).read_bytes()) != expected:
            raise ValueError("旧实验改变：" + n)
    return plan


class StreamResult:
    def __init__(self):
        self.content = []
        self.usage = {}
        self.finish_reason = None
        self.response_id = None
        self.fingerprint = None
        self.first_content_seconds = None
        self.last_content_seconds = None
        self.done = False
        self.frames = 0

    def feed(self, data, seconds):
        if data == "[DONE]":
            self.done = True
            return
        item = json.loads(data)
        if not isinstance(item, dict):
            raise ValueError("SSE数据类型错误")
        self.frames += 1
        if isinstance(item.get("usage"), dict) and item["usage"]:
            self.usage = usage_details(item)
        self.response_id = item.get("id") or self.response_id
        self.fingerprint = item.get("system_fingerprint") or self.fingerprint
        if item.get("error"):
            raise ValueError("服务返回流内错误")
        for choice in item.get("choices", []):
            if choice.get("finish_reason"):
                self.finish_reason = choice["finish_reason"]
            text = choice.get("delta", {}).get("content")
            if isinstance(text, str) and text:
                if self.first_content_seconds is None:
                    self.first_content_seconds = seconds
                self.last_content_seconds = seconds
                self.content.append(text)

    def value(self):
        if not self.done or self.finish_reason != "stop" or not self.content:
            raise ValueError("流式响应未完整结束")
        return base.decode("".join(self.content))


async def request(client, directory, phase, prompt, data, plan):
    budget = plan["budget"]
    used = base.totals(base.all_calls(directory))
    if any(
        used[k] >= budget[limit]
        for k, limit in (
            ("calls", "max_started_calls"),
            ("known_cny", "max_known_cny"),
            ("reported_tokens", "max_reported_tokens"),
        )
    ):
        raise RuntimeError("实验预算达到上限")
    settings = Settings()
    config = settings.api_config | dict(MODEL=base.MODELS["ranker"], ENABLE_THINKING="false", TEMPERATURE="0")
    key = api_key(config, settings.values)
    if not key:
        raise RuntimeError("本机模型密钥未配置")
    url, payload = build_request(config, prompt, json.dumps(data, ensure_ascii=False), 2500)
    payload.update(stream=True, stream_options=dict(include_usage=True))
    record = dict(
        call_id=str(uuid.uuid4()),
        at=base.stamp(),
        phase=phase,
        model=config["MODEL"],
        status="started",
        usage={},
        cost=dict(cny=None),
        payload_sha256=base.sha(payload),
    )
    base.append(directory / "requests_started.jsonl", record)
    began = perf_counter()
    parsed = StreamResult()
    result = None
    try:
        async with (
            asyncio.timeout(45),
            client.stream("POST", url, headers={"Authorization": "Bearer " + key}, json=payload) as response,
        ):
            record["headers_seconds"] = perf_counter() - began
            record["http_status"] = response.status_code
            response.raise_for_status()
            frame = []
            async for line in response.aiter_lines():
                if not line:
                    if frame:
                        parsed.feed("\n".join(frame), perf_counter() - began)
                        frame = []
                elif line.startswith("data:"):
                    frame.append(line[5:].lstrip(" "))
            if frame:
                parsed.feed("\n".join(frame), perf_counter() - began)
            result = parsed.value()
            record["status"] = "ok"
    except asyncio.CancelledError:
        record.update(status="cancelled", error_type="CancelledError")
        raise
    except Exception as error:
        record.update(status="error", error_type=type(error).__name__)
    finally:
        record.update(
            seconds=perf_counter() - began,
            usage=parsed.usage,
            raw_text="".join(parsed.content),
            first_content_seconds=parsed.first_content_seconds,
            last_content_seconds=parsed.last_content_seconds,
            finish_reason=parsed.finish_reason,
            response_id=parsed.response_id,
            fingerprint=parsed.fingerprint,
            stream_done=parsed.done,
            frame_count=parsed.frames,
        )
        record["cost"] = estimate_cost(
            parsed.usage, config["MODEL"], json.loads((directory / "prices.json").read_bytes())
        )
        base.append(directory / "calls.jsonl", record)
    return result, record["call_id"]


def budget():
    return dict(max_started_calls=300, max_known_cny=15, max_reported_tokens=5_000_000)


async def preflight(directory):
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "preflight.json").exists():
        return json.loads((directory / "preflight.json").read_bytes())
    base.save(directory / "prices.json", json.loads((SOURCE / "prices.json").read_bytes()))
    async with httpx.AsyncClient(timeout=httpx.Timeout(45, connect=5), follow_redirects=False) as client:
        value, ident = await request(
            client,
            directory,
            "preflight",
            '输出JSON：{"ranking":["probe"]}',
            dict(test="probe"),
            dict(budget=budget()),
        )
    record = next(r for r in base.all_calls(directory) if r["call_id"] == ident)
    if value != {"ranking": ["probe"]} or not record["usage"].get("input_tokens"):
        raise RuntimeError("流式预检或用量记录未通过；不启动正式实验")
    result = dict(
        at=base.stamp(),
        status="ok",
        call_id=ident,
        streaming=True,
        reported_usage=True,
        model=base.MODELS["ranker"],
    )
    base.save(directory / "preflight.json", result)
    return result


def prepare(directory):
    if (directory / "plan.json").exists():
        return frozen(directory)
    base.frozen(SOURCE)
    if json.loads((directory / "preflight.json").read_bytes())["status"] != "ok":
        raise ValueError("先通过流式预检")
    cases = json.loads((SOURCE / "cases.json").read_bytes())
    original = base.readl(SOURCE / "llm_rankings.jsonl")
    inputs = {r["case_id"]: r["input_order"] for r in original if r["repeat"] == 0}
    controls = []
    for case in cases:
        start = perf_counter()
        compact = [compress_job(j, case["query"], case["profile_facts"]) for j in case["candidates"]]
        for full, short in zip(case["candidates"], compact):
            check_spans(full, short)
        case["compressed_candidates"] = compact
        case["input_order"] = inputs[case["case_id"]]
        case["compression_seconds"] = perf_counter() - start
        controls.append(
            dict(
                case_id=case["case_id"],
                full_chars=sum(len(x["text"]) for j in case["candidates"] for x in j["lines"]),
                compact_chars=sum(len(x["text"]) for j in compact for x in j["lines"]),
                candidate_ids=case["candidate_ids"],
            )
        )
    labels = json.loads((SOURCE / "labels.json").read_bytes())
    base.save(directory / "cases.json", cases)
    base.save(directory / "labels.json", labels)
    base.save(directory / "compression.json", controls)
    base.save(directory / "prompts.json", dict(explain=base.RANK_PROMPT, rank=RANK_ONLY_PROMPT))
    rng = random.Random(20261117)
    ids = [c["case_id"] for c in cases]
    rng.shuffle(ids)
    schedule = [
        dict(case_id=cid, variant=VARIANTS[(i + k) % 4]) for i, cid in enumerate(ids) for k in range(4)
    ]
    base.save(directory / "schedule.json", schedule)
    plan = dict(
        at=base.stamp(),
        code_hashes=code_hashes(),
        source_hashes=source_hashes(),
        inputs={
            n: base.sha((directory / n).read_bytes())
            for n in (
                "cases.json",
                "labels.json",
                "compression.json",
                "prompts.json",
                "schedule.json",
                "prices.json",
                "preflight.json",
            )
        },
        variants=list(VARIANTS),
        model=base.MODELS["ranker"],
        budget=budget(),
        execution=dict(
            concurrency=2,
            attempts=2,
            timeout_seconds_per_attempt=45,
            temperature=0,
            enable_thinking=False,
            max_tokens=2500,
            repeats=1,
            stream=True,
            stream_options=dict(include_usage=True),
        ),
        compression=dict(
            max_chars_per_job=MAX_CHARS,
            max_unit_chars=UNIT_CHARS,
            method="Verbatim source spans, deterministic query/profile lexical overlap, task and qualification coverage; no labels or LLM summaries",
        ),
        controls="32 same fictional profiles/queries/candidates and frozen600 labels; same oldrun0 input order.4 contemporaneous arms in balanced case blocks; old results preserved. No tuning or production default selection.",
        limitations="Already exposed assistant cases/model labels; not fresh human blind test. Same Qwen-family proxy limits as original. Compression can omit relevant evidence. Streaming only newly instrumented, old transport nonstreaming.",
        timing="Admitted task: packing+request/retries+validation/fallback. Common offline compression time separate. TTFT is first nonempty content from request start; includes network/provider queue/prefill, not pure prefill. No service/RAG/UI time.",
    )
    base.save(directory / "plan.json", plan)
    with zipfile.ZipFile(directory / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in plan["code_hashes"]:
            archive.write(ROOT / name, name)
        archive.write(directory / "plan.json", "plan.json")
    return dict(
        cases=len(cases),
        pairs=len(labels),
        tasks=len(schedule),
        full_chars=sum(r["full_chars"] for r in controls),
        compact_chars=sum(r["compact_chars"] for r in controls),
    )


async def run(directory):
    plan = frozen(directory)
    cases = {c["case_id"]: c for c in json.loads((directory / "cases.json").read_bytes())}
    local = {r["case_id"]: r for r in base.readl(SOURCE / "local_rankings.jsonl")}
    schedule = json.loads((directory / "schedule.json").read_bytes())
    prompts = json.loads((directory / "prompts.json").read_bytes())
    done = {(r["case_id"], r["variant"]) for r in base.readl(directory / "rankings.jsonl")}
    sem = asyncio.Semaphore(2)
    async with httpx.AsyncClient(timeout=httpx.Timeout(45, connect=5), follow_redirects=False) as client:

        async def one(item):
            async with sem:
                begin = perf_counter()
                case = cases[item["case_id"]]
                variant = item["variant"]
                shown = view(case, variant)
                byid = {j["job_id"]: j for j in shown["candidates"]}
                data = dict(
                    query=case["query"],
                    profile_facts=case["profile_facts"],
                    candidates=[byid[k] for k in case["input_order"]],
                )
                calls, errors, result = [], [], None
                for _ in range(2):
                    value, ident = await request(
                        client,
                        directory,
                        f"rank:{case['case_id']}:{variant}",
                        prompts["explain" if variant.endswith("explain") else "rank"],
                        data,
                        plan,
                    )
                    calls.append(ident)
                    try:
                        result = valid_output(value, shown, variant)
                        break
                    except (ValueError, TypeError) as error:
                        errors.append(type(error).__name__)
                        base.append(
                            directory / "rejects.jsonl",
                            dict(
                                call_id=ident,
                                case_id=case["case_id"],
                                variant=variant,
                                error_type=type(error).__name__,
                            ),
                        )
                base.append(
                    directory / "rankings.jsonl",
                    dict(
                        at=base.stamp(),
                        case_id=case["case_id"],
                        variant=variant,
                        ranking=result["ranking"] if result else local[case["case_id"]]["ranking"]["rules"],
                        explanations=result.get("explanations", []) if result else [],
                        status="ok" if result else "fallback_rules",
                        call_ids=calls,
                        errors=errors,
                        seconds=perf_counter() - begin,
                    ),
                )

        tasks = [
            asyncio.create_task(one(item))
            for item in schedule
            if (item["case_id"], item["variant"]) not in done
        ]
        for i, task in enumerate(asyncio.as_completed(tasks), 1):
            await task
            if i % 8 == 0 or i == len(tasks):
                print(json.dumps(dict(completed_this_run=i, pending_at_start=len(tasks))), flush=True)
    return dict(rankings=len(base.readl(directory / "rankings.jsonl")))


def stats(values):
    values = [v for v in values if v is not None]
    return dict(
        n=len(values),
        mean=mean(values) if values else None,
        median=float(np.median(values)) if values else None,
        p95=float(np.percentile(values, 95)) if values else None,
    )


def per_metrics(cases, rankings, labels, calls, old=False):
    qrels = {(r["case_id"], r["job_id"]): r for r in labels}
    output = {}
    for row in rankings:
        c = cases[row["case_id"]]
        grades = {j: qrels[c["case_id"], j]["fit"] for j in c["candidate_ids"]}
        directions = {j: qrels[c["case_id"], j]["direction"] for j in c["candidate_ids"]}
        order = row["ranking"]
        if set(order) != set(c["candidate_ids"]) or len(order) != len(set(order)):
            raise ValueError("结果候选身份变化")
        m5, m10, d5 = (
            ranking_metrics(order, grades, 5),
            ranking_metrics(order, grades, 10),
            ranking_metrics(order, directions, 5),
        )
        cr = [calls[k] for k in row["call_ids"]]
        good = [r for r in cr if r["status"] == "ok"]
        last = good[-1] if good else None
        usable = row["status"] == "ok" and last is not None
        usage = last["usage"] if usable else {}
        unknown = any(r.get("cost", {}).get("cny") is None for r in cr)
        output[c["case_id"]] = dict(
            fit_ndcg5=m5["ndcg"],
            fit_ndcg10=m10["ndcg"],
            fit_p5=m5["precision"],
            direction_p5=d5["precision"],
            seconds=row["seconds"],
            http_seconds=sum(r["seconds"] for r in cr),
            local_seconds=row["seconds"] - sum(r["seconds"] for r in cr),
            first_content_seconds=last.get("first_content_seconds") if usable else None,
            content_window_seconds=(last["last_content_seconds"] - last["first_content_seconds"])
            if usable and last.get("first_content_seconds") is not None
            else None,
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            reasoning_tokens=usage.get("completion_tokens_details", {}).get("reasoning_tokens"),
            cached_tokens=usage.get(
                "prompt_cache_hit_tokens", usage.get("prompt_tokens_details", {}).get("cached_tokens")
            ),
            known_cost_cny=sum(r.get("cost", {}).get("cny") or 0 for r in cr),
            complete_cost_cny=None if unknown else sum(r["cost"]["cny"] for r in cr),
            unknown_cost_calls=sum(r.get("cost", {}).get("cny") is None for r in cr),
            status=row["status"],
            attempts=len(cr),
            old_nonstreaming=old,
        )
    return output


def summarize(directory):
    frozen(directory)
    case_list = json.loads((directory / "cases.json").read_bytes())
    cases = {c["case_id"]: c for c in case_list}
    labels = json.loads((directory / "labels.json").read_bytes())
    rows = base.readl(directory / "rankings.jsonl")
    if len(rows) != len(cases) * 4 or len({(r["case_id"], r["variant"]) for r in rows}) != len(rows):
        raise ValueError("四组结果未齐或重复")
    calls = {r["call_id"]: r for r in base.all_calls(directory)}
    methods = {
        variant: per_metrics(cases, [r for r in rows if r["variant"] == variant], labels, calls)
        for variant in VARIANTS
    }
    old_calls = {r["call_id"]: r for r in base.all_calls(SOURCE)}
    methods["old_full_explain"] = per_metrics(
        cases,
        [r for r in base.readl(SOURCE / "llm_rankings.jsonl") if r["repeat"] == 0],
        labels,
        old_calls,
        old=True,
    )
    result = dict(at=base.stamp(), splits={}, costs={}, label_sensitivity={})
    for split in ("dev", "test", "all"):
        subset = [c for c in case_list if split == "all" or c["split"] == split]
        output = {}
        for variant, scores in methods.items():
            selected = [scores[c["case_id"]] for c in subset]
            values = {
                k: stats([r[k] for r in selected])
                for k in (
                    "fit_ndcg5",
                    "fit_ndcg10",
                    "fit_p5",
                    "direction_p5",
                    "seconds",
                    "http_seconds",
                    "local_seconds",
                    "first_content_seconds",
                    "content_window_seconds",
                    "input_tokens",
                    "output_tokens",
                    "cached_tokens",
                    "known_cost_cny",
                    "complete_cost_cny",
                )
            }
            values["status"] = dict(Counter(r["status"] for r in selected))
            values["unknown_cost_calls"] = sum(r["unknown_cost_calls"] for r in selected)
            output[variant] = values
        contrasts = {}
        for a, b in (
            ("full_explain", "full_rank"),
            ("full_explain", "compact_explain"),
            ("full_explain", "compact_rank"),
            ("full_rank", "compact_rank"),
            ("compact_explain", "compact_rank"),
            ("old_full_explain", "full_explain"),
        ):
            contrasts[a + "_to_" + b] = {
                k: base.paired_interval(
                    {cid: r[k] for cid, r in methods[a].items()},
                    {cid: r[k] for cid, r in methods[b].items()},
                    subset,
                )
                for k in ("fit_ndcg5", "fit_p5", "seconds", "complete_cost_cny")
            }
        result["splits"][split] = dict(metrics=output, contrasts=contrasts)
    all_calls = list(calls.values())
    result["costs"] = {
        v: base.totals([r for r in all_calls if r["phase"].endswith(":" + v)]) for v in VARIANTS
    }
    result["costs"]["preflight"] = base.totals([r for r in all_calls if r["phase"] == "preflight"])
    result["costs"]["all"] = base.totals(all_calls)
    for label in ("judge_a", "judge_b"):
        diagnostic = {}
        for variant in VARIANTS:
            alternative = per_metrics(
                cases,
                [r for r in rows if r["variant"] == variant],
                base.readl(SOURCE / (label + ".jsonl")),
                calls,
            )
            diagnostic[variant] = {
                s: mean(
                    r["fit_ndcg5"] for cid, r in alternative.items() if s == "all" or cases[cid]["split"] == s
                )
                for s in ("all", "test")
            }
        result["label_sensitivity"][label] = diagnostic
    base.save(directory / "metrics.json", result)
    base.save(directory / "per_case_metrics.json", methods)
    return result


def verify(directory):
    plan = frozen(directory)
    cases = json.loads((directory / "cases.json").read_bytes())
    old = {c["case_id"]: c for c in json.loads((SOURCE / "cases.json").read_bytes())}
    checks = 0
    for c in cases:
        ref = old[c["case_id"]]
        for k in ref:
            if c[k] != ref[k]:
                raise ValueError("旧画像/查询/候选改变：" + k)
        for full, short in zip(c["candidates"], c["compressed_candidates"]):
            check_spans(full, short)
            checks += 1
    if (directory / "labels.json").read_bytes() != (SOURCE / "labels.json").read_bytes():
        raise ValueError("标签改变")
    old_plan = json.loads((SOURCE / "plan.json").read_bytes())
    for n, expected in old_plan["preserved_old_hashes"].items():
        if base.sha((ROOT / n).read_bytes()) != expected:
            raise ValueError("更早的证据改变")
    rows = base.readl(directory / "rankings.jsonl")
    calls = base.all_calls(directory)
    if any(r["status"] == "no_completion_record" for r in calls):
        raise ValueError("有开始后未结束的请求")
    result = dict(
        at=base.stamp(),
        cases=len(cases),
        profile_job_pairs=checks,
        source_files_unchanged=len(plan["source_hashes"]),
        older_evidence_unchanged=len(old_plan["preserved_old_hashes"]),
        rankings=len(rows),
        status=dict(Counter(r["status"] for r in rows)),
        reported_cost=base.totals(calls),
        no_completion_record=0,
        production_default_changed=False,
        business_service_calls=0,
    )
    base.save(directory / "verification.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "prepare", "run", "summarize", "verify"))
    parser.add_argument("--output", type=type(DEFAULT), default=DEFAULT)
    args = parser.parse_args()
    if args.action in {"preflight", "run"}:
        result = asyncio.run(globals()[args.action](args.output))
    else:
        result = globals()[args.action](args.output)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
