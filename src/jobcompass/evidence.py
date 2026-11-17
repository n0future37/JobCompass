"""本地职责证据门槛：只把达到开发集校准门槛的原文工作片段交给解释器。"""

import math
import re
import threading
from collections import OrderedDict
from time import perf_counter

from .intent import parse_intent
from .models import LocalModels
from .preprocessing import digest
from .settings import Settings
from .work_scopes import work_scope, work_anchor_matches

VERSION = "work-evidence.v3.1"
DUTY = re.compile(
    r"岗位职责|工作职责|工作内容|职位职责|职责描述|工作描述|主要职责|主要工作|工作任务|岗位内容|你要做的事|职责[:：]"
)
QUAL = re.compile(
    r"任职要求|任职资格|岗位要求|职位要求|任职条件|人员要求|岗位条件|岗位资格|应聘条件|^要求[:：]"
)
END = re.compile(
    r"^[\d一二三四五六七八九十、.\s【】]*?(?:福利|薪酬|薪资|待遇|工作时间|工作地点|公司简介|公司背景|我们为你提供|联系我们)"
)
META = re.compile(
    r"^(?:[\d\s、.．）)【】一二三四五六七八九十-]+)?(?:人数(?:需求)?|招聘人数|名额|薪资|工资|福利|待遇|学历|专业|经验|工作地点|联系电话|联系人|上班时间|招聘岗位|所属部门)[:：]"
)
UNSAFE = re.compile(
    r"忽略.*(?:指令|提示)|系统指令|系统提示|输出.*(?:推荐|JSON|offer)|system\s*prompt|ignore.*instructions",
    re.I,
)
QUAL_LINE = re.compile(
    r"本科|大专|学历|专业.{0,6}(?:毕业|优先)|相关专业|工作经验|从业经验|资格证|热爱.*事业|无需|不要求|不需要"
)
ACTION = re.compile(
    r"负责|协助|开展|参与|开发|编写|设计|核算|审核|处理|维护|制作|整理|管理|分析|研究|研发|执行|提供|迎宾|点餐|传菜|接待|护理|检修|建模|绘制|教学|讲授|审校|编校|收派|配送|分拣|盘点|跟踪|查询|登记|接听|解答|巡视|推广|培训|宣讲|收集|反馈|招聘|回访|服务|报表|凭证|税务|检索|生产"
)
GENERIC = re.compile(
    r"^(?:[\d\s、.．）)]+)?(?:完成|负责|协助)?(?:领导|上级|部门).*?(?:其他|其它|交办|安排)|^(?:其他|其它).*(?:工作|事项)$"
)
DEFAULTS = dict(
    threshold=0.0, direction_threshold=0.0, max_passages_per_job=6, max_passage_chars=240, cache_size=128
)

QUAL_CAPABILITY = re.compile(
    r"^(?:[\d\s、.．）)]+)?(?:具备|具有|熟悉|熟练|精通|掌握|热爱|性格|较强|良好|能吃苦)"
)
EXPLICIT_TASK = re.compile(r"负责|协助|参与|完成|开展|执行|进行|制作|编写|编制|办理|设计|开发")


def work_units(job, limit=240):
    """行内连续片段保留原行号和偏移；不让资格专业或岗位标题替代职责证据。"""
    units, section = [], "unknown"
    for number, original in enumerate(job["jd_text"].split("\n"), 1):
        text = original.strip()
        if not text or UNSAFE.search(text):
            continue
        if DUTY.search(text):
            section = "duties"
            match = DUTY.search(text)
            text = text[match.end() :].lstrip("：:】] ")
            if len(text) < 5:
                continue
        elif QUAL.search(text):
            section = "qualification"
            continue
        elif END.search(text):
            section = "other"
            continue
        if section in {"qualification", "other"}:
            continue
        if QUAL_CAPABILITY.search(text) and not EXPLICIT_TASK.search(text):
            continue
        if META.search(text) or GENERIC.search(text) or len(text) < 6:
            continue
        if section == "unknown" and (not ACTION.search(text) or QUAL_LINE.search(text)):
            continue
        # 单行很长时按句/有重叠窗口处理，引用恰为实际送入评分的连续原文。
        base = original.find(text)
        for match in re.finditer(r"[^。；;]+[。；;]?", text):
            fragment = match.group()
            if len(fragment.strip()) < 6 or GENERIC.search(fragment.strip()):
                continue
            step = max(1, limit - 40)
            for offset in range(0, len(fragment), step):
                piece = fragment[offset : offset + limit]
                leading = len(piece) - len(piece.lstrip())
                quote = piece.strip()
                if len(quote) < 6:
                    continue
                start = base + match.start() + offset + leading
                assert original[start : start + len(quote)] == quote
                units.append(dict(line=number, text=quote, start=start, end=start + len(quote)))
                if offset + limit >= len(fragment):
                    break
    return units


def direction_query(request):
    """去掉城市/类型与明确历史前缀；不读取profile，不把筛选词当职责。"""
    query = request.query
    intent = parse_intent(request)
    chars = list(query)
    for source in intent.sources:
        if (
            source.source == "query"
            and source.field
            in {
                "cities",
                "excluded_cities",
                "employment_types",
                "excluded_employment_types",
                "education_max",
                "include_unknown_education",
            }
            and source.start is not None
        ):
            chars[source.start : source.end] = " " * (source.end - source.start)
    query = "".join(chars)
    query = re.sub(r"(?:之前|以前|过去).*?(?:现在|目前)", "", query)
    # 资格自述和防编造提醒不是工作方向；保留职责偏好及否定词的原义。
    clauses = re.split(r"[，,。；;]", query)
    query = "，".join(
        c
        for c in clauses
        if not re.search(
            r"(?:别把|不能把|不要假设|不要编造|不代表我)|(?:暂时|目前|我|本人).{0,6}(?:没有|没做过|缺少|不具备|未获得).{0,30}(?:论文|竞赛|经历|经验|学历|学位|证书)",
            c,
        )
    )
    query = re.sub(
        r"求职方向是|求职方向|招聘类型|招聘信息|岗位名称|工作地点|地点|城市|不限|岗位|职位|希望|想找|想做|帮我找|请帮我|请|帮我|只看|只接受|不接受|不考虑|都可以|都行|优先|最好|必须|至少|明确提及|找|要",
        " ",
        query,
    )
    query = re.sub(r"[，。；;、：:！？!?或和及与的在也]+", " ", query)
    return re.sub(r"\s+", " ", query).strip()[:120]


def choose_units(units, query, budget):
    """词面仅用来分配有限评分预算，不能据此通过相关性门槛。"""
    if len(units) <= budget:
        return units
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9+#.]*|[\u4e00-\u9fff]{1,2}", query.casefold())
    scored = sorted(
        range(len(units)), key=lambda i: (-sum(token in units[i]["text"].casefold() for token in tokens), i)
    )
    picked = []
    for i in scored[: max(1, budget - 2)]:
        if any(token in units[i]["text"].casefold() for token in tokens):
            picked.append(i)
    # 同时覆盖全文首尾及均匀位置，避免证据只从前部截取。
    for n in range(budget):
        i = round(n * (len(units) - 1) / (budget - 1)) if budget > 1 else 0
        if i not in picked:
            picked.append(i)
        if len(picked) == budget:
            break
    return [units[i] for i in picked]


class EvidenceGuard:
    def __init__(self, settings=None, *, scorer=None, model_provider=None):
        self.settings = settings or Settings()
        self.spec = DEFAULTS | getattr(self.settings, "config", {}).get("evidence", {})
        self.scorer = scorer
        self.model_provider = model_provider
        self.model = None
        self.cache = OrderedDict()
        self.cache_lock = threading.Lock()

    def _model_scores(self, query, passages):
        if self.model_provider:
            model = self.model_provider()
        else:
            if self.model is None:
                self.model = LocalModels(self.settings)
            model = self.model
        return model.score_passages(query, passages)

    def prepare(self, results, request):
        started = perf_counter()
        raw_focus = direction_query(request)
        scope, anchor, normalized = work_scope(raw_focus)
        focus = normalized + "；查询：" + raw_focus if scope else raw_focus
        plans, pending, passages = [], [], []
        for row in results:
            job = row["job"]
            key = digest(
                [
                    VERSION,
                    focus,
                    job["title"],
                    job["jd_text"],
                    self.spec,
                    getattr(self.settings, "config", {}).get("reranker", {}),
                ]
            )
            with self.cache_lock:
                cached = self.cache.get(key)
                if cached is not None:
                    self.cache.move_to_end(key)
            all_units = work_units(job, self.spec["max_passage_chars"]) if raw_focus else []
            eligible_units = [
                u for u in all_units if scope is None or work_anchor_matches(scope, anchor, u["text"])
            ]
            units = choose_units(eligible_units, focus, self.spec["max_passages_per_job"])
            context_units = choose_units(all_units, focus, self.spec["max_passages_per_job"])
            plan = dict(row=row, cache_key=key, units=units, scores=cached, error=None)
            plans.append(plan)
            title = re.sub(r"[（(].*?[）)]", "", job["title"])[:60]
            context = (
                "岗位名称："
                + title
                + "\n实际工作内容："
                + "；".join(u["text"][:90] for u in context_units)[:350]
            )
            plan["context"] = context
            if units and cached is None:
                pending.append(plan)
                passages.append(context)
                passages.extend(u["text"] for u in units)
        if passages:
            try:
                score = self.scorer or self._model_scores
                values = score(focus, passages)
                if len(values) != len(passages) or any(
                    isinstance(x, bool) or not math.isfinite(float(x)) for x in values
                ):
                    raise ValueError("职责评分缺失或不合法")
                offset = 0
                for plan in pending:
                    size = 1 + len(plan["units"])
                    plan["scores"] = [float(x) for x in values[offset : offset + size]]
                    offset += size
                    with self.cache_lock:
                        self.cache[plan["cache_key"]] = plan["scores"]
                        while len(self.cache) > self.spec["cache_size"]:
                            self.cache.popitem(last=False)
            except Exception as error:
                for plan in pending:
                    plan["error"] = type(error).__name__
        packs, checks = [], []
        for plan in plans:
            job, units = plan["row"]["job"], plan["units"]
            values = plan["scores"] or []
            direction_score = values[0] if values else None
            scores = values[1:] if values else []
            qualified = [
                (value, unit)
                for value, unit in zip(scores, units)
                if value >= self.spec["threshold"] and direction_score >= self.spec["direction_threshold"]
            ]
            # 一个来源行至多提供一个已评分片段，避免行号映射歧义。
            chosen, seen = [], set()
            for value, unit in sorted(qualified, key=lambda p: (-p[0], p[1]["line"], p[1]["start"])):
                if unit["line"] not in seen:
                    chosen.append(unit)
                    seen.add(unit["line"])
                if len(chosen) == 3:
                    break
            reason = (
                "support_model_unavailable"
                if plan["error"]
                else "no_direction"
                if not raw_focus
                else "no_work_content"
                if not units
                else "supported_work_content"
                if chosen
                else "below_support_threshold"
            )
            checks.append(
                dict(
                    job_key=job["job_key"],
                    status="supported" if chosen else "insufficient",
                    reason=reason,
                    max_score=max(scores) if scores else None,
                    direction_score=direction_score,
                    error_type=plan["error"],
                    lines=chosen,
                )
            )
            if chosen:
                packs.append(
                    dict(
                        job_key=job["job_key"],
                        title=job["title"],
                        company=job.get("company"),
                        lines=[dict(line=u["line"], text=u["text"]) for u in chosen],
                    )
                )
        errors = sorted({c["error_type"] for c in checks if c["error_type"]})
        return dict(
            packs=packs,
            checks=checks,
            focus=raw_focus,
            scoring_query=focus,
            scope=scope,
            scope_mode="anchored" if scope else "semantic_fallback" if raw_focus else "field_only",
            version=VERSION,
            threshold=self.spec["threshold"],
            direction_threshold=self.spec["direction_threshold"],
            candidate_count=len(results),
            supported_count=len(packs),
            scored_passages=len(passages),
            status="degraded" if errors else "ready",
            errors=errors,
            seconds=perf_counter() - started,
        )


def apply_evidence_result(retrieval, answer):
    """API/Agent只展示实际带支持引用的结果；原始召回不直接出现在推荐列表。"""
    if "evidence_guard" not in answer:
        return retrieval
    if answer["evidence_guard"].get("field_only"):
        return dict(retrieval, result_kind="filtered_only")
    selected = {r["job_key"] for r in answer["recommendations"]}
    result = dict(retrieval, results=[r for r in retrieval["results"] if r["job"]["job_key"] in selected])
    errors = answer["evidence_guard"]["errors"]
    if errors:
        result["degradations"] = list(retrieval.get("degradations", [])) + [
            dict(stage="evidence_guard", fallback="supported_cached_only", error_type=errors[0])
        ]
    if not result["results"] and retrieval["results"]:
        result["no_result_reason"] = (
            "工作内容证据校验暂不可用，本次无法确认推荐依据，请稍后重试。"
            if answer["evidence_guard"]["errors"]
            else "本次候选缺少足以支持求职方向的工作内容证据，暂不形成推荐；不代表岗位库中一定没有合适岗位。"
        )
        answer["answer"] = result["no_result_reason"]
    return result
