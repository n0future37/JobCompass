"""可追溯的本地条件解析。职责级语义与没有权威字段的条件保留未支持。"""

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .conditions import CITIES, SKILLS, FilterText, term_pattern

from .qualifications import DEGREES, Education, education_intent

VERSION = "intent.v3.1"
ALIASES = {"校招": "校园", "校园招聘": "校园", "兼职": "兼职/临时", "临时": "兼职/临时"}
HARD = r"必须|只看|只接受|只要|仅限|至少|仅接受|一定要|只能|需要|要求"
NEGATIVE = r"不接受|不考虑|不要|不得|不能|排除|不想去|不想在|不愿去|不愿在|不在|不去|不选|不找|拒绝"
SOFT = r"最好|倾向|优先|加分|尽量|首选"
UNSUPPORTED = r"远程|双休|上市公司|国企|外企|五险|社保|到岗|每周|月薪|年薪|工资|薪资|薪酬|工作年限|正式工作经验|[一二三四五六七八九十两\d]+年.*经验|同时.*(?:办公|工作)"
ROLE_WORDS = "工程师 会计 财务 招聘 专员 法务 教师 老师 护士 物流 平面 设计师 土木 销售 代表 市场 编辑 客服 机械 电气 医药 研发 餐饮 服务 数据 软件 测试 人事 视觉 产品 经理 助理 运营".split()
FILLER = "现在 目前 只看 只接受 只要 仅限 必须 至少 仅接受 一定要 只能 需要 要求 不接受 不考虑 不要 不得 不能 排除 不去 不选 不找 拒绝 明确 提及 包含 出现 招聘信息 岗位 职位 后端 前端 开发 工作 招聘 熟悉 掌握 具有 具备 使用 经验 技能 支持 涉及 名称 类型 地点 城市 上班 办公 的 和 与 且 在 会 要 找 看 有 吗 都 可以 行".split()


class HardConditions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    education_max: Education | None = None
    include_unknown_education: bool = False
    cities: list[FilterText] = Field(default_factory=list, max_length=10)
    skills_all: list[FilterText] = Field(default_factory=list, max_length=10)
    employment_types: list[FilterText] = Field(default_factory=list, max_length=5)
    excluded_cities: list[FilterText] = Field(default_factory=list, max_length=10)
    excluded_employment_types: list[FilterText] = Field(default_factory=list, max_length=5)


class ConditionSource(BaseModel):
    field: str
    value: str
    source: Literal["query", "request"]
    strength: Literal["hard", "soft", "history", "unsupported"]
    quote: str
    start: int | None = None
    end: int | None = None


class Intent(BaseModel):
    version: str = VERSION
    hard: HardConditions
    soft_preferences: dict[str, list[str]] = Field(default_factory=dict)
    sources: list[ConditionSource] = Field(default_factory=list)
    unresolved: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)

    @property
    def status(self):
        return "conflict" if self.conflicts else "unsupported" if self.unresolved else "ready"

    def conditions(self):
        return self.hard.model_dump()

    def public(self):
        return self.model_dump() | {"status": self.status}


def city_name(value):
    return re.sub(r"市$", "", value.strip())


def employment_name(value):
    return ALIASES.get(value.strip(), value.strip())


def unique(values):
    return list(dict.fromkeys(values))


def parse_intent(request):
    query = request.query
    education_max, pending, education_sources, masked_query = education_intent(query)
    if request.education_max:
        education_max = min([x for x in (education_max, request.education_max) if x], key=DEGREES.get)
        education_sources.append(
            dict(
                field="education_max",
                value=request.education_max,
                source="request",
                strength="hard",
                quote=request.education_max,
            )
        )
    pending = pending or request.include_unknown_education
    explicit = dict(
        cities=unique(city_name(x) for x in request.cities),
        skills_all=unique(request.skills_all),
        employment_types=unique(employment_name(x) for x in request.employment_types),
        excluded_cities=unique(city_name(x) for x in request.excluded_cities),
        excluded_employment_types=unique(employment_name(x) for x in request.excluded_employment_types),
    )
    natural = {field: [] for field in explicit}
    soft = {"cities": [], "employment_types": [], "notes": []}
    sources, unresolved, conflicts = [ConditionSource(**x) for x in education_sources], [], []
    for field, values in explicit.items():
        for value in values:
            sources.append(
                ConditionSource(
                    field=field,
                    value=value,
                    source="request",
                    strength="hard",
                    quote=next(
                        (
                            x
                            for x in getattr(request, field)
                            if (
                                city_name(x)
                                if "cities" in field
                                else employment_name(x)
                                if "employment_types" in field
                                else x
                            )
                            == value
                        ),
                        value,
                    ),
                )
            )

    def source(field, value, strength, start, end):
        sources.append(
            ConditionSource(
                field=field,
                value=value,
                source="query",
                strength=strength,
                quote=query[start:end],
                start=start,
                end=end,
            )
        )

    def merge(field, values, extend=False):
        values = unique(values)
        if not values:
            return
        if natural[field] and not field.startswith("excluded_") and field != "skills_all" and not extend:
            common = [v for v in natural[field] if v in values]
            if not common:
                conflicts.append(f"自然语言{field}条件互相冲突")
            else:
                natural[field] = common
        else:
            natural[field] = unique(natural[field] + values)

    segments = []
    for match in re.finditer(r"[^，,。；;\n]+", masked_query):
        text, offset = match.group(), match.start()
        # 同一分句中的显式修正保留修正后文字，旧地点不并入OR。
        current = re.search(r"现在|目前|改成|改为", text)
        if current and current.start() > 0:
            offset += current.start()
            text = text[current.start() :]
        split = re.search(SOFT, text)
        if split and split.start() > 0 and re.search(HARD, text[: split.start()]):
            segments.extend(
                [(text[: split.start()], offset), (text[split.start() :], offset + split.start())]
            )
        else:
            segments.append((text, offset))
    for text, offset in segments:
        history = bool(re.search(r"之前|以前|过去|曾经", text)) and not re.search(
            r"现在|目前|改成|改为", text
        )
        background = bool(
            re.search(r"(?:我|本人).*(?:读书|上学|毕业|居住|住在|经历|做过)", text)
        ) and not re.search(r"找|求职|岗位|职位|只看", text)
        city_matches = list(
            re.finditer("|".join(re.escape(c) + "市?" for c in sorted(CITIES, key=len, reverse=True)), text)
        )
        type_matches = list(re.finditer(r"兼职/临时|校园招聘|校园|校招|实习|全职|兼职|临时", text))
        if history or background:
            for m in city_matches:
                source("cities", city_name(m.group()), "history", offset + m.start(), offset + m.end())
            continue
        is_soft = bool(re.search(SOFT, text)) and not re.search(HARD + "|" + NEGATIVE, text)
        if re.search(r"不限城市|城市不限|不限定城市|全国均可", text):
            natural["cities"] = []
        if re.search(r"不限.*招聘类型|招聘类型不限|类型不限|工作形式不限", text):
            natural["employment_types"] = []
        if is_soft:
            for field, matches, norm in (
                ("cities", city_matches, city_name),
                ("employment_types", type_matches, employment_name),
            ):
                for m in matches:
                    value = norm(m.group())
                    soft[field].append(value)
                    source(field, value, "soft", offset + m.start(), offset + m.end())
            soft["notes"].append(text.strip())
            continue
        city_and = len(city_matches) > 1 and bool(re.search(r"同时|两个城市.*办公", text))
        if city_and:
            unresolved.append(text.strip())
            city_matches = []
        values = {field: [] for field in explicit}
        matched = False
        for field, matches, norm in (
            ("cities", city_matches, city_name),
            ("employment_types", type_matches, employment_name),
        ):
            for m in matches:
                prefix = text[: m.start()]
                # 否定作用于其后的同一组OR值，转折后的新指令单独判。
                negative = bool(re.search(NEGATIVE, prefix.split("但")[-1])) or bool(
                    re.match(r"(?:也)?不要", text[m.end() :])
                )
                destination = "excluded_" + field if negative else field
                value = norm(m.group())
                values[destination].append(value)
                source(destination, value, "hard", offset + m.start(), offset + m.end())
                matched = True
        required = bool(re.search(HARD, text))
        terms = [term for term in SKILLS if re.search(term_pattern(term), text, re.I)]
        skill_or = required and len(terms) > 1 and bool(re.search(r"或|任选", text))
        if required and not re.search(NEGATIVE, text) and not skill_or:
            for term in terms:
                m = re.search(term_pattern(term), text, re.I)
                values["skills_all"].append(term)
                source("skills_all", term, "hard", offset + m.start(), offset + m.end())
                matched = True
        for field, content in values.items():
            merge(field, content, extend=bool(re.search(r"也(?:可以|接受|行)|都可以|都行", text)))
        unknown = bool(re.search(UNSUPPORTED, text)) or bool(
            re.search(
                r"(?:学历|本科|硕士|博士|大专|专科).*(?:硬门槛|硬性要求|必须)|(?:学历|最低学历)(?:要求)?[:：]?[^不限\s]",
                text,
            )
        )
        # 来源字段之外的明确否定保留不支持，不冒充职责级语义判断。
        if re.search(NEGATIVE, text) and not any(
            values[k] for k in ("excluded_cities", "excluded_employment_types")
        ):
            unknown = True
        if skill_or:
            unknown = True
        # 具有地点标记但无可识别城市，不能静默忽略。
        if re.search(r"地点|城市|只去", text) and not city_matches and not re.search(r"不限|其他|全国", text):
            unknown = True
        if required and not unknown:
            remainder = text
            for term in sorted(
                SKILLS + CITIES + list(ALIASES) + ["全职", "实习", "校园", "兼职/临时"], key=len, reverse=True
            ):
                remainder = re.sub(
                    term_pattern(term) + ("市?" if term in CITIES else ""), "", remainder, flags=re.I
                )
            for word in sorted(ROLE_WORDS + FILLER + ["或者", "或", "也"], key=len, reverse=True):
                remainder = remainder.replace(word, "")
            remainder = re.sub(r"[\s,，:：、()（）?？]+", "", remainder)
            if not matched or remainder:
                unknown = True
        if unknown:
            unresolved.append(text.strip())
            source("unsupported", text.strip(), "unsupported", offset, offset + len(text))
    combined = {}
    for field in explicit:
        if field in ("cities", "employment_types") and explicit[field] and natural[field]:
            common = [v for v in explicit[field] if v in natural[field]]
            if not common:
                conflicts.append(f"自然语言与显式{field}条件冲突")
                combined[field] = explicit[field]
            else:
                combined[field] = common
        else:
            combined[field] = unique(explicit[field] + natural[field])
    for field in ("cities", "employment_types"):
        if combined[field] and set(combined[field]) <= set(combined["excluded_" + field]):
            conflicts.append(f"{field}正向条件全部被排除")
    for field, values in combined.items():
        limit = 5 if "employment_types" in field else 10
        if len(values) > limit:
            unresolved.append(f"{field}条件数量超过{limit}项")
            combined[field] = values[:limit]
    unresolved = unique(unresolved + conflicts)
    return Intent(
        hard=HardConditions(**combined, education_max=education_max, include_unknown_education=pending),
        soft_preferences={k: unique(v) for k, v in soft.items() if v},
        sources=sources,
        unresolved=unresolved,
        conflicts=unique(conflicts),
    )
