"""明确硬条件与软查询分离；未知字段不判为满足。"""

import re
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

from .qualifications import Education, education_check

FilterText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=50)]

SKILLS = [
    "Python",
    "Java",
    "C++",
    "C#",
    "SQL",
    "MySQL",
    "Redis",
    "Milvus",
    "FastAPI",
    "Django",
    "Flask",
    "PyTorch",
    "Docker",
    "Kubernetes",
    "RAG",
    "Agent",
    "LangChain",
    "机器学习",
    "计算机视觉",
    "推荐算法",
    "数据分析",
    "自动化测试",
    "Excel",
    "ERP",
    "SAP",
    "CAD",
    "Photoshop",
    "Word",
    "PPT",
    "财务报表",
    "质量管理",
]
CITIES = [
    "深圳",
    "北京",
    "上海",
    "广州",
    "杭州",
    "南京",
    "苏州",
    "武汉",
    "成都",
    "西安",
    "天津",
    "重庆",
    "长沙",
    "郑州",
    "大连",
    "长春",
    "沈阳",
    "济南",
    "青岛",
    "无锡",
    "宁波",
    "福州",
    "厦门",
    "哈尔滨",
]


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=1000)
    mode: str = "hybrid"
    top_k: int = Field(default=5, ge=1, le=20)
    rerank: bool = True
    generate: bool = True
    cities: list[FilterText] = Field(default_factory=list, max_length=10)
    skills_all: list[FilterText] = Field(default_factory=list, max_length=10)
    employment_types: list[FilterText] = Field(default_factory=list, max_length=5)
    excluded_cities: list[FilterText] = Field(default_factory=list, max_length=10)
    excluded_employment_types: list[FilterText] = Field(default_factory=list, max_length=5)
    education_max: Education | None = None
    include_unknown_education: bool = False
    profile: str = Field(default="", max_length=4000)

    @field_validator("query")
    @classmethod
    def query_not_blank(cls, value):
        if not value.strip():
            raise ValueError("查询不能为空白")
        return value.strip()

    @field_validator("mode")
    @classmethod
    def valid_mode(cls, value):
        if value not in {"bm25", "dense", "hybrid"}:
            raise ValueError("检索模式必须为bm25/dense/hybrid")
        return value


def term_pattern(term):
    if re.fullmatch(r"[A-Za-z0-9+#.]+", term):
        return r"(?<![A-Za-z0-9_])" + re.escape(term) + r"(?![A-Za-z0-9_])"
    return re.escape(term)


def mention_lines(job, term):
    pattern = re.compile(term_pattern(term), re.IGNORECASE)
    result = []
    for index, line in enumerate(job["jd_text"].splitlines(), 1):
        if pattern.search(line):
            # 明确否定不能作正向匹配依据。
            if re.search(
                r"(?:不要求|无需|不使用|不涉及)[^，,。；;\n]{0,12}" + term_pattern(term),
                line,
                re.IGNORECASE,
            ):
                continue
            result.append(index)
    return result


def profile_mentions(profile):
    """只表示画像自述提及；否定限于同一分句，不推断已获得任职资格。"""
    return [
        term
        for term in SKILLS
        if re.search(term_pattern(term), profile, re.I)
        and not re.search(
            r"(?:不会|未掌握|不熟悉|没有)[^，,。；;\n]{0,8}" + term_pattern(term), profile, re.I
        )
    ]


def parse_conditions(request):
    from .intent import parse_intent

    intent = parse_intent(request)
    return intent.conditions(), intent.unresolved


def assess(job, conditions):
    from .intent import city_name, employment_name

    checks = []
    actual_cities = [city_name(c) for c in (job.get("cities") or [])]
    if conditions["cities"]:
        requested = [city_name(c) for c in conditions["cities"]]
        matched = [s for s in requested if s in actual_cities]
        checks.append(
            dict(
                field="cities",
                satisfied=bool(matched),
                unknown=not actual_cities,
                requested=requested,
                evidence=actual_cities,
            )
        )
    excluded = conditions.get("excluded_cities", [])
    if excluded:
        permitted = [c for c in actual_cities if c not in excluded]
        if conditions["cities"]:
            permitted = [c for c in permitted if c in conditions["cities"]]
        checks.append(
            dict(
                field="excluded_cities",
                satisfied=bool(permitted),
                unknown=not actual_cities,
                requested=excluded,
                evidence=actual_cities,
                permitted=permitted,
            )
        )
    actual = employment_name(job.get("employment_type_text") or "")
    known = actual in {"全职", "实习", "校园", "兼职/临时"}
    if conditions["employment_types"]:
        requested = [employment_name(x) for x in conditions["employment_types"]]
        checks.append(
            dict(
                field="employment_type",
                satisfied=known and actual in requested,
                unknown=not known,
                requested=requested,
                evidence=actual or None,
            )
        )
    excluded = conditions.get("excluded_employment_types", [])
    if excluded:
        checks.append(
            dict(
                field="excluded_employment_types",
                satisfied=known and actual not in excluded,
                unknown=not known,
                requested=excluded,
                evidence=actual or None,
            )
        )
    for term in conditions["skills_all"]:
        lines = mention_lines(job, term)
        checks.append(
            dict(
                field="jd_mentions",
                term=term,
                satisfied=bool(lines),
                unknown=not lines,
                requested=term,
                lines=lines,
            )
        )
    if conditions.get("education_max"):
        checks.append(
            education_check(
                job, conditions["education_max"], conditions.get("include_unknown_education", False)
            )
        )
    return all(x["satisfied"] or x.get("allowed_pending", False) for x in checks), checks
