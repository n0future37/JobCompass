"""学历下限证据与本轮学历约束；冲突及缺失不冒充满足。"""

import re
from typing import Literal

Education = Literal["初中", "高中", "中专", "大专", "本科", "硕士", "博士"]
DEGREES = {"初中": 1, "高中": 2, "中专": 2, "大专": 3, "本科": 4, "硕士": 5, "博士": 6}
ALIASES = {
    "专科": "大专",
    "研究生": "硕士",
    "中技": "中专",
    "中专/中技": "中专",
    "大学本科": "本科",
    "大学专科": "大专",
    "硕士研究生": "硕士",
    "博士研究生": "博士",
    "初中及以下": "初中",
}
DEGREE = "|".join(sorted(set(DEGREES) | set(ALIASES), key=len, reverse=True))


def normalize_degree(value):
    return ALIASES.get(value, value)


def education_requirement(job):
    """分别记录来源字段和正文要求；优先/加分不当硬门槛。"""
    evidence = []
    raw = (job.get("education_text") or "").strip()
    if re.fullmatch(r"(?:学历)?不限|无学历要求|不要求学历", raw):
        evidence.append(dict(source="education_text", text=raw, minimum=0))
    else:
        m = re.fullmatch(r"(" + DEGREE + r")(?:及以上|以上)?(?:学历|学位)?", raw)
        if m:
            evidence.append(dict(source="education_text", text=raw, minimum=DEGREES[normalize_degree(m[1])]))
    headings = re.compile(
        r"(?P<duties>岗位职责|工作职责|工作内容|主要职责|主要工作|工作任务|岗位内容|你要做的事)"
        r"|(?P<qualification>任职要求|任职资格|岗位要求|岗位条件|职位要求|应聘条件|学历要求|专业要求|核心要求)"
        r"|(?P<other>福利待遇|薪酬福利|薪资福利|职位福利|福利|待遇|我们为你提供|工作时间|工作地点|个人发展|公司简介)"
    )
    section = "unknown"
    for number, line in enumerate(job.get("jd_text", "").split("\n"), 1):
        segments, start = [], 0
        for marker in headings.finditer(line):
            segments.append((section, line[start : marker.start()]))
            section, start = marker.lastgroup, marker.start()
        segments.append((section, line[start:]))
        for kind, text in segments:
            if kind in {"duties", "other"}:
                continue
            for clause in re.split(r"[，,。；;]", text):
                if re.search(r"学历不限|不限学历|无学历要求|不要求学历", clause):
                    evidence.append(dict(source="jd_text", line=number, text=clause.strip(), minimum=0))
                    continue
                if re.search(r"同等经验|同等能力|不要求|无需|学历提升|落户|补贴|报销", clause):
                    continue
                matches = list(
                    re.finditer(
                        r"(" + DEGREE + r")(?:(?:及以上|以上)(?:学历|学位|毕业)?|(?:学历|学位|毕业))", clause
                    )
                )
                if not matches:
                    matches = list(
                        re.finditer(r"(?:学历要求|最低学历|学历门槛|学历)[:：为\s]*(" + DEGREE + r")", clause)
                    )
                for match in matches:
                    # “本科以上学历，机械专业优先”仍有学历硬门槛；只有直接修饰学历的优先才忽略。
                    if re.match(r"(?:者)?(?:优先|加分|更佳|为佳)", clause[match.end() :].strip()):
                        continue
                    evidence.append(
                        dict(
                            source="jd_text",
                            line=number,
                            text=clause.strip(),
                            minimum=DEGREES[normalize_degree(match[1])],
                        )
                    )
    minima = sorted({e["minimum"] for e in evidence})
    return dict(
        status="conflict" if len(minima) > 1 else "known" if minima else "unknown",
        minimum=minima[0] if len(minima) == 1 else None,
        known_minima=minima,
        evidence=evidence,
    )


def education_check(job, maximum, include_unknown=False):
    requirement = education_requirement(job)
    bound = DEGREES[maximum]
    known_failure = any(n > bound for n in requirement["known_minima"])
    satisfied = requirement["status"] == "known" and not known_failure
    unknown = requirement["status"] != "known" and not known_failure
    return dict(
        field="education",
        requested=maximum,
        satisfied=satisfied,
        unknown=unknown,
        status="fail" if known_failure else "pass" if satisfied else "pending",
        allowed_pending=bool(unknown and include_unknown),
        requirement_status=requirement["status"],
        evidence=requirement["evidence"],
    )


def education_intent(query):
    """返回可证明的学历上限、原文跨度及同长度掩码，供通用解析继续检查其他条件。"""
    chars = list(query)
    sources, bounds = [], []
    pending = False

    def consume(match, value, field="education_max", hard=True):
        if field == "education_max" and hard:
            bounds.append(value)
        sources.append(
            dict(
                field=field,
                value=str(value),
                source="query",
                strength="hard" if hard else "history",
                quote=match[0],
                start=match.start(),
                end=match.end(),
            )
        )
        chars[match.start() : match.end()] = " " * (match.end() - match.start())

    fact = re.compile(
        r"(?:我(?:是|只有|有|为)?|本人(?:是|只有|有|为)?)(?:目前|已经|今年)?("
        + DEGREE
        + r")(?:毕业|学历|学位|应届生)?"
    )
    for m in fact.finditer(query):
        prefix = re.split(r"[，,。；;]", query[: m.start()])[-1]
        consume(m, normalize_degree(m[1]), hard=not bool(re.search(r"之前|以前|曾经|过去", prefix)))
    cap = re.compile(
        r"(?:学历(?:门槛|要求|下限)?|最低学历)(?:最高|不高于|不超过|至多|最多|为|[:：])\s*("
        + DEGREE
        + r")(?:及以下|以下)?|("
        + DEGREE
        + r")(?:及以下|以下)(?:学历|门槛)?"
    )
    for m in cap.finditer(query):
        consume(m, normalize_degree(m[1] or m[2]))
    exclude = re.compile(
        r"(?:不接受|不考虑|排除|不要|没有|无)(?:必须|要求|强制要求)?("
        + DEGREE
        + r")(?:及以上|以上)?(?:学历)?(?:硬门槛|门槛|硬性要求|要求)"
    )
    for m in exclude.finditer(query):
        degree = normalize_degree(m[1])
        lower = next((d for d, n in DEGREES.items() if n == DEGREES[degree] - 1), None)
        if lower:
            consume(m, lower)
    unknown = re.compile(
        r"学历(?:没写|未写|未说明|没注明|缺失|未知)[^，,。；;]{0,12}(?:待确认|待核实|也可以|也展示|保留|显示)"
    )
    for m in unknown.finditer(query):
        pending = True
        consume(m, "pending", "include_unknown_education")
    maximum = min(bounds, key=DEGREES.get) if bounds else None
    return maximum, pending, sources, "".join(chars)
