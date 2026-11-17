"""保守的中文简历规则解析；所有候选值均需用户核对确认。"""

import re

from .conditions import CITIES, SKILLS, term_pattern

PARSER_VERSION = "resume-local.v1"
HEADINGS = {
    "education": r"教育背景|教育经历|学历信息|Education",
    "skills": r"专业技能|技术技能|技能清单|职业技能|技能|Skills",
    "projects": r"项目经历|项目经验|个人项目|科研项目|Projects?",
    "work": r"工作经历|工作经验|实习经历|工作实习经历|Experience",
    "intent": r"求职意向|意向岗位|职业意向|求职目标|职业目标",
    "other": r"个人信息|基本信息|联系方式|自我评价|个人评价|荣誉奖项|获奖经历|证书|语言能力",
}
TERMS = list(
    dict.fromkeys(
        SKILLS + ["LangGraph", "Go", "JavaScript", "TypeScript", "React", "Vue", "Linux", "Git", "PostgreSQL"]
    )
)


def parse_resume(text):
    lines = text.splitlines()
    sections, evidence = [], {}
    warnings = ["解析候选需逐项核对；关键词提及不代表能力已认证，缺失信息保持未知"]
    section = None
    for number, line in enumerate(lines, 1):
        content = line
        for name, pattern in HEADINGS.items():
            match = re.fullmatch(r"(?:" + pattern + r")\s*(?:[:：]\s*(.*))?", line, re.I)
            if match:
                section, content = name, match[1] or ""
                break
        sections.append((number, line, section, content))

    def add(field, number, line):
        item = dict(line=number, quote=line)
        if item not in evidence.setdefault(field, []):
            evidence[field].append(item)

    education = []
    for number, line, section, content in sections:
        if section != "education" and not re.search(r"(?:最高)?学历\s*[:：]", line):
            continue
        if re.search(r"在读|拟入学|计划|预计|申请|意向", line):
            warnings.append("发现在读或计划学历，未自动计为已获得学历")
            continue
        for degree, rank in (("大专", 1), ("本科", 2), ("硕士", 3), ("博士", 4)):
            if degree in content:
                education.append((rank, degree, number, line))
    suggestions = {}
    if education:
        _, degree, number, line = max(education)
        suggestions["education"] = degree
        add("education", number, line)
        warnings.append("请确认所选学历是否已经取得，学校与毕业状态未自动核验")
    years = []
    for number, line, _, _ in sections:
        if re.search(r"计划|预计|目标|希望|实习经验|实习经历|没有|不足|不到|\d\s*[-至~]\s*\d", line):
            continue
        match = re.search(r"(?:工作经验\s*[:：]?\s*(\d+(?:\.\d+)?)\s*年|(\d+(?:\.\d+)?)\s*年工作经验)", line)
        if match and 0 <= float(match[1] or match[2]) <= 60:
            years.append((float(match[1] or match[2]), number, line))
    if years and len({r[0] for r in years}) == 1:
        suggestions["experience_years"] = years[0][0]
        add("experience_years", years[0][1], years[0][2])
    else:
        warnings.append("未找到唯一明确的工作年限，请手动填写；不累加在校、实习或项目日期")
    for number, line, section, content in sections:
        if section == "education":
            match = re.search(r"专业\s*[:：]\s*([^|,，;；]{1,100})", content)
            if match and "major" not in suggestions:
                suggestions["major"] = match[1].strip()
                add("major", number, line)
    skills = []
    for term in TERMS:
        for number, line, section, content in sections:
            if section not in {"skills", "projects", "work"}:
                continue
            if not re.search(term_pattern(term), content, re.I):
                continue
            # 对学习意愿和否定表述保持保守，避免把期望当能力。
            if re.search(
                r"不会|未掌握|不熟悉|没有|未接触|计划学习|希望学习|想学|待学习|学习计划|尚未", content
            ):
                continue
            skills.append(term)
            add("skills:" + term, number, line)
            break
    if skills:
        suggestions["skills"] = skills
    project_lines = [
        (number, line, content)
        for number, line, section, content in sections
        if section == "projects" and content
    ]
    if project_lines:
        projects = "\n".join(content for _, _, content in project_lines)
        if len(projects) > 4000:
            warnings.append("项目经历超过4000字符，预览已截取；完整提取原文仍可查看")
        suggestions["projects"] = projects[:4000]
        for number, line, _ in project_lines[:15]:
            add("projects", number, line)
    intent = [(n, line, content) for n, line, section, content in sections if section == "intent" and content]
    cities = [city for city in CITIES if any(city in c for _, _, c in intent)]
    if cities:
        suggestions["preferred_cities"] = cities
        for number, line, content in intent:
            if any(c in content for c in cities):
                add("preferred_cities", number, line)
    if not suggestions:
        warnings.append("未识别到可靠字段，可参考提取原文手动填写画像")
    profile = []
    for key, label in (("education", "学历"), ("experience_years", "工作经验(年)"), ("major", "专业")):
        if key in suggestions:
            profile.append(f"{label}：{suggestions[key]}")
    if skills:
        profile.append("自述技能：" + "、".join(skills))
    if suggestions.get("projects"):
        profile.append("项目经历：\n" + suggestions["projects"])
    return dict(
        parser_version=PARSER_VERSION,
        suggestions=suggestions,
        evidence=evidence,
        warnings=list(dict.fromkeys(warnings)),
        suggested_profile_text="\n".join(profile)[:4000],
    )
