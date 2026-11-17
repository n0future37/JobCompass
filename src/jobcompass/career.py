"""账号私有的背景、岗位偏好和投递记录；不代为向招聘网站投递。"""

import re
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import JSON, Column, ForeignKey, Integer, String, Table, select, update

from .conditions import FilterText, mention_lines, profile_mentions
from .qualifications import DEGREES, Education, education_check, education_requirement
from .identity import users
from .storage import jobs, metadata


class Background(BaseModel):
    model_config = ConfigDict(extra="forbid")
    education: Education | None = None
    experience_years: float | None = Field(default=None, ge=0, le=60)
    major: str = Field(default="", max_length=100)
    skills: list[FilterText] = Field(default_factory=list, max_length=30)
    projects: str = Field(default="", max_length=4000)
    interests: list[FilterText] = Field(default_factory=list, max_length=15)
    preferred_cities: list[FilterText] = Field(default_factory=list, max_length=10)


class Preference(BaseModel):
    model_config = ConfigDict(extra="forbid")
    preference: Literal["like", "neutral", "dislike"]
    note: str = Field(default="", max_length=1000)


class Application(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["not_applied", "applied", "interviewing", "offer", "rejected"]
    note: str = Field(default="", max_length=1000)
    expected_version: int | None = Field(default=None, ge=0)


preferences = Table(
    "rag_job_preferences",
    metadata,
    Column("owner", String(36), ForeignKey("rag_users.id", ondelete="CASCADE"), primary_key=True),
    Column("job_key", String(192, collation="utf8mb4_bin"), ForeignKey("rag_jobs.job_key"), primary_key=True),
    Column("preference", String(16), nullable=False),
    Column("note", String(1000), nullable=False),
    Column("updated_at", String(40), nullable=False),
)
applications = Table(
    "rag_applications",
    metadata,
    Column("owner", String(36), ForeignKey("rag_users.id", ondelete="CASCADE"), primary_key=True),
    Column("job_key", String(192, collation="utf8mb4_bin"), ForeignKey("rag_jobs.job_key"), primary_key=True),
    Column("status", String(20), nullable=False),
    Column("note", String(1000), nullable=False),
    Column("version", Integer, nullable=False),
    Column("snapshot", JSON, nullable=False),
    Column("updated_at", String(40), nullable=False),
)


class ConflictError(ValueError):
    pass


class Career:
    def __init__(self, store):
        self.store = store

    def background(self, owner, value=None):
        with self.store.engine.begin() as c:
            row = c.execute(select(users.c.profile).where(users.c.id == owner).with_for_update()).first()
            if not row:
                raise LookupError("账号不存在")
            profile = dict(row[0])
            if value is not None:
                profile["background"] = value.model_dump()
                c.execute(update(users).where(users.c.id == owner).values(profile=profile))
            return Background.model_validate(profile.get("background", {})).model_dump()

    def records(self, owner, table):
        with self.store.engine.connect() as c:
            return [
                dict(r)
                for r in c.execute(
                    select(table).where(table.c.owner == owner).order_by(table.c.updated_at.desc())
                ).mappings()
            ]

    def put(self, owner, key, value):
        is_application = isinstance(value, Application)
        table = applications if is_application else preferences
        with self.store.engine.begin() as c:
            # 同一用户的写入串行；同时避免“首次创建”竞态及画像覆盖。
            if not c.execute(select(users.c.id).where(users.c.id == owner).with_for_update()).first():
                raise LookupError("账号不存在")
            job = c.execute(select(jobs.c.payload, jobs.c.active).where(jobs.c.job_key == key)).first()
            if not job:
                raise LookupError("岗位不存在")
            where = (table.c.owner == owner, table.c.job_key == key)
            old = c.execute(select(table).where(*where).with_for_update()).mappings().first()
            data = value.model_dump(exclude={"expected_version"})
            # 相同提交幂等，包括客户端未收到上次成功响应后的重试。
            if old and all(old[k] == v for k, v in data.items()):
                return dict(old)
            if is_application:
                version = old["version"] if old else 0
                if value.expected_version is not None and version != value.expected_version:
                    raise ConflictError("记录已变化，请刷新后重试")
                data.update(
                    version=version + 1,
                    snapshot={k: job[0].get(k) for k in ("title", "company", "url", "version")},
                )
            data.update(updated_at=datetime.now(UTC).isoformat())
            if old:
                c.execute(update(table).where(*where).values(**data))
            else:
                c.execute(table.insert().values(owner=owner, job_key=key, **data))
            return dict(c.execute(select(table).where(*where)).mappings().one())


def background_checks(job, background):
    """只判断可解析的学历/经验下限；缺失、复杂或矛盾表述保持未知。"""
    checks = []
    for field, fact in (("education", background.education), ("experience", background.experience_years)):
        raw = job.get("education_text" if field == "education" else "experience_text")
        raw = raw.strip() if isinstance(raw, str) else ""
        minimum = None
        if raw and re.fullmatch(r"(?:学历|经验|工作经验)?不限|无(?:工作)?经验|应届(?:毕业生)?", raw):
            minimum = 0
        elif field == "education":
            match = re.fullmatch(r"(大专|本科|硕士|博士)(?:及以上|以上|学历)?", raw)
            if match:
                minimum = DEGREES[match[1]]
        else:
            match = re.fullmatch(r"(\d+)(?:\s*[-~至—]\s*\d+)?年(?:以上|及以上)?(?:工作经验)?", raw)
            if match:
                minimum = int(match[1])
        if field == "education":
            requirement = education_requirement(job)
            minimum = requirement["minimum"]
            if fact is not None:
                check = education_check(job, fact)
                checks.append(
                    dict(
                        field=field,
                        requirement=raw or None,
                        user_value=fact,
                        minimum=minimum,
                        status="fail"
                        if check["status"] == "fail"
                        else "pass"
                        if check["satisfied"]
                        else "unknown",
                        evidence=check["evidence"],
                        requirement_status=check["requirement_status"],
                    )
                )
                continue
        actual = DEGREES.get(fact) if field == "education" else fact
        state = (
            "unknown"
            if minimum is None or (actual is None and minimum > 0)
            else "pass"
            if minimum == 0 or actual >= minimum
            else "fail"
        )
        checks.append(
            dict(field=field, requirement=raw or None, user_value=fact, minimum=minimum, status=state)
        )
    return checks


def soft_signals(job, background, preference=None, application=None):
    skills = [s for s in background.skills if mention_lines(job, s)]
    # 项目说明保留原文作为用户依据；不从项目名臆造技能。
    project_skills = [s for s in profile_mentions(background.projects) if mention_lines(job, s)]
    interests = [
        s for s in background.interests if s.casefold() in (job["title"] + "\n" + job["jd_text"]).casefold()
    ]
    city = sorted(set(background.preferred_cities) & set(job.get("cities", [])))
    liked = (preference or {}).get("preference") == "like"
    status = (application or {}).get("status")
    score = (
        min(len(skills), 5) * 0.3
        + min(len(project_skills), 5) * 0.2
        + min(len(interests), 3) * 0.2
        + bool(city) * 0.3
        + liked * 0.8
        - (status in {"applied", "interviewing"}) * 0.2
    )
    return dict(
        score=score,
        skills=skills,
        project_skills=sorted(project_skills),
        project_evidence=background.projects if project_skills else None,
        interests=interests,
        cities=city,
        preference=(preference or {}).get("preference", "neutral"),
        application_status=status,
    )
