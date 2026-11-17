"""私有简历原件/解析草稿与经过用户确认的画像；所有处理均在本地。"""

import json
import subprocess
import sys
import tempfile
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import (
    JSON,
    Column,
    ForeignKey,
    LargeBinary,
    String,
    Table,
    UniqueConstraint,
    delete,
    select,
    update,
)
from sqlalchemy.dialects.mysql import MEDIUMBLOB

from .career import Background, ConflictError
from .identity import users
from .preprocessing import digest
from .resume_extract import MAX_BYTES
from .resume_parser import PARSER_VERSION, parse_resume
from .storage import metadata

resumes = Table(
    "rag_resumes",
    metadata,
    Column("id", String(36), primary_key=True),
    Column("owner", String(36), ForeignKey("rag_users.id", ondelete="CASCADE"), nullable=False, index=True),
    Column("sha256", String(64), nullable=False),
    Column("parser_version", String(40), nullable=False),
    Column("filename", String(200), nullable=False),
    Column("status", String(20), nullable=False),
    Column("created_at", String(40), nullable=False),
    Column("original", LargeBinary().with_variant(MEDIUMBLOB(), "mysql"), nullable=False),
    Column("payload", JSON, nullable=False),
    UniqueConstraint("owner", "sha256", "parser_version", name="uq_resume_owner_content_parser"),
)
parse_slots = threading.BoundedSemaphore(2)


class ResumeError(ValueError):
    pass


class ConfirmResume(BaseModel):
    model_config = ConfigDict(extra="forbid")
    profile_etag: str = Field(pattern=r"^[a-f0-9]{64}$")
    background: Background
    profile_text: str = Field(max_length=4000)


def profile_etag(profile):
    return digest({"text": profile.get("text", ""), "background": profile.get("background", {})})


def extract_bounded(content, extension, timeout=15):
    if not parse_slots.acquire(blocking=False):
        raise RuntimeError("简历解析繁忙，请稍后重试")
    try:
        with tempfile.TemporaryDirectory(prefix="jc_resume_") as temp:
            path = Path(temp) / ("input" + extension)
            path.write_bytes(content)
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "jobcompass.resume_extract", str(path), extension],
                    capture_output=True,
                    timeout=timeout,
                    check=False,
                    creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
                )
            except subprocess.TimeoutExpired:
                raise ResumeError("解析超过15秒，请精简文件或导出TXT后重传") from None
        try:
            value = json.loads(result.stdout)
        except (ValueError, UnicodeError):
            raise ResumeError("文件解析未完成，请导出标准文件后重传") from None
        if not value.get("ok"):
            raise ResumeError(value.get("error", "文件解析失败"))
        return value
    finally:
        parse_slots.release()


class ResumeStore:
    def __init__(self, store):
        self.store = store

    def get(self, owner, resume_id, include_original=False):
        with self.store.engine.connect() as c:
            row = (
                c.execute(
                    select(
                        *[column for column in resumes.c if include_original or column.name != "original"]
                    ).where(resumes.c.id == resume_id, resumes.c.owner == owner)
                )
                .mappings()
                .first()
            )
        if not row:
            raise LookupError("简历不存在")
        return dict(row)

    def list(self, owner):
        with self.store.engine.connect() as c:
            return [
                dict(r)
                for r in c.execute(
                    select(resumes.c.id, resumes.c.filename, resumes.c.created_at, resumes.c.status)
                    .where(resumes.c.owner == owner)
                    .order_by(resumes.c.created_at.desc())
                    .limit(20)
                ).mappings()
            ]

    def upload(self, owner, filename, content):
        filename = filename.replace("\\", "/").split("/")[-1]
        if not filename or len(filename) > 200 or any(ord(c) < 32 for c in filename):
            raise ResumeError("文件名无效或过长")
        extension = Path(filename).suffix.lower()
        if extension not in {".pdf", ".docx", ".txt"}:
            raise ResumeError("仅支持PDF、DOCX和UTF-8 TXT")
        if not content or len(content) > MAX_BYTES:
            raise ResumeError("文件为空或超过5MB")
        sha = digest(content)
        with self.store.engine.connect() as c:
            old = c.execute(
                select(resumes.c.id).where(
                    resumes.c.owner == owner,
                    resumes.c.sha256 == sha,
                    resumes.c.parser_version == PARSER_VERSION,
                )
            ).first()
        if old:
            return self.preview(owner, old[0])
        extracted = extract_bounded(content, extension)
        parsed = parse_resume(extracted["text"])
        parsed.update(text=extracted["text"], warnings=extracted["warnings"] + parsed["warnings"])
        with self.store.engine.begin() as c:
            if not c.execute(select(users.c.id).where(users.c.id == owner).with_for_update()).first():
                raise LookupError("账号不存在")
            # 每账号保留最多20个文件；同内容重传幂等，不占新名额。
            existing = c.execute(select(resumes.c.id).where(resumes.c.owner == owner)).scalars().all()
            old = c.execute(
                select(resumes.c.id).where(
                    resumes.c.owner == owner,
                    resumes.c.sha256 == sha,
                    resumes.c.parser_version == PARSER_VERSION,
                )
            ).first()
            if old:
                resume_id = old[0]
            else:
                if len(existing) >= 20:
                    raise ResumeError("已保存20份简历，请删除不再需要的原件后重传")
                resume_id = str(uuid.uuid4())
                c.execute(
                    resumes.insert().values(
                        id=resume_id,
                        owner=owner,
                        sha256=sha,
                        parser_version=PARSER_VERSION,
                        filename=filename,
                        status="draft",
                        created_at=datetime.now(UTC).isoformat(),
                        original=content,
                        payload=parsed,
                    )
                )
        return self.preview(owner, resume_id)

    def preview(self, owner, resume_id):
        row = self.get(owner, resume_id)
        with self.store.engine.connect() as c:
            profile = c.execute(select(users.c.profile).where(users.c.id == owner)).scalar_one()
        background = Background.model_validate(profile.get("background", {})).model_dump()
        # 草稿仅填原来空缺的字段；已有值保留，差异明示让用户决定。
        conflicts = []
        for field, value in row["payload"]["suggestions"].items():
            if background[field] in (None, "", []):
                background[field] = value
            elif background[field] != value:
                conflicts.append(field)
        return dict(
            id=row["id"],
            filename=row["filename"],
            status=row["status"],
            created_at=row["created_at"],
            **{key: value for key, value in row["payload"].items() if key != "last_confirmation"},
            proposed_background=background,
            proposed_profile_text=profile.get("text") or row["payload"]["suggested_profile_text"],
            profile_etag=profile_etag(profile),
            conflicts=conflicts,
        )

    def confirm(self, owner, resume_id, value):
        fingerprint = digest(value.model_dump())
        with self.store.engine.begin() as c:
            row = c.execute(select(users.c.profile).where(users.c.id == owner).with_for_update()).first()
            if not row:
                raise LookupError("账号不存在")
            profile = dict(row[0])
            resume = (
                c.execute(
                    select(resumes)
                    .where(resumes.c.id == resume_id, resumes.c.owner == owner)
                    .with_for_update()
                )
                .mappings()
                .first()
            )
            if not resume:
                raise LookupError("简历不存在")
            payload = dict(resume["payload"])
            if payload.get("last_confirmation") == fingerprint:
                return dict(status="confirmed", already_applied=True, profile=profile)
            if value.profile_etag != profile_etag(profile):
                raise ConflictError("画像已在其他位置修改，请重新载入预览后核对保存")
            profile.update(
                text=value.profile_text,
                background=value.background.model_dump(),
                resume_source=dict(
                    id=resume_id,
                    sha256=resume["sha256"],
                    parser_version=resume["parser_version"],
                    confirmed_at=datetime.now(UTC).isoformat(),
                ),
            )
            payload["last_confirmation"] = fingerprint
            c.execute(update(users).where(users.c.id == owner).values(profile=profile))
            c.execute(
                update(resumes).where(resumes.c.id == resume_id).values(status="confirmed", payload=payload)
            )
        return dict(status="confirmed", already_applied=False, profile=profile)

    def delete(self, owner, resume_id):
        with self.store.engine.begin() as c:
            c.execute(select(users.c.id).where(users.c.id == owner).with_for_update())
            result = c.execute(delete(resumes).where(resumes.c.id == resume_id, resumes.c.owner == owner))
            if not result.rowcount:
                raise LookupError("简历不存在")
