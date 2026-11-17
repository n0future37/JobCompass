"""JSON/CSV外部岗位导入。原文件保持不变，统一字段严格校验。"""

import csv
import io
import json
import re
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator
from typing import Literal

from .conditions import FilterText

from .preprocessing import SCHEMA_VERSION, clean_text, digest, write_jsonl


class ImportJob(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: str = Field(pattern=r"^[a-z][a-z0-9_]{0,31}$")
    source_job_id: str = Field(min_length=1, max_length=128)
    title: str = Field(min_length=1, max_length=500)
    jd_text: str = Field(min_length=1, max_length=100000)
    company: str | None = None
    cities: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    location_text: str | None = None
    salary_text: str | None = None
    education_text: str | None = None
    experience_text: str | None = None
    employment_type_text: str | None = None
    attendance_text: str | None = None
    duration_text: str | None = None
    source_published_at_text: str | None = None
    source_updated_at_text: str | None = None
    url: str | None = None
    active: bool = True

    @field_validator("source_job_id", "title", "jd_text")
    @classmethod
    def nonblank(cls, value):
        value = value.strip()
        if not value:
            raise ValueError("必填字段不能只含空白")
        return value


def parse_file(path):
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.casefold() == ".csv":
        values = list(csv.DictReader(io.StringIO(text)))
        for row in values:
            for field in ("cities", "skills"):
                row[field] = [x.strip() for x in re.split(r"[|,，]", row.get(field) or "") if x.strip()]
            if "active" in row:
                if row["active"].lower() not in {"true", "false", "1", "0"}:
                    raise ValueError("CSV active只接受true/false/1/0")
    else:
        values = json.loads(text)
    if not isinstance(values, list) or not 1 <= len(values) <= 10000:
        raise ValueError("导入需要1至10000条岗位的JSON数组或CSV")
    return [ImportJob.model_validate(value) for value in values]


def make_snapshot(path, output):
    observations = parse_file(path)
    now = datetime.now(UTC).isoformat()
    rows = {}
    from .settings import ROOT

    provenance = dict(
        raw_file=path.relative_to(ROOT).as_posix(),
        sha256=digest(path.read_bytes()),
        collected_at=now,
        jd_source_field="jd_text",
    )
    for job in observations:
        row = job.model_dump()
        raw = row["jd_text"]
        row["jd_text"] = clean_text(raw, html="<p" in raw.lower() or "<div" in raw.lower())
        if not row["jd_text"]:
            raise ValueError("清洗后JD为空")
        row.update(
            schema_version=SCHEMA_VERSION,
            job_key=f"{row['source']}:{row['source_job_id']}",
        )
        row["content_hash"] = digest(row)
        row.update(
            raw_jd=raw,
            raw_jd_format="imported_text",
            jd_source_field="jd_text",
            source_status_raw=None,
            collected_at=now,
            provenance=[provenance],
            version=1,
            duplicate_group=digest(
                [
                    row["company"],
                    row["title"],
                    re.sub(r"\s+", "", row["jd_text"]).casefold(),
                ]
            ),
        )
        key = row["job_key"]
        if key in rows and rows[key]["content_hash"] != row["content_hash"]:
            raise ValueError("同一文件含相同岗位ID但不同内容，请明确版本后分批导入")
        rows[key] = row
    if output.exists():
        # 仅重用本任务创建的快照，先核对归属；不递归删除用户路径。
        manifest_path = output / "manifest.json"
        if (
            manifest_path.exists()
            and json.loads(manifest_path.read_bytes()).get("raw_fingerprint") == provenance["sha256"]
        ):
            return output
        raise ValueError("任务快照已存在但指纹不符")
    output.mkdir(parents=True)
    write_jsonl(output / "jobs.jsonl", rows.values())
    write_jsonl(output / "versions.jsonl", rows.values())
    manifest = dict(
        schema_version=SCHEMA_VERSION,
        raw_fingerprint=provenance["sha256"],
        jobs_sha256=digest((output / "jobs.jsonl").read_bytes()),
        jobs=len(rows),
    )
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    return output


class TextImportRequest(BaseModel):
    """用户填写的一个岗位；不从JD猜测未填写的岗位条件。"""

    model_config = ConfigDict(extra="forbid")
    title: str = Field(min_length=1, max_length=500)
    jd_text: str = Field(min_length=1, max_length=100000)
    company: str | None = Field(default=None, max_length=300)
    cities: list[FilterText] = Field(default_factory=list, max_length=10)
    employment_type_text: Literal["全职", "实习", "校招", "兼职"] | None = None
    salary_text: str | None = Field(default=None, max_length=100)
    education_text: str | None = Field(default=None, max_length=100)
    experience_text: str | None = Field(default=None, max_length=100)
    url: HttpUrl | None = None
    source_job_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_.:-]{1,80}$")

    @field_validator("title", "jd_text")
    @classmethod
    def not_blank(cls, value):
        if not value.strip():
            raise ValueError("岗位名称和JD正文不能为空白")
        return value  # 包装原件保留用户粘贴的文本，清洗仍由worker执行。

    @field_validator("company", "salary_text", "education_text", "experience_text")
    @classmethod
    def optional_text(cls, value):
        return value.strip() or None if value is not None else None


def prepare_text_import(value, owner):
    row = value.model_dump(mode="json", exclude={"source_job_id"})
    row["source"] = "manual"
    # 同用户有稳定编号或URL时更新同一岗位；无身份依据时只能按完整内容区分。
    if value.source_job_id:
        identity = "id_" + value.source_job_id
    elif value.url:
        from urllib.parse import urlsplit, urlunsplit

        url = urlsplit(str(value.url))
        identity = "url_" + digest(urlunsplit(url._replace(fragment="")))[:40]
    else:
        normalized = ImportJob(source_job_id="pending", **row).model_dump(exclude={"source_job_id"})
        identity = "text_" + digest(normalized)[:40]
    row["source_job_id"] = digest(owner)[:24] + "_" + identity
    ImportJob.model_validate(row)
    content = json.dumps([row], ensure_ascii=False)
    return dict(filename="pasted_job.json", content=content, idempotency_key="text:" + digest(content))
