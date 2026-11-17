"""从原始响应重新生成统一岗位、版本和可追溯来源；不修改raw。"""

import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path

SCHEMA_VERSION = "job.v1"


def digest(value):
    if not isinstance(value, bytes):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(value).hexdigest()


class TextParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self.hidden = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.hidden += 1
        if tag in {"p", "br", "div", "li", "section", "tr"} and not self.hidden:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in {"script", "style"}:
            self.hidden = max(0, self.hidden - 1)
        if tag in {"p", "div", "li", "section", "tr"} and not self.hidden:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def clean_text(value, html=False):
    value = str(value or "")
    if html:
        parser = TextParser()
        parser.feed(value)
        parser.close()
        value = "".join(parser.parts)
    value = value.replace("\u200b", "").replace("\ufeff", "").replace("\xa0", " ")
    value = value.replace("\uf0b7", "•")
    return "\n".join(
        line for line in (re.sub(r"[^\S\n]+", " ", line).strip() for line in value.splitlines()) if line
    )


def optional(value):
    if not isinstance(value, (str, int, float)):
        return None
    text = clean_text(value)
    return None if text.casefold() in {"", "未知", "未提供", "不限", "unknown"} else text


def field_text(value):
    # “不限”是来源明确给出的值，必须与未知区分。
    return clean_text(value) if isinstance(value, str) and value.strip() else None


def name_values(value):
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        return [s for item in value for s in name_values(item)]
    if isinstance(value, dict):
        for key in ("name", "value", "cityName"):
            if key in value:
                return name_values(value[key])
    return []


def normalize_zhilian(data, meta):
    position = data["data"]["detailedPosition"]
    company = data["data"].get("detailedCompany") or {}
    sid = str(position.get("number") or position.get("positionNumber") or "").strip()
    descriptions = {
        key: position[key]
        for key in ("jobDescPC", "jobDesc")
        if isinstance(position.get(key), str) and clean_text(position[key], html=True)
    }
    if not descriptions:
        raise ValueError("缺少完整JD描述")
    jd_field = max(descriptions, key=lambda k: len(clean_text(descriptions[k], html=True)))
    cities = list(
        dict.fromkeys(name_values(position.get("workCity")) + name_values(position.get("positionWorkCity")))
    )
    row = dict(
        source="zhilian",
        source_job_id=sid,
        title=field_text(position.get("positionName") or position.get("name")),
        company=field_text(company.get("companyName") or position.get("companyName")),
        cities=cities,
        location_text=field_text(position.get("workAddress")),
        salary_text=field_text(position.get("salary60")),
        education_text=field_text(position.get("education")),
        experience_text=field_text(position.get("workingExp") or position.get("positionWorkingExp")),
        employment_type_text=field_text(position.get("workType") or position.get("emplType")),
        skills=list(dict.fromkeys(name_values(position.get("skillLabel")))),
        attendance_text=None,
        duration_text=None,
        source_published_at_text=field_text(
            position.get("positionPublishTime") or position.get("publishTime")
        ),
        source_updated_at_text=None,
        source_status_raw=position.get("positionStatus"),
        url=field_text(position.get("positionUrl") or position.get("url")),
        raw_jd=descriptions[jd_field],
        raw_jd_format="html",
        jd_text=clean_text(descriptions[jd_field], html=True),
        jd_source_field=f"data.detailedPosition.{jd_field}",
    )
    return row


def normalize_shixiseng(body, meta):
    from .crawlers.shixiseng import parse_detail

    old = parse_detail(body, meta, "")
    return dict(
        source="shixiseng",
        source_job_id=old["source_job_id"],
        title=old["title"],
        company=old["company"],
        cities=[old["city"]] if old["city"] else [],
        location_text=old["location_text"],
        salary_text=old["salary_text"],
        education_text=old["education_text"],
        experience_text=None,
        employment_type_text="实习",
        skills=[],
        attendance_text=old["attendance_text"],
        duration_text=old["duration_text"],
        source_published_at_text=None,
        source_updated_at_text=old["source_updated_at_text"],
        source_status_raw=None,
        url=old["url"],
        raw_jd=old["jd_text"],
        raw_jd_format="extracted_text",
        jd_text=old["jd_text"],
        jd_source_field="HTML class=job_detail",
    )


def write_jsonl(path, rows):
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def preprocess(raw_root: Path, output: Path, allowed_raw_files=None):
    if output.exists():
        raise ValueError("输出快照已存在，拒绝覆盖；请指定新版本目录")
    selected = None
    if allowed_raw_files is not None:
        if not isinstance(allowed_raw_files, (list, tuple, set)) or not all(
            isinstance(s, str) for s in allowed_raw_files
        ):
            raise ValueError("原始文件选择清单必须是相对路径数组")
        selected = set(allowed_raw_files)
        root = raw_root.parent.parent.resolve()
        for value in selected:
            path = root / value
            if (
                not value.startswith("data/raw/")
                or "\\" in value
                or ".." in Path(value).parts
                or not value.endswith(".body")
                or not path.resolve().is_relative_to(raw_root.resolve())
                or not path.is_file()
            ):
                raise ValueError("选择清单路径不存在或越界")
    observations, skipped, rejected = [], [], []
    raw_hashes = [
        {
            "path": p.relative_to(raw_root.parent.parent).as_posix(),
            "sha256": digest(p.read_bytes()),
        }
        for p in sorted(raw_root.rglob("*"))
        if p.is_file() and (p.suffix == ".body" or p.name.endswith(".meta.json"))
    ]
    for path in sorted(raw_root.rglob("*.body")):
        relative = path.relative_to(raw_root.parent.parent).as_posix()
        if selected is not None and relative not in selected:
            skipped.append({"raw_file": relative, "reason": "不在本轮冻结的原始文件选择清单"})
            continue
        meta_path = path.with_suffix(".meta.json")
        body = path.read_bytes()
        if not meta_path.exists():
            rejected.append({"raw_file": relative, "reason": "缺少请求元数据"})
            continue
        meta_bytes = meta_path.read_bytes()
        try:
            meta = json.loads(meta_bytes)
            if not isinstance(meta, dict):
                raise ValueError("请求元数据必须是对象")
        except ValueError:
            rejected.append({"raw_file": relative, "reason": "请求元数据不是有效JSON对象"})
            continue
        if meta.get("status") != 200:
            rejected.append({"raw_file": relative, "reason": f"HTTP_{meta.get('status')}"})
            continue
        source = path.relative_to(raw_root).parts[1]
        try:
            if source == "zhilian":
                data = json.loads(body)
                if not isinstance(data.get("data"), dict) or "detailedPosition" not in data["data"]:
                    skipped.append({"raw_file": relative, "reason": "列表/城市等非详情响应"})
                    continue
                if data.get("code") != 200 or data["data"].get("isVerification"):
                    raise ValueError("上游错误或验证响应")
                row = normalize_zhilian(data, meta)
            elif source == "shixiseng":
                if "/intern/inn_" not in meta.get("url", ""):
                    skipped.append({"raw_file": relative, "reason": "非详情响应"})
                    continue
                row = normalize_shixiseng(body, meta)
            else:
                skipped.append({"raw_file": relative, "reason": "未支持的来源"})
                continue
            if not row["source_job_id"] or not row["title"] or not row["jd_text"]:
                raise ValueError("岗位ID/标题/JD为空")
            if "\ufffd" in row["jd_text"]:
                raise ValueError("正文含Unicode解码替换字符")
            collected = meta["collected_at"]
            if datetime.fromisoformat(collected).tzinfo is None:
                raise ValueError("采集时间无时区")
            row["schema_version"] = SCHEMA_VERSION
            row["job_key"] = f"{source}:{row['source_job_id']}"
            content = {
                k: row[k]
                for k in row
                if k
                not in {
                    "raw_jd",
                    "raw_jd_format",
                    "jd_source_field",
                    "schema_version",
                    "source_published_at_text",
                    "source_updated_at_text",
                    "source_status_raw",
                }
            }
            row["content_hash"] = digest(content)
            row["collected_at"] = collected
            row["provenance"] = [
                {
                    "raw_file": relative,
                    "meta_file": meta_path.relative_to(raw_root.parent.parent).as_posix(),
                    "sha256": digest(body),
                    "collected_at": collected,
                    "jd_source_field": row["jd_source_field"],
                }
            ]
            observations.append(row)
        except (ValueError, KeyError, TypeError) as error:
            rejected.append({"raw_file": relative, "reason": str(error)})
    versions, duplicate_events = {}, []
    for row in observations:
        key = row["job_key"], row["content_hash"]
        if key in versions:
            versions[key]["provenance"].extend(row["provenance"])
            duplicate_events.append(
                {
                    "kind": "same_identity_same_content",
                    "job_key": row["job_key"],
                    "raw_file": row["provenance"][0]["raw_file"],
                }
            )
            if datetime.fromisoformat(row["collected_at"]) > datetime.fromisoformat(
                versions[key]["collected_at"]
            ):
                versions[key]["collected_at"] = row["collected_at"]
        else:
            versions[key] = row
    by_job = defaultdict(list)
    for row in versions.values():
        by_job[row["job_key"]].append(row)
    current = []
    for key, items in sorted(by_job.items()):
        items.sort(key=lambda x: (datetime.fromisoformat(x["collected_at"]), x["content_hash"]))
        for number, row in enumerate(items, 1):
            row["version"] = number
        current.append(items[-1])
    exact = defaultdict(list)
    for row in current:
        # 同正文不同ID只分组，不删除；地点/薪资差异不能被正文去重吞掉。
        key = digest(
            [
                row["company"],
                row["title"],
                re.sub(r"\s+", "", row["jd_text"]).casefold(),
            ]
        )
        row["duplicate_group"] = key
        exact[key].append(row["job_key"])
    groups = [
        {"group": k, "job_keys": v, "kind": "same_company_title_jd_different_identity"}
        for k, v in exact.items()
        if len(v) > 1
    ]
    output.mkdir(parents=True)
    write_jsonl(output / "jobs.jsonl", current)
    write_jsonl(
        output / "versions.jsonl",
        sorted(versions.values(), key=lambda x: (x["job_key"], x["version"])),
    )
    write_jsonl(output / "duplicates.jsonl", duplicate_events + groups)
    write_jsonl(output / "rejected.jsonl", rejected)
    write_jsonl(output / "skipped.jsonl", skipped)
    manifest = dict(
        schema_version=SCHEMA_VERSION,
        snapshot=output.name,
        raw_fingerprint=digest(raw_hashes),
        raw_files=raw_hashes,
        body_count=len(list(raw_root.rglob("*.body"))),
        raw_selection={
            "policy": "explicit_allowlist",
            "selected_raw_files": sorted(selected),
            "sha256": digest(sorted(selected)),
        }
        if selected is not None
        else None,
        observations=len(observations),
        meta_only_count=sum(
            not p.with_name(p.name.removesuffix(".meta.json") + ".body").exists()
            for p in raw_root.rglob("*.meta.json")
        ),
        jobs=len(current),
        versions=len(versions),
        duplicate_observations=len(duplicate_events),
        cross_identity_duplicate_groups=len(groups),
        rejected=len(rejected),
        skipped=len(skipped),
        source_counts=dict(Counter(x["source"] for x in current)),
        jobs_sha256=digest((output / "jobs.jsonl").read_bytes()),
        missing={
            k: sum(x.get(k) in (None, "", []) for x in current)
            for k in (
                "company",
                "cities",
                "salary_text",
                "education_text",
                "experience_text",
                "skills",
            )
        },
        limits=[
            "来源状态编码未经核实，不据此声称当前在招",
            "不把校园类型自动当成实习",
            "无实习僧真实样本，本次只验证智联",
            "相似正文仅候选，未做语义近重复合并",
        ],
    )
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
