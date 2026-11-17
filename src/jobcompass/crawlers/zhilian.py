"""智联小批量采集：复用旧验证爬虫，保留响应，独立批次输出。

智联公开岗位响应采集与原始记录保存。
原文件SHA256：55307e0086151821cadbf8b8e6bb8d46d5fd233cfb64df1002e3d7a0dccdf7a5
本次移植：项目根路径、关键词/数量参数、独立批次路径、响应失败检查。
"""

import argparse
import json
import re
import time
from datetime import datetime, timezone
from html import unescape
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[3]
RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
RAW = ROOT / "data/raw"
FIELDS = (
    "job_id title company location_text city salary_text jd_text url source collected_at raw_file".split()
)


def now():
    return datetime.now(timezone.utc).isoformat()


def raw_directory(source, collected_at=None):
    stamp = datetime.fromisoformat(collected_at) if collected_at else datetime.now(timezone.utc)
    if stamp.tzinfo is None:
        raise ValueError("采集时间必须包含时区")
    day = stamp.astimezone(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d")
    return RAW / day / source / RUN_ID


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class Probe:
    def __init__(self, source):
        self.source = source
        self.directory = raw_directory(source)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.last = 0.0
        self.opener = build_opener(NoRedirect())

    def request(self, label, url, payload=None, headers=None):
        time.sleep(max(0, 11 - (time.monotonic() - self.last)))
        self.last = time.monotonic()
        meta = {
            "url": url,
            "method": "POST" if payload is not None else "GET",
            "collected_at": now(),
        }
        self.directory = raw_directory(self.source, meta["collected_at"])
        self.directory.mkdir(parents=True, exist_ok=True)
        destination = self.directory / f"{label}.body"
        if destination.exists() or destination.with_suffix(".meta.json").exists():
            raise RuntimeError(f"Refusing to repeat recorded request: {label}")
        request_headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        }
        request_headers.update(headers or {})
        data = None
        if payload is not None:
            data = json.dumps(payload).encode()
            request_headers["Content-Type"] = "application/json"
            meta["request_payload"] = payload
        try:
            try:
                response = self.opener.open(Request(url, data=data, headers=request_headers), timeout=30)
            except HTTPError as error:
                response = error
            with response:
                body = response.read()
                meta.update(
                    status=response.status,
                    content_type=response.headers.get("Content-Type", ""),
                    bytes=len(body),
                )
                if response.headers.get("Location"):
                    parts = urlsplit(response.headers["Location"])
                    meta["redirect_target"] = urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))
                # Do not persist cookies, auth headers or redirect tokens in metadata.
            destination.write_bytes(body)
        except (URLError, TimeoutError, OSError) as error:
            meta.update(status=None, error=type(error).__name__)
            body = b""
        destination.with_suffix(".meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "source": self.source,
                    "step": label,
                    "status": meta.get("status"),
                    "bytes": len(body),
                    "error": meta.get("error"),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return body, meta, str(destination.relative_to(ROOT)).replace("\\", "/")


def write_summary(source, summary, rows, directory=None):
    summary["complete_jd_count"] = len(rows)
    summary["missing_fields"] = [
        {"job_id": row["job_id"], "fields": [key for key in FIELDS if not row.get(key)]} for row in rows
    ]
    path = (directory or raw_directory(source)) / "summary.json"
    with path.open("x", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def json_body(body):
    try:
        value = json.loads(body)
        return value if isinstance(value, dict) else {}
    except (ValueError, UnicodeError):
        return {}


def clean_html(value):
    value = re.sub(r"<br\s*/?>|</(?:p|div|li)>", "\n", value, flags=re.I)
    return unescape(re.sub(r"<[^>]+>", "", value)).strip()


def zhilian(keyword="Python", limit=1):
    probe = Probe("zhilian")
    summary = dict(
        source="zhilian",
        keyword=keyword,
        city_code="765",
        city_code_evidence="本批次cities.body：深圳/765",
        list_requests=0,
        list_count=None,
        detail_attempts=0,
        login_required="未确认",
        failure="",
    )
    rows = []
    headers = {
        "Origin": "https://www.zhaopin.com",
        "Referer": "https://www.zhaopin.com/",
    }
    body, meta, _ = probe.request(
        "cities", "https://fe-api.zhaopin.com/c/i/search/base/data", headers=headers
    )
    base = json_body(body)
    summary["live_city_verified"] = "深圳" in body.decode("utf-8", errors="replace") and "765" in body.decode(
        "utf-8", errors="replace"
    )
    if meta.get("status") != 200 or base.get("code") != 200 or (base.get("data") or {}).get("isVerification"):
        summary["failure"] = "城市元数据请求即遇到拒绝/限流/验证；停止平台"
        return write_summary("zhilian", summary, rows, probe.directory)
    payload = dict(
        S_SOU_FULL_INDEX=keyword,
        S_SOU_WORK_CITY="765",
        pageIndex=1,
        pageSize=limit,
        platform=13,
        version="0.0.0",
        eventScenario="pcSearchedSouSearch",
        anonymous=1,
        clickFilterBlackCompany=False,
    )
    summary["list_requests"] = 1
    body, meta, _ = probe.request(
        "list_page1",
        "https://fe-api.zhaopin.com/c/i/search/positions",
        payload,
        headers,
    )
    parsed = json_body(body)
    data = parsed.get("data") or {}
    if meta.get("status") != 200 or parsed.get("code") != 200 or data.get("isVerification"):
        summary["failure"] = (
            f"列表不可用：HTTP {meta.get('status')}, code={parsed.get('code')}, isVerification={data.get('isVerification')}"
        )
        return write_summary("zhilian", summary, rows, probe.directory)
    items = data.get("list") or []
    if not isinstance(items, list) or any(not isinstance(card, dict) for card in items):
        summary["failure"] = "列表字段结构异常，停止采集"
        return write_summary("zhilian", summary, rows, probe.directory)
    summary["list_count"] = len(items)
    for index, card in enumerate(items[:limit], 1):
        number = card.get("number")
        if not number:
            continue
        summary["detail_attempts"] += 1
        body, meta, raw_file = probe.request(
            f"detail_{index}",
            "https://fe-api.zhaopin.com/c/i/jobs/position-detail-new?" + urlencode({"number": number}),
            headers=headers,
        )
        detail = json_body(body)
        if (
            meta.get("status") != 200
            or detail.get("code") != 200
            or (detail.get("data") or {}).get("isVerification")
        ):
            summary["failure"] = f"详情不可用：HTTP {meta.get('status')}, code={detail.get('code')}；停止平台"
            break
        position = (detail.get("data") or {}).get("detailedPosition") or {}
        company = (detail.get("data") or {}).get("detailedCompany") or {}
        description = position.get("jobDescPC") or position.get("jobDesc") or ""
        jd = clean_html(description)
        if not jd:
            summary["failure"] = "详情缺少 jobDescPC/jobDesc；未将列表摘要作为 JD"
            continue
        rows.append(
            dict(
                job_id=str(position.get("number") or position.get("positionNumber") or number),
                title=position.get("positionName") or position.get("name") or card.get("name") or "",
                company=company.get("companyName")
                or position.get("companyName")
                or card.get("companyName")
                or "",
                location_text=position.get("workAddress") or "",
                city=position.get("workCity")
                or position.get("positionWorkCity")
                or card.get("workCity")
                or "",
                salary_text=position.get("salary60") or card.get("salary60") or "",
                jd_text=jd,
                url=position.get("positionUrl")
                or position.get("url")
                or card.get("positionURL")
                or card.get("positionUrl")
                or "",
                source="zhilian",
                collected_at=meta["collected_at"],
                raw_file=raw_file,
            )
        )
    if rows:
        summary["login_required"] = "本次无需登录"
    if not items:
        summary["failure"] = "列表返回 0 条"
    return write_summary("zhilian", summary, rows, probe.directory)


def resolve_city(base, name):
    matches = {
        str(item["code"]) for item in base.get("hotCity", []) if item.get("name") == name and item.get("code")
    }
    if len(matches) != 1:
        raise ValueError(f"城市未在本批次元数据中唯一确认：{name}")
    return matches.pop()


def select_card(card, target, existing):
    number = str(card.get("number") or "")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", number):
        return False, "missing_or_invalid_id"
    if "zhilian:" + number in existing:
        return False, "known_identity"
    if card.get("workCity") != target["city"]:
        return False, "city_not_confirmed"
    if target.get("employment_types") and card.get("workType") not in target["employment_types"]:
        return False, "employment_not_confirmed"
    # 关键词用于获取候选；即使标题偏离方向也保留，相关性由查询时判断。
    # 旧冻结计划的 title_any 只作历史记录，不再作为准入条件。
    return True, "selected"


def prior_collection_jobs(checkpoints):
    """跨批次跳过已采来源ID；只接受本项目已完成的智联检查点。"""
    from jobcompass.preprocessing import digest

    keys = set()
    if not isinstance(checkpoints, list):
        raise ValueError("前序检查点必须是数组")
    for name in checkpoints:
        path = (ROOT / name).resolve()
        if not path.is_relative_to((ROOT / "artifacts/collections").resolve()) or not path.is_file():
            raise ValueError("前序检查点不存在或越界")
        record = json.loads(path.read_bytes())
        if record.get("source") != "zhilian" or record.get("status") != "complete":
            raise ValueError("前序批次未完成；停止后续采集")
        for item in record["accepted"]:
            key = item["job_key"]
            raw = (ROOT / item["raw_file"]).resolve()
            if not re.fullmatch(r"zhilian:[A-Za-z0-9_-]{1,128}", key):
                raise ValueError("前序岗位ID无效")
            if not raw.is_relative_to(RAW.resolve()) or not raw.is_file():
                raise ValueError("前序raw不存在或越界")
            if digest(raw.read_bytes()) != item["body_sha256"]:
                raise ValueError("前序raw指纹变化")
            keys.add(key)
    return keys


def collect_selected(plan_path, metadata_batch=None):
    """按预先冻结城市/方向配额补采；只记raw与选择证据，不在此处替换业务库。"""
    from collections import Counter
    from jobcompass.preprocessing import digest, normalize_zhilian

    global RUN_ID
    plan = json.loads(Path(plan_path).read_bytes())
    targets = plan["targets"]
    if not 1 <= len(targets) <= 32 or not 1 <= plan["max_new_jobs"] <= 240:
        raise ValueError("选择计划规模不合法")
    if sum(t["quota"] for t in targets) > plan["max_new_jobs"] or any(
        not 1 <= t["quota"] <= 12 for t in targets
    ):
        raise ValueError("配额超出计划上限")
    if not 1 <= plan["max_pages_per_target"] <= 2 or not 1 <= plan["max_detail_attempts"] <= 320:
        raise ValueError("请求上限不合法")
    snapshot = ROOT / plan["existing_snapshot"]
    initial = [json.loads(s) for s in (snapshot / "jobs.jsonl").read_text(encoding="utf-8").split("\n") if s]
    previous_jobs = prior_collection_jobs(plan.get("previous_collections", []))
    existing = {j["job_key"] for j in initial} | previous_jobs
    if metadata_batch:
        metadata_batch = Path(metadata_batch).resolve()
        if metadata_batch.parent.name != "zhilian" or not metadata_batch.is_relative_to(RAW.resolve()):
            raise ValueError("元数据批次必须在本项目智联raw目录")
        RUN_ID = metadata_batch.name
    report_dir = ROOT / "artifacts/collections" / RUN_ID
    report_dir.mkdir(parents=True, exist_ok=False)
    (report_dir / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    report = dict(
        status="running",
        run_id=RUN_ID,
        source="zhilian",
        plan_sha256=digest(Path(plan_path).read_bytes()),
        existing_jobs=len(initial),
        previous_collection_jobs=len(previous_jobs),
        previous_collections=plan.get("previous_collections", []),
        existing_snapshot_sha256=digest((snapshot / "jobs.jsonl").read_bytes()),
        requests=[],
        accepted=[],
        rejected_details=[],
        targets=[],
        detail_attempts=0,
        list_requests=0,
        skips={},
        stop_reason=None,
        started_at=now(),
        caps=dict(
            max_new_jobs=plan["max_new_jobs"],
            max_detail_attempts=plan["max_detail_attempts"],
            max_list_requests=len(targets) * plan["max_pages_per_target"],
        ),
        rationale="City/role/type coverage only; not anchored to expected test answers. No hidden-login or verification bypass.",
    )
    skips = Counter()
    seen = set(existing)
    attempted = set()
    probe = Probe("zhilian")
    headers = {"Origin": "https://www.zhaopin.com", "Referer": "https://www.zhaopin.com/"}

    def save():
        report["skips"] = dict(skips)
        temporary = report_dir / "checkpoint.tmp"
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(report_dir / "checkpoint.json")

    def record(label, body, meta, path, reused=False):
        report["requests"].append(
            dict(
                label=label,
                raw_file=path,
                status=meta.get("status"),
                collected_at=meta.get("collected_at"),
                body_sha256=digest(body),
                reused=reused,
            )
        )
        save()

    def fetch(label, url, payload=None):
        body, meta, path = probe.request(label, url, payload, headers)
        record(label, body, meta, path)
        value = json_body(body)
        data = value.get("data") or {}
        if (
            meta.get("status") != 200
            or value.get("code") != 200
            or not isinstance(data, dict)
            or data.get("isVerification")
        ):
            report["stop_reason"] = (
                f"平台请求失败/验证，停止，不重试：{label}, HTTP={meta.get('status')}, code={value.get('code')}"
            )
            save()
            raise RuntimeError("stop_platform")
        return value, meta, path

    try:
        if metadata_batch:
            file = metadata_batch / "cities.body"
            meta = json.loads(file.with_suffix(".meta.json").read_bytes())
            if (
                meta.get("status") != 200
                or (datetime.now(timezone.utc) - datetime.fromisoformat(meta["collected_at"])).total_seconds()
                > 7200
            ):
                raise ValueError("元数据非本轮成功响应或超过2小时")
            base = json_body(file.read_bytes())
            if base.get("code") != 200 or (base.get("data") or {}).get("isVerification"):
                raise ValueError("元数据上游拒绝")
            record("cities", file.read_bytes(), meta, file.relative_to(ROOT).as_posix(), True)
            probe.last = time.monotonic()
        else:
            base, _, _ = fetch("cities", "https://fe-api.zhaopin.com/c/i/search/base/data")
        for target in targets:
            code = resolve_city(base["data"], target["city"])
            progress = dict(target=target, city_code=code, accepted=0, pages=0, detail_attempts=0)
            report["targets"].append(progress)
            save()
            for page in range(1, plan["max_pages_per_target"] + 1):
                if (
                    len(report["accepted"]) >= plan["max_new_jobs"]
                    or report["detail_attempts"] >= plan["max_detail_attempts"]
                ):
                    break
                payload = dict(
                    S_SOU_FULL_INDEX=target["keyword"],
                    S_SOU_WORK_CITY=code,
                    pageIndex=page,
                    pageSize=20,
                    platform=13,
                    version="0.0.0",
                    eventScenario="pcSearchedSouSearch",
                    anonymous=1,
                    clickFilterBlackCompany=False,
                )
                cached = metadata_batch / "list_probe_beijing_python_intern.body" if metadata_batch else None
                if target["id"] == "bj_python_intern" and page == 1 and cached and cached.exists():
                    meta = json.loads(cached.with_suffix(".meta.json").read_bytes())
                    if meta.get("request_payload") != payload or meta.get("status") != 200:
                        raise ValueError("探测列表与目标请求不一致")
                    value = json_body(cached.read_bytes())
                    record(
                        "list_probe_beijing_python_intern",
                        cached.read_bytes(),
                        meta,
                        cached.relative_to(ROOT).as_posix(),
                        True,
                    )
                    if value.get("code") != 200 or (value.get("data") or {}).get("isVerification"):
                        raise ValueError("缓存列表不可用")
                else:
                    value, _, _ = fetch(
                        f"list_{target['id']}_p{page}",
                        "https://fe-api.zhaopin.com/c/i/search/positions",
                        payload,
                    )
                progress["pages"] += 1
                report["list_requests"] += 1
                data = value["data"]
                cards = data.get("list") or []
                if not isinstance(cards, list) or any(not isinstance(c, dict) for c in cards):
                    raise ValueError("列表结构异常")
                for card in cards:
                    if (
                        progress["accepted"] >= target["quota"]
                        or len(report["accepted"]) >= plan["max_new_jobs"]
                        or report["detail_attempts"] >= plan["max_detail_attempts"]
                    ):
                        break
                    selected, reason = select_card(card, target, seen | attempted)
                    if not selected:
                        skips[reason] += 1
                        continue
                    number = str(card["number"])
                    key = "zhilian:" + number
                    attempted.add(key)
                    report["detail_attempts"] += 1
                    progress["detail_attempts"] += 1
                    save()
                    value, meta, path = fetch(
                        f"detail_{report['detail_attempts']:04d}",
                        "https://fe-api.zhaopin.com/c/i/jobs/position-detail-new?"
                        + urlencode({"number": number}),
                    )
                    try:
                        row = normalize_zhilian(value, meta)
                        if row["source_job_id"] != number:
                            raise ValueError("来源岗位ID不一致")
                        if not row["title"] or not row["jd_text"] or "\ufffd" in row["jd_text"]:
                            raise ValueError("标题或完整正文不可用")
                        if target["city"] not in row["cities"]:
                            raise ValueError("详情城市与选择目标不符或未知")
                        if (
                            target["employment_types"]
                            and row["employment_type_text"] not in target["employment_types"]
                        ):
                            raise ValueError("详情类型与选择目标不符或未知")
                    except (ValueError, KeyError, TypeError) as error:
                        report["rejected_details"].append(dict(job_key=key, raw_file=path, reason=str(error)))
                        save()
                        continue
                    seen.add(key)
                    progress["accepted"] += 1
                    report["accepted"].append(
                        dict(
                            job_key=key,
                            title=row["title"],
                            company=row["company"],
                            cities=row["cities"],
                            employment_type=row["employment_type_text"],
                            education=row["education_text"],
                            experience=row["experience_text"],
                            target_id=target["id"],
                            sampling_family=target.get("family"),
                            source_job_family=value["data"]["detailedPosition"].get("jobTypeLevelName"),
                            source_sub_job_type=value["data"]["detailedPosition"].get("subJobType"),
                            raw_file=path,
                            collected_at=meta["collected_at"],
                            body_sha256=digest((ROOT / path).read_bytes()),
                            jd_chars=len(row["jd_text"]),
                        )
                    )
                    save()
                    print(
                        json.dumps(
                            {
                                "target": target["id"],
                                "accepted": progress["accepted"],
                                "total_new": len(report["accepted"]),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                if progress["accepted"] >= target["quota"] or not cards or data.get("isEndPage"):
                    break
            progress["quota_filled"] = progress["accepted"] >= target["quota"]
            save()
        report["status"] = "complete"
    except Exception as error:
        report["status"] = "stopped"
        report["stop_reason"] = report["stop_reason"] or type(error).__name__ + ": " + str(error)
    finally:
        report["ended_at"] = now()
        save()
        output = probe.directory / "summary.json"
        if not output.exists():
            output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "new_jobs": len(report["accepted"]),
                "details": report["detail_attempts"],
                "lists": report["list_requests"],
                "report": str(report_dir),
                "stop_reason": report["stop_reason"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="智联小批量采集，默认Python/深圳/1条详情")
    parser.add_argument("--keyword", default="Python", help="搜索关键词")
    parser.add_argument("--limit", type=int, choices=range(1, 6), default=1, help="最多请求1至5条详情")
    parser.add_argument("--plan", type=Path, help="按冻结配额选择性补采；与小样本入口共用限速")
    parser.add_argument("--metadata-batch", type=Path, help="复用本轮两小时内城市响应")
    parser.add_argument("--network", action="store_true", help="显式允许联网采集")
    args = parser.parse_args()
    if not args.keyword.strip():
        parser.error("关键词不能为空白")
    if not args.network:
        parser.error("实际采集需要--network；查看参数使用--help")
    if args.plan:
        result = collect_selected(args.plan, args.metadata_batch)
        raise SystemExit(0 if result["status"] == "complete" else 1)
    if args.metadata_batch:
        parser.error("--metadata-batch需与--plan同时使用")
    result = zhilian(args.keyword.strip(), args.limit)
    raise SystemExit(1 if result["failure"] else 0)
