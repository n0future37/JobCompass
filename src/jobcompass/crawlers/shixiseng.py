"""实习僧独立采集入口：列表→详情→原始响应及采集摘要。"""

import argparse
import json
import re
from html.parser import HTMLParser
from urllib.parse import urlencode, urljoin, urlsplit, urlunsplit

if __package__:
    from . import zhilian as transport
else:
    import zhilian as transport

SOURCE = "shixiseng"
SOURCE_URL = "https://www.shixiseng.com/interns?" + urlencode(
    {"city": "深圳", "from": "menu", "keyword": "Python"}
)
DETAIL_PATTERN = r"/intern/inn_[a-z0-9]+$"
FIELDS = {
    "new_job_name",
    "job_detail",
    "job_money",
    "job_position",
    "job_academic",
    "job_week",
    "job_time",
    "job_date",
    "com-name",
    "com_position",
}


class Page(HTMLParser):
    """仅提取明确的正文容器，避免把导航、推荐岗位或脚本拼成JD。"""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.links, self.stack, self.captures = [], [], []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "a" and attrs.get("href"):
            self.links.append(attrs["href"])
        if tag in {"br", "hr", "img", "meta", "input", "link", "source", "wbr"}:
            if tag == "br":
                self.handle_data("\n")
            return
        if tag in {"p", "div", "li"}:
            self.handle_data("\n")
        keys = set(attrs.get("class", "").split()) & FIELDS
        self.stack.append({"tag": tag, "keys": keys, "text": []})

    def handle_startendtag(self, tag, attrs):
        if tag in {"br", "hr", "img", "meta", "input", "link", "source", "wbr"}:
            self.handle_starttag(tag, attrs)
        else:
            self.handle_starttag(tag, attrs)
            self.handle_endtag(tag)

    def handle_data(self, data):
        if any(frame["tag"] in {"script", "style"} for frame in self.stack):
            return
        for frame in self.stack:
            if frame["keys"]:
                frame["text"].append(data)

    def handle_endtag(self, tag):
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index]["tag"] == tag:
                for frame in self.stack[index:]:
                    if frame["keys"]:
                        self.captures.append(frame)
                del self.stack[index:]
                break
        if tag in {"p", "div", "li"}:
            self.handle_data("\n")

    def field(self, key):
        for frame in self.captures:
            if key in frame["keys"]:
                lines = [
                    re.sub(r"[^\S\n]+", " ", s).strip()
                    for s in "".join(frame["text"]).splitlines()
                ]
                return "\n".join(s for s in lines if s)
        return ""


def usable_text(text):
    # 不猜测字体映射，不把无法识别的字符写成薪资/学历等事实。
    if not text or any(0xE000 <= ord(c) <= 0xF8FF or ord(c) >= 0xF0000 for c in text):
        return None
    return text


def response_issue(body, meta):
    status = meta.get("status")
    if status != 200:
        return f"HTTP_{status}: redirect/rejection/network error; stopped without retry"
    text = body.decode("utf-8", errors="replace")
    markers = (
        "aliyun_waf",
        "acw_sc__v2",
        "_____tmd_____",
        "请完成安全验证",
        "访问过于频繁",
    )
    found = [m for m in markers if m in text]
    return f"verification_page: {','.join(found)}" if found else ""


def detail_urls(body, url):
    page = Page()
    page.feed(body.decode("utf-8", errors="replace"))
    result = []
    host = urlsplit(url).hostname
    for href in page.links:
        parts = urlsplit(urljoin(url, href))
        if parts.scheme not in {"http", "https"} or parts.hostname != host:
            continue
        if re.search(DETAIL_PATTERN, parts.path):
            target = urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))
            if target not in result:
                result.append(target)
    return result


def parse_detail(body, meta, raw_file):
    page = Page()
    page.feed(body.decode("utf-8", errors="replace"))
    title = usable_text(page.field("new_job_name"))
    jd = usable_text(page.field("job_detail"))
    if not title or not jd:
        raise ValueError("缺少可读标题/明确JD容器，或正文存在字体编码，拒绝生成岗位")
    match = re.search(r"/intern/(inn_[a-z0-9]+)", meta["url"])
    if not match:
        raise ValueError("缺少来源岗位标识")
    return dict(
        job_id=match[1],
        source_job_id=match[1],
        source=SOURCE,
        title=title,
        company=usable_text(page.field("com-name")),
        city=usable_text(page.field("job_position")),
        location_text=usable_text(page.field("com_position")),
        salary_text=usable_text(page.field("job_money")),
        education_text=usable_text(page.field("job_academic")),
        attendance_text=usable_text(page.field("job_week")),
        duration_text=usable_text(page.field("job_time")),
        source_updated_at_text=usable_text(page.field("job_date")),
        jd_text=jd,
        url=meta["url"],
        collected_at=meta["collected_at"],
        raw_file=raw_file,
    )


def collect(limit=1):
    source = SOURCE
    probe = transport.Probe(source)
    rows = []
    summary = dict(
        source=source,
        list_url=SOURCE_URL,
        mode="live",
        status="unavailable",
        requests=0,
        complete_jd_count=0,
        failure="",
    )
    try:
        body, meta, _ = probe.request("list", SOURCE_URL)
        summary["requests"] += 1
        problem = response_issue(body, meta)
        if problem:
            summary.update(status="blocked", failure=problem)
            return finish(source, summary, rows, probe.directory)
        urls = detail_urls(body, SOURCE_URL)
        summary["detail_links_found"] = len(urls)
        if not urls:
            summary["failure"] = "no_verified_detail_links: 页面可达不等于列表可采"
        for index, url in enumerate(urls[:limit], 1):
            body, meta, raw = probe.request(f"detail_{index}", url)
            summary["requests"] += 1
            problem = response_issue(body, meta)
            if problem:
                summary.update(status="blocked", failure=problem)
                break
            rows.append(parse_detail(body, meta, raw))
        if rows:
            summary["status"] = "partial" if summary["failure"] else "passed"
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as error:
        summary.update(
            status="partial" if rows else "unavailable",
            failure=f"{type(error).__name__}: {error}",
        )
    return finish(source, summary, rows, probe.directory)


def finish(source, summary, rows, directory):
    summary["complete_jd_count"] = len(rows)
    with (directory / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, choices=range(1, 4), default=1)
    parser.add_argument(
        "--network", action="store_true", help="实际联网，默认Python/深圳"
    )
    args = parser.parse_args()
    if args.network:
        result = collect(args.limit)
        raise SystemExit(0 if result["status"] == "passed" else 1)
    else:
        parser.error("指定--network采集；--help查看参数")
