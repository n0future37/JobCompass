"""冻结扩展检索探针：人工12查询、24岗位×2问法及12控制项，标签类型分开。"""

import json
import re
from collections import defaultdict

from .evaluation import QUERIES
from .preprocessing import digest

BUCKETS = [
    ("python_backend", r"[Pp]ython.*后端|后端.*[Pp]ython", "Python后端接口开发"),
    ("rag", r"RAG|知识库|大模型应用", "RAG知识库应用开发"),
    ("agent", r"智能体|[Aa]gent", "Agent智能体应用开发"),
    ("java", r"[Jj][Aa][Vv][Aa]", "Java后端开发"),
    ("frontend", r"前端|[Rr]eact|[Vv]ue", "React Vue前端开发"),
    ("data_analysis", r"数据分析", "SQL数据分析"),
    ("data_engineering", r"数据开发|大数据开发|数据工程|数仓", "ETL数据工程与数据仓库"),
    ("devops", r"运维|DevOps", "Docker Kubernetes运维"),
    ("vision", r"视觉|图像", "计算机视觉图像算法"),
    ("nlp", r"NLP|自然语言|语言模型", "NLP自然语言处理算法"),
    ("recommendation", r"推荐|搜广推|搜索算法", "推荐系统与搜索排序算法"),
    ("test", r"测试", "软件自动化测试开发"),
]


def make_queries(jobs, directory):
    original = [
        {"query_id": qid, "split": split, "query": query, "group": qid, "kind": "human", "request": {}}
        for qid, split, query, _ in QUERIES
    ]

    def normalized(job):
        return re.sub(r"\s+", "", job["jd_text"])

    duplicates = defaultdict(list)
    for job in jobs:
        duplicates[digest(normalized(job))].append(job["job_key"])
    selected, seen = [], set()
    for bucket, pattern, topic in BUCKETS:
        pool = [j for j in jobs if re.search(pattern, j["title"])]
        if bucket == "recommendation":
            pool += [j for j in jobs if re.search(r"推荐系统|推荐算法|召回.*排序", j["jd_text"])]
        pool = sorted(
            {j["job_key"]: j for j in pool}.values(),
            key=lambda j: digest(["probes.v1", bucket, j["job_key"]]),
        )
        anchors = []
        for job in pool:
            fingerprint = digest(normalized(job))
            if fingerprint not in seen:
                anchors.append(job)
                seen.add(fingerprint)
            if len(anchors) == 2:
                break
        if len(anchors) != 2:
            raise ValueError("该岗位方向不足两个不同JD，不能伪造样本:" + bucket)
        for n, job in enumerate(anchors):
            lines = [line for line in job["jd_text"].splitlines() if len(line) >= 20]
            if not lines:
                raise ValueError("锚定岗位缺少可用职责")
            # 明确标记原文驱动探针；它们并不模拟独立真实用户相关性金标。
            for style in range(2):
                detail = re.sub(
                    r"必须|只看|只接受|只要|仅限|不接受|不能|至少|不要|不得|仅接受|一定要",
                    "",
                    lines[min(style, len(lines) - 1)][:110],
                )
                query = (
                    f"想找{topic}岗位，希望涉及：{detail}"
                    if style == 0
                    else f"寻找{job['title']}，工作内容包括：{detail}"
                )
                selected.append(
                    {
                        "query_id": f"A_{bucket}_{n + 1}_{style + 1}",
                        "split": "dev" if n == 0 else "test",
                        "group": f"anchor_{bucket}_{n + 1}",
                        "family": bucket,
                        "kind": "anchor",
                        "query": query,
                        "request": {},
                        "anchor_job": job["job_key"],
                        "expected_keys": duplicates[digest(normalized(job))],
                    }
                )
    controls = [
        ("不存在城市", "Python后端", {"cities": ["不存在的城市"]}, "empty"),
        ("不存在技能", "开发", {"skills_all": ["nonexistent_skill_xyz123"]}, "empty"),
        ("上市条件未支持", "必须上市公司", {}, "unresolved"),
        ("排除条件不反转", "不要Java", {}, "unresolved"),
        ("空白输入", "   ", {}, "validation_error"),
        ("城市显式冲突", "必须北京的Python开发", {"cities": ["深圳"]}, "unresolved"),
        ("无证据查询", "qxzv_987654321", {}, "empty"),
        ("薪资未支持", "必须月薪30k以上", {}, "unresolved"),
        ("深圳Python", "Python后端接口", {"cities": ["深圳"], "skills_all": ["Python"]}, "nonempty_valid"),
        ("上海Java", "Java开发", {"cities": ["上海"], "skills_all": ["Java"]}, "valid"),
        ("实习岗位", "开发", {"employment_types": ["实习"]}, "valid"),
        ("不接受加班", "不接受加班", {}, "unresolved"),
    ]
    control_rows = [
        {
            "query_id": f"C{n:02}",
            "split": "control",
            "group": f"C{n:02}",
            "kind": "control",
            "name": name,
            "query": text,
            "request": request,
            "expected": expected,
        }
        for n, (name, text, request, expected) in enumerate(controls, 1)
    ]
    queries = original + selected + control_rows
    protocol = {
        "version": "expanded-probes.v1",
        "corpus_hash": digest([[j["job_key"], j["content_hash"]] for j in jobs]),
        "query_hash": digest(queries),
        "queries": 72,
        "human_queries": 12,
        "anchor_queries": 48,
        "anchor_groups": 24,
        "families": 12,
        "controls": 12,
        "limits": [
            "48锚定样例来自JD原文，不代表自然用户分布，也不是48个独立岗位意图",
            "同一岗位两种问法按group成组划分，dev/test各12岗位，但方向/模板共享；不称外部盲测",
            "AnchorHit不是开放式相关性Recall；未命中指定岗位仍可能返回其他相关岗位",
        ],
    }
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "manifest.json").exists():
        if json.loads((directory / "manifest.json").read_bytes()) != protocol:
            raise ValueError("冻结查询或语料已变化，需新版本")
    else:
        for name, data in (("queries.json", queries), ("manifest.json", protocol)):
            (directory / name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return queries, protocol
