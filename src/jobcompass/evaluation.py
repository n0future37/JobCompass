"""冻结查询与全语料规则代理标签，分开报告检索指标和证据校验。"""

import json
import math
import platform
import re
from datetime import UTC, datetime
from time import perf_counter

import numpy as np

from .conditions import SearchRequest, assess
from .generation import Generator
from .preprocessing import digest
from .retrieval import Retriever
from .settings import ROOT, Settings

QUERIES = [
    ("D01", "dev", "Python 后端接口开发", [["Python"], ["后端", "接口", "服务端"]]),
    (
        "D02",
        "dev",
        "RAG 知识库与检索增强生成开发",
        [["RAG", "检索增强"], ["知识库", "知识问答"]],
    ),
    ("D03", "dev", "推荐系统算法工程师", [["推荐"], ["算法", "模型"]]),
    ("D04", "dev", "数据分析与SQL", [["数据分析"], ["SQL", "MySQL"]]),
    ("D05", "dev", "计算机视觉图像处理", [["计算机视觉", "图像"], ["算法", "模型"]]),
    ("D06", "dev", "Java 后端微服务", [["Java"], ["微服务", "Spring"]]),
    ("T01", "test", "Python 自动化测试开发", [["Python"], ["测试", "自动化"]]),
    ("T02", "test", "Agent 大模型应用开发", [["Agent", "智能体"], ["大模型", "LLM"]]),
    ("T03", "test", "数据工程 ETL 数据仓库", [["ETL", "数据仓库", "数仓"], ["数据"]]),
    (
        "T04",
        "test",
        "Docker Kubernetes 云原生运维",
        [["Docker", "Kubernetes", "K8s"], ["运维", "云原生"]],
    ),
    ("T05", "test", "自然语言处理 NLP 算法", [["NLP", "自然语言"], ["算法", "模型"]]),
    ("T06", "test", "前端 React Vue 开发", [["React", "Vue"], ["前端"]]),
]


def grade(job, groups):
    text = job["title"] + "\n" + job["jd_text"]
    matches = [any(re.search(re.escape(term), text, re.IGNORECASE) for term in group) for group in groups]
    return 2 if all(matches) else 1 if any(matches) else 0


def ranking_metrics(ranking, labels, k=10):
    """固定K分母；未标注不当作0，重复身份视为无效评测输入。"""
    if k < 1:
        raise ValueError("K必须大于0")
    if len(set(ranking)) != len(ranking):
        raise ValueError("岗位级排名包含重复ID")
    if any(value not in (0, 1, 2) for value in labels.values()):
        raise ValueError("相关性只接受0/1/2")
    head = ranking[:k]
    unjudged = sum(key not in labels for key in head)
    relevant = {key for key, value in labels.items() if value == 2}
    hits = len(set(head) & relevant)
    dcg = sum((2 ** labels.get(key, 0) - 1) / math.log2(rank + 2) for rank, key in enumerate(head))
    ideal = sum(
        (2**value - 1) / math.log2(rank + 2)
        for rank, value in enumerate(sorted(labels.values(), reverse=True)[:k])
    )
    return {
        "k": k,
        "returned": len(head),
        "relevant_in_labeled_corpus": len(relevant),
        "hits_grade_2": hits,
        "unjudged": unjudged,
        "precision": hits / k if not unjudged else None,
        "recall": hits / len(relevant) if relevant and not unjudged else None,
        "ndcg": dcg / ideal if ideal and not unjudged else None,
        "recall_ceiling": min(k, len(relevant)) / len(relevant) if relevant else None,
    }


def metrics(ranking, labels, k=10):
    values = ranking_metrics(ranking, labels, k)
    return {
        "recall_at_10": values["recall"],
        "ndcg_at_10": values["ndcg"],
        "precision_at_10": values["precision"],
        "recall_at_10_ceiling": values["recall_ceiling"],
        "relevant_jobs": values["relevant_in_labeled_corpus"],
    }


def evaluate(settings=None):
    settings = settings or Settings()
    retriever = Retriever(settings)
    generator = Generator(settings)
    corpus_hash = digest([[j["job_key"], j["content_hash"]] for j in retriever.jobs])
    version_snapshot = digest([[j["job_key"], j["content_hash"], j["version"]] for j in retriever.jobs])
    directory = ROOT / "artifacts/evaluations" / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    directory.mkdir(parents=True)
    labels = {qid: {j["job_key"]: grade(j, groups) for j in retriever.jobs} for qid, _, _, groups in QUERIES}
    definitions = [
        {"query_id": qid, "split": split, "query": text, "rule_groups": groups}
        for qid, split, text, groups in QUERIES
    ]
    frozen = ROOT / "data/eval/rule_proxy_v1"
    frozen.mkdir(parents=True, exist_ok=True)
    protocol = {
        "corpus_hash": corpus_hash,
        "queries_sha256": digest(definitions),
        "label_kind": "rule_proxy_not_human_gold",
        "version": "1",
    }
    if (frozen / "manifest.json").exists():
        if json.loads((frozen / "manifest.json").read_bytes()) != protocol:
            raise ValueError("固定评测输入已变化，请建立新版本，不覆盖旧基线")
    else:
        (frozen / "manifest.json").write_text(
            json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (frozen / "queries.json").write_text(
            json.dumps(definitions, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (frozen / "qrels.json").write_text(json.dumps(labels, ensure_ascii=False), encoding="utf-8")
    (directory / "queries.json").write_text(
        json.dumps(definitions, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (directory / "qrels_rule_proxy.json").write_text(json.dumps(labels, ensure_ascii=False), encoding="utf-8")
    rows, citation_total, citation_valid = [], 0, 0
    try:
        for mode, rerank in [
            ("bm25", False),
            ("dense", False),
            ("hybrid", False),
            ("hybrid", True),
        ]:
            method = mode + ("_reranked" if rerank else "")
            for qid, split, text, _ in QUERIES:
                start = perf_counter()
                request = SearchRequest(query=text, mode=mode, top_k=10, rerank=rerank, generate=False)
                result = retriever.search(request, include_trace=True)
                trace = result.get("evaluation_trace", {})
                candidates = trace.get("before_rerank", [])
                budget = settings.config["retrieval"]["recall_k"]
                ranked = [x["job"]["job_key"] for x in result["results"]]
                answer = generator.answer(result, request)
                for item in answer["recommendations"]:
                    job = retriever.by_key[item["job_key"]]
                    lines = job["jd_text"].splitlines()
                    for c in item["citations"]:
                        citation_total += 1
                        citation_valid += int(
                            1 <= c["line"] <= len(lines) and lines[c["line"] - 1] == c["quote"]
                        )
                row = dict(
                    query_id=qid,
                    split=split,
                    method=method,
                    **metrics(ranked, labels[qid]),
                    ranking=ranked,
                    recommendations=answer["recommendations"],
                    candidate_recall_at_budget=ranking_metrics(candidates, labels[qid], budget)["recall"],
                    candidate_union_recall=ranking_metrics(candidates, labels[qid], max(1, len(candidates)))[
                        "recall"
                    ],
                    candidate_count=len(candidates),
                    stage_rankings=trace,
                    total_ms=(perf_counter() - start) * 1000,
                    phase_ms=result["timings_ms"],
                    degradations=result["degradations"],
                )
                rows.append(row)
                print(
                    json.dumps(
                        {
                            k: row[k]
                            for k in (
                                "query_id",
                                "method",
                                "recall_at_10",
                                "ndcg_at_10",
                                "total_ms",
                            )
                        }
                    ),
                    flush=True,
                )
        hard_cases = [
            SearchRequest(
                query="Python 后端",
                cities=["深圳"],
                skills_all=["Python"],
                generate=False,
                rerank=False,
            ),
            SearchRequest(query="开发", cities=["不存在的城市"], generate=False, rerank=False),
            SearchRequest(query="必须上市公司", generate=False, rerank=False),
        ]
        hard_rows = []
        for request in hard_cases:
            result = retriever.search(request)
            total = len(result["results"])
            violated = sum(not assess(x["job"], result["conditions"])[0] for x in result["results"])
            hard_rows.append(
                dict(
                    query=request.model_dump(),
                    results=total,
                    violations=violated,
                    unresolved=result["unresolved_conditions"],
                    reason=result["no_result_reason"],
                )
            )
        summary = []
        for method in sorted({r["method"] for r in rows}):
            selected = [r for r in rows if r["method"] == method]
            # 各方法首条单列，避免把初次模型加载混入warm统计。
            warm = [r["total_ms"] for r in selected[1:]]
            for split in ("dev", "test"):
                group = [r for r in selected if r["split"] == split]
                summary.append(
                    dict(
                        method=method,
                        split=split,
                        recall_at_10=float(
                            np.mean([r["recall_at_10"] for r in group if r["recall_at_10"] is not None])
                        ),
                        ndcg_at_10=float(
                            np.mean([r["ndcg_at_10"] for r in group if r["ndcg_at_10"] is not None])
                        ),
                        precision_at_10=float(np.mean([r["precision_at_10"] for r in group])),
                        recall_at_10_ceiling=float(np.mean([r["recall_at_10_ceiling"] for r in group])),
                        candidate_recall_at_budget=float(
                            np.mean([r["candidate_recall_at_budget"] for r in group])
                        ),
                        candidate_union_recall=float(np.mean([r["candidate_union_recall"] for r in group])),
                        first_query_ms=selected[0]["total_ms"],
                        warm_p50_ms=float(np.percentile(warm, 50)),
                        warm_p95_ms=float(np.percentile(warm, 95)),
                    )
                )
        import torch

        report = dict(
            report_protocol="layered_rule_proxy.v2",
            version_snapshot=version_snapshot,
            candidate_budget=settings.config["retrieval"]["recall_k"],
            candidate_union_note="混合并集最多2*recall_k；公平比较用candidate_recall_at_budget（当前40），并集召回仅诊断",
            corpus_jobs=len(retriever.jobs),
            corpus_hash=corpus_hash,
            queries=len(QUERIES),
            label_kind="全语料确定性规则代理标签，不是人工金标，不能作为最终语义相关性结论",
            query_protocol="6开发查询+6固定测试查询；共享同一冻结检索语料，无训练或按测试调参",
            measures=summary,
            hard_conditions=hard_rows,
            hard_condition_violation_rate=sum(x["violations"] for x in hard_rows)
            / max(1, sum(x["results"] for x in hard_rows)),
            citation_exactness=citation_valid / max(1, citation_total),
            citations_checked=citation_total,
            unsupported_statement_policy="程序只展示已逐字验证的原文及明确元数据；未完成语义引用人工评测",
            generation_model_calls=0,
            currency_cost=None,
            hardware=dict(
                python=platform.python_version(),
                torch=torch.__version__,
                device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            ),
            config=settings.config,
            limits=[
                "标签由规则生成，存在词面偏置",
                "查询集小，非市场代表性",
                "原文逐字正确不等于语义支持",
                "耗时含MySQL查询及模型推理，不含UI网络或外部生成",
            ],
        )
        if version_snapshot != digest(
            [[j["job_key"], j["content_hash"], j["version"]] for j in retriever.store.all_jobs()]
        ):
            raise ValueError("评测期间语料变化，本次不保存可比较指标")
        (directory / "per_query.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (directory / "summary.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return {
            "report": str(directory / "summary.json"),
            "measures": summary,
            "hard_condition_violation_rate": report["hard_condition_violation_rate"],
            "citation_exactness": report["citation_exactness"],
        }
    finally:
        if retriever.client:
            retriever.client.close()
        retriever.store.close()
