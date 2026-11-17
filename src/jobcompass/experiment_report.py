"""从冻结产物生成对照报告；不调用模型或修改人工标签。"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from .settings import ROOT

NAMES = {
    "line700_o1__original__hybrid_rerank": "700字符/重叠1行（基线）",
    "token256_o32__original__hybrid_rerank": "256 token/重叠32",
    "token384_o48__original__hybrid_rerank": "384 token/重叠48",
    "token384_o0__original__hybrid_rerank": "384 token/无重叠",
    "line700_o1__rewrite__hybrid_rerank": "基线切块 + 查询改写",
    "line700_o1__hyde__hybrid_rerank": "基线切块 + HyDE（含回退）",
    "line700_o1__original__bm25": "BM25，无重排",
    "line700_o1__original__dense": "纯向量，无重排",
}


def report_experiment(directory):
    summary = json.loads((directory / "summary.json").read_bytes())
    if summary["status"] != "complete":
        raise ValueError("实验尚未完整完成")
    rows = [
        json.loads(s) for s in (directory / "per_query.jsonl").read_text(encoding="utf-8").split("\n") if s
    ]
    queries = json.loads((ROOT / "data/eval/expanded_probes_v1/queries.json").read_bytes())
    audit = json.loads((ROOT / "artifacts/data_audit/jobs_v2.json").read_bytes())
    if audit["corpus_hash"] != summary["corpus_hash"]:
        raise ValueError("近重复检查不是同一语料")
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(keys):
        for key in keys[1:]:
            parent[find(key)] = find(keys[0])

    for keys in audit["exact_jd_groups"]:
        union(keys)
    for pair in audit["near_candidates"]:
        union(pair["job_keys"])
    clusters = defaultdict(list)
    for query in queries:
        if query["kind"] == "anchor" and query["anchor_job"] not in [
            r["job_key"] for r in clusters[find(query["anchor_job"])]
        ]:
            clusters[find(query["anchor_job"])].append(
                dict(job_key=query["anchor_job"], split=query["split"])
            )
    cross = [v for v in clusters.values() if len({r["split"] for r in v}) > 1]
    query_splits = defaultdict(set)
    for query in queries:
        if query["kind"] != "control":
            query_splits["".join(query["query"].split()).casefold()].add(query["split"])
    fallback = {
        qid
        for c in summary["transform_calls"]
        if c["kind"] == "hyde" and c["status"] != "ok"
        for qid in c["query_ids"]
    }
    by_variant = defaultdict(dict)
    for row in rows:
        by_variant[row["variant"]][row["query_id"]] = row
    base = by_variant["line700_o1__original__hybrid_rerank"]
    hyde = by_variant["line700_o1__hyde__hybrid_rerank"]
    successful = [
        qid
        for qid, r in hyde.items()
        if r["kind"] == "anchor" and r["split"] == "test" and qid not in fallback
    ]
    changes = []
    for variant, items in by_variant.items():
        changed = [
            dict(query_id=qid, before=base[qid].get("anchor_hit_at_10"), after=row.get("anchor_hit_at_10"))
            for qid, row in items.items()
            if row["kind"] == "anchor"
            and row["split"] == "test"
            and row.get("anchor_hit_at_10") != base[qid].get("anchor_hit_at_10")
        ]
        if changed:
            changes.append(dict(variant=variant, changes=changed))
    supplements = dict(
        near_duplicate_audit=dict(
            exact_groups=len(audit["exact_jd_groups"]),
            near_pairs=len(audit["near_candidates"]),
            cross_split_anchor_clusters=cross,
            repeated_queries_across_splits=sum(len(v) > 1 for v in query_splits.values()),
            limits=audit["limits"],
        ),
        hyde_success_only_test_anchor=dict(
            queries=len(successful),
            groups=len({hyde[q]["group"] for q in successful}),
            baseline_hit=mean(base[q]["anchor_hit_at_10"] for q in successful),
            hyde_hit=mean(hyde[q]["anchor_hit_at_10"] for q in successful),
        ),
        changed_test_anchors=changes,
        returned_jobs=sum(len(r["ranking"]) for r in rows),
        hard_violations=sum(r.get("hard_violations", 0) for r in rows),
        transformation_seconds=sum(c["seconds"] for c in summary["transform_calls"]),
        transform_success_queries={
            kind: sum(
                len(c["query_ids"])
                for c in summary["transform_calls"]
                if c["kind"] == kind and c["status"] == "ok"
            )
            for kind in ("rewrite", "hyde")
        },
    )
    (directory / "analysis.json").write_text(
        json.dumps(supplements, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    text = [
        "# 切块、查询改写与 HyDE 对照",
        "",
        "冻结1004个岗位，8组×72查询=576次检索。设备为RTX 4070 Laptop GPU，BGE-small-zh-v1.5与BGE-reranker-v2-m3。词法/向量每路最多40个岗位，混合并集最多80，前15重排，报告岗位级Top10；不是混合总召回40。",
        "",
        "72条由12个人工已标查询（6开发/6固定测试）、48条原文锚定查询（24岗位×2问法，12方向）、12个条件/异常控制项组成。测试锚定部分24问法仅12岗位组；方向和模板共享，不是外部盲测。原文锚定更容易被关键词命中。",
        "",
        "| 方案 | 指定岗位Hit@10（24测试问法） | MRR@10 | 原6测试查询人标P@10范围 | 已标候选占比 |",
        "|---|---:|---:|---:|---:|",
    ]
    for variant in NAMES:
        r = next(m for m in summary["measures"] if m["variant"] == variant and m["split"] == "test")
        low, high = r["human_p10_lower_bound"], r["human_p10_upper_bound"]
        bound = f"{low:.4f}" if abs(low - high) < 1e-10 else f"[{low:.4f}, {high:.4f}]"
        text.append(
            f"| {NAMES[variant]} | {r['anchor_hit_at_10']:.2%} | {r['anchor_mrr_at_10']:.4f} | {bound} | {r['human_judged_fraction']:.2%} |"
        )
    text += [
        "",
        "P@10以标签2为强相关；范围是假设所有未标注分别不相关/强相关所得的识别边界，不是置信区间。新召回不能套用旧人标或当作0；60个新增查询—岗位对在human_review_pending.jsonl。不同方案的完整标注查询子集不同，不能拿子集均值直接比。",
        "",
        "原人工v2的6固定测试：基线混合重排候选池Recall@10=0.6105、nDCG@10=0.9199；BM25分别0.5441、0.8590。原候选池并非1004岗位全量相关性金标。新策略完整nDCG/开放式Recall待补标，锚定Hit不能替代它们。",
        "",
        "## 切块与结论",
        "",
        "| 切块 | 块数 | 输入最大token | 超过512的块数 |",
        "|---|---:|---:|---:|",
    ]
    for r in summary["indexes"]:
        text.append(f"| {r['chunk_variant']} | {r['chunks']} | {r['token_max']} | {r['over_512_count']} |")
    text += [
        "",
        "token切块消除了本轮模型输入截断（原528/1416块超限），代价为块数增加；目前未显示更高锚定命中。改写也未提高测试锚定Hit。HyDE为24/24，基线23/24，只有一个新增命中；按12岗位组成组bootstrap的差值95%区间[0,0.125]包含0，不宣称显著提升。BM25同样24/24，说明这批原文锚定样本偏容易、不能据此称HyDE最好。保留生产默认参数，未按固定测试挑最佳组合。",
        "",
        f"实际变换请求{len(summary['transform_calls'])}次，成功改写60/60、HyDE48/60；HyDE两批ReadTimeout共12查询按原查询编码回退。报告主表是含回退的运行策略；成功子集配对结果见analysis.json。供应商已回传用量{summary['transform_tokens']} tokens，超时请求用量未知；货币价格未配置，成本为null。批量生成总用时{supplements['transformation_seconds']:.1f}秒，不是单次线上请求延迟。重算复用了缓存，无新增变换调用。",
        "",
        "## 延迟与约束",
        "",
        "| 方案 | 检索P50 ms | 检索P95 ms |",
        "|---|---:|---:|",
    ]
    for variant in NAMES:
        r = next(m for m in summary["measures"] if m["variant"] == variant and m["split"] == "test")
        text.append(f"| {NAMES[variant]} | {r['warm_p50_ms']:.1f} | {r['warm_p95_ms']:.1f} |")
    text += [
        "",
        f"延迟为每方案60个非控制查询剔除首项后的59次顺序本机GPU检索，合并dev/test；包括本地编码/检索/融合/重排，不含批量改写生成、回答生成、API、数据库写历史或网页。不是完整HyDE线上耗时，也不是Docker CPU性能。控制项各方案12/12；全部返回{supplements['returned_jobs']}岗位位置，已支持硬条件违反{supplements['hard_violations']}，条件真值仍基于保守解析规则。",
        "",
        f"沿用同语料的20个正文精确重复组、63对近重复候选，并按连通组核查锚定dev/test；跨split锚定组{len(cross)}、跨split相同查询0。LSH可能漏检；检索语料两split共享、未训练模型。该检查不证明无模板或语义泄漏。",
        "",
        "6条固定测试只能做回归冒烟。下一轮需要独立真实求职表达、模糊意图、否定/冲突/多约束、证据不足、无结果与对抗JD，按意图组固定划分并盲标；优先从50–100个独立意图起步，再按稳定性扩充。当前72条扩大了覆盖，未变成72条独立人工语义评测。",
        "",
        "运行：`python -m jobcompass.experiments --directory artifacts/experiments/<新目录>`会调用真实变换模型；将已有transforms_*.json复制到新目录可复用。`python -m jobcompass.experiment_report --directory <已有目录>`只读原始产物生成本报告。临时Milvus集合已全部定向删除，开发索引保持不变。",
        "",
        "本报告不评测回答生成/语义引用支持率。既有507引用人标属于原抽取式答案，不能沿用为新Agent或新检索方案的生成质量。",
    ]
    (directory / "REPORT.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    return supplements


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    report_experiment(parser.parse_args().directory)
