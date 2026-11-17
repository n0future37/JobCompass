"""扩充语料回归：复用版本一致的代理标签，以共同候选池比较两轮检索。"""

import argparse
import asyncio
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

import numpy as np

from .evaluation import ranking_metrics
from .experiment_report import NAMES
from .preprocessing import digest
from .scenario_judging import PROMPT_VERSION, judge, make_batches
from .scenario_label_cache import recover
from .settings import Settings


def readl(path):
    return [json.loads(s) for s in path.read_text(encoding="utf-8").split("\n") if s.strip()]


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def seed_labels(directory, previous):
    """只复用同一查询、同一JD版本及可核对原文行的标签，不重新包装成人标。"""
    queries = json.loads((directory / "queries.json").read_bytes())
    if queries != json.loads((previous / "queries.json").read_bytes()):
        raise ValueError("两轮查询不同，不能直接复用或配对比较")
    source = previous / "judge/direction_annotations.jsonl"
    source_hash = digest(source.read_bytes())
    pairs = {p["pair_id"]: p for b in make_batches(directory) for p in b}
    output = directory / "judge"
    output.mkdir(exist_ok=True)
    if (output / "checkpoint.json").exists():
        meta = json.loads((output / "reuse.json").read_bytes())
        if meta["source_sha256"] != source_hash or meta["ranking_sha256"] != digest(
            (directory / "per_query.jsonl").read_bytes()
        ):
            raise ValueError("复用缓存来源或排名指纹变化")
        return meta
    adopted = []
    for row in readl(source):
        if row["pair_id"] not in pairs:
            continue
        pair = pairs[row["pair_id"]]
        job = pair["job"]
        if row["content_hash"] != job["content_hash"] or row["query_id"] != pair["query_id"]:
            raise ValueError("复用标签岗位版本不符")
        for evidence in row["evidence"]:
            if job["jd_text"].split("\n")[evidence["line"] - 1] != evidence["text"]:
                raise ValueError("复用标签原文证据不符")
        adopted.append(dict(row, reused_from=str(previous)))
    if len({r["pair_id"] for r in adopted}) != len(adopted):
        raise ValueError("重复复用标签")
    config = Settings().api_config
    signature = digest(
        [PROMPT_VERSION, config, digest((directory / "per_query.jsonl").read_bytes()), sorted(pairs)]
    )
    signature = digest([signature, "pending-mixed8.v1"])
    meta = dict(
        source=str(source),
        source_sha256=source_hash,
        ranking_sha256=digest((directory / "per_query.jsonl").read_bytes()),
        seeded=len(adopted),
        new_pairs=len(pairs) - len(adopted),
        total_pairs=len(pairs),
        policy="Exact query and content version, original annotation_kind preserved; no human gold added",
    )
    save(output / "reuse.json", meta)
    (output / "seeded_annotations.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in adopted), encoding="utf-8"
    )
    (output / "annotations.jsonl").write_text(
        (output / "seeded_annotations.jsonl").read_text(encoding="utf-8"), encoding="utf-8"
    )
    save(
        output / "checkpoint.json",
        dict(
            signature=signature,
            prompt_version=PROMPT_VERSION,
            model=config["MODEL"],
            annotation_kind="model",
            status="partial_unjudged",
            calls=[],
            annotations=adopted,
            limits=dict(max_calls=120, max_reported_tokens=1200000, concurrency=2),
            reported_tokens=0,
            currency_cost=None,
            unjudged_pairs=len(pairs) - len(adopted),
        ),
    )
    return meta


async def judge_new(directory, previous, max_calls=120, max_tokens=1200000):
    meta = seed_labels(directory, previous)
    print(json.dumps(dict(stage="cache_reuse", **meta), ensure_ascii=False), flush=True)
    for attempt in (1, 2):
        await judge(directory, max_calls, max_tokens, attempt, repack_pending=True)
        recovered = recover(directory)
        print(
            json.dumps(
                dict(
                    stage="direction_recovery",
                    attempt=attempt,
                    labeled=recovered["direction_labeled"],
                    remaining=recovered["direction_unjudged"],
                )
            ),
            flush=True,
        )
        if not recovered["direction_unjudged"]:
            return recovered
    return recovered


def paired_interval(deltas, groups):
    grouped = defaultdict(list)
    for qid, value in deltas.items():
        grouped[groups[qid]].append(value)
    if not grouped:
        return dict(queries=0, delta=None, interval95=None)
    values = list(grouped.values())
    sums = np.array([sum(v) for v in values])
    sizes = np.array([len(v) for v in values])
    index = np.random.default_rng(20261103).integers(0, len(values), size=(2000, len(values)))
    sampled = sums[index].sum(axis=1) / sizes[index].sum(axis=1)
    return dict(
        queries=len(deltas),
        groups=len(values),
        delta=mean(deltas.values()),
        interval95=[float(x) for x in np.percentile(sampled, [2.5, 97.5])],
    )


def compare(directory, previous):
    new_summary = json.loads((directory / "summary.json").read_bytes())
    old_summary = json.loads((previous / "summary.json").read_bytes())
    if new_summary["status"] != "complete" or old_summary["status"] != "complete":
        raise ValueError("检索未完成")
    queries = {q["query_id"]: q for q in json.loads((directory / "queries.json").read_bytes())}
    if list(queries.values()) != json.loads((previous / "queries.json").read_bytes()):
        raise ValueError("配对查询不一致")
    old_labels = readl(previous / "judge/direction_annotations.jsonl")
    new_labels = readl(directory / "judge/direction_annotations.jsonl")
    corpora = {
        "old": {j["job_key"]: j for j in readl(previous / "corpus.jsonl")},
        "new": {j["job_key"]: j for j in readl(directory / "corpus.jsonl")},
    }
    labels = defaultdict(dict)
    for source, values in (("old", old_labels), ("new", new_labels)):
        for row in values:
            job = corpora[source][row["job_key"]]
            if row["content_hash"] != job["content_hash"]:
                raise ValueError("标签岗位版本不符")
            for e in row["evidence"]:
                if job["jd_text"].split("\n")[e["line"] - 1] != e["text"]:
                    raise ValueError("标签证据原文不符")
            qid, key = row["query_id"], row["job_key"]
            if key in labels[qid] and labels[qid][key] != row["direction_grade"]:
                raise ValueError("两轮共享候选的方向标签冲突")
            labels[qid][key] = row["direction_grade"]
    rows = {
        side: [r for r in readl(path / "per_query.jsonl") if r["split"] == "test"]
        for side, path in (("old", previous), ("new", directory))
    }
    union = defaultdict(set)
    for rr in rows.values():
        for r in rr:
            union[r["query_id"]].update(r["ranking"][:10])
    complete = {qid for qid, keys in union.items() if keys <= labels[qid].keys()}
    positive = {qid for qid in complete if any(v == 2 for v in labels[qid].values())}
    per, measures = [], []
    for side, rr in rows.items():
        for row in rr:
            qid = row["query_id"]
            metrics = ranking_metrics(row["ranking"], labels[qid])
            per.append(
                dict(
                    corpus=side,
                    query_id=qid,
                    variant=row["variant"],
                    direction_p10=metrics["precision"],
                    pooled_recall10=metrics["recall"] if qid in positive else None,
                    pooled_ndcg10=metrics["ndcg"] if qid in positive else None,
                    returned=len(row["ranking"][:10]),
                    unjudged=metrics["unjudged"],
                    parser_blocked=bool(row.get("unresolved_conditions")),
                )
            )
        for variant in sorted({r["variant"] for r in rr}):
            items = [p for p in per if p["corpus"] == side and p["variant"] == variant]
            if len(items) != 60:
                raise ValueError("不是完整60测试查询")
            timing = [r["total_ms"] for r in rr if r["variant"] == variant and r["eligible_jobs"] > 0][1:]
            measures.append(
                dict(
                    corpus=side,
                    variant=variant,
                    direction_p10=mean(p["direction_p10"] for p in items)
                    if all(p["direction_p10"] is not None for p in items)
                    else None,
                    pooled_recall10=mean(
                        p["pooled_recall10"] for p in items if p["pooled_recall10"] is not None
                    )
                    if positive
                    else None,
                    pooled_ndcg10=mean(p["pooled_ndcg10"] for p in items if p["pooled_ndcg10"] is not None)
                    if positive
                    else None,
                    common_positive_queries=len(positive),
                    parser_blocked=sum(p["parser_blocked"] for p in items),
                    nonempty=sum(p["returned"] > 0 for p in items),
                    unjudged=sum(p["unjudged"] for p in items),
                    warm_p50_ms=float(np.percentile(timing, 50)) if timing else None,
                    warm_p95_ms=float(np.percentile(timing, 95)) if timing else None,
                )
            )
    index = {(p["corpus"], p["variant"], p["query_id"]): p for p in per}
    groups = {qid: q["group"] for qid, q in queries.items()}
    comparisons = []
    for variant in sorted({m["variant"] for m in measures}):
        delta = {}
        for qid, query in queries.items():
            if query["split"] != "test":
                continue
            a, b = index["old", variant, qid], index["new", variant, qid]
            if a["direction_p10"] is not None and b["direction_p10"] is not None:
                delta[qid] = b["direction_p10"] - a["direction_p10"]
        comparisons.append(dict(variant=variant, **paired_interval(delta, groups)))
    # 只复用旧报告明确列出的城市参考；不是拿新结果反推过滤条件。
    wanted = {
        r["query_id"]: r["wanted"]
        for r in json.loads((previous / "city_field_audit.json").read_bytes())["items"]
    }
    city_audit = []
    for side, rr in rows.items():
        for variant in sorted({r["variant"] for r in rr}):
            checks = [
                dict(
                    query_id=r["query_id"],
                    job_key=k,
                    wanted=wanted[r["query_id"]],
                    actual=corpora[side][k]["cities"],
                    violated=not bool(set(wanted[r["query_id"]]) & set(corpora[side][k]["cities"])),
                )
                for r in rr
                if r["variant"] == variant and r["query_id"] in wanted
                for k in r["ranking"][:10]
            ]
            city_audit.append(
                dict(
                    corpus=side,
                    variant=variant,
                    positions=len(checks),
                    violations=sum(c["violated"] for c in checks),
                    items=checks,
                )
            )
    method_comparisons = []
    baseline = "line700_o1__original__hybrid_rerank"
    for variant in sorted({m["variant"] for m in measures} - {baseline}):
        delta = {}
        for qid, query in queries.items():
            if query["split"] != "test":
                continue
            a, b = index["new", baseline, qid], index["new", variant, qid]
            if a["direction_p10"] is not None and b["direction_p10"] is not None:
                delta[qid] = b["direction_p10"] - a["direction_p10"]
        method_comparisons.append(dict(variant=variant, **paired_interval(delta, groups)))
    index_comparison = {
        side: summary.get("indexes", []) for side, summary in (("old", old_summary), ("new", new_summary))
    }
    checkpoint = json.loads((directory / "judge/checkpoint.json").read_bytes())
    reuse = json.loads((directory / "judge/reuse.json").read_bytes())
    report = dict(
        protocol="corpus-paired-regression.v1",
        old_corpus_jobs=old_summary["corpus_jobs"],
        new_corpus_jobs=new_summary["corpus_jobs"],
        test_queries=60,
        dev_queries=60,
        label_kind="model direction proxy + preserved assistant completions; not human gold",
        new_pool_pairs=len(new_labels),
        shared_label_pool_pairs=sum(map(len, labels.values())),
        complete_query_pools=len(complete),
        common_positive_queries=len(positive),
        direction_complete=all(m["unjudged"] == 0 for m in measures),
        strict_metrics_validated=False,
        reused_annotations=reuse["seeded"],
        newly_direction_labeled=len(new_labels) - reuse["seeded"],
        new_label_sources=dict(Counter(r["annotation_kind"] for r in new_labels)),
        judge_calls=len(checkpoint["calls"]),
        judge_reported_tokens=checkpoint["reported_tokens"],
        new_transform_calls=len(new_summary["transform_calls"]),
        new_transform_tokens=new_summary["transform_tokens"],
        total_new_reported_tokens=checkpoint["reported_tokens"] + new_summary["transform_tokens"],
        currency_cost=None,
        measures=measures,
        corpus_paired_changes=comparisons,
        method_paired_changes=method_comparisons,
        indexes=index_comparison,
        query_categories=dict(Counter(q.get("category", "unspecified") for q in queries.values())),
        city_field_audit=city_audit,
        limitations=[
            "Same120 exposed assistant scenarios, not blind or broad multi-occupation validation",
            "Common old/new Top10 union pool, not global corpus recall; no unjudged-as-zero",
            "Reused direction labels preserve original model errors and shared-provider bias",
            "Strict eligibility judgments remain uncalibrated; city audit covers only6 predeclared queries",
            "No new generated-answer/citation semantic benchmark; native GPU timing excludes model transforms, generation/API/UI",
            "P10 macro includes blocked/empty queries at fixed denominator600; paired bootstrap only query sampling under fixed proxy labels",
        ],
    )
    save(directory / "rerun_metrics.json", report)
    save(
        directory / "city_field_audit.json",
        dict(
            summary=[{k: v for k, v in r.items() if k != "items"} for r in city_audit],
            items=[
                dict(corpus=r["corpus"], variant=r["variant"], **i) for r in city_audit for i in r["items"]
            ],
        ),
    )
    (directory / "common_pool_per_query.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in per), encoding="utf-8"
    )

    def fmt(value):
        return "未知" if value is None else f"{value:.4f}"

    text = [
        "# 1574岗位语料重测",
        "",
        "固定原120场景（dev/test各60）与8方案，未调参；旧1004、新1574，两轮查询与冻结改写/HyDE文本完全一致。不是新盲测或多职业完整评测。",
        "",
        f"新测试候选池{len(new_labels)}对，复用{reuse['seeded']}、新补{len(new_labels) - reuse['seeded']}；模型代理而非人工金标。方向标签是否完整：{report['direction_complete']}。旧/新统一在{report['shared_label_pool_pairs']}对共同标签池上重计，Recall/nDCG共用{len(positive)}条有正例查询；非全语料Recall。",
        "",
        "|方案|旧P@10|新P@10|旧池Recall|新池Recall|旧池nDCG|新池nDCG|",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in sorted({m["variant"] for m in measures}):
        a = next(m for m in measures if m["corpus"] == "old" and m["variant"] == variant)
        b = next(m for m in measures if m["corpus"] == "new" and m["variant"] == variant)
        text.append(
            f"|{NAMES[variant]}|{fmt(a['direction_p10'])}|{fmt(b['direction_p10'])}|{fmt(a['pooled_recall10'])}|{fmt(b['pooled_recall10'])}|{fmt(a['pooled_ndcg10'])}|{fmt(b['pooled_ndcg10'])}|"
        )
    text += [
        "",
        "P@10固定60×10；未支持条件产生空结果仍计入。扩大共同标注池会改变Recall/nDCG的分母，旧列也重算，不能直接拿旧报告不同池分数比较。",
        "",
        "## 条件、耗时与成本",
        "",
        f"新查询变换调用{report['new_transform_calls']}，新标注调用{report['judge_calls']}，新增已报告tokens {report['total_new_reported_tokens']}；单价未知，金额null。旧变换历史tokens不重复计入本轮。原6次HyDE超时回退文本保持不变。",
        "",
        "城市字段检查只覆盖原先明确标出的6条返回查询；不是整体硬条件违反率。严格资格判分仍不可靠，不作算法选型或用户胜任结论。",
        "",
        "|方案|新非空/60|解析阻断|城市字段违反/返回位置|检索P50/P95 ms|",
        "|---|---:|---:|---:|---:|",
    ]
    for m in measures:
        if m["corpus"] == "new":
            city = next(c for c in city_audit if c["corpus"] == "new" and c["variant"] == m["variant"])
            text.append(
                f"|{NAMES[m['variant']]}|{m['nonempty']}|{m['parser_blocked']}|{city['violations']}/{city['positions']}|{m['warm_p50_ms']:.1f}/{m['warm_p95_ms']:.1f}|"
            )
    text += [
        "",
        "耗时为本机GPU检索，剔除快速退出与各方案首项，不含变换/回答生成/API/UI；不当线上端到端延迟或容量。配对分组bootstrap区间见rerun_metrics.json，仅反映固定代理标签下查询抽样误差，不包含模型错标。",
        "",
        "软件首次检查并发领取出现MySQL1213死锁；已增加完整领取事务最多3次重试，其他错误不隐藏。原失败保留，修后测试证据单列。生产检索参数、语料、索引和旧实验不因评测更改。",
        "",
        "## 复现",
        "",
        "冻结协议与哈希见rerun_protocol.json/plan.json；检索用--snapshot指向jobs_v6、--queries指向scenario_intents_v2、--transform-cache复用旧JSON。",
        "`python -m jobcompass.corpus_rerun --directory <本目录> --previous artifacts/experiments/scenario_20261103_v1 --report-only`只读计分；不传--report-only则有界补新候选标注。",
    ]
    text += ["", "## 切块与模型输入预算", "", "|语料|切块方案|块数|超过512 token的块|", "|---|---|---:|---:|"]
    for side, stats in index_comparison.items():
        for chunk in stats:
            text.append(f"|{side}|{chunk['chunk_variant']}|{chunk['chunks']}|{chunk['over_512_count']}|")
    text += [
        "",
        "字符长度不等于token长度；原字符块超限时编码截断，token块按预算切分。效果对照见主表，不因块数多或无超限就推断质量更高。",
        "",
        "## 新语料内的方法配对差值",
        "",
        "|相对原混合重排|方向P差值|分组bootstrap95%区间|",
        "|---|---:|---|",
    ]
    for comparison in method_comparisons:
        text.append(f"|{NAMES[comparison['variant']]}|{fmt(comparison['delta'])}|{comparison['interval95']}|")
    (directory / "REPORT.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--max-calls", type=int, default=120)
    parser.add_argument("--max-tokens", type=int, default=1200000)
    args = parser.parse_args()
    if not args.report_only:
        asyncio.run(judge_new(args.directory, args.previous, args.max_calls, args.max_tokens))
    result = compare(args.directory, args.previous)
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "direction_complete",
                    "reused_annotations",
                    "newly_direction_labeled",
                    "judge_calls",
                    "judge_reported_tokens",
                )
            },
            ensure_ascii=False,
        )
    )
