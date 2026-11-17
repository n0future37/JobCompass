"""对新场景实验做本地汇总；模型标签与人工金标严格分开。"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

import numpy as np

from .evaluation import ranking_metrics
from .experiment_report import NAMES
from .preprocessing import digest


def readl(path):
    return [json.loads(s) for s in path.read_text(encoding="utf-8").split("\n") if s.strip()]


def summarize(directory):
    summary = json.loads((directory / "summary.json").read_bytes())
    if summary["status"] != "complete":
        raise ValueError("检索未完成")
    rows = readl(directory / "per_query.jsonl")
    queries = {q["query_id"]: q for q in json.loads((directory / "queries.json").read_bytes())}
    checkpoint = json.loads((directory / "judge/checkpoint.json").read_bytes())
    label_file = (
        "judge/direction_annotations.jsonl"
        if (directory / "judge/direction_annotations.jsonl").exists()
        else "judge/annotations.jsonl"
    )
    labels = readl(directory / label_file)
    assert len({(r["query_id"], r["job_key"]) for r in labels}) == len(labels)
    by_query = defaultdict(dict)
    direction = defaultdict(dict)
    states = {}
    for row in labels:
        if row["grade"] is not None:
            by_query[row["query_id"]][row["job_key"]] = row["grade"]
        direction[row["query_id"]][row["job_key"]] = row["direction_grade"]
        states[row["query_id"], row["job_key"]] = row["hard_status"]
    test = [r for r in rows if r["split"] == "test"]
    union = defaultdict(set)
    for r in test:
        union[r["query_id"]].update(r["ranking"])
    complete_pool = {q for q, keys in union.items() if keys and keys <= by_query[q].keys()}
    positive_pool = {q for q in complete_pool if any(g == 2 for g in by_query[q].values())}
    direction_complete_pool = {q for q, keys in union.items() if keys and keys <= direction[q].keys()}
    direction_positive_pool = {
        q for q in direction_complete_pool if any(g == 2 for g in direction[q].values())
    }
    baseline = "line700_o1__original__hybrid_rerank"
    baserows = {r["query_id"]: r for r in test if r["variant"] == baseline}
    per = []
    measures = []
    for row in test:
        qid = row["query_id"]
        rank = row["ranking"][:10]
        g = by_query[qid]
        missing = [k for k in rank if k not in g]
        exact = ranking_metrics(rank, g)
        strict = exact["precision"]
        direct_metrics = ranking_metrics(rank, direction[qid])
        direct = direct_metrics["precision"]
        per.append(
            dict(
                query_id=qid,
                variant=row["variant"],
                group=queries[qid]["group"],
                category=queries[qid]["category"],
                p10=strict,
                direction_p10=direct,
                direction_p10_lower=sum(direction[qid].get(k) == 2 for k in rank) / 10,
                direction_p10_upper=(
                    sum(direction[qid].get(k) == 2 for k in rank) + sum(k not in direction[qid] for k in rank)
                )
                / 10,
                direction_pool_recall10=direct_metrics["recall"] if qid in direction_positive_pool else None,
                direction_pool_ndcg10=direct_metrics["ndcg"] if qid in direction_positive_pool else None,
                p10_lower=sum(g.get(k) == 2 for k in rank) / 10,
                p10_upper=(sum(g.get(k) == 2 for k in rank) + len(missing)) / 10,
                pool_recall10=exact["recall"] if qid in positive_pool else None,
                pool_ndcg10=exact["ndcg"] if qid in positive_pool else None,
                returned=len(rank),
                unjudged=len(missing),
                parser_blocked=bool(row.get("unresolved_conditions")),
                hard_violations=sum(states.get((qid, k)) == "violated" for k in rank),
                hard_unknown=sum(states.get((qid, k)) == "unknown" for k in rank),
                judged_positions=sum((qid, k) in states for k in rank),
            )
        )
    for variant in sorted({r["variant"] for r in test}):
        items = [r for r in per if r["variant"] == variant]
        runrows = [r for r in test if r["variant"] == variant]
        if len(items) != 60:
            raise ValueError("不是完整60测试查询")
        timing = [r["total_ms"] for r in runrows if r["eligible_jobs"] > 0]
        warmed = timing[1:] if len(timing) > 1 else timing
        positions = sum(r["judged_positions"] for r in items)
        answered = [r for r in items if not r["parser_blocked"]]
        measures.append(
            dict(
                variant=variant,
                queries=60,
                p10=mean(r["p10"] for r in items) if all(r["p10"] is not None for r in items) else None,
                p10_lower=mean(r["p10_lower"] for r in items),
                p10_upper=mean(r["p10_upper"] for r in items),
                direction_p10=mean(r["direction_p10"] for r in items)
                if all(r["direction_p10"] is not None for r in items)
                else None,
                direction_p10_lower=mean(r["direction_p10_lower"] for r in items),
                direction_p10_upper=mean(r["direction_p10_upper"] for r in items),
                direction_pool_metric_queries=len(direction_positive_pool),
                direction_pool_recall10=mean(
                    r["direction_pool_recall10"] for r in items if r["direction_pool_recall10"] is not None
                )
                if direction_positive_pool
                else None,
                direction_pool_ndcg10=mean(
                    r["direction_pool_ndcg10"] for r in items if r["direction_pool_ndcg10"] is not None
                )
                if direction_positive_pool
                else None,
                unblocked_direction_p10=mean(r["direction_p10"] for r in answered)
                if all(r["direction_p10"] is not None for r in answered)
                else None,
                unblocked_queries=len(answered),
                unblocked_p10=mean(r["p10"] for r in answered)
                if all(r["p10"] is not None for r in answered)
                else None,
                pool_metric_queries=len(positive_pool),
                pool_recall10=mean(r["pool_recall10"] for r in items if r["pool_recall10"] is not None)
                if positive_pool
                else None,
                pool_ndcg10=mean(r["pool_ndcg10"] for r in items if r["pool_ndcg10"] is not None)
                if positive_pool
                else None,
                returned=sum(r["returned"] for r in items),
                judged_positions=positions,
                hard_violation_rate=sum(r["hard_violations"] for r in items) / positions
                if positions
                else None,
                hard_unknown_rate=sum(r["hard_unknown"] for r in items) / positions if positions else None,
                retrieval_unblocked_p50_ms=float(np.percentile(warmed, 50)) if warmed else None,
                retrieval_unblocked_p95_ms=float(np.percentile(warmed, 95)) if warmed else None,
                parser_blocked=sum(r["parser_blocked"] for r in items),
            )
        )
    base = {r["query_id"]: r for r in per if r["variant"] == baseline}
    comparisons = []
    for variant in sorted({r["variant"] for r in per} - {baseline}):
        current = {r["query_id"]: r for r in per if r["variant"] == variant}
        groups = defaultdict(list)
        for qid in base:
            if base[qid]["direction_p10"] is not None and current[qid]["direction_p10"] is not None:
                groups[base[qid]["group"]].append(current[qid]["direction_p10"] - base[qid]["direction_p10"])
        if sum(map(len, groups.values())) != 60:
            comparisons.append(
                dict(
                    variant=variant,
                    delta_p10=None,
                    interval95=None,
                    reason="部分返回候选未标，不比较不同子集均值",
                )
            )
            continue
        values = list(groups.values())
        rng = np.random.default_rng(20261103)
        indices = rng.integers(0, len(values), size=(4000, len(values)))
        sampled = [mean(x for i in sample for x in values[int(i)]) for sample in indices]
        comparisons.append(
            dict(
                variant=variant,
                queries=60,
                groups=len(values),
                delta_p10=mean(x for v in values for x in v),
                interval95=[float(x) for x in np.percentile(sampled, [2.5, 97.5])],
                metric="direction_p10",
                method="paired group bootstrap4000; query-weighted macro; only sampling uncertainty under fixed model labels",
            )
        )
    corpus = {j["job_key"]: j for j in readl(directory / "corpus.jsonl")}
    for label in labels:
        job = corpus[label["job_key"]]
        if label["content_hash"] != job["content_hash"]:
            raise ValueError("标注岗位版本变化")
        for e in label["evidence"]:
            if job["jd_text"].split("\n")[e["line"] - 1] != e["text"]:
                raise ValueError("引用校验失败")
    blocked = [
        dict(
            query_id=qid,
            query=queries[qid]["query"],
            reference=queries[qid]["hard_constraints_reference"],
            unresolved=r["unresolved_conditions"],
        )
        for qid, r in baserows.items()
        if r.get("unresolved_conditions")
    ]
    silent = [
        dict(
            query_id=qid,
            query=queries[qid]["query"],
            reference=queries[qid]["hard_constraints_reference"],
            recognized=r["conditions"],
        )
        for qid, r in baserows.items()
        if queries[qid]["hard_constraints_reference"]
        and not r.get("unresolved_conditions")
        and not any(r["conditions"].values())
    ]
    qualifier_flags = [
        dict(query_id=r["query_id"], job_key=r["job_key"], reason=r["reason"])
        for r in labels
        if not queries[r["query_id"]]["hard_constraints_reference"]
        and r["hard_status"] == "unknown"
        and any(word in r["reason"] for word in ("学历", "年限", "经验", "资格"))
    ]
    corpus_coverage = dict(
        cities=dict(Counter(c for j in corpus.values() for c in j.get("cities", []))),
        employment_types=dict(Counter(j.get("employment_type_text") for j in corpus.values())),
    )
    report = dict(
        protocol="scenario-model-assisted-report.v2",
        test_queries=60,
        dev_queries=60,
        corpus_jobs=summary["corpus_jobs"],
        corpus_hash=summary["corpus_hash"],
        label_source="model+assistant unresolved completions; not human gold",
        label_sources=dict(Counter(r["annotation_kind"] for r in labels)),
        label_model=checkpoint["model"],
        labeled_pairs=len(labels),
        test_union_pairs=sum(len(k) for k in union.values()),
        label_status="direction_complete"
        if len(labels) == sum(len(k) for k in union.values())
        else "direction_partial",
        raw_judge_status=checkpoint["status"],
        strict_metrics_validated=False,
        strict_metrics_warning="Confirmed invented qualification and unspecified-qualification penalties; do not use strict diagnostics for algorithm selection",
        assistant_sample_audit=json.loads((directory / "audit/comparison.json").read_bytes())
        if (directory / "audit/comparison.json").exists()
        else None,
        city_field_audit=json.loads((directory / "city_field_audit.json").read_bytes())
        if (directory / "city_field_audit.json").exists()
        else None,
        judge_status_counts=dict(Counter(c["status"] for c in checkpoint["calls"])),
        label_file=label_file,
        direction_pool_metric_queries=len(direction_positive_pool),
        direction_labeled_complete=len(labels) == sum(len(k) for k in union.values()),
        comparison_metric="direction_p10",
        qualifier_unknown_flags=qualifier_flags,
        corpus_coverage=corpus_coverage,
        primary_status="Exploratory direction-relevance proxy; strict matching diagnostics affected by judge qualification bias, not selection evidence",
        positive_pool_metric_queries=len(positive_pool),
        pool_complete_queries=len(complete_pool),
        measures=measures,
        paired_comparisons=comparisons,
        parser_blocked_test=blocked,
        hard_references_without_detected_filter=silent,
        judge_tokens=checkpoint["reported_tokens"],
        judge_calls=len(checkpoint["calls"]),
        judge_failures=sum(c["status"] != "ok" for c in checkpoint["calls"]),
        transform_tokens=summary["transform_tokens"],
        transform_calls=len(summary["transform_calls"]),
        total_reported_tokens=summary["transform_tokens"] + checkpoint["reported_tokens"],
        currency_cost=None,
        input_hashes={
            name: digest((directory / name).read_bytes())
            for name in [
                "plan.json",
                "queries.json",
                "corpus.jsonl",
                "per_query.jsonl",
                label_file,
            ]
        },
        limitations=[
            "Constructed assistant scenarios, not real user logs or external blind test; source/domain shared across splits",
            "Fixed provider alias used for both query transform and model judgment; systematic shared-model bias possible",
            "Judge sometimes treats unspecified user qualifications as unknown even when query does not require them; strict matching not validated",
            "Direction results are exploratory diagnostics after detecting judge bias, not a preregistered human benchmark",
            "Model qrels with line validation are not independently verified human gold; bootstrap does not include labeling errors",
            "Candidate-pool recall/nDCG only on same complete positive-pool subset, not global recall; no-result correctness unknown",
            "Production parser gaps unchanged; model-judged hard violations do not equal formally proven violations",
            "No generated-answer semantic citation or unsupported-claim quality evaluation in this experiment",
            "Token usage provider-reported; prices unconfigured, actual currency cost unknown; no end-to-end per-query latency from batched transforms",
        ],
    )
    (directory / "model_assisted_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (directory / "model_assisted_per_query.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in per), encoding="utf-8"
    )

    def fmt(x):
        return "未知" if x is None else f"{x:.4f}"

    text = [
        "# 新120场景检索对照实验",
        "",
        f"固定{summary['corpus_jobs']}岗位；8方案×120查询=960次检索；60开发、60测试，未调参。当前报告质量部分仅60测试查询，模型辅助标签，非人工金标。",
        "",
        "原700字符/一行重叠、token256/32、384/48、384/0；BGE-small-zh-v1.5与BGE-reranker-v2-m3，词法/向量每路40岗位、RRF60、前15重排、Top10。改写替换词法和向量查询；HyDE只替换向量输入，原始查询用于条件解析和重排；假想JD不作最终依据。",
        "",
        f"测试候选并集{report['test_union_pairs']}对，已标{len(labels)}，来源{report['label_sources']}；对模型隐藏方案/排名，提供完整公开JD及构造查询。方向标签状态{report['label_status']}；原批次严格校验状态{checkpoint['status']}，6对严格等级仍结构矛盾，另有严格判分语义偏差。质量字段含义：方向P@10只看工作方向，严格P@10同时考虑硬条件；明确冲突0、关键未知至多1，软偏好不硬过滤。",
        "",
        "模型判分出现了把查询未限定的学历/年限也当成画像未知的错误，严格匹配分数尚未可靠验证。下表优先展示方向相关性代理；严格P@10只作诊断，不作为选型结论。方向评分仍可能受模型偏差影响，不是人工金标。",
        "",
        "|方案|方向P@10（全部60）|方向候选池Recall@10|方向候选池nDCG@10|",
        "|---|---:|---:|---:|",
    ]
    for m in measures:
        text.append(
            f"|{NAMES[m['variant']]}|{fmt(m['direction_p10']) if m['direction_p10'] is not None else str([round(m['direction_p10_lower'], 4), round(m['direction_p10_upper'], 4)])}|{fmt(m['direction_pool_recall10'])}|{fmt(m['direction_pool_ndcg10'])}|"
        )
    text += [
        "",
        f"方向Recall/nDCG只在同一{len(direction_positive_pool)}条测试查询上计算：候选并集标全且至少一项2级。其余查询的分母未知，不当作0。P@10以全部60条×10为分母，未支持条件导致的空结果贡献0，不通过删掉困难题抬分。候选池不是1004岗位穷尽标注；新指标不能与旧6题的分数作直接提升比较。",
        "",
        "## 判分抽核与限制",
        "",
        "13类各1项助手首评在揭示模型标签前冻结；方向精确一致11/13，严格等级10/13。两处方向差异为产品化AI/DevOps相邻职责边界，不自动认定谁是金标。另发现明确错误：S_T023查询没有大专背景，模型却声称大专与岗位本科冲突；S_T003未限制学历年限却被判未知。这使严格匹配判分不适合作为效果指标。抽核不是第二人类标注，11/13不能称1300对总体准确率，详见audit/REVIEW.md。",
        "",
        "主表只保留方向相关性代理。严格匹配P/Recall/nDCG与模型判硬冲突率仅在JSON保留为不可靠诊断，strict_metrics_validated=false，不填造可靠的总体硬条件违反率或用户胜任率。",
        "",
        "## 覆盖与耗时",
        "",
        f"结构化城市统计：{corpus_coverage['cities']}；招聘类型：{corpus_coverage['employment_types']}。结构化岗位均深圳，正文可能还写其他办公选项，不能以该统计证明所有外地查询全语料无结果。当前语料对跨城和实习需求覆盖有限。所有查询request为空，未代填结构化筛选器；本实验包含自然语言解析缺口，不代表显式城市筛选失效。",
        "",
        f"检测到{len(qualifier_flags)}对无作者硬条件参考、却因学历/年限/经验等被模型判unknown的待复核项。此规则只标诊断候选，不自动修改标签或当作已确认错标率；原始输出与证据可追溯。",
        "",
        f"现有解析器在全120条中阻断27条；测试阻断{len(blocked)}条。另有{len(silent)}条测试查询带作者硬条件参考，但无解析过滤且未阻断；这是需复核的能力缺口，不直接等于每项都违规。详见JSON清单。生产条件解析逻辑未因本轮结果修改。",
        "",
        "独立字段检查（非模型判分）：测试有6条明确城市参考的查询返回了岗位，每方案共60个返回位置，其结构化cities均不满足所需城市。city_field_audit.json逐项保留，反映自然语言条件没有转成过滤；不是整体硬条件违规率，也不证明JD正文的其他地点选项不可用。",
        "",
        "|方案|测试非空查询数|未阻断查询数|未阻断方向P@10|检索P50 ms|检索P95 ms|",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for m in measures:
        nonempty = sum(bool(r["ranking"]) for r in test if r["variant"] == m["variant"])
        text.append(
            f"|{NAMES[m['variant']]}|{nonempty}/60|{m['unblocked_queries']}|{fmt(m['unblocked_direction_p10'])}|{m['retrieval_unblocked_p50_ms']:.1f}|{m['retrieval_unblocked_p95_ms']:.1f}|"
        )
    text += [
        "",
        "检索耗时剔除未支持条件/无可检索岗位的快速退出，再剔除各方案首条；仅本机GPU顺序检索，含解析、局部数据筛选、编码、融合和重排，不含批量查询变换、回答生成、API/UI。不是完整在线P50；变换批量用时见原始调用记录。",
        "",
        "## 配对差值与限制",
        "",
        "|相对原查询混合重排|方向P@10差值|按场景组bootstrap95%区间|",
        "|---|---:|---|",
    ]
    for c in comparisons:
        text.append(f"|{NAMES[c['variant']]}|{fmt(c['delta_p10'])}|{c['interval95']}|")
    text += [
        "",
        "区间只在固定模型标签下估计查询抽样不确定性，不涵盖模型错标、同模型生成/判分偏差，且同时比较多个方案，不能自动据此宣称显著总体提升或选择生产默认。",
        "",
        f"查询变换{report['transform_calls']}请求/{report['transform_tokens']}已报告tokens；模型标注{report['judge_calls']}请求/{report['judge_tokens']}已报告tokens，非完整有效批次{report['judge_failures']}（包括部分标签合法，不全是服务错误）。合计{report['total_reported_tokens']}已报告tokens；失败请求与至多2个中断中请求可能有未回报用量，单价未配置，金额未知。质量标签未使用真实用户画像/简历。",
        "",
        "本实验没有评估生成回答的语义引用支持率、完整回答无依据陈述或无结果判断的真实准确率。旧763人工标注不覆盖新查询。开发集只运行检索诊断，未追加模型标注；质量表为60条测试。",
        "",
        "## 复现",
        "",
        "`python -m jobcompass.experiments --queries data/eval/scenario_intents_v2 --directory artifacts/experiments/<新目录>`使用已配置真实变换服务；旧72条入口不传--queries。",
        "`python -m jobcompass.scenario_judging --directory <实验目录>`标注测试候选，已有checkpoint复用且失败留未知；`python -m jobcompass.scenario_report --directory <实验目录>`只读现有产物计分，零外部调用。",
        "`python -m jobcompass.scenario_label_cache --directory <实验目录>`只从缓存恢复合法方向判断，容许最多6条真实行证据，严格等级矛盾保持null；原模型标签不覆盖。该步骤不调用模型。",
        "复现实验应保存并复用transforms缓存和judge checkpoint；已完成实验目录禁止覆盖。集合在测试数据库隔离创建并清理，开发索引和1004岗位不改。",
        "",
    ]
    (directory / "REPORT.md").write_text("\n".join(text), encoding="utf-8")
    return dict(
        status="complete",
        labeled=len(labels),
        test_pool=report["test_union_pairs"],
        positive_pool_queries=len(positive_pool),
        measures=measures,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(args.directory), ensure_ascii=False, indent=2))
