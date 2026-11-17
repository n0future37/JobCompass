"""固定语料的召回/重排消融。诊断参考条件仅在本实验进程内注入，不进入生产链路。"""

import argparse
import json
import platform
import re
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from time import perf_counter
from unittest.mock import patch

import numpy as np
from sqlalchemy import text

from .conditions import SearchRequest
from .evaluation import ranking_metrics
from .experiments import verify_corpus_snapshot
from .intent import HardConditions, Intent
from .preprocessing import digest
from .retrieval import Retriever
from .settings import ROOT, Settings
from .storage import Store

VARIANTS = [
    ("bm25", "bm25", False, "current"),
    ("bm25_rerank", "bm25", True, "current"),
    ("dense", "dense", False, "current"),
    ("dense_rerank", "dense", True, "current"),
    ("hybrid", "hybrid", False, "current"),
    ("hybrid_rerank", "hybrid", True, "current"),
    ("legacy_conditions", "hybrid", True, "legacy"),
    ("reference_conditions", "hybrid", True, "reference"),
]


def readl(path):
    return [json.loads(s) for s in path.read_text(encoding="utf-8").split("\n") if s.strip()]


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def reference_check(job, conditions):
    """独立判分器：不用生产assess/解析器；仅验证参考定义的来源字段和明确提词。"""
    states = []
    c = conditions

    def clean_city(x):
        return x.strip().removesuffix("市")

    raw_cities = job.get("cities")
    actual = {clean_city(x) for x in raw_cities or [] if isinstance(x, str) and x.strip()}
    include, exclude = set(c["cities"]), set(c["excluded_cities"])
    if include or exclude:
        if not actual:
            states.append("unknown")
        else:
            permitted = (actual & include if include else actual) - exclude
            states.append("satisfied" if permitted else "violated")
    if c["employment_types"] or c["excluded_employment_types"]:
        aliases = {"校招": "校园", "校园招聘": "校园", "兼职": "兼职/临时", "临时": "兼职/临时"}
        raw = job.get("employment_type_text") or ""
        actual_type = aliases.get(raw.strip(), raw.strip())
        if actual_type not in {"全职", "实习", "校园", "兼职/临时"}:
            states.append("unknown")
        else:
            ok = (not c["employment_types"] or actual_type in c["employment_types"]) and actual_type not in c[
                "excluded_employment_types"
            ]
            states.append("satisfied" if ok else "violated")
    for skill in c["skills_all"]:
        # 明确提词只在JD核验；不把个人技能、标题或模糊同义推断当证据。
        pattern = re.escape(skill)
        if skill.isascii():
            pattern = r"(?<![a-zA-Z0-9])" + pattern + r"(?![a-zA-Z0-9])"
        matches = [
            line
            for line in job["jd_text"].split("\n")
            if re.search(pattern, line, re.I)
            and not re.search(r"(?:无需|不要求|不需要|不涉及|不使用)[^，。；;]{0,12}" + pattern, line, re.I)
        ]
        states.append("satisfied" if matches else "unknown")
    return "violated" if "violated" in states else "unknown" if "unknown" in states else "satisfied"


def load_queries(dataset):
    manifest = json.loads((dataset / "manifest.json").read_bytes())
    rows = []
    for split in ("dev", "test"):
        name = split + ".jsonl"
        if digest((dataset / name).read_bytes()) != manifest["hashes"][name]:
            raise ValueError("冻结条件查询指纹不符")
        part = readl(dataset / name)
        if len(part) != manifest["splits"][split] or any(r["split"] != split for r in part):
            raise ValueError("split数量或字段不符")
        rows.extend(part)
    if len({r["query_id"] for r in rows}) != len(rows):
        raise ValueError("重复查询ID")
    grouped = defaultdict(set)
    for row in rows:
        grouped[row["group"]].add(row["split"])
    if any(len(x) > 1 for x in grouped.values()):
        raise ValueError("同组跨split")
    return rows


def injected_intent(query, legacy, source):
    if source == "reference":
        status = query["expected_status"]
        return Intent(
            version="intent.reference.diagnostic",
            hard=HardConditions(**query["expected_conditions"]),
            unresolved=[] if status == "ready" else ["参考条件不可执行"],
            conflicts=["参考条件冲突"] if status == "conflict" else [],
        )
    return Intent(
        version="intent.v1.frozen.diagnostic",
        hard=HardConditions(**legacy["conditions"]),
        unresolved=legacy["unresolved"],
    )


def run(directory, dataset, snapshot, baseline):
    queries = load_queries(dataset)
    legacy = {r["query_id"]: r for r in readl(baseline)}
    if set(legacy) != {q["query_id"] for q in queries}:
        raise ValueError("旧解析输出未覆盖相同查询")
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "per_query.jsonl").exists():
        raise ValueError("实验目录已含结果，请用新目录，不能覆盖")
    settings = Settings()
    plan = dict(
        protocol="retrieval-stage-ablation.v1",
        created_at=datetime.now(UTC).isoformat(),
        dataset=str(dataset),
        dataset_manifest_sha256=digest((dataset / "manifest.json").read_bytes()),
        baseline_sha256=digest(baseline.read_bytes()),
        variants=VARIANTS,
        source_hashes={
            name: digest((ROOT / name).read_bytes())
            for name in (
                "src/jobcompass/intent.py",
                "src/jobcompass/conditions.py",
                "src/jobcompass/retrieval.py",
                "src/jobcompass/stage_ablation.py",
                "configs/rag.toml",
            )
        },
        config=settings.config,
        evaluation_policy=dict(
            conditions="all64test; original assistant-authored references",
            relevance="only48test non-boundary occupation scenarios, shared top10 pool; model proxy",
            candidate_budget="each branch40; raw RRF union<=80; retain first40 for stage coverage; rerank first15/top10 unchanged",
            stage_coverage="known positives in top10-union pool only, NOT corpus recall@40",
            no_generation=True,
            no_query_transform=True,
            no_profile=True,
            labels="direction only; field compliance independently evaluated; model strict grade ignored",
            tuning="no selection or parameter tuning on test; templates reused across16occupations",
            latency="serial native GPU including DB/version checks; first request cold marked; excludes generation/API/UI",
        ),
        judge_budget=dict(max_calls=180, max_reported_tokens=1800000, concurrency=2, attempts_per_pair=2),
        platform=platform.platform(),
    )
    save(directory / "plan.json", plan)
    store, test_store = Store(settings), Store(Settings("test"))
    lock = test_store.engine.connect()
    locked = False
    retriever = None
    summary = dict(status="running", protocol=plan["protocol"], external_calls=0)
    save(directory / "summary.json", summary)
    try:
        locked = bool(lock.execute(text("SELECT GET_LOCK('jobcompass:test:validation',0)")).scalar())
        if not locked:
            raise RuntimeError("其他真实验证正在运行")
        retriever = Retriever(settings=settings, store=store)
        summary["corpus"] = verify_corpus_snapshot(retriever.jobs, snapshot)
        manifest_path = ROOT / "artifacts/indexes/dev/current.json"
        before_index = digest(manifest_path.read_bytes())
        before_versions = digest([[j["job_key"], j["content_hash"], j["version"]] for j in retriever.jobs])
        by_key = retriever.by_key
        (directory / "corpus.jsonl").write_text(
            "".join(json.dumps(j, ensure_ascii=False) + "\n" for j in retriever.jobs), encoding="utf-8"
        )
        export_queries = []
        for query in queries:
            export_queries.append(
                query
                | dict(
                    kind="assistant_scenario",
                    hard_constraints_reference=[
                        f"{field}={','.join(values)}"
                        for field, values in query["expected_conditions"].items()
                        if values
                    ],
                    soft_preferences_reference=[
                        f"{field}={','.join(values)}"
                        for field, values in query["expected_soft"].items()
                        if values
                    ],
                )
            )
        save(directory / "queries.json", export_queries)
        rows = []
        start_all = perf_counter()
        for qi, query in enumerate(queries):
            # 按查询稳定轮换方法顺序，避免所有冷启动集中在同一方案。
            offset = qi % len(VARIANTS)
            variants = VARIANTS[offset:] + VARIANTS[:offset]
            for variant, mode, rerank, source in variants:
                request = SearchRequest(
                    query=query["query"],
                    **query["request"],
                    mode=mode,
                    rerank=rerank,
                    generate=False,
                    top_k=10,
                )
                started = perf_counter()
                if source == "current":
                    result = retriever.search(request, include_trace=True)
                else:
                    intent = injected_intent(query, legacy[query["query_id"]], source)
                    with patch("jobcompass.retrieval.parse_intent", return_value=intent):
                        result = retriever.search(request, include_trace=True)
                ranking = [r["job"]["job_key"] for r in result["results"]]
                trace = result.get("evaluation_trace", {})
                checks = {k: reference_check(by_key[k], query["expected_conditions"]) for k in ranking}
                row = dict(
                    query_id=query["query_id"],
                    split=query["split"],
                    category=query["category"],
                    group=query["group"],
                    variant=variant,
                    mode=mode,
                    rerank=rerank,
                    parser_source=source,
                    ranking=ranking,
                    returned=len(ranking),
                    quality_evaluation=query["expected_status"] == "ready"
                    and query["category"] != "boundary",
                    conditions=result["conditions"],
                    intent_status=result["intent"]["status"],
                    unresolved_conditions=result["unresolved_conditions"],
                    false_rejection=query["expected_status"] == "ready"
                    and bool(result["unresolved_conditions"]),
                    missed_rejection=query["expected_status"] != "ready"
                    and not result["unresolved_conditions"],
                    reference_checks=checks,
                    degradations=result["degradations"],
                    no_result_reason=result["no_result_reason"],
                    timings_ms=result["timings_ms"],
                    total_ms=(perf_counter() - started) * 1000,
                    cold_start=not rows,
                    trace={
                        "bm25": trace.get("bm25", []),
                        "dense": trace.get("dense", []),
                        "raw_union_size": len(trace.get("before_rerank", [])),
                        "candidate40": trace.get("before_rerank", [])[:40],
                        "rerank_input15": trace.get("before_rerank", [])[:15],
                    },
                )
                rows.append(row)
                with (directory / "per_query.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            if (qi + 1) % 8 == 0:
                print(
                    json.dumps(
                        {
                            "queries": qi + 1,
                            "runs": len(rows),
                            "seconds": round(perf_counter() - start_all, 1),
                        }
                    ),
                    flush=True,
                )
        summary.update(
            status="complete",
            queries=len(queries),
            runs=len(rows),
            seconds=perf_counter() - start_all,
            condition_test_queries=sum(q["split"] == "test" for q in queries),
            relevance_test_queries=sum(
                q["split"] == "test" and q["expected_status"] == "ready" and q["category"] != "boundary"
                for q in queries
            ),
            corpus_unchanged=before_versions
            == digest([[j["job_key"], j["content_hash"], j["version"]] for j in store.all_jobs()]),
            production_index_unchanged=before_index == digest(manifest_path.read_bytes()),
            production_index_sha256=before_index,
            degradations=sum(bool(r["degradations"]) for r in rows),
            external_calls=0,
            completed_at=datetime.now(UTC).isoformat(),
        )
        if not summary["corpus_unchanged"] or not summary["production_index_unchanged"]:
            raise ValueError("实验期间生产语料或索引变化，结果不能验收")
        save(directory / "summary.json", summary)
        return summary
    except Exception as error:
        summary.update(status="failed", error_type=type(error).__name__)
        save(directory / "summary.json", summary)
        raise
    finally:
        if retriever and retriever.client:
            retriever.client.close()
        store.close()
        if locked:
            lock.execute(text("SELECT RELEASE_LOCK('jobcompass:test:validation')"))
        lock.close()
        test_store.close()


def summarize(directory):
    summary = json.loads((directory / "summary.json").read_bytes())
    if summary["status"] != "complete":
        raise ValueError("检索未完成")
    queries = {q["query_id"]: q for q in json.loads((directory / "queries.json").read_bytes())}
    jobs = {j["job_key"]: j for j in readl(directory / "corpus.jsonl")}
    runs = readl(directory / "per_query.jsonl")
    label_path = directory / "judge/direction_annotations.jsonl"
    if not label_path.exists():
        label_path = directory / "judge/annotations.jsonl"
    labels = readl(label_path) if label_path.exists() else []
    direction = defaultdict(dict)
    constrained = defaultdict(dict)
    for label in labels:
        qid, key = label["query_id"], label["job_key"]
        if label["content_hash"] != jobs[key]["content_hash"] or key in direction[qid]:
            raise ValueError("重复或错误版本标签")
        for evidence in label["evidence"]:
            lines = jobs[key]["jd_text"].split("\n")
            if not 1 <= evidence["line"] <= len(lines) or lines[evidence["line"] - 1] != evidence["text"]:
                raise ValueError("引用不是原文")
        direction[qid][key] = label["direction_grade"]
        constrained[qid][key] = (
            label["direction_grade"]
            if reference_check(jobs[key], queries[qid]["expected_conditions"]) == "satisfied"
            else 0
        )
    test = [r for r in runs if r["split"] == "test"]
    quality = [r for r in test if r["quality_evaluation"]]
    union = defaultdict(set)
    for row in quality:
        union[row["query_id"]].update(row["ranking"])
    qids = {r["query_id"] for r in quality}
    complete = {q for q in qids if union[q] <= direction[q].keys()}
    positive = {q for q in complete if any(g == 2 for g in constrained[q].values())}
    per = []
    for row in quality:
        qid = row["query_id"]
        rank = row["ranking"]
        d, c = direction[qid], constrained[qid]
        scores = ranking_metrics(rank, c)
        known = {k for k, v in c.items() if v == 2}
        per.append(
            dict(
                query_id=qid,
                variant=row["variant"],
                category=row["category"],
                returned=len(rank),
                direction_p10=ranking_metrics(rank, d)["precision"],
                constrained_p10=scores["precision"],
                p10_lower=sum(c.get(k) == 2 for k in rank) / 10,
                p10_upper=(sum(c.get(k) == 2 for k in rank) + sum(k not in c for k in rank)) / 10,
                pool_recall10=scores["recall"] if qid in positive else None,
                pool_ndcg10=scores["ndcg"] if qid in positive else None,
                known_positive_coverage40=len(known & set(row["trace"]["candidate40"])) / len(known)
                if qid in positive
                else None,
                known_positive_coverage15=len(known & set(row["trace"]["rerank_input15"])) / len(known)
                if qid in positive
                else None,
            )
        )
    measures = []
    for variant, _, _, _ in VARIANTS:
        current = [r for r in test if r["variant"] == variant]
        items = [r for r in per if r["variant"] == variant]
        states = Counter(state for r in current for state in r["reference_checks"].values())
        latencies = [r["total_ms"] for r in current if not r["cold_start"] and r["returned"]]

        def macro(field):
            values = [r[field] for r in items]
            return mean(values) if values and all(v is not None for v in values) else None

        def pooled(field):
            values = [r[field] for r in items if r[field] is not None]
            return mean(values) if values else None

        measures.append(
            dict(
                variant=variant,
                test_queries=len(current),
                relevance_queries=len(items),
                direction_p10=macro("direction_p10"),
                constrained_p10=macro("constrained_p10"),
                p10_lower=macro("p10_lower"),
                p10_upper=macro("p10_upper"),
                complete_pool_queries=len(complete),
                positive_pool_queries=len(positive),
                pool_recall10=pooled("pool_recall10"),
                pool_ndcg10=pooled("pool_ndcg10"),
                known_positive_coverage40=pooled("known_positive_coverage40"),
                known_positive_coverage15=pooled("known_positive_coverage15"),
                returned_all64=sum(r["returned"] for r in current),
                returned_relevance48=sum(r["returned"] for r in items),
                empty_queries=sum(not r["ranking"] for r in current),
                false_rejections=sum(r["false_rejection"] for r in current),
                missed_rejections=sum(r["missed_rejection"] for r in current),
                hard_violation_positions=states["violated"],
                hard_unknown_positions=states["unknown"],
                checked_positions=sum(states.values()),
                hard_violation_rate=states["violated"] / sum(states.values()) if states else None,
                returned_query_p50_ms=float(np.percentile(latencies, 50)) if latencies else None,
                returned_query_p95_ms=float(np.percentile(latencies, 95)) if latencies else None,
            )
        )
    lookup = {(r["variant"], r["query_id"]): r for r in per}
    contrasts = []
    pairs = [
        ("bm25", "bm25_rerank"),
        ("dense", "dense_rerank"),
        ("hybrid", "hybrid_rerank"),
        ("bm25", "hybrid"),
        ("dense", "hybrid"),
        ("legacy_conditions", "hybrid_rerank"),
        ("hybrid_rerank", "reference_conditions"),
    ]
    for base, other in pairs:
        changes = [
            lookup[other, q]["constrained_p10"] - lookup[base, q]["constrained_p10"]
            for q in sorted(qids)
            if (other, q) in lookup
            and (base, q) in lookup
            and lookup[other, q]["constrained_p10"] is not None
            and lookup[base, q]["constrained_p10"] is not None
        ]
        contrasts.append(
            dict(
                baseline=base,
                variant=other,
                queries=len(changes),
                delta_constrained_p10=mean(changes) if len(changes) == len(qids) and changes else None,
                wins=sum(x > 0 for x in changes),
                ties=sum(x == 0 for x in changes),
                losses=sum(x < 0 for x in changes),
                significance="descriptive only; repeated templates, model labels, no human gold or universal significance claim",
            )
        )
    checkpoint = directory / "judge/checkpoint.json"
    judge = json.loads(checkpoint.read_bytes()) if checkpoint.exists() else {}
    result = dict(
        protocol="retrieval-stage-report.v1",
        run_summary=summary,
        labels=len(labels),
        label_kind="model direction proxy; not human gold; strict grade ignored",
        report_input_sha256={
            "per_query.jsonl": digest((directory / "per_query.jsonl").read_bytes()),
            "labels": digest(label_path.read_bytes()) if label_path.exists() else None,
        },
        costs=dict(
            calls=len(judge.get("calls", [])),
            reported_tokens=judge.get("reported_tokens", 0),
            currency=None,
            generation_calls=0,
            query_transform_calls=0,
        ),
        measures=measures,
        contrasts=contrasts,
        caveats=[
            "128 assistant-authored template references; only48test occupation queries scored for relevance",
            "pool Recall/nDCG and stage coverage share the finite top10 union labels, not exhaustive corpus recall",
            "compliance covers source city/type and explicit JD mentions, NOT personal qualification",
            "latency is native GPU serial retrieval, includes DB version checks, excludes API/UI and all LLM generation",
            "natural-language soft city/type stored but no dedicated preference bonus; role keywords still affect retrieval",
        ],
    )
    save(directory / "metrics.json", result)
    (directory / "scored_queries.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in per), encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--legacy-baseline", type=Path)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if args.report_only:
        result = summarize(args.directory)
        print(json.dumps({"labels": result["labels"], "measures": result["measures"]}, ensure_ascii=False))
    else:
        if not all([args.dataset, args.snapshot, args.legacy_baseline]):
            parser.error("运行需指定dataset/snapshot/legacy-baseline")
        print(
            json.dumps(
                run(args.directory, args.dataset, args.snapshot, args.legacy_baseline), ensure_ascii=False
            )
        )
