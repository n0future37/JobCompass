"""隔离Milvus实验：切块、原查询/真实改写/HyDE对照，保留不完全人标的未知项。"""

import argparse
import json
import platform
from pathlib import Path
import time
from collections import defaultdict
from datetime import UTC, datetime

import numpy as np
from pydantic import ValidationError
from sqlalchemy import text

from .chunking import token_chunks
from .conditions import SearchRequest, assess, parse_conditions
from .evaluation import ranking_metrics
from .experiment_data import make_queries
from .models import LocalModels
from .preprocessing import digest
from .query_transform import transform_queries
from .retrieval import BM25, chunks, milvus, rrf
from .settings import ROOT, Settings
from .storage import Store

CHUNKS = [
    ("line700_o1", None, None),
    ("token256_o32", 256, 32),
    ("token384_o48", 384, 48),
    ("token384_o0", 384, 0),
]


def dense_query_prefix(transform, query_id, kind, hyde_fallback_ids):
    return transform != "hyde" or kind == "control" or query_id in hyde_fallback_ids


def human_metrics(ranking, labels):
    head = ranking[:10]
    unjudged = [key for key in head if key not in labels]
    hits = sum(labels.get(key) == 2 for key in head)
    exact = ranking_metrics(ranking, labels, 10)
    return {
        "unjudged_keys": unjudged,
        "judged_fraction": (len(head) - len(unjudged)) / 10,
        "precision_lower_bound": hits / 10,
        "precision_upper_bound": (hits + len(unjudged)) / 10,
        "precision_at_10": exact["precision"],
        "pooled_ndcg_at_10": exact["ndcg"] if not unjudged else None,
    }


def bootstrap_delta(rows, baseline, variant, kind="anchor", repetitions=2000):
    base = {
        r["query_id"]: r
        for r in rows
        if r["variant"] == baseline and r["kind"] == kind and r["split"] == "test"
    }
    other = {
        r["query_id"]: r
        for r in rows
        if r["variant"] == variant and r["kind"] == kind and r["split"] == "test"
    }
    grouped = defaultdict(list)
    for key in base.keys() & other.keys():
        if "anchor_hit_at_10" in base[key] and "anchor_hit_at_10" in other[key]:
            grouped[base[key]["group"]].append(other[key]["anchor_hit_at_10"] - base[key]["anchor_hit_at_10"])
    if not grouped:
        return {"groups": 0, "delta": None, "interval_95": None}
    values = np.array([np.mean(v) for v in grouped.values()])
    rng = np.random.default_rng(20261103)
    samples = rng.choice(values, (repetitions, len(values)), replace=True).mean(axis=1)
    return {
        "groups": len(values),
        "delta": float(values.mean()),
        "interval_95": [float(x) for x in np.percentile(samples, [2.5, 97.5])],
        "method": "按锚定岗位group成组配对bootstrap，不把两种问法当独立样本",
    }


def load_scenario_queries(directory):
    """只读冻结查询；参考条件不是生产解析器的输出，不能偷偷转成过滤器。"""
    from pathlib import Path

    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_bytes())
    for name, expected in manifest["hashes"].items():
        if digest((directory / name).read_bytes()) != expected:
            raise ValueError("冻结查询文件指纹不符")
    queries = json.loads((directory / "queries.json").read_bytes())
    if len(queries) != manifest["queries"] or len({q["query_id"] for q in queries}) != len(queries):
        raise ValueError("查询数量或ID不合法")
    grouped = defaultdict(set)
    for query in queries:
        if query["kind"] != "assistant_scenario" or query["anchor_job"] is not None:
            raise ValueError("场景集不能冒充已有金标或锚定查询")
        SearchRequest(query=query["query"], **query["request"])
        grouped[query["group"]].add(query["split"])
    if any(len(splits) != 1 for splits in grouped.values()):
        raise ValueError("同一场景组跨开发测试集")
    return queries, manifest


def verify_corpus_snapshot(jobs, snapshot):
    """显式验证新语料快照，保留默认旧人标入口的指纹保护。"""
    manifest = json.loads((snapshot / "manifest.json").read_bytes())
    body = (snapshot / "jobs.jsonl").read_bytes()
    if digest(body) != manifest["jobs_sha256"]:
        raise ValueError("语料快照文件指纹不符")
    frozen = [json.loads(line) for line in body.decode("utf-8").split("\n") if line.strip()]
    if len(frozen) != manifest["jobs"] or len({j["job_key"] for j in frozen}) != len(frozen):
        raise ValueError("语料快照数量或身份重复")

    def identity(rows):
        return sorted((j["job_key"], j["content_hash"], j["version"]) for j in rows)

    if identity(jobs) != identity(frozen):
        raise ValueError("开发库与指定语料快照的岗位/版本不符")
    return dict(
        snapshot=str(snapshot),
        jobs=len(frozen),
        jobs_sha256=digest(body),
        manifest_sha256=digest((snapshot / "manifest.json").read_bytes()),
    )


def run(settings=None, directory=None, query_dataset=None, corpus_snapshot=None, transform_cache=None):
    if corpus_snapshot and not query_dataset:
        raise ValueError("新语料必须显式指定冻结查询集，不能混用旧人标入口")
    settings = settings or Settings("dev")
    directory = directory or ROOT / "artifacts/experiments" / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "summary.json").exists():
        existing = json.loads((directory / "summary.json").read_bytes())
        if existing.get("status") == "complete":
            raise ValueError("已完成实验不可覆盖，请使用新目录")
    if query_dataset:
        scenario_queries, scenario_manifest = load_scenario_queries(query_dataset)
        plan = {
            "protocol": "scenario-retrieval-ablation.v1",
            "query_manifest_sha256": digest((query_dataset / "manifest.json").read_bytes()),
            "queries_sha256": digest((query_dataset / "queries.json").read_bytes()),
            "config": settings.config,
            "chunks": CHUNKS,
            "transform_limits": {"max_calls": 40, "max_reported_tokens": 240000},
            "conditions": "original query parsed by unchanged production parser; no oracle filters",
            "generation": "query transformation only; fictional scenario queries, no account/profile data",
            "splits": "same frozen settings for dev and test; no tuning after inspecting either split",
            "quality": "unjudged candidates stay unknown until independently labeled",
            **(
                {
                    "corpus_snapshot": str(corpus_snapshot),
                    "corpus_manifest_sha256": digest((corpus_snapshot / "manifest.json").read_bytes()),
                }
                if corpus_snapshot
                else {}
            ),
            **(
                {
                    "transform_cache": str(transform_cache),
                    "transform_cache_sha256": digest(transform_cache.read_bytes()),
                }
                if transform_cache
                else {}
            ),
        }
        plan_path = directory / "plan.json"
        if plan_path.exists() and json.loads(plan_path.read_bytes()) != json.loads(json.dumps(plan)):
            raise ValueError("续跑计划指纹不符")
        plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    store, test_store = Store(settings), Store(Settings("test"))
    lock = test_store.engine.connect()
    locked = False
    client, created = None, []
    results = []
    report = {
        "protocol": "chunk-query-ablation.v1",
        "started_at": datetime.now(UTC).isoformat(),
        "directory": str(directory),
        "status": "running",
    }

    def save():
        (directory / "summary.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    try:
        locked = bool(lock.execute(text("SELECT GET_LOCK('jobcompass:test:validation',0)")).scalar())
        if not locked:
            raise RuntimeError("其他真实测试正在运行，不能并行使用测试命名空间")
        jobs = store.all_jobs()
        corpus_hash = digest([[j["job_key"], j["content_hash"]] for j in jobs])
        if corpus_snapshot:
            report["corpus_snapshot"] = verify_corpus_snapshot(jobs, corpus_snapshot)
            report["old_human_labels_used"] = False
        else:
            frozen = json.loads((ROOT / "data/eval/review_human_e09ee0b905fb/source.json").read_bytes())
            if corpus_hash != frozen["corpus_hash"]:
                raise ValueError("开发语料已变化，不能混用旧人标")
        (directory / "corpus.jsonl").write_text(
            "".join(json.dumps(j, ensure_ascii=False) + "\n" for j in jobs), encoding="utf-8"
        )
        if query_dataset:
            queries, protocol = scenario_queries, scenario_manifest
            report["protocol"] = "scenario-retrieval-ablation.v1"
            report["query_dataset"] = str(query_dataset)
            (directory / "queries.json").write_text(
                json.dumps(queries, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        else:
            queries, protocol = make_queries(jobs, ROOT / "data/eval/expanded_probes_v1")
        report.update(
            corpus_jobs=len(jobs),
            corpus_hash=corpus_hash,
            query_protocol=protocol,
            config=settings.config,
            source_hashes={p.name: digest(p.read_bytes()) for p in (ROOT / "src/jobcompass").glob("*.py")},
        )
        save()
        client = milvus(Settings("test"))
        client.list_collections()
        if transform_cache:
            target = directory / transform_cache.name
            if target.exists() and target.read_bytes() != transform_cache.read_bytes():
                raise ValueError("变换缓存文件已存在且指纹不同")
            target.write_bytes(transform_cache.read_bytes())
        transforms = transform_queries(
            settings,
            queries,
            directory,
            **({"max_calls": 40, "max_tokens": 240000} if query_dataset else {}),
        )
        fallback_ids = {
            kind: {
                qid
                for c in transforms["calls"]
                if c["kind"] == kind and c["status"] != "ok"
                for qid in c["query_ids"]
            }
            for kind in ("rewrite", "hyde")
        }
        hyde_fallback_ids = {
            qid
            for c in transforms["calls"]
            if c["kind"] == "hyde" and c["status"] != "ok"
            for qid in c["query_ids"]
        }
        report["transform_calls"] = [] if transform_cache else transforms["calls"]
        report["transform_tokens"] = 0 if transform_cache else transforms["reported_total_tokens"]
        if transform_cache:
            report["transform_cache_reused"] = dict(
                source=str(transform_cache),
                sha256=digest(transform_cache.read_bytes()),
                cached_calls=len(transforms["calls"]),
                historical_tokens=transforms["reported_total_tokens"],
                new_calls=0,
                new_tokens=0,
            )
        report["currency_cost"] = None
        report["transform_fallback_batches"] = sum(c["status"] != "ok" for c in transforms["calls"])
        report["generation_note"] = "生成仅用于query改写/假想JD检索，不是用户答案；实验不调用回答生成模型"
        save()
        model = LocalModels(settings)
        model.encode(["模型预热"])
        import torch

        report["hardware"] = {
            "os": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        }
        by_key = {j["job_key"]: j for j in jobs}
        lexical = BM25(jobs)
        human = defaultdict(dict)
        for line in (
            []
            if corpus_snapshot
            else (
                (ROOT / "data/eval/review_human_e09ee0b905fb/annotations.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            )
        ):
            row = json.loads(line)
            if row["kind"] == "relevance":
                human[row["query_id"]][row["job_key"]] = row["label"]
        collections, index_reports = {}, []
        for name, budget, overlap in CHUNKS:
            started = time.perf_counter()
            passages = (
                chunks(jobs) if budget is None else token_chunks(jobs, model.embed_tokenizer, budget, overlap)
            )
            lengths = []
            for offset in range(0, len(passages), 100):
                lengths.extend(
                    model.embed_tokenizer(
                        [p["text"] for p in passages[offset : offset + 100]],
                        truncation=False,
                        return_length=True,
                    )["length"]
                )
            embedding_start = time.perf_counter()
            vectors = model.encode([p["text"] for p in passages])
            embedding_seconds = time.perf_counter() - embedding_start
            collection = "jc_exp_" + digest([str(directory), name])[:24]
            if client.has_collection(collection_name=collection):
                raise ValueError("本轮集合已存在，拒绝覆盖")
            client.create_collection(
                collection_name=collection,
                dimension=512,
                primary_field_name="id",
                id_type="string",
                max_length=64,
                vector_field_name="vector",
                metric_type="COSINE",
                consistency_level="Strong",
            )
            created.append(collection)
            report["created_collections"] = list(created)
            save()
            for offset in range(0, len(passages), 100):
                client.upsert(
                    collection_name=collection,
                    data=[
                        {**p, "vector": v.tolist()}
                        for p, v in zip(passages[offset : offset + 100], vectors[offset : offset + 100])
                    ],
                )
            client.flush(collection_name=collection)
            assert int(client.get_collection_stats(collection_name=collection)["row_count"]) == len(passages)
            collections[name] = collection
            index_reports.append(
                {
                    "chunk_variant": name,
                    "chunks": len(passages),
                    "token_mean": float(np.mean(lengths)),
                    "token_max": max(lengths),
                    "over_512_count": sum(n > 512 for n in lengths),
                    "embedding_seconds": embedding_seconds,
                    "build_seconds": time.perf_counter() - started,
                }
            )
            report["indexes"] = index_reports
            save()
            print(json.dumps({"indexed": name, "chunks": len(passages)}), flush=True)
        variants = (
            [(name, "original", "hybrid", True) for name, _, _ in CHUNKS]
            + [("line700_o1", kind, "hybrid", True) for kind in ("rewrite", "hyde")]
            + [("line700_o1", "original", mode, False) for mode in ("bm25", "dense")]
        )
        output = (directory / "per_query.jsonl").open("w", encoding="utf-8")
        try:
            for chunk_name, transform, mode, rerank in variants:
                variant = chunk_name + "__" + transform + "__" + mode + ("_rerank" if rerank else "")
                for query in queries:
                    started = time.perf_counter()
                    row = {k: query[k] for k in ("query_id", "kind", "split", "group")}
                    row.update(
                        variant=variant,
                        chunk_variant=chunk_name,
                        transform=transform,
                        mode=mode,
                        effective_transform="original"
                        if query["kind"] == "control"
                        or query["query_id"] in fallback_ids.get(transform, set())
                        else transform,
                    )
                    try:
                        request = SearchRequest(
                            query=query["query"], generate=False, rerank=rerank, top_k=10, **query["request"]
                        )
                        conditions, unresolved = parse_conditions(request)
                    except ValidationError:
                        row.update(
                            ranking=[],
                            validation_error=True,
                            control_pass=query.get("expected") == "validation_error",
                            total_ms=(time.perf_counter() - started) * 1000,
                        )
                        results.append(row)
                        output.write(json.dumps(row) + "\n")
                        continue
                    row.update(conditions=conditions, unresolved_conditions=unresolved)
                    eligible = (
                        {j["job_key"] for j in jobs if assess(j, conditions)[0]} if not unresolved else set()
                    )
                    row["eligible_jobs"] = len(eligible)
                    text_query = (
                        query["query"]
                        if transform == "original" or query["kind"] == "control"
                        else transforms["texts"][transform][query["query_id"]]
                    )
                    sparse_query = text_query if transform == "rewrite" else query["query"]
                    t = time.perf_counter()
                    sparse_scores = lexical.scores(sparse_query)
                    sparse = [
                        j["job_key"]
                        for n, j in sorted(
                            enumerate(jobs), key=lambda pair: (-sparse_scores[pair[0]], pair[1]["job_key"])
                        )
                        if j["job_key"] in eligible and sparse_scores[n] > 0
                    ][:40]
                    phase = {"bm25_ms": (time.perf_counter() - t) * 1000}
                    dense, dense_scores = [], {}
                    if eligible and mode != "bm25":
                        t = time.perf_counter()
                        vector = model.encode(
                            [text_query],
                            query=dense_query_prefix(
                                transform, query["query_id"], query["kind"], hyde_fallback_ids
                            ),
                        )[0].tolist()
                        hits = client.search(
                            collection_name=collections[chunk_name],
                            data=[vector],
                            filter="job_key in " + json.dumps(sorted(eligible)),
                            limit=min(200, len(eligible) * 3),
                            output_fields=["job_key"],
                            search_params={"metric_type": "COSINE"},
                            consistency_level="Strong",
                        )[0]
                        for hit in hits:
                            key = hit["entity"]["job_key"]
                            dense_scores[key] = max(dense_scores.get(key, -1), float(hit["distance"]))
                        dense = sorted(dense_scores, key=lambda key: (-dense_scores[key], key))[:40]
                        phase["dense_ms"] = (time.perf_counter() - t) * 1000
                    ordered = (
                        sparse
                        if mode == "bm25" or not dense
                        else dense
                        if mode == "dense"
                        else rrf([sparse, dense], 60)[0]
                    )
                    if not sparse and (not dense or max(dense_scores.values(), default=0) < 0.55):
                        ordered = []
                    row["retrieval_degradations"] = (
                        ["dense_empty_bm25_fallback"] if mode == "dense" and not dense and sparse else []
                    )
                    before_rerank = list(ordered)
                    if rerank and ordered:
                        t = time.perf_counter()
                        head = ordered[:15]
                        scores = model.rerank(query["query"], [by_key[key] for key in head])
                        ordered = [
                            key for _, key in sorted(zip(scores, head), key=lambda p: (-p[0], p[1]))
                        ] + ordered[15:]
                        phase["rerank_ms"] = (time.perf_counter() - t) * 1000
                    ranked = ordered[:10]
                    row.update(
                        ranking=ranked,
                        recall_candidates=before_rerank,
                        stage_ms=phase,
                        total_ms=(time.perf_counter() - started) * 1000,
                        hard_violations=sum(not assess(by_key[k], conditions)[0] for k in ranked),
                    )
                    if query["kind"] in {"human", "assistant_scenario"}:
                        row.update(human_metrics(ranked, human[query["query_id"]]))
                    elif query["kind"] == "anchor":
                        positions = [n for n, k in enumerate(ranked, 1) if k in query["expected_keys"]]
                        row.update(
                            anchor_hit_at_10=int(bool(positions)),
                            anchor_mrr_at_10=1 / min(positions) if positions else 0,
                            expected_keys=query["expected_keys"],
                        )
                    else:
                        expected = query["expected"]
                        row["control_pass"] = (
                            (bool(unresolved) and not ranked)
                            if expected == "unresolved"
                            else not ranked
                            if expected == "empty"
                            else bool(ranked) and row["hard_violations"] == 0
                            if expected == "nonempty_valid"
                            else row["hard_violations"] == 0
                            if expected == "valid"
                            else False
                        )
                        row["unresolved"] = unresolved
                    results.append(row)
                    output.write(json.dumps(row, ensure_ascii=False) + "\n")
                    output.flush()
                print(json.dumps({"completed_variant": variant, "queries": len(queries)}), flush=True)
        finally:
            output.close()
        measures = []
        for variant in sorted({r["variant"] for r in results}):
            for split in ("dev", "test"):
                human_rows = [
                    r
                    for r in results
                    if r["variant"] == variant
                    and r["split"] == split
                    and r["kind"] in {"human", "assistant_scenario"}
                ]
                anchor_rows = [
                    r
                    for r in results
                    if r["variant"] == variant and r["split"] == split and r["kind"] == "anchor"
                ]
                valid = [r for r in human_rows if r["precision_at_10"] is not None]
                latencies = [
                    r["total_ms"] for r in results if r["variant"] == variant and r["kind"] != "control"
                ][1:]
                measures.append(
                    {
                        "variant": variant,
                        "split": split,
                        "human_queries": len(human_rows),
                        "fully_judged_queries": len(valid),
                        "human_judged_fraction": float(np.mean([r["judged_fraction"] for r in human_rows]))
                        if human_rows
                        else None,
                        "human_p10_lower_bound": float(
                            np.mean([r["precision_lower_bound"] for r in human_rows])
                        )
                        if human_rows
                        else None,
                        "human_p10_upper_bound": float(
                            np.mean([r["precision_upper_bound"] for r in human_rows])
                        )
                        if human_rows
                        else None,
                        "human_p10_on_complete_queries": float(np.mean([r["precision_at_10"] for r in valid]))
                        if valid
                        else None,
                        "anchor_queries": len(anchor_rows),
                        "anchor_groups": len({r["group"] for r in anchor_rows}),
                        "anchor_hit_at_10": float(np.mean([r["anchor_hit_at_10"] for r in anchor_rows]))
                        if anchor_rows
                        else None,
                        "anchor_mrr_at_10": float(np.mean([r["anchor_mrr_at_10"] for r in anchor_rows]))
                        if anchor_rows
                        else None,
                        "warm_p50_ms": float(np.percentile(latencies, 50)),
                        "warm_p95_ms": float(np.percentile(latencies, 95)),
                    }
                )
        missing = defaultdict(set)
        for row in results:
            missing[row["query_id"]].update(row.get("unjudged_keys", []))
        (directory / "human_review_pending.jsonl").write_text(
            "".join(
                json.dumps(
                    {
                        "query_id": qid,
                        "query": next(q["query"] for q in queries if q["query_id"] == qid),
                        "job_key": key,
                        "title": by_key[key]["title"],
                        "jd_text": by_key[key]["jd_text"],
                        "label": None,
                        "annotation_kind": "",
                        "reason": "",
                    },
                    ensure_ascii=False,
                )
                + "\n"
                for qid in sorted(missing)
                for key in sorted(missing[qid])
            ),
            encoding="utf-8",
        )
        baseline = "line700_o1__original__hybrid_rerank"
        report.update(
            status="complete",
            measures=measures,
            control_results=[
                {
                    "variant": v,
                    "passed": sum(
                        r.get("control_pass", False)
                        for r in results
                        if r["variant"] == v and r["kind"] == "control"
                    ),
                    "total": sum(q["kind"] == "control" for q in queries),
                }
                for v in sorted({r["variant"] for r in results})
            ],
            anchor_paired_bootstrap=[
                {"variant": v, **bootstrap_delta(results, baseline, v)}
                for v in sorted({r["variant"] for r in results})
                if v != baseline
            ],
            pending_human_pairs=sum(len(v) for v in missing.values()),
            limits=[
                "人工P上下界只表示未标注候选带来的范围，不是置信区间；完整标注查询子集随方法变化不能直接比较子集均值",
                "48锚定样例为原文构造、仅24岗位组且方向/模板共享，不作为48独立真实用户查询",
                "同一模型/语料/原始rerank查询/40候选预算和15重排预算；HyDE只替换向量查询，词法仍用原查询",
                "分块对照固定原查询；改写/HyDE固定原700字符索引，未挑测试集最优组合",
                "耗时为本机GPU检索，不含线上API/UI/用户答案生成；模型加载首项不代表稳定延迟",
            ],
        )
        if query_dataset:
            report["scenario_coverage"] = [
                {
                    "variant": variant,
                    "split": split,
                    "queries": len(selected),
                    "parser_blocked": sum(bool(r.get("unresolved_conditions")) for r in selected),
                    "nonempty": sum(bool(r["ranking"]) for r in selected),
                    "empty_after_supported_filters": sum(
                        not r["ranking"] and not r.get("unresolved_conditions") for r in selected
                    ),
                    "returned_positions": sum(len(r["ranking"]) for r in selected),
                    "warm_p50_ms": float(np.percentile([r["total_ms"] for r in selected][1:], 50)),
                    "warm_p95_ms": float(np.percentile([r["total_ms"] for r in selected][1:], 95)),
                }
                for variant in sorted({r["variant"] for r in results})
                for split in ("dev", "test")
                if (selected := [r for r in results if r["variant"] == variant and r["split"] == split])
            ]
            report["limits"] = [
                "120助手构造场景不是实际用户日志；108名义组不保证统计独立",
                "未支持条件导致的空结果单列；现有解析器未识别的条件也可能被忽略，不能把hard_violations=0当完整条件正确率",
                "未标注候选保持未知，质量指标需另建来源明确的标签版本",
                "检索耗时不含批量查询变换、回答生成、API/UI；变换失败回退原查询",
                "固定同一语料/模型/40每路/15重排/RRF60，所有方案统一运行，不按测试调参",
            ]
        if corpus_hash != digest([[j["job_key"], j["content_hash"]] for j in store.all_jobs()]):
            raise ValueError("运行期间开发语料变化，结果不可比")
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        if client:
            for name in created:
                client.drop_collection(collection_name=name)
            report["temporary_collections_removed"] = list(created)
            client.close()
        if locked:
            lock.execute(text("SELECT RELEASE_LOCK('jobcompass:test:validation')"))
        lock.close()
        test_store.close()
        store.close()
        report["ended_at"] = datetime.now(UTC).isoformat()
        save()
    return directory / "summary.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=str)
    parser.add_argument("--queries", type=str, help="冻结场景查询目录；不指定时使用原72查询")
    parser.add_argument("--snapshot", type=Path, help="显式新语料快照，核对开发库且不使用旧人标")
    parser.add_argument("--transform-cache", type=Path, help="复用同查询/模型签名的冻结变换，新增调用0")
    args = parser.parse_args()

    print(
        run(
            directory=Path(args.directory) if args.directory else None,
            query_dataset=Path(args.queries) if args.queries else None,
            corpus_snapshot=args.snapshot,
            transform_cache=args.transform_cache,
        )
    )
