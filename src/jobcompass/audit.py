"""可复现的近重复候选检查与评测拆分检查；候选不自动删除。"""

import hashlib
import json
import re
from collections import defaultdict
from time import perf_counter

import numpy as np

from .evaluation import QUERIES
from .preprocessing import digest
from .settings import ROOT


def near_duplicates(jobs, threshold=0.85):
    shingles = []
    buckets = defaultdict(list)
    pairs = set()
    exact = defaultdict(list)
    for i, job in enumerate(jobs):
        text = re.sub(r"[\W_]+", "", job["jd_text"].casefold())
        grams = {text[n : n + 5] for n in range(max(1, len(text) - 4))}
        shingles.append(grams)
        exact[digest(text)].append(job["job_key"])
        hashes = np.array(
            [
                int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "big")
                for s in sorted(grams)
            ],
            dtype=np.uint64,
        )
        votes = ((hashes[:, None] >> np.arange(64, dtype=np.uint64)) & 1).sum(axis=0)
        fingerprint = sum(1 << bit for bit, count in enumerate(votes) if count >= len(hashes) / 2)
        for band in range(8):
            key = band, (fingerprint >> (band * 8)) & 255
            pairs.update((prior, i) for prior in buckets[key])
            buckets[key].append(i)
    candidates = []
    for a, b in sorted(pairs):
        x, y = shingles[a], shingles[b]
        if min(len(x), len(y)) / max(1, max(len(x), len(y))) < threshold:
            continue
        jaccard = len(x & y) / max(1, len(x | y))
        if jaccard >= threshold:
            candidates.append(
                dict(job_keys=[jobs[a]["job_key"], jobs[b]["job_key"]], jaccard_5gram=round(jaccard, 6))
            )
    return dict(
        exact_jd_groups=[v for v in exact.values() if len(v) > 1],
        near_candidates=candidates,
        algorithm="simhash64,8x8bit候选;中文/英文归一化字符5gram Jaccard>=0.85复核",
        limits="仅近重复候选检查，LSH可能漏检；跨公司模板相似不等于同一岗位；不自动合并",
    )


def audit(snapshot=None):
    snapshot = snapshot or ROOT / "data/processed/jobs_v2"
    jobs = [json.loads(s) for s in (snapshot / "jobs.jsonl").read_text(encoding="utf-8").split("\n") if s]
    started = perf_counter()
    report = near_duplicates(jobs)
    dev = {re.sub(r"\s+", "", q.casefold()) for _, s, q, _ in QUERIES if s == "dev"}
    test = {re.sub(r"\s+", "", q.casefold()) for _, s, q, _ in QUERIES if s == "test"}
    report.update(
        jobs=len(jobs),
        corpus_hash=digest([[j["job_key"], j["content_hash"]] for j in jobs]),
        identical_queries_across_splits=sorted(dev & test),
        split_policy="按查询划分开发/固定测试，共享冻结检索语料；未训练模型，代理标签不能证明语义独立性",
        label_policy="规则代理标签，不是人工金标；相同/近似JD仍按来源岗位ID计数，指标可能受到模板相似影响",
        seconds=perf_counter() - started,
    )
    target = ROOT / "artifacts/data_audit" / (snapshot.name + ".json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return dict(
        report=str(target),
        exact_jd_groups=len(report["exact_jd_groups"]),
        near_candidates=len(report["near_candidates"]),
        query_overlap=len(dev & test),
        seconds=report["seconds"],
    )
