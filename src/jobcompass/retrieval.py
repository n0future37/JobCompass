"""BM25、Milvus向量基线、RRF、岗位聚合及本地重排。"""

import json
import math
import re
import threading
from collections import Counter, OrderedDict, defaultdict
from copy import deepcopy
from time import perf_counter

import numpy as np

from .conditions import assess, profile_mentions, mention_lines
from .intent import parse_intent
from .models import LocalModels
from .preprocessing import digest
from .settings import ROOT, Settings
from .storage import Store, corpus_fingerprint


def tokenize(text):
    text = text.casefold()
    terms = re.findall(r"[a-z][a-z0-9+#.]*", text)
    for sequence in re.findall(r"[\u4e00-\u9fff]+", text):
        terms.extend(sequence[i : i + 2] for i in range(len(sequence) - 1))
        if len(sequence) == 1:
            terms.append(sequence)
    return terms


class BM25:
    def __init__(self, jobs):
        self.counts = [Counter(tokenize(j["title"] + "\n" + j["title"] + "\n" + j["jd_text"])) for j in jobs]
        self.lengths = np.array([sum(c.values()) for c in self.counts])
        self.average = float(np.mean(self.lengths)) if len(jobs) else 1.0
        self.document_frequency = Counter(t for c in self.counts for t in c)
        postings = defaultdict(lambda: ([], []))
        for index, counts in enumerate(self.counts):
            for term, frequency in counts.items():
                postings[term][0].append(index)
                postings[term][1].append(frequency)
        self.postings = {
            term: (np.asarray(indices), np.asarray(frequencies, dtype=np.float64))
            for term, (indices, frequencies) in postings.items()
        }

    def scores(self, query):
        scores = np.zeros(len(self.counts), dtype=np.float64)
        size = len(self.counts)
        for term in set(tokenize(query)):
            if term not in self.postings:
                continue
            df = self.document_frequency[term]
            idf = math.log(1 + (size - df + 0.5) / (df + 0.5))
            indices, tf = self.postings[term]
            scores[indices] += (
                idf * tf * 2.5 / (tf + 1.5 * (0.25 + 0.75 * self.lengths[indices] / self.average))
            )
        return scores


def rrf(rankings, k=60):
    scores = defaultdict(float)
    for ranking in rankings:
        for rank, key in enumerate(dict.fromkeys(ranking), 1):
            scores[key] += 1 / (k + rank)
    return sorted(scores, key=lambda key: (-scores[key], key)), dict(scores)


def chunks(jobs, limit=700):
    result = []
    for job in jobs:
        lines = job["jd_text"].splitlines()
        start = 0
        while start < len(lines):
            end, count = start, 0
            while end < len(lines) and (count + len(lines[end]) <= limit or end == start):
                count += len(lines[end]) + 1
                end += 1
            text = "\n".join(lines[start:end])
            result.append(
                dict(
                    id=digest([job["job_key"], job["content_hash"], start, end])[:32],
                    job_key=job["job_key"],
                    version=job["version"],
                    content_hash=job["content_hash"],
                    line_start=start + 1,
                    line_end=end,
                    text=job["title"] + "\n" + text,
                )
            )
            start = end if end == len(lines) or end == start + 1 else end - 1
    return result


def milvus(settings):
    from pymilvus import MilvusClient

    return MilvusClient(
        uri=settings.milvus_uri,
        user=settings.db_user,
        password=settings.milvus_password,
        db_name=settings.db_name,
        timeout=20,
    )


class IndexValidationError(RuntimeError):
    """派生索引未通过校验，允许worker重试；不是用户导入文件错误。"""


def validate_index(client, collection, passages):
    # 统计row_count可能包括upsert后尚未回收的旧记录，不能作为有效记录数。
    response = client.query(
        collection_name=collection,
        filter="",
        output_fields=["count(*)"],
        consistency_level="Strong",
        timeout=20,
    )
    count = response[0].get("count(*)") if len(response) == 1 else None
    if type(count) is not int or count != len(passages):
        raise IndexValidationError(f"索引有效计数不符：{count}/{len(passages)}")
    fields = ("id", "job_key", "content_hash", "version")
    for start in range(0, len(passages), 100):
        batch = passages[start : start + 100]
        rows = client.query(
            collection_name=collection,
            filter="id in " + json.dumps([row["id"] for row in batch]),
            output_fields=list(fields),
            consistency_level="Strong",
            timeout=20,
        )
        expected = {row["id"]: tuple(row[field] for field in fields) for row in batch}
        actual = {row.get("id"): tuple(row.get(field) for field in fields) for row in rows}
        if len(rows) != len(batch) or actual != expected:
            raise IndexValidationError("索引文本块身份、版本或内容指纹不符")
    return count


def build_index(settings=None):
    settings = settings or Settings()
    store = Store(settings)
    jobs = store.all_jobs()
    snapshot = digest([[j["job_key"], j["content_hash"], j["version"]] for j in jobs])
    embedding_spec = {k: v for k, v in settings.config["embedding"].items() if k != "path"}
    index_version = digest([snapshot, embedding_spec, settings.config["retrieval"]])
    collection = f"jc_chunks_{settings.stage}_{index_version[:16]}"
    if not jobs:
        manifest = dict(
            snapshot=snapshot,
            collection=None,
            jobs=0,
            chunks=0,
            index_version=index_version,
        )
        directory = ROOT / "artifacts/indexes" / settings.stage
        directory.mkdir(parents=True, exist_ok=True)
        temporary = directory / "current.json.tmp"
        temporary.write_text(json.dumps(manifest), encoding="utf-8")
        temporary.replace(directory / "current.json")
        store.index_state(snapshot, "ready", manifest)
        store.close()
        return manifest
    model = LocalModels(settings)
    passages = chunks(jobs, settings.config["retrieval"]["chunk_chars"])
    start = perf_counter()
    store.index_state(snapshot, "building", {"collection": collection})
    client = None
    try:
        vectors = model.encode([c["text"] for c in passages])
        client = milvus(settings)
        if not client.has_collection(collection_name=collection):
            client.create_collection(
                collection_name=collection,
                dimension=vectors.shape[1],
                primary_field_name="id",
                id_type="string",
                max_length=64,
                vector_field_name="vector",
                metric_type="COSINE",
                consistency_level="Strong",
            )
        for start_row in range(0, len(passages), 100):
            data = [
                {**c, "vector": v.tolist()}
                for c, v in zip(
                    passages[start_row : start_row + 100],
                    vectors[start_row : start_row + 100],
                )
            ]
            client.upsert(collection_name=collection, data=data)
        client.flush(collection_name=collection)
        validate_index(client, collection, passages)
        manifest = dict(
            snapshot=snapshot,
            index_version=index_version,
            collection=collection,
            jobs=len(jobs),
            chunks=len(passages),
            dimension=int(vectors.shape[1]),
            embedding=settings.config["embedding"],
            retrieval=settings.config["retrieval"],
            seconds=perf_counter() - start,
        )
        directory = ROOT / "artifacts/indexes" / settings.stage
        directory.mkdir(parents=True, exist_ok=True)
        temporary = directory / "current.json.tmp"
        temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(directory / "current.json")
        store.index_state(snapshot, "ready", manifest)
        return manifest
    except Exception as error:
        store.index_state(
            snapshot,
            "failed",
            {"collection": collection, "error_type": type(error).__name__},
        )
        raise
    finally:
        if client:
            client.close()
        store.close()


class Retriever:
    def __init__(self, settings=None, store=None, models=None, *, index_manifest=None):
        self.settings = settings or Settings()
        self.store = store or Store(self.settings)
        self.models = models or LocalModels(self.settings)
        self.jobs = self.store.all_jobs()
        self.by_key = {j["job_key"]: j for j in self.jobs}
        self.lexical = BM25(self.jobs)
        self.client = None
        self.index_manifest = index_manifest
        self.snapshot_lock = threading.Lock()
        self.client_lock = threading.Lock()
        self.filter_lock = threading.Lock()
        self.filter_cache = OrderedDict()
        self.revision = digest([[j["job_key"], j["content_hash"], j["version"]] for j in self.jobs])

    def snapshot(self):
        revision = corpus_fingerprint(self.store)
        with self.snapshot_lock:
            if revision != self.revision:
                latest = self.store.all_jobs()
                lexical = BM25(latest)
                self.jobs = latest
                self.by_key = {j["job_key"]: j for j in latest}
                self.lexical = lexical
                self.revision = digest([[j["job_key"], j["content_hash"], j["version"]] for j in latest])
            # 查询持有自己的快照引用；后续刷新不会改变正在执行的查询。
            return self.revision, self.jobs, self.by_key, self.lexical

    def eligibility(self, revision, by_key, conditions):
        key = digest([revision, conditions])
        with self.filter_lock:
            if key not in self.filter_cache:
                eligible, checks = [], {}
                for job_key, job in by_key.items():
                    passed, row_checks = assess(job, conditions)
                    if passed:
                        eligible.append(job_key)
                        checks[job_key] = row_checks
                self.filter_cache[key] = (eligible, checks)
                if len(self.filter_cache) > 32:
                    self.filter_cache.popitem(last=False)
            self.filter_cache.move_to_end(key)
            return self.filter_cache[key]

    def search(self, request, *, include_trace=False, eligible_keys=None, query_plan=None):
        start = perf_counter()
        # MySQL是权威来源；新增、更新和下架后不继续使用旧词法快照。
        revision, current_jobs, current, lexical = self.snapshot()
        refresh_ms = (perf_counter() - start) * 1000
        intent = parse_intent(request)
        conditions, unresolved = intent.conditions(), intent.unresolved
        timings, degraded = {"corpus_snapshot": refresh_ms}, []
        if unresolved:
            return dict(
                results=[],
                conditions=conditions,
                intent=intent.public(),
                unresolved_conditions=unresolved,
                degradations=[],
                timings_ms={},
                no_result_reason="存在未支持的明确硬条件，请使用可核验字段",
                corpus_revision=revision,
            )
        t = perf_counter()
        eligible, checks = self.eligibility(revision, current, conditions)
        if eligible_keys is not None:
            permitted = set(eligible_keys)
            eligible = [key for key in eligible if key in permitted]
        timings["eligibility"] = (perf_counter() - t) * 1000
        if not eligible:
            return dict(
                results=[],
                conditions=conditions,
                intent=intent.public(),
                unresolved_conditions=[],
                degradations=[],
                timings_ms={},
                no_result_reason="没有可确认满足全部硬条件的岗位；未知字段不算满足",
                corpus_revision=revision,
            )
        allowed = set(eligible)
        t = perf_counter()
        lexical_scores = lexical.scores(query_plan["lexical_query"] if query_plan else request.query)
        sparse = [
            j["job_key"]
            for i, j in sorted(
                enumerate(current_jobs),
                key=lambda p: (-lexical_scores[p[0]], p[1]["job_key"]),
            )
            if j["job_key"] in allowed and lexical_scores[i] > 0
        ][: self.settings.config["retrieval"]["recall_k"]]
        timings["bm25"] = (perf_counter() - t) * 1000
        dense, dense_scores = [], {}
        if request.mode in {"dense", "hybrid"}:
            t = perf_counter()
            try:
                manifest = self.index_manifest or json.loads(
                    (ROOT / "artifacts/indexes" / self.settings.stage / "current.json").read_bytes()
                )
                if manifest["snapshot"] != revision:
                    raise RuntimeError("当前向量索引尚未同步，使用最新词法检索")
                if {k: v for k, v in manifest["embedding"].items() if k != "path"} != {
                    k: v for k, v in self.settings.config["embedding"].items() if k != "path"
                } or manifest["retrieval"] != self.settings.config["retrieval"]:
                    raise RuntimeError("索引参数已改变，需重建")
                with self.client_lock:
                    if self.client is None:
                        self.client = milvus(self.settings)
                vector = self.models.encode(
                    [query_plan["dense_query"] if query_plan else request.query],
                    query=query_plan["dense_is_query"] if query_plan else True,
                )[0].tolist()
                hits = self.client.search(
                    collection_name=manifest["collection"],
                    data=[vector],
                    filter="job_key in " + json.dumps(eligible),
                    limit=min(200, len(eligible) * 3),
                    output_fields=["job_key", "content_hash", "version"],
                    search_params={"metric_type": "COSINE"},
                    consistency_level="Strong",
                )[0]
                for hit in hits:
                    entity = hit["entity"]
                    key = entity["job_key"]
                    if (
                        key in current
                        and entity["content_hash"] == current[key]["content_hash"]
                        and entity["version"] == current[key]["version"]
                    ):
                        dense_scores[key] = max(dense_scores.get(key, -1), float(hit["distance"]))
                dense = sorted(dense_scores, key=lambda key: (-dense_scores[key], key))[
                    : self.settings.config["retrieval"]["recall_k"]
                ]
            except Exception as error:
                degraded.append(
                    {
                        "stage": "dense",
                        "fallback": "bm25",
                        "error_type": type(error).__name__,
                    }
                )
            timings["dense"] = (perf_counter() - t) * 1000
        if request.mode == "bm25" or not dense:
            ordered = sparse
        elif request.mode == "dense":
            ordered = dense
        else:
            ordered, _ = rrf([sparse, dense], self.settings.config["retrieval"]["rrf_k"])
        # 防止任意无关查询也被向量强制填满；可见地返回证据不足。
        if not sparse and (not dense or max(dense_scores.values(), default=0) < 0.55):
            ordered = []
        recall_order = list(ordered) if include_trace else None
        if request.rerank and ordered:
            t = perf_counter()
            head = ordered[: self.settings.config["retrieval"]["rerank_k"]]
            try:
                scores = self.models.rerank(request.query, [current[k] for k in head])
                ordered = [k for _, k in sorted(zip(scores, head), key=lambda p: (-p[0], p[1]))] + ordered[
                    len(head) :
                ]
            except Exception as error:
                degraded.append(
                    {
                        "stage": "reranker",
                        "fallback": "recall_order",
                        "error_type": type(error).__name__,
                    }
                )
            timings["reranker"] = (perf_counter() - t) * 1000
        results = []
        # 画像是软偏好，只在已满足本轮硬条件的召回集合内给小幅加分。
        preferred = profile_mentions(request.profile)
        if preferred:
            original_rank = {key: n for n, key in enumerate(ordered, 1)}
            ordered = sorted(
                ordered,
                key=lambda key: (
                    -(
                        1 / original_rank[key]
                        + 0.05
                        * sum(bool(mention_lines(current[key], term)) for term in preferred)
                        / len(preferred)
                    )
                ),
            )
        latest_rows = self.store.current(ordered[: request.top_k])
        for rank, key in enumerate(ordered[: request.top_k], 1):
            job = current[key]
            # 再核验一次，避免检索过程中岗位下架或版本变化。
            latest = latest_rows.get(key)
            if (
                not latest
                or latest["content_hash"] != job["content_hash"]
                or latest["version"] != job["version"]
            ):
                continue
            results.append(
                dict(
                    rank=rank,
                    job=deepcopy(job),
                    hard_condition_checks=deepcopy(checks[key]),
                    dense_score=dense_scores.get(key),
                )
            )
        timings["retrieval_total"] = (perf_counter() - start) * 1000
        result = dict(
            results=results,
            conditions=conditions,
            intent=intent.public(),
            unresolved_conditions=[],
            degradations=degraded,
            timings_ms=timings,
            no_result_reason=None if results else "本次检索没有找到足够相关的岗位证据",
            soft_preferences=preferred,
            corpus_revision=revision,
        )

        if include_trace:
            result["evaluation_trace"] = {
                "bm25": sparse,
                "dense": dense,
                "before_rerank": recall_order,
                "after_preferences": list(ordered),
                "recall_k": self.settings.config["retrieval"]["recall_k"],
                "rerank_k": self.settings.config["retrieval"]["rerank_k"],
            }
        return result
