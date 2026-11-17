"""完整RAG服务编排：检索→证据选择→校验→历史持久化。"""

import hashlib
import json
import logging
import threading
import uuid
from time import perf_counter

import redis

from .explanations import ExplanationService
from .personalized_ranking import PersonalizedRanker
from .generation import Generator
from .evidence import apply_evidence_result
from .retrieval import Retriever
from .settings import ROOT, Settings
from .storage import Store

logger = logging.getLogger("jobcompass")


def cache_client(settings, *, port=6379):
    # 缓存已有业务降级、限流已有503；禁止SDK退避重试扩大故障等待。
    return redis.Redis(
        host=settings.redis_host,
        port=port,
        db=settings.redis_db,
        password=settings.redis_password,
        decode_responses=True,
        socket_connect_timeout=2,
        socket_timeout=2,
        max_connections=20,
        retry=redis.retry.Retry(redis.backoff.NoBackoff(), 0),
    )


class RagService:
    def __init__(self, settings=None):
        self.settings = settings or Settings()
        self.store = Store(self.settings)
        self.generator = Generator(self.settings, model_provider=lambda: self.get_retriever().models)
        self.retriever = None
        self.start_lock = threading.Lock()
        self.limit = threading.BoundedSemaphore(4)
        self.ranker = PersonalizedRanker(self.settings)
        self.explanations = ExplanationService(self)
        self.redis = cache_client(self.settings)

    def get_retriever(self):
        with self.start_lock:
            if self.retriever is None:
                self.retriever = Retriever(self.settings, self.store)
        return self.retriever

    def query(self, request, owner="cli", persist=True, request_id=None):
        if not self.limit.acquire(timeout=2):
            raise RuntimeError("当前查询较多，请稍后重试")
        request_id, started = request_id or str(uuid.uuid4()), perf_counter()
        try:
            retriever = self.get_retriever()
            # 把整个有效语料版本纳入缓存键，新增和下架也会使缓存失效。
            from .preprocessing import digest

            corpus = digest([[j["job_key"], j["content_hash"], j["version"]] for j in self.store.all_jobs()])
            cache_key = (
                self.settings.prefix
                + "recall:"
                + hashlib.sha256(
                    json.dumps(
                        [
                            corpus,
                            self.settings.config,
                            request.model_dump(exclude={"generate"}),
                        ],
                        ensure_ascii=False,
                        sort_keys=True,
                    ).encode()
                ).hexdigest()
            )
            cached, cache_state = None, "miss"
            cache_started = perf_counter()
            try:
                raw = self.redis.get(cache_key)
                if raw:
                    candidate = json.loads(raw)
                    current = self.store.current([x["job"]["job_key"] for x in candidate["results"]])
                    if candidate["results"] and all(
                        current.get(x["job"]["job_key"], {}).get("content_hash") == x["job"]["content_hash"]
                        and current.get(x["job"]["job_key"], {}).get("version") == x["job"]["version"]
                        for x in candidate["results"]
                    ):
                        cached, cache_state = candidate, "hit"
            except (redis.RedisError, ValueError):
                cache_state = "unavailable"
            # Retriever持有可刷新的词法结构和模型；同一实例搜索时串行更新快照。
            if cached is not None:
                retrieval = dict(
                    cached,
                    cached_origin_timings_ms=cached["timings_ms"],
                    timings_ms={"cache_lookup": (perf_counter() - cache_started) * 1000},
                )
                retrieval["results"] = [
                    dict(row, job=current[row["job"]["job_key"]]) for row in retrieval["results"]
                ]
            else:
                with self.start_lock:
                    retrieval = retriever.search(request)
            if cached is None and cache_state != "unavailable" and not retrieval["degradations"]:
                try:
                    self.redis.set(cache_key, json.dumps(retrieval, ensure_ascii=False), ex=120)
                except redis.RedisError:
                    cache_state = "unavailable"
            answer = self.generator.answer(retrieval, request)
            retrieval = apply_evidence_result(retrieval, answer)
            # 生成可能耗时数秒；返回前再次排除期间更新/下架的岗位及其引用。
            latest = self.store.current([r["job"]["job_key"] for r in retrieval["results"]])
            valid = {
                r["job"]["job_key"]
                for r in retrieval["results"]
                if latest.get(r["job"]["job_key"], {}).get("content_hash") == r["job"]["content_hash"]
                and latest.get(r["job"]["job_key"], {}).get("version") == r["job"]["version"]
            }
            retrieval = dict(
                retrieval,
                results=[r for r in retrieval["results"] if r["job"]["job_key"] in valid],
            )
            answer["recommendations"] = [r for r in answer["recommendations"] if r["job_key"] in valid]
            if not valid:
                answer["answer"] = retrieval.get("no_result_reason") or "候选已更新或下架，请重新检索"
            result = dict(
                request_id=request_id,
                request=request.model_dump(),
                **retrieval,
                **answer,
                cache=cache_state,
                total_ms=(perf_counter() - started) * 1000,
                timing_scope="查询进入至检索和证据组装完成；不含历史/文件写入与页面传输；完整HTTP用时见请求日志",
            )
            if persist:
                result["query_id"] = self.store.save_history(owner, result)
                directory = ROOT / "artifacts/runs" / self.settings.stage
                directory.mkdir(parents=True, exist_ok=True)
                (directory / f"{result['query_id']}.json").write_text(
                    json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
            logger.info(
                json.dumps(
                    {
                        "event": "rag_query",
                        "request_id": request_id,
                        "query_id": result.get("query_id"),
                        "results": len(result["results"]),
                        "generation_mode": result["generation"]["mode"],
                        "total_ms": result["total_ms"],
                    }
                )
            )
            return result
        finally:
            self.limit.release()

    def close(self):
        if self.retriever and self.retriever.client:
            self.retriever.client.close()
        self.store.close()
        self.redis.close()
