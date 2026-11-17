"""真实SDK连接失败验收：仅将本进程客户端指向关闭的本机端口，不停共享服务。"""

import json
import socket
from datetime import UTC, datetime
from time import perf_counter

from fastapi.testclient import TestClient

from .conditions import SearchRequest
from .retrieval import Retriever
from .service import RagService, cache_client
from .settings import ROOT, Settings


def probe():
    from . import web

    report = {
        "protocol": "closed-port-sdk.v1",
        "external_model_calls": 0,
        "shared_services_stopped": False,
        "business_writes": 0,
    }
    request = SearchRequest(query="Python 后端", mode="hybrid", rerank=False, generate=False)
    # 绑定但不监听，确保不会误连其他进程；MySQL只读取既有dev语料。
    with socket.socket() as closed:
        closed.bind(("127.0.0.1", 0))
        port = closed.getsockname()[1]
        settings = Settings("dev")
        settings.milvus_uri = f"http://127.0.0.1:{port}"
        retriever = Retriever(settings)
        start = perf_counter()
        try:
            result = retriever.search(request)
            report["milvus_unavailable"] = {
                "elapsed_seconds": perf_counter() - start,
                "results": len(result["results"]),
                "degradations": result["degradations"],
            }
            assert result["results"] and any(d["stage"] == "dense" for d in result["degradations"])
        finally:
            if retriever.client:
                retriever.client.close()
            retriever.store.close()
        svc = RagService(Settings("dev"))
        svc.redis.close()
        svc.settings.redis_host = "127.0.0.1"
        svc.redis = cache_client(svc.settings, port=port)
        start = perf_counter()
        try:
            fallback = svc.query(
                SearchRequest(query="Python", mode="bm25", rerank=False, generate=False), persist=False
            )
            assert fallback["cache"] == "unavailable" and fallback["results"]
            report["redis_cache_unavailable"] = {
                "cache": fallback["cache"],
                "results": len(fallback["results"]),
                "elapsed_seconds": perf_counter() - start,
            }
            web.app.dependency_overrides[web.service] = lambda: svc
            # 无需创建用户，只验证限流基础服务不可用时的API明示503。
            client = TestClient(web.app)
            limiter_start = perf_counter()
            try:
                response = client.post(
                    "/api/auth/register",
                    json={"username": "fault_probe_unused", "password": "fixed-unsubmitted-probe"},
                )
                assert response.status_code == 503 and response.headers.get("x-request-id")
                report["redis_limiter_unavailable"] = {
                    "http_status": response.status_code,
                    "request_id_present": True,
                    "elapsed_seconds": perf_counter() - limiter_start,
                }
            finally:
                client.close()
        finally:
            web.app.dependency_overrides.clear()
            svc.close()
    report["passed"] = True
    directory = ROOT / "artifacts/faults" / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    directory.mkdir(parents=True)
    (directory / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"report": str(directory / "summary.json"), **report}


if __name__ == "__main__":
    print(json.dumps(probe(), ensure_ascii=False, indent=2))
