"""显式运行的真实服务验收：进程崩溃恢复、HTTP并发和持续负载；不调用外部模型。"""

import argparse
import ctypes
import json
import os
import platform
import secrets
import shutil
import socket
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

import httpx
import numpy as np
import redis
from sqlalchemy import delete, func, select, text, update

from .identity import sessions, users
from .preprocessing import digest
from .settings import ROOT, Settings
from .storage import Store, feedback, history, imports, indexes, jobs, versions
from .tasks import TaskQueue, tasks


def memory_bytes(pid):
    """Windows进程工作集，不等于Python堆，也不含GPU显存。其他系统不伪造数值。"""
    if sys.platform != "win32":
        return None
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
            (name, ctypes.c_size_t)
            for name in (
                "peak",
                "working",
                "qpeak",
                "q",
                "pp",
                "p",
                "page",
                "pagepeak",
                "private",
            )
        ]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    handle = kernel.OpenProcess(0x0410, False, pid)
    if not handle:
        return None
    try:
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        ok = psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
        return {"working_set": counters.working, "private_bytes": counters.private} if ok else None
    finally:
        kernel.CloseHandle(handle)


def summarize(rows):
    durations = [r["http_ms"] for r in rows if r["status"] == 200]
    return {
        "requests": len(rows),
        "status_counts": dict(Counter(str(r["status"]) for r in rows)),
        "validation_failures": sum(bool(r.get("error")) for r in rows),
        "cache_counts": dict(Counter(r.get("cache", "none") for r in rows)),
        "p50_http_ms": float(np.percentile(durations, 50)) if durations else None,
        "p95_http_ms": float(np.percentile(durations, 95)) if durations else None,
        "p99_http_ms": float(np.percentile(durations, 99)) if durations else None,
    }


def cleanup(store, owners, before):
    """只删本轮owner及exclusive锁期间新增的派生主键，保留测试库已有资料。"""
    with store.engine.begin() as c:
        keys = list(c.execute(select(jobs.c.job_key).where(jobs.c.owner.in_(owners))).scalars())
        for table in (feedback, history, tasks):
            c.execute(delete(table).where(table.c.owner.in_(owners)))
        c.execute(delete(sessions).where(sessions.c.user_id.in_(owners)))
        c.execute(delete(users).where(users.c.id.in_(owners)))
        if keys:
            c.execute(delete(versions).where(versions.c.job_key.in_(keys)))
        c.execute(delete(jobs).where(jobs.c.owner.in_(owners)))
        for table in (imports, indexes):
            column = list(table.c)[0]
            created = set(c.execute(select(column)).scalars()) - before[table.name]
            if created:
                c.execute(delete(table).where(column.in_(created)))


def validate(minutes=30, interval=3, port=18001):
    if not 1 <= minutes <= 120 or interval < 2:
        raise ValueError("持续1至120分钟，间隔至少2秒以遵守正式限流")
    settings = Settings("test")
    assert settings.db_name == "jobcompass_test" and settings.redis_db == 1
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = "check_" + secrets.token_hex(8)
    directory = ROOT / "artifacts/validation" / stamp
    isolated = directory / "runtime"
    isolated.mkdir(parents=True)
    shutil.copytree(ROOT / "configs", isolated / "configs")
    # 本地测试子进程只接收所需服务凭据，不复制.env或个人画像。
    child_env = {k: v for k, v in os.environ.items() if not k.endswith(("_API_KEY", "_API_TOKEN"))}
    child_env.update({k: v for k, v in settings.values.items() if k.startswith(("MYSQL_", "MILVUS_"))})
    child_env.update(
        JOBCOMPASS_ROOT=str(isolated),
        JOBCOMPASS_ENV="test",
        JOBCOMPASS_RUN_NAMESPACE=run_id,
        REDIS_PASSWORD=settings.redis_password or "",
        PYTHONIOENCODING="utf-8",
        PYTHONPATH=str(ROOT / "src"),
        EMBEDDING_PATH=settings.config["embedding"]["path"],
        RERANKER_PATH=settings.config["reranker"]["path"],
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
    )
    store, api = Store(settings), None
    lock = store.engine.connect()
    lock_name = "jobcompass:test:validation"
    locked = False
    owners = [run_id]
    client_milvus = cache = None
    created_collections, before = set(), {}
    previous_collections = None
    report = {
        "run_id": run_id,
        "started_at": datetime.now(UTC).isoformat(),
        "protocol": "process-http-soak.v2",
        "external_model_calls": 0,
        "environment": "本机Python+真实MySQL/Redis/Milvus测试命名空间",
        "hardware": {
            "os": platform.platform(),
            "python": platform.python_version(),
            "logical_cpus": os.cpu_count(),
        },
        "source_hashes": {p.name: digest(p.read_bytes()) for p in (ROOT / "src/jobcompass").glob("*.py")},
        "scope": "完整HTTP往返，包含认证、检索、证据、历史/文件保存；不含UI和外部生成",
        "requested_minutes": minutes,
        "interval_seconds": interval,
        "limits": [
            "有界持续负载，不是最大容量或长期生产稳定性保证",
            "工作集不含GPU显存",
        ],
    }

    def save():
        (directory / "summary.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def child(code, **kw):
        return subprocess.Popen([sys.executable, "-c", code], cwd=ROOT, env=child_env, **kw)

    if sys.platform == "win32":
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        locked = bool(lock.execute(text("SELECT GET_LOCK(:n, 0)"), {"n": lock_name}).scalar())
        if not locked:
            raise RuntimeError("其他真实测试正在运行，未改动数据")
        tables = (
            jobs,
            versions,
            imports,
            indexes,
            users,
            sessions,
            history,
            feedback,
            tasks,
        )
        with store.engine.connect() as c:
            counts = {t.name: c.execute(select(func.count()).select_from(t)).scalar() for t in tables}
            if any(counts.values()):
                raise RuntimeError("验收需要空测试业务表，不会清空已有数据")
            before = {t.name: set(c.execute(select(list(t.c)[0])).scalars()) for t in (imports, indexes)}
        # 保留已有collection；收尾只清理本轮确定新建的测试collection。
        from .retrieval import milvus

        client_milvus = milvus(settings)
        previous_collections = set(client_milvus.list_collections())
        cache = redis.Redis(host=settings.redis_host, password=settings.redis_password, db=1)
        namespace = settings.prefix + run_id + ":"
        fixture = isolated / "data/imports" / run_id / "jobs.json"
        fixture.parent.mkdir(parents=True)
        fixture.write_text(
            json.dumps(
                [
                    {
                        "source": "fixture",
                        "source_job_id": run_id,
                        "title": "Python恢复验证",
                        "jd_text": "使用Python开发接口\n熟悉MySQL事务",
                        "cities": ["深圳"],
                    }
                ],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        queue = TaskQueue(store)
        task = queue.submit(
            run_id,
            "crash",
            {
                "path": fixture.relative_to(isolated).as_posix(),
                "file_sha256": digest(fixture.read_bytes()),
            },
        )
        # 在SQL已提交、索引尚未建立的真实进程边界退出，不改生产worker逻辑。
        crash_code = """import os
import platform
from jobcompass.worker import work_once
from jobcompass.settings import Settings
def crash(settings, store):
    os._exit(23)
work_once(Settings('test'), index_builder=crash)
"""
        p = child(crash_code, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        assert p.wait(timeout=90) == 23
        crashed = queue.get(run_id, task["id"])
        assert crashed["status"] == "running" and len(store.all_jobs()) == 1
        with store.engine.begin() as c:
            # 加速租约过期，明确记录不是实际等待300秒的实验。
            c.execute(update(tasks).where(tasks.c.id == task["id"]).values(lease_until=0))
        with (directory / "worker.log").open("w", encoding="utf-8") as log:
            p = child(
                "from jobcompass.worker import work_once; work_once()",
                stdout=log,
                stderr=log,
            )
            assert p.wait(timeout=240) == 0
        recovered = queue.get(run_id, task["id"])
        with store.engine.connect() as c:
            version_count = c.execute(select(func.count()).select_from(versions)).scalar()
        assert recovered["status"] == "completed" and recovered["attempts"] == 2
        assert len(store.all_jobs()) == 1 and version_count == 1
        created_collections |= set(client_milvus.list_collections()) - previous_collections
        report["process_recovery"] = {
            "crash_exit": 23,
            "sql_committed_before_exit": True,
            "lease_expiry_accelerated": True,
            "task_status": recovered["status"],
            "attempts": recovered["attempts"],
            "jobs_after_retry": 1,
            "versions_after_retry": version_count,
            "index_jobs": recovered["result"]["index"]["jobs"],
        }
        cleanup(store, owners, before)
        store.upsert_import(ROOT / "data/processed/jobs_v2", owner=run_id)
        with (directory / "build.log").open("w", encoding="utf-8") as log:
            p = child(
                "from jobcompass.worker import rebuild; from jobcompass.settings import Settings; from jobcompass.storage import Store; s=Settings(); st=Store(s); rebuild(s,st); st.close()",
                stdout=log,
                stderr=log,
            )
            assert p.wait(timeout=300) == 0
        created_collections |= set(client_milvus.list_collections()) - previous_collections
        manifest = json.loads((isolated / "artifacts/indexes/test/current.json").read_bytes())
        report["index"] = manifest
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", port))
        with (directory / "api.log").open("w", encoding="utf-8") as log:
            api = child(
                f"import uvicorn; uvicorn.run('jobcompass.web:app',host='127.0.0.1',port={port},log_level='warning')",
                stdout=log,
                stderr=log,
            )
        url = f"http://127.0.0.1:{port}"
        with httpx.Client(base_url=url, timeout=90) as http:
            deadline = time.monotonic() + 60
            while True:
                try:
                    health = http.get("/api/health").json()
                    assert all(v == "ok" for v in health.values())
                    break
                except Exception:
                    if time.monotonic() > deadline or api.poll() is not None:
                        raise RuntimeError("隔离API未启动") from None
                    time.sleep(1)
            credentials = {"username": run_id, "password": secrets.token_urlsafe(24)}
            registered = http.post("/api/auth/register", json=credentials)
            assert registered.status_code == 201
            owners.append(registered.json()["user_id"])
            assert http.post("/api/auth/login", json=credentials).status_code == 200
            queries = [
                "Python 后端接口开发",
                "RAG 知识库检索",
                "Java Spring微服务",
                "React Vue前端",
                "NLP自然语言算法",
                "SQL数据分析",
            ]

            def request_one(n, phase):
                started = time.perf_counter()
                row = {
                    "phase": phase,
                    "n": n,
                    "elapsed_s": time.monotonic() - soak_start,
                }
                try:
                    response = http.post(
                        "/api/search",
                        json={
                            "query": queries[n % len(queries)],
                            "mode": "hybrid",
                            "rerank": True,
                            "generate": False,
                        },
                    )
                    row.update(
                        status=response.status_code,
                        http_ms=(time.perf_counter() - started) * 1000,
                    )
                    if response.status_code == 200:
                        data = response.json()
                        row.update(
                            cache=data["cache"],
                            query_id=data["query_id"],
                            results=len(data["results"]),
                        )
                        by_key = {r["job"]["job_key"]: r["job"] for r in data["results"]}
                        valid = bool(by_key) and not data["degradations"] and data["generation"]["calls"] == 0
                        valid &= response.headers.get("x-request-id") == data["request_id"]
                        for rec in data["recommendations"]:
                            lines = by_key[rec["job_key"]]["jd_text"].splitlines()
                            valid &= all(
                                1 <= c["line"] <= len(lines) and lines[c["line"] - 1] == c["quote"]
                                for c in rec["citations"]
                            )
                        if not valid:
                            row["error"] = "business_or_citation_assertion"
                    return row
                except Exception as exc:
                    return dict(
                        row,
                        status="exception",
                        http_ms=(time.perf_counter() - started) * 1000,
                        error=type(exc).__name__,
                    )

            soak_start = time.monotonic()
            # 冷启动单列，不将模型加载掺入稳态分位数。
            report["cold_request"] = request_one(0, "cold")
            assert report["cold_request"]["status"] == 200 and not report["cold_request"].get("error")
            report["bursts"] = []
            for concurrency in (1, 4, 8):
                with ThreadPoolExecutor(max_workers=concurrency) as pool:
                    batch = list(pool.map(lambda n: request_one(n, "burst"), range(concurrency)))
                report["bursts"].append({"concurrency": concurrency, **summarize(batch), "rows": batch})
            rows, samples = [], []
            soak_start = time.monotonic()
            deadline = soak_start + minutes * 60
            next_tick = soak_start
            with (directory / "requests.jsonl").open("w", encoding="utf-8") as output:
                while time.monotonic() < deadline:
                    if api.poll() is not None:
                        raise RuntimeError("验收API提前退出")
                    row = request_one(len(rows), "soak")
                    rows.append(row)
                    output.write(json.dumps(row, ensure_ascii=False) + "\n")
                    output.flush()
                    if len(rows) % 10 == 1:
                        samples.append(
                            {
                                "elapsed_s": time.monotonic() - soak_start,
                                "memory": memory_bytes(api.pid),
                            }
                        )
                        report.update(
                            soak=summarize(rows),
                            memory_samples=samples,
                            elapsed_seconds=time.monotonic() - soak_start,
                        )
                        save()
                        print(
                            json.dumps(
                                {
                                    "completed_requests": len(rows),
                                    "elapsed_seconds": round(time.monotonic() - soak_start),
                                    "failures": report["soak"]["validation_failures"],
                                }
                            ),
                            flush=True,
                        )
                    next_tick += interval
                    time.sleep(max(0, min(next_tick, deadline) - time.monotonic()))
            report.update(
                soak=summarize(rows),
                memory_samples=samples,
                elapsed_seconds=time.monotonic() - soak_start,
            )
            with store.engine.connect() as c:
                history_count = c.execute(
                    select(func.count()).select_from(history).where(history.c.owner == owners[-1])
                ).scalar()
            successes = sum(r["status"] == 200 for r in rows) + int(report["cold_request"]["status"] == 200)
            successes += sum(b["status_counts"].get("200", 0) for b in report["bursts"])
            report["history_consistency"] = {
                "successful_searches": successes,
                "history_rows": history_count,
                "matches": successes == history_count,
            }
            report["passed"] = (
                report["soak"]["validation_failures"] == 0
                and report["soak"]["status_counts"] == {"200": len(rows)}
                and successes == history_count
                and all(
                    b["validation_failures"] == 0 and b["status_counts"] == {"200": b["requests"]}
                    for b in report["bursts"]
                )
            )
    except Exception as exc:
        report.update(passed=False, error_type=type(exc).__name__)
        raise
    finally:
        if api and api.poll() is None:
            api.terminate()
            api.wait(timeout=30)
        if locked and before:
            cleanup(store, owners, before)
            if client_milvus and previous_collections is not None:
                created_collections |= set(client_milvus.list_collections()) - previous_collections
                for name in created_collections:
                    if name.startswith("jc_chunks_test_"):
                        client_milvus.drop_collection(collection_name=name)
            if cache:
                for key in cache.scan_iter(match=namespace + "*"):
                    cache.delete(key)
            with store.engine.connect() as c:
                report["cleanup_counts"] = {
                    t.name: c.execute(select(func.count()).select_from(t)).scalar() for t in tables
                }
        if report.get("cleanup_counts") and any(report["cleanup_counts"].values()):
            report.update(passed=False, cleanup_residue=True)
        if cache:
            cache.close()
        if client_milvus:
            client_milvus.close()
        if locked:
            lock.execute(text("SELECT RELEASE_LOCK(:n)"), {"n": lock_name})
        lock.close()
        store.close()
        report["ended_at"] = datetime.now(UTC).isoformat()
        if sys.platform == "win32":
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        save()
    return directory / "summary.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minutes", type=int, default=30)
    parser.add_argument("--interval", type=float, default=3)
    parser.add_argument("--port", type=int, default=18001)
    args = parser.parse_args()
    report_path = validate(args.minutes, args.interval, args.port)
    print(report_path)
    sys.exit(0 if json.loads(report_path.read_bytes()).get("passed") else 1)
