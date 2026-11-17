"""独立worker：读取持久化任务、导入MySQL、恢复派生索引。"""

import json
import logging
import threading
import time

from sqlalchemy import select, text

from .importing import make_snapshot
from .preprocessing import digest
from .retrieval import build_index
from .settings import ROOT, Settings
from .storage import Store, indexes
from .tasks import TaskQueue

logger = logging.getLogger("jobcompass.worker")


def rebuild(settings, store):
    # 同一项目环境的所有worker/CLI共用数据库命名锁，不互相覆盖索引指针。
    with store.engine.connect() as lock:
        name = f"jobcompass:{settings.stage}:index"
        if not lock.execute(text("SELECT GET_LOCK(:name, 0)"), {"name": name}).scalar():
            raise RuntimeError("索引正在由另一worker更新")
        try:
            return build_index(settings)
        finally:
            lock.execute(text("SELECT RELEASE_LOCK(:name)"), {"name": name})


def work_once(settings=None, index_builder=rebuild):
    settings = settings or Settings()
    store = Store(settings)
    queue = TaskQueue(store)
    try:
        task = queue.claim()
        if not task:
            # 崩溃发生在业务提交后，仍能由当前权威语料计算待同步索引。
            jobs = store.all_jobs()
            snapshot = digest([[j["job_key"], j["content_hash"], j["version"]] for j in jobs])
            with store.engine.connect() as c:
                state = c.execute(select(indexes).where(indexes.c.snapshot == snapshot)).mappings().first()
            if not state or state["status"] != "ready":
                payload = state["payload"] if state else {}
                if payload.get("attempts", 0) < 3 and payload.get("retry_at", 0) <= time.time():
                    try:
                        index_builder(settings, store)
                    except Exception as error:
                        attempts = payload.get("attempts", 0) + 1
                        store.index_state(
                            snapshot,
                            "failed",
                            dict(
                                attempts=attempts,
                                retry_at=time.time() + min(300, 10 * 2**attempts),
                                error_type=type(error).__name__,
                            ),
                        )
                        logger.warning(json.dumps(dict(stage="index", error_type=type(error).__name__)))
                    return True
            return False
        stop = threading.Event()

        def heartbeat():
            while not stop.wait(30):
                try:
                    if not queue.heartbeat(task):
                        return
                except Exception:
                    logger.warning(json.dumps(dict(task_id=task["id"], stage="lease", status="unavailable")))

        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        try:
            path = (ROOT / task["payload"]["path"]).resolve()
            if (
                not path.is_relative_to((ROOT / "data/imports").resolve())
                or digest(path.read_bytes()) != task["payload"]["file_sha256"]
            ):
                raise ValueError("导入原件路径或指纹校验失败")
            snapshot = make_snapshot(path, path.parent / "snapshot")
            imported = store.upsert_import(snapshot, owner=task["owner"])
            index = index_builder(settings, store)
            status = queue.finish(task, dict(imported=imported, index=index))
        except Exception as error:
            status = queue.finish(task, {}, error)
        finally:
            stop.set()
            thread.join(timeout=2)
        logger.info(
            json.dumps(
                dict(
                    task_id=task["id"],
                    stage="import",
                    status=status,
                    attempt=task["attempts"],
                )
            )
        )
        return True
    finally:
        store.close()


def run(settings=None, once=False):
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if once:
        return {"worked": work_once(settings)}
    try:
        while True:
            try:
                worked = work_once(settings)
            except Exception as error:
                logger.error(json.dumps(dict(stage="worker", error_type=type(error).__name__)))
                worked = False
            if not worked:
                time.sleep(2)
    except KeyboardInterrupt:
        return {"stopped": True}
