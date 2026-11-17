"""MySQL持久化任务队列：幂等提交、租约、重试和任务归属。"""

import time
import uuid

from sqlalchemy import JSON, Column, Float, Integer, String, Table, or_, select, update
from sqlalchemy.exc import OperationalError
from sqlalchemy.dialects.mysql import insert

from .preprocessing import digest
from .storage import metadata

tasks = Table(
    "rag_tasks",
    metadata,
    Column("id", String(36), primary_key=True),
    Column("owner", String(64), nullable=False),
    Column("idempotency", String(64), unique=True, nullable=False),
    Column("status", String(32), nullable=False),
    Column("attempts", Integer, nullable=False),
    Column("available_at", Float(53), nullable=False),
    Column("lease_until", Float(53), nullable=False),
    Column("lease_token", String(36), nullable=True),
    Column("payload", JSON, nullable=False),
    Column("result", JSON, nullable=False),
)


class TaskQueue:
    def __init__(self, store):
        self.store = store

    def submit(self, owner, key, payload):
        identifier = digest([owner, key])
        task_id = str(uuid.uuid4())
        with self.store.engine.begin() as c:
            c.execute(
                insert(tasks)
                .values(
                    id=task_id,
                    owner=owner,
                    idempotency=identifier,
                    status="queued",
                    attempts=0,
                    available_at=time.time(),
                    lease_until=0,
                    lease_token=None,
                    payload=payload,
                    result={},
                )
                .on_duplicate_key_update(id=tasks.c.id)
            )
            row = dict(c.execute(select(tasks).where(tasks.c.idempotency == identifier)).mappings().one())
            if row["payload"].get("file_sha256") != payload.get("file_sha256"):
                raise ValueError("幂等键已用于不同文件")
        return row

    def get(self, owner, task_id):
        with self.store.engine.connect() as c:
            row = (
                c.execute(select(tasks).where(tasks.c.id == task_id, tasks.c.owner == owner))
                .mappings()
                .first()
            )
            return dict(row) if row else None

    def claim(self):
        # InnoDB死锁/锁等待只重试整个领取事务；退出begin上下文后已回滚。
        for attempt in range(3):
            try:
                return self._claim_once()
            except OperationalError as error:
                code = error.orig.args[0] if getattr(error.orig, "args", ()) else None
                if code not in {1213, 1205} or attempt == 2:
                    raise
                time.sleep(0.02 * 2**attempt)

    def _claim_once(self):
        now, token = time.time(), str(uuid.uuid4())
        with self.store.engine.begin() as c:
            # SKIP LOCKED允许多个worker领取不同任务；过期租约允许崩溃恢复。
            row = (
                c.execute(
                    select(tasks)
                    .where(
                        tasks.c.attempts < 3,
                        tasks.c.available_at <= now,
                        or_(
                            tasks.c.status.in_(["queued", "retrying"]),
                            (tasks.c.status == "running") & (tasks.c.lease_until < now),
                        ),
                    )
                    .order_by(tasks.c.available_at, tasks.c.id)
                    .limit(1)
                    .with_for_update(skip_locked=True)
                )
                .mappings()
                .first()
            )
            if not row:
                # 最后一次尝试中进程崩溃，也必须形成可见终态。
                c.execute(
                    update(tasks)
                    .where(
                        tasks.c.status == "running",
                        tasks.c.lease_until < now,
                        tasks.c.attempts >= 3,
                    )
                    .values(
                        status="failed",
                        result={"error_type": "LeaseExpired", "retryable": True},
                    )
                )
                return None
            row = dict(row)
            c.execute(
                update(tasks)
                .where(tasks.c.id == row["id"])
                .values(
                    status="running",
                    attempts=row["attempts"] + 1,
                    lease_token=token,
                    lease_until=now + 300,
                )
            )
            row.update(lease_token=token, attempts=row["attempts"] + 1)
        return row

    def heartbeat(self, task):
        with self.store.engine.begin() as c:
            return (
                c.execute(
                    update(tasks)
                    .where(
                        tasks.c.id == task["id"],
                        tasks.c.lease_token == task["lease_token"],
                        tasks.c.status == "running",
                    )
                    .values(lease_until=time.time() + 300)
                ).rowcount
                == 1
            )

    def finish(self, task, result, error=None):
        terminal = isinstance(error, (ValueError, PermissionError)) or task["attempts"] >= 3
        status = "completed" if error is None else "failed" if terminal else "retrying"
        result = (
            result
            if error is None
            else {
                "error_type": type(error).__name__,
                "retryable": not isinstance(error, (ValueError, PermissionError)),
            }
        )
        with self.store.engine.begin() as c:
            count = c.execute(
                update(tasks)
                .where(
                    tasks.c.id == task["id"],
                    tasks.c.lease_token == task["lease_token"],
                    tasks.c.status == "running",
                )
                .values(
                    status=status,
                    result=result,
                    lease_until=0,
                    lease_token=None,
                    available_at=time.time() + min(300, 5 * 2 ** task["attempts"]),
                )
            ).rowcount
        return status if count else "lease_lost"

    def retry(self, owner, task_id):
        with self.store.engine.begin() as c:
            return (
                c.execute(
                    update(tasks)
                    .where(
                        tasks.c.id == task_id,
                        tasks.c.owner == owner,
                        tasks.c.status == "failed",
                    )
                    .values(
                        status="queued",
                        attempts=0,
                        available_at=time.time(),
                        lease_until=0,
                        result={},
                    )
                ).rowcount
                == 1
            )
