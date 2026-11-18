"""MySQL是业务权威数据源；Milvus结果必须回查当前版本。"""

import json
import uuid
from datetime import UTC, datetime

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
    func,
    select,
    update,
)
from sqlalchemy.dialects.mysql import insert

from .preprocessing import digest
from .settings import Settings

metadata = MetaData()
jobs = Table(
    "rag_jobs",
    metadata,
    Column("job_key", String(192, collation="utf8mb4_bin"), primary_key=True),
    Column("source", String(32), nullable=False),
    Column("source_job_id", String(128, collation="utf8mb4_bin"), nullable=False),
    Column("version", Integer, nullable=False),
    Column("content_hash", String(64), nullable=False),
    Column("active", Boolean, nullable=False, default=True),
    Column("owner", String(64), nullable=False, default="system"),
    Column("payload", JSON, nullable=False),
)
versions = Table(
    "rag_versions",
    metadata,
    Column("job_key", String(192, collation="utf8mb4_bin"), primary_key=True),
    Column("version", Integer, primary_key=True),
    Column("content_hash", String(64), nullable=False),
    Column("payload", JSON, nullable=False),
)
imports = Table(
    "rag_imports",
    metadata,
    Column("fingerprint", String(64), primary_key=True),
    Column("status", String(32), nullable=False),
    Column("payload", JSON, nullable=False),
)
indexes = Table(
    "rag_indexes",
    metadata,
    Column("snapshot", String(64), primary_key=True),
    Column("status", String(32), nullable=False),
    Column("payload", JSON, nullable=False),
)
history = Table(
    "rag_history",
    metadata,
    Column("query_id", String(36), primary_key=True),
    Column("owner", String(64), nullable=False, index=True),
    Column("created_at", String(40), nullable=False),
    Column("payload", JSON, nullable=False),
)
feedback = Table(
    "rag_feedback",
    metadata,
    Column("feedback_id", String(36), primary_key=True),
    Column("query_id", String(36), nullable=False),
    Column("owner", String(64), nullable=False),
    Column("payload", JSON, nullable=False),
)


def engine(settings=None):
    settings = settings or Settings()
    return create_engine(
        settings.mysql_url,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=5,
        pool_recycle=1800,
        hide_parameters=True,
        connect_args={"connect_timeout": 5, "read_timeout": 20, "write_timeout": 20},
    )


class Store:
    def __init__(self, settings=None):
        self.settings = settings or Settings()
        self.engine = engine(self.settings)

    def corpus_revision(self):
        """只取当前有效岗位的身份/版本，避免为缓存键反复解码全部JD正文。"""
        from .preprocessing import digest

        with self.engine.connect() as connection:
            rows = connection.execute(
                select(jobs.c.job_key, jobs.c.content_hash, jobs.c.version)
                .where(jobs.c.active.is_(True))
                .order_by(jobs.c.job_key)
            )
            return digest([list(row) for row in rows])

    def upsert_import(self, snapshot, owner="system"):
        manifest = json.loads((snapshot / "manifest.json").read_bytes())
        # JSON字符串中的U+2028等字符不是JSONL分隔符，不能使用str.splitlines()。
        rows = [
            json.loads(s)
            for s in (snapshot / "jobs.jsonl").read_text(encoding="utf-8").split("\n")
            if s.strip()
        ]
        old_versions = [
            json.loads(s)
            for s in (snapshot / "versions.jsonl").read_text(encoding="utf-8").split("\n")
            if s.strip()
        ]
        fingerprint = digest(
            [
                owner,
                manifest["raw_fingerprint"],
                manifest["schema_version"],
                manifest["jobs_sha256"],
            ]
        )
        with self.engine.begin() as connection:
            exists = (
                connection.execute(select(imports).where(imports.c.fingerprint == fingerprint))
                .mappings()
                .first()
            )
            if exists and exists["status"] == "completed":
                return {"status": "already_imported", "jobs": len(rows)}
            connection.execute(
                insert(imports)
                .values(fingerprint=fingerprint, status="running", payload=manifest)
                .on_duplicate_key_update(status="running", payload=manifest)
            )
            for row in rows:
                row = dict(row)
                existing = (
                    connection.execute(select(jobs).where(jobs.c.job_key == row["job_key"]).with_for_update())
                    .mappings()
                    .first()
                )
                if existing and existing["owner"] != owner:
                    raise PermissionError("不能覆盖其他导入者的岗位")
                if not existing:
                    for observation in old_versions:
                        if observation["job_key"] == row["job_key"]:
                            connection.execute(
                                insert(versions)
                                .values(
                                    job_key=observation["job_key"],
                                    version=observation["version"],
                                    content_hash=observation["content_hash"],
                                    payload=observation,
                                )
                                .on_duplicate_key_update(version=versions.c.version)
                            )
                if existing:
                    old = existing["payload"]
                    if datetime.fromisoformat(row["collected_at"]) < datetime.fromisoformat(
                        old["collected_at"]
                    ):
                        continue  # 迟到的旧观察不能回滚当前岗位。
                    row["version"] = existing["version"] + int(
                        existing["content_hash"] != row["content_hash"]
                    )
                    if existing["content_hash"] == row["content_hash"]:
                        merged = {digest(p): p for p in old["provenance"] + row["provenance"]}
                        row["provenance"] = list(merged.values())
                connection.execute(
                    insert(versions)
                    .values(
                        job_key=row["job_key"],
                        version=row["version"],
                        content_hash=row["content_hash"],
                        payload=row,
                    )
                    .on_duplicate_key_update(payload=row)
                )
                values = dict(
                    job_key=row["job_key"],
                    source=row["source"],
                    source_job_id=row["source_job_id"],
                    version=row["version"],
                    content_hash=row["content_hash"],
                    active=row.get("active", True),
                    owner=owner,
                    payload=row,
                )
                connection.execute(insert(jobs).values(**values).on_duplicate_key_update(**values))
            connection.execute(
                update(imports).where(imports.c.fingerprint == fingerprint).values(status="completed")
            )
            # 与权威岗位更新同一事务保存待建索引状态，build失败可重复执行恢复。
            pairs = connection.execute(
                select(jobs.c.job_key, jobs.c.content_hash, jobs.c.version)
                .where(jobs.c.active.is_(True))
                .order_by(jobs.c.job_key)
            ).all()
            index_snapshot = digest([list(p) for p in pairs])
            connection.execute(
                insert(indexes)
                .values(snapshot=index_snapshot, status="pending", payload={})
                .on_duplicate_key_update(snapshot=index_snapshot)
            )
        return {"status": "completed", "jobs": len(rows), "fingerprint": fingerprint}

    def all_jobs(self):
        with self.engine.connect() as connection:
            return [
                dict(r[0])
                for r in connection.execute(
                    select(jobs.c.payload).where(jobs.c.active.is_(True)).order_by(jobs.c.job_key)
                )
            ]

    def current(self, keys):
        if not keys:
            return {}
        with self.engine.connect() as connection:
            return {
                r.job_key: dict(r.payload)
                for r in connection.execute(
                    select(jobs).where(jobs.c.job_key.in_(list(keys)), jobs.c.active.is_(True))
                )
            }

    def page(self, offset=0, limit=20, city=None, keyword=None):
        filters = [jobs.c.active.is_(True)]
        if city:
            pattern = "%" + city.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
            filters.append(func.json_search(jobs.c.payload, "one", pattern, "\\", "$.cities[*]").is_not(None))
        if keyword:
            filters.append(
                jobs.c.payload["title"].as_string().icontains(keyword, autoescape=True)
                | jobs.c.payload["jd_text"].as_string().icontains(keyword, autoescape=True)
            )
        with self.engine.connect() as c:
            total = c.execute(select(func.count()).select_from(jobs).where(*filters)).scalar_one()
            values = c.execute(
                select(jobs.c.payload).where(*filters).order_by(jobs.c.job_key).offset(offset).limit(limit)
            ).scalars()
            return {"total": total, "items": [dict(v) for v in values]}

    def index_state(self, snapshot, status, payload):
        with self.engine.begin() as connection:
            connection.execute(
                insert(indexes)
                .values(snapshot=snapshot, status=status, payload=payload)
                .on_duplicate_key_update(status=status, payload=payload)
            )

    def save_history(self, owner, payload):
        qid = str(uuid.uuid4())
        with self.engine.begin() as connection:
            connection.execute(
                history.insert().values(
                    query_id=qid,
                    owner=owner,
                    created_at=datetime.now(UTC).isoformat(),
                    payload=payload,
                )
            )
        return qid

    def list_history(self, owner, limit=20):
        with self.engine.connect() as connection:
            return [
                dict(r)
                for r in connection.execute(
                    select(history)
                    .where(history.c.owner == owner)
                    .order_by(history.c.created_at.desc())
                    .limit(limit)
                ).mappings()
            ]

    def add_feedback(self, owner, query_id, payload):
        with self.engine.begin() as connection:
            if not connection.execute(
                select(history.c.query_id).where(history.c.query_id == query_id, history.c.owner == owner)
            ).first():
                raise ValueError("查询不存在或不属于当前会话")
            fid = str(uuid.uuid4())
            connection.execute(
                feedback.insert().values(feedback_id=fid, query_id=query_id, owner=owner, payload=payload)
            )
        return fid

    def close(self):
        self.engine.dispose()


def corpus_fingerprint(store):
    """业务存储使用轻量SQL；离线评测的只读内存适配器保持兼容。"""
    if hasattr(store, "corpus_revision"):
        return store.corpus_revision()
    from .preprocessing import digest

    return digest([[j["job_key"], j["content_hash"], j["version"]] for j in store.all_jobs()])
