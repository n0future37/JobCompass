import os
import uuid

import pytest

from jobcompass.preprocessing import digest


@pytest.fixture
def job_factory():
    def make(sid="1", jd="使用Python开发接口\n熟悉MySQL", **changes):
        row = dict(
            source="fixture",
            source_job_id=sid,
            job_key=f"fixture:{sid}",
            title="Python后端开发",
            company="测试公司",
            cities=["深圳"],
            location_text=None,
            salary_text=None,
            education_text=None,
            experience_text=None,
            employment_type_text="全职",
            skills=[],
            attendance_text=None,
            duration_text=None,
            source_published_at_text=None,
            source_updated_at_text=None,
            source_status_raw=None,
            url=None,
            jd_text=jd,
            raw_jd=jd,
            raw_jd_format="text",
            jd_source_field="jd_text",
            schema_version="job.v1",
            collected_at="2026-10-02T00:00:00+00:00",
            version=1,
            provenance=[{"raw_file": "fixture.json"}],
        )
        row.update(changes)
        row["content_hash"] = digest(
            {
                k: row[k]
                for k in (
                    "job_key",
                    "title",
                    "jd_text",
                    "cities",
                    "employment_type_text",
                )
            }
        )
        row["duplicate_group"] = digest(row["jd_text"])
        return row

    return make


@pytest.fixture
def real_store():
    if os.getenv("JOBCOMPASS_INTEGRATION") != "1":
        pytest.skip("设置JOBCOMPASS_INTEGRATION=1以使用真实隔离测试库")
    from sqlalchemy import delete
    from jobcompass.career import applications, preferences
    from jobcompass.resumes import resumes
    from jobcompass.identity import sessions, users
    from jobcompass.settings import Settings
    from jobcompass.storage import (
        Store,
        feedback,
        history,
        imports,
        indexes,
        jobs,
        versions,
    )
    from jobcompass.tasks import tasks

    store = Store(Settings("test"))
    from sqlalchemy import text

    from jobcompass.storage import engine

    # 协调锁使用独立连接池，不占用被测业务池，也不影响checkedout断言。
    validation_engine = engine(store.settings)
    validation_lock = validation_engine.connect()
    lock_name = "jobcompass:test:validation"
    if not validation_lock.execute(text("SELECT GET_LOCK(:n, 0)"), {"n": lock_name}).scalar():
        validation_lock.close()
        validation_engine.dispose()
        store.close()
        pytest.fail("另一个真实服务验收正在运行；未改动测试数据")
    assert store.settings.db_name == "jobcompass_test" and store.settings.redis_db == 1
    prefix = uuid.uuid4().hex[:24]
    store.test_owner = prefix
    store.test_prefix = prefix
    # 只清理该fixture新建的主键；测试库已有内容全部保留。
    with store.engine.connect() as c:
        old_imports = set(c.execute(imports.select()).scalars())
        old_indexes = set(c.execute(indexes.select()).scalars())
    try:
        yield store
    finally:
        with store.engine.begin() as c:
            user_ids = list(c.execute(select_user_ids(users, prefix)).scalars())
            for table in (resumes, applications, preferences, feedback, history, tasks):
                c.execute(delete(table).where(table.c.owner.like(prefix + "%") | table.c.owner.in_(user_ids)))
            c.execute(delete(sessions).where(sessions.c.user_id.in_(select_user_ids(users, prefix))))
            c.execute(delete(users).where(users.c.username.like(prefix + "%")))
            keys = (
                c.execute(jobs.select().where(jobs.c.owner.like(prefix + "%") | jobs.c.owner.in_(user_ids)))
                .mappings()
                .all()
            )
            if keys:
                c.execute(delete(versions).where(versions.c.job_key.in_([r["job_key"] for r in keys])))
            c.execute(delete(jobs).where(jobs.c.owner.like(prefix + "%") | jobs.c.owner.in_(user_ids)))
            # 仅删除本用例创建的派生状态行；测试不并行运行此fixture。
            for table, old in ((imports, old_imports), (indexes, old_indexes)):
                new = set(c.execute(table.select()).scalars()) - old
                if new:
                    c.execute(delete(table).where(list(table.c)[0].in_(new)))
        validation_lock.execute(text("SELECT RELEASE_LOCK(:n)"), {"n": lock_name})
        validation_lock.close()
        validation_engine.dispose()
        store.close()


def select_user_ids(users, prefix):
    from sqlalchemy import select

    return select(users.c.id).where(users.c.username.like(prefix + "%"))


@pytest.fixture(autouse=True)
def fixed_local_evidence_response(monkeypatch):
    # 软件自动化不加载真实模型。门槛/拒绝用例另注入固定正负分数；真实模型质量独立评测。
    from jobcompass.evidence import EvidenceGuard

    monkeypatch.setattr(EvidenceGuard, "_model_scores", lambda self, query, passages: [20.0] * len(passages))


@pytest.fixture(autouse=True)
def fixed_personalized_ranking_response(monkeypatch):
    # 普通软件/真实存储测试均不向外部模型发送画像；专门协议用例显式注入MockTransport。
    import json
    from jobcompass.personalized_ranking import PersonalizedRanker

    original = PersonalizedRanker._post

    async def fixed(self, url, payload, key):
        if self.transport is not None:
            return await original(self, url, payload, key)
        candidates = json.loads(payload["messages"][-1]["content"])["candidates"]
        return dict(
            choices=[
                dict(
                    finish_reason="stop",
                    message=dict(content=json.dumps(dict(ranking=[j["job_id"] for j in candidates]))),
                )
            ],
            usage=dict(
                prompt_tokens=100,
                completion_tokens=30,
                prompt_cache_hit_tokens=0,
                completion_tokens_details=dict(reasoning_tokens=0),
            ),
        )

    monkeypatch.setattr(PersonalizedRanker, "_post", fixed)
