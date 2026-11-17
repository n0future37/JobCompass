"""纯文本导入与JSON/CSV共用任务；验证身份、原文、版本及归属。"""

import json

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy import select

from jobcompass.importing import TextImportRequest, prepare_text_import
from jobcompass.storage import jobs, versions
from jobcompass.tasks import tasks


def test_text_import_wraps_raw_jd_and_leaves_missing_unknown():
    jd = "  使用 Python 开发接口\n熟悉 MySQL\u2028知识库检索  "
    packaged = prepare_text_import(TextImportRequest(title="后端开发", jd_text=jd), "user-one")
    row = json.loads(packaged["content"])[0]
    assert row["source"] == "manual" and row["jd_text"] == jd
    assert row["cities"] == [] and row["education_text"] is None
    assert row["experience_text"] is None and row["company"] is None
    assert packaged["filename"].endswith(".json")
    assert packaged == prepare_text_import(TextImportRequest(title="后端开发", jd_text=jd), "user-one")


def test_text_import_stable_identity_and_owner_namespaces():
    one = TextImportRequest(title="后端", jd_text="Python接口", source_job_id="Job-A")

    def key(value, owner="one"):
        return json.loads(prepare_text_import(value, owner)["content"])[0]["source_job_id"]

    changed = one.model_copy(update={"jd_text": "Python和SQL接口"})
    assert key(one) == key(changed) and key(one) != key(one, "two")
    assert (
        prepare_text_import(one, "one")["idempotency_key"]
        != prepare_text_import(changed, "one")["idempotency_key"]
    )
    url = TextImportRequest.model_validate(
        one.model_dump() | {"source_job_id": None, "url": "https://example.invalid/job/1#detail"}
    )
    assert key(url) == key(url.model_copy(update={"jd_text": "new requirements"}))
    no_id = TextImportRequest(title="后端", jd_text="Python")
    assert key(no_id) != key(no_id.model_copy(update={"jd_text": "Java"}))


def test_text_import_rejects_bad_fields():
    for change in (
        {"title": " "},
        {"jd_text": "\n"},
        {"url": "javascript:alert(1)"},
        {"source_job_id": "../bad"},
        {"owner": "other"},
        {"employment_type_text": "invented"},
    ):
        with pytest.raises(ValidationError):
            TextImportRequest.model_validate(dict(title="后端", jd_text="Python接口") | change)


@pytest.mark.integration
def test_text_api_task_worker_versions_and_ownership(real_store, tmp_path, monkeypatch):
    from jobcompass import web, worker, settings
    from jobcompass.identity import Identity
    from jobcompass.service import RagService
    from jobcompass.tasks import TaskQueue

    s = real_store
    s.settings.prefix += "text_import_" + s.test_prefix + ":"
    for module in (web, worker, settings):
        monkeypatch.setattr(module, "ROOT", tmp_path)
    identity = Identity(s)
    first_name = s.test_prefix + "a"
    uid = identity.register(first_name, "fixed-import-password")
    identity.register(s.test_prefix + "b", "fixed-import-password")
    svc = RagService(s.settings)
    monkeypatch.setattr(web, "RagService", lambda: svc)
    web.app.dependency_overrides[web.service] = lambda: svc
    try:
        with TestClient(web.app) as client:
            assert (
                client.post("/api/imports/text", json={"title": "岗位", "jd_text": "Python"}).status_code
                == 401
            )
            client.cookies.set("jc_auth", identity.login(first_name, "fixed-import-password"))
            content = dict(
                title="合成文本导入岗位",
                jd_text="Python开发接口\n熟悉MySQL",
                company="合成公司",
                cities=["深圳"],
                source_job_id="fixed-one",
            )
            first = client.post("/api/imports/text", json=content)
            assert first.status_code == 202
            task_id = first.json()["id"]
            assert client.post("/api/imports/text", json=content).json()["id"] == task_id
            assert client.post("/api/imports/text", json={**content, "title": " "}).status_code == 422
            with s.engine.connect() as c:
                payload = c.execute(select(tasks.c.payload).where(tasks.c.id == task_id)).scalar_one()
            raw = json.loads((tmp_path / payload["path"]).read_text(encoding="utf-8"))[0]
            key = "manual:" + raw["source_job_id"]
            assert raw["jd_text"] == content["jd_text"]
            assert worker.work_once(s.settings, index_builder=lambda *args: {"kind": "fixed-index-response"})
            assert TaskQueue(s).get(uid, task_id)["status"] == "completed"
            job = s.current([key])[key]
            assert job["version"] == 1 and job["education_text"] is None
            updated = client.post("/api/imports/text", json={**content, "jd_text": "Python和SQL开发接口"})
            assert updated.status_code == 202 and updated.json()["id"] != task_id
            assert worker.work_once(s.settings, index_builder=lambda *args: {"kind": "fixed-index-response"})
            assert s.current([key])[key]["version"] == 2
            with s.engine.connect() as c:
                assert len(c.execute(select(versions).where(versions.c.job_key == key)).all()) == 2
                assert len(c.execute(select(jobs).where(jobs.c.job_key == key)).all()) == 1
            client.cookies.set("jc_auth", identity.login(s.test_prefix + "b", "fixed-import-password"))
            assert client.get("/api/imports/" + task_id).status_code == 404
            assert client.get("/api/raw", params={"path": payload["path"]}).status_code == 404
    finally:
        web.app.dependency_overrides.clear()
        keys = list(svc.redis.scan_iter(match=s.settings.prefix + "*"))
        if keys:
            svc.redis.delete(*keys)
        svc.close()
