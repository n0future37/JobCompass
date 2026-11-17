import json
import logging
import uuid

from fastapi.testclient import TestClient

from jobcompass.web import app


def test_rejected_origin_keeps_request_id_and_structured_log(caplog):
    # 不进入生命周期或业务依赖；错误来源应在访问数据库前被拒绝。
    client = TestClient(app)
    try:
        with caplog.at_level(logging.INFO, logger="jobcompass.api"):
            response = client.post("/api/auth/register", headers={"Origin": "https://untrusted.invalid"})
        assert response.status_code == 403
        identifier = str(uuid.UUID(response.headers["x-request-id"]))
        assert response.headers["x-content-type-options"] == "nosniff"
        assert response.headers["cache-control"] == "no-store"
        records = [json.loads(r.message) for r in caplog.records if r.name == "jobcompass.api"]
        assert any(r["request_id"] == identifier and r["status"] == 403 for r in records)
    finally:
        client.close()
