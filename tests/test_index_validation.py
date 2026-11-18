"""索引重复构建、错误记录及失败发布的回归；不加载模型或服务。"""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from jobcompass import retrieval


class FixedModels:
    def __init__(self, settings):
        pass

    def encode(self, texts):
        return np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (len(texts), 1))


class FakeMilvus:
    def __init__(self):
        self.rows = {}
        self.physical_count = 0
        self.closed = False
        self.corrupt = None

    def has_collection(self, **kwargs):
        return bool(self.rows)

    def create_collection(self, **kwargs):
        pass

    def upsert(self, *, data, **kwargs):
        self.rows.update({row["id"]: row for row in data})
        self.physical_count += len(data)

    def flush(self, **kwargs):
        pass

    def get_collection_stats(self, **kwargs):
        pytest.fail("不得以物理统计数验收有效记录")

    def query(self, *, filter, output_fields, consistency_level, timeout, **kwargs):
        assert consistency_level == "Strong" and timeout == 20
        if output_fields == ["count(*)"]:
            return [{"count(*)": len(self.rows) - int(self.corrupt == "count")}]
        ids = json.loads(filter.removeprefix("id in "))
        rows = [
            {key: row[key] for key in output_fields} for identifier in ids for row in [self.rows[identifier]]
        ]
        if self.corrupt == "missing":
            return rows[:-1]
        if self.corrupt in {"job_key", "content_hash", "version"}:
            rows[0][self.corrupt] = "incorrect"
        return rows

    def close(self):
        self.closed = True


@pytest.fixture
def build_fixture(tmp_path, monkeypatch, job_factory):
    rows = [job_factory("a"), job_factory("b")]
    states = []
    stores = []

    class FakeStore:
        def __init__(self, settings):
            self.closed = False
            stores.append(self)

        def all_jobs(self):
            return rows

        def index_state(self, snapshot, status, payload):
            states.append((status, payload))

        def close(self):
            self.closed = True

    client = FakeMilvus()
    monkeypatch.setattr(retrieval, "ROOT", tmp_path)
    monkeypatch.setattr(retrieval, "Store", FakeStore)
    monkeypatch.setattr(retrieval, "LocalModels", FixedModels)
    monkeypatch.setattr(retrieval, "milvus", lambda settings: client)
    config = SimpleNamespace(
        stage="test", config={"embedding": {"revision": "fixed"}, "retrieval": {"chunk_chars": 700}}
    )
    pointer = tmp_path / "artifacts/indexes/test/current.json"
    return config, client, states, stores, pointer


def test_repeated_build_accepts_upsert_tombstones_and_keeps_same_index(build_fixture):
    config, client, states, stores, pointer = build_fixture
    manifests = [retrieval.build_index(config) for _ in range(3)]
    assert client.physical_count == 6 and len(client.rows) == 2
    assert len({m["collection"] for m in manifests}) == 1
    assert all(m["chunks"] == 2 for m in manifests)
    assert json.loads(pointer.read_text())["index_version"] == manifests[0]["index_version"]
    assert [status for status, _ in states] == ["building", "ready"] * 3
    assert all(s.closed for s in stores) and client.closed


@pytest.mark.parametrize("corrupt", ["count", "missing", "job_key", "content_hash", "version"])
def test_invalid_index_is_retryable_and_does_not_replace_ready_pointer(build_fixture, corrupt):
    config, client, states, stores, pointer = build_fixture
    retrieval.build_index(config)
    before = pointer.read_bytes()
    client.corrupt = corrupt
    with pytest.raises(retrieval.IndexValidationError):
        retrieval.build_index(config)
    assert pointer.read_bytes() == before
    assert states[-1][0] == "failed"
    assert states[-1][1]["error_type"] == "IndexValidationError"
    assert all(s.closed for s in stores) and client.closed


def test_index_validation_exception_is_not_a_permanent_input_error():
    error = retrieval.IndexValidationError("fixed inconsistent index")
    assert isinstance(error, RuntimeError) and not isinstance(error, (ValueError, PermissionError))
