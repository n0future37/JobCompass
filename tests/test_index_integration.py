"""真实Milvus重复upsert校验；固定向量，不调用外部模型。"""

import json

import numpy as np
import pytest

from jobcompass import retrieval, settings
from jobcompass.importing import make_snapshot


@pytest.mark.integration
def test_real_milvus_rebuild_and_corrupt_version_recovery(real_store, tmp_path, monkeypatch):
    s = real_store
    assert not s.all_jobs(), "需要空测试岗位库，不会清空既有数据"
    monkeypatch.setattr(settings, "ROOT", tmp_path)
    monkeypatch.setattr(retrieval, "ROOT", tmp_path)

    class FixedModels:
        def __init__(self, config):
            pass

        def encode(self, texts):
            return np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (len(texts), 1))

    monkeypatch.setattr(retrieval, "LocalModels", FixedModels)
    raw = tmp_path / "jobs.json"
    raw.write_text(
        json.dumps(
            [
                dict(
                    source="fixture",
                    source_job_id=s.test_prefix,
                    title="Python开发",
                    jd_text="负责Python后端接口开发",
                )
            ]
        ),
        encoding="utf-8",
    )
    s.upsert_import(make_snapshot(raw, tmp_path / "snapshot"), s.test_owner)
    client = retrieval.milvus(s.settings)
    before = set(client.list_collections())
    try:
        manifests = [retrieval.build_index(s.settings) for _ in range(3)]
        assert len({m["collection"] for m in manifests}) == 1
        collection = manifests[0]["collection"]
        passages = retrieval.chunks(s.all_jobs())
        assert client.query(
            collection_name=collection, filter="", output_fields=["count(*)"], consistency_level="Strong"
        ) == [{"count(*)": 1}]
        # 条数正确但版本错误，也不能验收；下次完整构建应恢复正确版本。
        client.upsert(
            collection_name=collection, data=[{**passages[0], "version": 999, "vector": [1.0, 0.0]}]
        )
        with pytest.raises(retrieval.IndexValidationError, match="版本"):
            retrieval.validate_index(client, collection, passages)
        repaired = retrieval.build_index(s.settings)
        assert repaired["collection"] == collection
        assert retrieval.validate_index(client, collection, passages) == 1
    finally:
        for name in set(client.list_collections()) - before:
            assert name.startswith("jc_chunks_test_")
            client.drop_collection(name)
        client.close()
