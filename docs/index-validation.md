# 索引构建验收与重试

Milvus 的 `get_collection_stats().row_count` 是统计计数，重复 upsert 后可能包含尚未回收的旧记录，不能用它与文本块数量作精确验收。

构建完成后使用 `query(filter="", output_fields=["count(*)"], consistency_level="Strong", timeout=20)` 读取有效记录数。再按100块一批读取并比较 `id`、`job_key`、`version` 和 `content_hash`。这样既接受同一索引的重复构建，也能发现条数相同但身份或版本错误的索引；不依赖后台压缩时机。

验收失败抛出 `IndexValidationError`，属于可重试的索引故障。MySQL 保留失败状态，worker 使用原有最多3次退避重试；用户文件校验产生的 `ValueError` 和归属冲突仍直接失败。通过全部校验才发布索引指针，失败时保留原指针。返回结果仍按MySQL当前状态和版本检查。

校验只覆盖有效数量与块元数据，不证明向量语义质量。真实模型质量通过独立RAG评测判断。

固定响应回归：

```powershell
python -m pytest tests/test_index_validation.py -q
python -m pytest -m "not integration" -q
```

真实存储回归需要独立、空的测试岗位库和已配置的MySQL、Redis、Milvus；不会替你清空现有数据：

```powershell
$env:JOBCOMPASS_INTEGRATION = "1"
python -m pytest tests/test_index_integration.py tests/test_integration.py -q
```

覆盖真实Milvus连续重建、同条数错误版本拒绝及恢复，MySQL已提交后的校验失败重试，以及API导入与检索闭环。模型响应固定，不调用外部生成接口。

参考：[Milvus统计计数说明](https://milvus.io/api-reference/pymilvus/v2.6.x/MilvusClient/Collections/get_collection_stats.md)、[官方仓库对有效计数的说明](https://github.com/milvus-io/milvus/discussions/35631)。
