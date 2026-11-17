# 导入模块

导入解决“不同来源的招聘信息如何进入统一岗位库”。网页支持 JSON 数组、UTF-8 CSV 文件，或在表单中粘贴纯文本 JD；输出持久化任务 ID。worker 才负责解析、标准化、MySQL 事务提交和 Milvus 索引。

## 文件规范

必填 `source`、`source_job_id`、`title`、`jd_text`。`source` 为小写英文平台名（可含数字/下划线），同一来源的岗位 ID 稳定不变。`jd_text` 为完整正文。不要将接口列表卡片或旧导出当作原始 HTTP 响应。

可选：`company`、`cities`、`skills`、`location_text`、`salary_text`、`education_text`、`experience_text`、`employment_type_text`、`attendance_text`、`duration_text`、`source_published_at_text`、`source_updated_at_text`、`url`、`active`。缺失字段保持未知，不推断薪资、学历、经验或在招状态。`active` 仅表示是否允许本地检索。多余字段拒绝导入，避免拼写错误悄悄丢数据。

JSON 的 `cities`、`skills` 为数组；CSV 用竖线分隔。CSV 的 `active` 为 true/false/1/0。单文件最多 5MB、10000 行。整个文件先校验，任一必填项错误时不提交部分业务数据。

下面是虚构的格式样例，不是采集结果，不应混入真实评测语料：

```json
[{"source":"example","source_job_id":"sample001","title":"Python后端开发","company":"虚构示例公司","cities":["深圳"],"jd_text":"使用Python开发业务接口。\n熟悉MySQL。","employment_type_text":"全职"}]
```

## 粘贴纯文本 JD

在网页“数据导入”切换至“粘贴 JD”。岗位名称、JD 正文必填；公司、城市、类型及展开的学历、经验、薪资、来源链接、来源编号可选。缺少的字段保持未知，不从正文自动猜测。游客可先填写，登录后草稿保留；切换到另一个账号会清除表单。

`POST /api/imports/text` 接收结构化 JSON（不是直接发送 text/plain），例如 `{"title":"Python后端开发","jd_text":"使用Python开发接口。"}`。类型约束见 `TextImportRequest`；文本按原样封装为 JSON 数组，复用文件落盘、限流、幂等任务和 worker。来源记为 `manual`，不伪装成网页原始响应。

岗位身份按“来源编号 → 来源 URL（忽略片段）→ 规范化字段内容”确定，并加账号命名空间。同一输入重复提交返回同一任务；相同编号或 URL 的正文更新进入同一岗位的新版本。无编号/URL时，修改内容会产生另一个身份，系统不能推断两段文本是否同一岗位。不同账号的手动录入不会互相覆盖。任务与导入原件私有；成功导入后的岗位进入共享岗位库，网页明确提示这一点。

## 调用与重试

登录后 `POST /api/imports`：`{"filename":"jobs.json","content":"文件完整文本","idempotency_key":"本次批次稳定标识"}`。相同用户、相同幂等键返回同一任务；该键换成另一份文件返回409。原件按 SHA256 保存，不覆盖。

`GET /api/imports/{id}` 查看 `queued/running/retrying/completed/failed`、尝试次数和失败类型；失败后可 `POST /api/imports/{id}/retry`。`completed` 表示该任务数据已提交且索引构建成功。另一个导入可能在其后继续改变语料，查询仍会回查当前状态。

worker 的正式入口为 `python -m jobcompass worker`；`worker --once` 处理一次任务/索引恢复后退出。任务最多自动尝试3次，超时等暂时性错误退避重试；字段错误和归属冲突直接失败。租约300秒、每30秒续期，过期任务可再次领取，旧租约不能覆盖新任务终态。

同一岗位相同内容合并来源观察；内容变化分配新的单调版本号。A→B→A 也保留三个事件版本。迟到旧观察不能回滚当前岗位。跨用户不能覆盖同一来源岗位；公共历史语料归属 system。岗位下架接口仅允许其导入者操作。

索引状态与岗位更新在同一 MySQL 事务中写入。MySQL成功、Milvus失败时，任务和待索引状态保留，重复执行不会重复增加岗位；worker或 `build` 可以恢复。在向量索引同步前，查询明确降级为当前关键词结果；旧向量不能绕过状态和版本检查。

关键文件：`importing.py` 解析格式；`tasks.py` 持久化队列；`worker.py` 编排；`storage.py` 事务和版本；`web.py` 登录、归属校验及API。测试见 `tests/test_integration.py`、`tests/test_text_import.py`。新模块4项测试验证原文保留、身份规则、字段边界及真实测试库的API/任务/版本/越权流程；索引构建使用固定响应，既有集成测试另验证真实Milvus。完整回归Windows71通过（11真实集成）、Linux无服务60通过，结果不相加。产物见 `artifacts/imports/20261103_text/verification.json` 及 `artifacts/tests/text-import-*.xml`。
