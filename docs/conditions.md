# 条件解析与岗位筛选

## 2026-11-16：学历约束修复（当前行为）

普通搜索和Agent共享`qualifications.py`的学历证据检查。`SearchRequest.education_max`表示岗位最低学历要求的可接受上限（初中/高中/中专/大专/本科/硕士/博士），不是要求岗位恰好写这个学历；本轮明确自述或上限语句会进入intent.v3.1。若文本和显式参数同时给上限，按更窄范围核验。

示例“我是大专毕业，想看没有本科硬门槛的测试开发岗位，学历没写的请标待确认”：排除正文或来源字段明确要求本科/硕士的候选；来源缺失或冲突保持pending。仅`include_unknown_education=true`或用户明确说待确认时，允许展示未知候选；检查仍为`satisfied=false, unknown=true, status=pending`，网页写“学历待确认”。不能把允许展示理解成已满足。

核对来源`education_text`及JD明确学历要求，保留各自证据。JD按任职要求与职责/福利段落区分，福利落户或服务对象学历不作员工最低学历；“本科及以上”无学历后缀也检查。两个来源冲突时不静默择优；任一可靠门槛已超过用户上限即排除，即使允许未知也不能绕过。优先/加分不当硬下限；同等经验等复杂条件仍未知。经验、薪资、远程及职责否定并未由本轮学历修复获得全面支持。

验证：`python -m pytest tests/test_education_conditions.py tests/test_intent_v2.py tests/test_career_agent.py -q`。最终本机193测试通过含11真实隔离集成，记录`artifacts/tests/repair-final-full.xml`；Linux同套182非集成通过（`repair-linux-configured.xml`），次数不相加。真实语料反例与初次失败分别保留在`artifacts/experiments/evidence_guard_20261116_v3/`，当前RAG配置rag.v6使旧缓存失效，未重建开发索引。

以下为此前城市/类型及意图v2的规则与历史验收；历史命令不可覆盖旧评测目录，新实验见`docs/evaluation.md`。

用户输入“广州找会计全职，不接受兼职，最好有Excel培训”，普通搜索先形成城市广州、类型全职、排除兼职三项硬条件；Excel培训只是偏好。原查询仍交给BM25/向量检索，不把“会计”变成只能标题精确命中的过滤器。

数据流是SearchRequest→intent.parse_intent→HardConditions与来源span→conditions.assess→召回/重排→有效版本回查。个人画像不覆盖本轮条件；来源span是用户原话的位置，不是JD证据或个人能力证明。API返回intent.version/status/hard/soft_preferences/sources/unresolved/conflicts，岗位匹配结果另外返回hard_condition_checks。

## 规则与边界

同字段备选值按OR，例如北京或上海；不同硬字段按AND。表单选北京/上海、文本只看北京，取交集北京；表单北京、文本只看广州，返回冲突，不自动放宽成两地。之前考虑北京、现在只看天津，以当前要求为准。

城市支持配置词表和“市”后缀规范化；校招/校园招聘归校园，兼职/临时归兼职/临时。显式排除城市和类型使用单独字段excluded_cities/excluded_employment_types；未知不是满足。多城市岗位至少有一个同时满足正向范围且未被排除的办公选项才通过，不声称所有办公地点都满足。

skills_all表示JD明确提词，不是“用户会这项技能”或“用户具备任职资格”。否定提词不作为满足证据。自然语言技能识别仍是有限词表；当前学历门槛见上方v3，年限门槛的有限支持另见推荐模块。职责级“不要纯销售”、每周到岗/薪资/远程等缺少可靠支持时，明确返回unsupported。复杂语义和词表外表达未获总体准确率验证。

软城市/招聘类型记录在intent中，不硬筛选；当前尚无专门排序奖励。原查询中的偏好词仍可能影响词法/向量分数，画像技能则保留既有小幅排序逻辑。不要把“已识别软偏好”写成已完成学习用户偏好或跨职业语义理解。

## 文件与验证

- conditions.py定义请求字段及岗位事实核验，intent.py负责带来源的有限解析，retrieval.py把解析接入原检索链路。
- rag.toml的rag.v2使旧召回缓存失效，向量/块/融合/重排参数不变；当前索引无需重建。
- intent_evaluation.py只读冻结条件参考；stage_ablation.py在独立实验进程中注入旧条件或参考条件作诊断。参考条件不会传给生产过滤器。

固定data/eval/multiuser_intents_v1含128助手撰写模板用例，dev/test各64。实现前冻结，首次test前锁定源码，首次新版dev/test均64通过；原parser为24/21。作者同时实现解析器，模板重复，因此是功能回归证据，不是人类盲测或真实用户准确率。测试已暴露，后续变更按回归处理。报告在artifacts/experiments/intent_v2_20261103/REPORT.md。

在项目目录使用job解释器：

```powershell
& 'python' -m pytest tests/test_intent_v2.py tests/test_stage_ablation.py -q
& 'python' -m jobcompass.intent_evaluation --dataset data/eval/multiuser_intents_v1 --split test --output artifacts/experiments/intent_v2_20261103
```

面试可说明：本轮把可核验条件从模糊查询中分离，保留条件来源、否定和冲突，检索后按MySQL当前版本核验；用冻结条件参考与旧解析输出定位问题，再独立比较召回和重排。不能把有限规则的通过率推广为通用意图理解能力。
