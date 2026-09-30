# Adapter 契约

Adapter 负责原始来源的字段、监督和分组映射；已有适配格式可直接通过配置调用。
跨源近重复、配额选择及完整母题筛选等额外策略需要项目另行实现。

```python
adapter(source, paths, audit, review) -> Iterable[case]
```

每个 case 只能包含一个 decision，并保存以下元数据：

```text
source, source_id, original_id, group_key,
source_family, task_family, domain, source_split, original_split,
source_url, revision, license, supervision
```

其中 `source_family`、`task_family`、`domain` 与 `source_url` 取自 source 配置
（配置里的键名是 `url`，落到 `_meta` 时写作 `source_url`），由构建引擎注入，
adapter 使用 `make_case` 时无需自行填写。全局合并阶段还会追加两个字段并改写一个：
`split`（最终划分）、`source_group_key`（adapter 声明的原始 group_key），
而 `_meta.group_key` 会被改写为传递关联组件的 ID。

其中：

- `source_id` 定位原始记录或监督；
- `original_id` 将同一原监督派生的多个视图关联起来；
- `group_key` 隔离所有不能跨 split 的相关记录；
- `_meta.integration.link_keys` 可选，用于连接跨文件、跨来源的同一病例、案件、
  问题、对话或文档；共享核心会计算这些锚点的传递闭包；
- `task_family` 描述监督语义，不能只填写数据集名称；
- `supervision` 区分原生硬标签、软标签、规则派生或其他来源。

Adapter 必须遵守以下规则：

1. 待预测目标及其标注解释不得泄漏到 `state` 或 `questions`；待评价的候选回答、
   任务已提供的证据可以作为输入。对话任务不得使用当时尚不可见的未来轮次。
2. 未标注候选不能自动解释为负例。
3. 类别编号不能因为是数字就转换成 Score。
4. 多标签任务不能直接改成互斥单选。
5. Score 的每一级必须具有不同、可排序的含义。
6. 中英文输入与候选尽量保持同一语言。
7. 不确定、截断或无法可靠解析的记录调用 `audit` 拒收。

每个 adapter 至少测试正常样本、否定语义、边界标签、损坏记录、分组一致性和候选顺序。
