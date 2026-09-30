# 架构与数据流

General 与垂域数据共享同一条构造骨架。领域差异收敛为配置与插件，而不是复制新的主流程。

```text
固定原始版本
  → 来源 adapter
  → schema 与概率格式校验
  → 临时候选池、关联组件与精确输入去重
  → 冲突排除、留出保护与组件级划分
  → train / calibration / validation / test cases + manifest
  → build 自动调用独立复读验收
  → 可选 prepare：BERT tokenizer 编译与逐 decision 长度拒收
  → items + 独立编译报告
```

## 共享部分

| 层 | 统一行为 |
|---|---|
| Case | `state / questions / gold / _meta` |
| 监督 | `choice / score / noul` 与概率目标 |
| 谱系 | 来源、原 ID、母单位、版本、许可与监督来源 |
| 分组划分 | `group_key` 与显式 `link_keys` 的关联组件不跨 split |
| 去重 | 序列化 state 与有序 questions 的精确去重；相同输入的冲突 gold 全部排除 |
| 编译 | 使用包内 `checked_sequence` 与实际 BERT tokenizer |
| 交付 | manifest、审计文件、文件哈希与独立复读 |

## 项目策略

| 策略 | 需要按数据集确定 |
|---|---|
| 来源读取 | 原始字段、文件和官方 split |
| 任务语义 | 原标签能够证明的判断范围 |
| 分组锚点 | 问题、病例、案件、文档或完整对话 |
| 选择策略 | 在配置来源和 adapter 规则中确定纳入范围；构造后按 manifest 审计分布 |
| 输入预算 | tokenizer、总长度、head 长度与截断规则 |
| 发布权限 | 许可、隐私、再分发和访问控制 |

构造程序负责执行已经确定的规则。标签含义、负例成立条件、分组单位和许可边界应先经过研究审查，再固化为 adapter、配置与测试。

内部划分采用组件哈希，不实施标签分层或题型配额。`prepare` 是独立阶段，不回写
cases 和构造 manifest，也不自动保证拒收后仍保留完整多选母题。最终训练数量与
任务完整性需按编译结果确认。

## 执行后端

所有领域使用同一个 `build` 命令：

- `--workers 1`：串行转换，适合开发与回归；
- `--workers 2..16`：来源级并发转换，父进程执行唯一的全局合并。

单个来源不会按 `workers` 自动拆成多个分片。远端下载、断点上传和平台认证不在
当前执行后端内。

## 并发边界

当前实现并发执行各来源的原始解析与 case 格式验证。以下步骤由父进程统一执行：

- 跨来源谱系连接；
- 精确重复和冲突 gold；
- train/calibration/validation/test 所有权；
- 最终文件写出和统计汇总。

因此并行后端先生成临时 case spool，再进行一次全局合并。adapter 提供的
`group_key` 与 `integration.link_keys` 在全局阶段计算传递闭包，不能在 worker
内部独立决定最终 split。

并行临时 case 文件未压缩，合并和验收使用 SQLite；关联锚点及部分 adapter 的
辅助表仍驻留内存。磁盘和内存需求取决于转换后行数、关联组数及来源格式，不能
仅按最终 gzip 大小估算。
