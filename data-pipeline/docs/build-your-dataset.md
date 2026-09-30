# 从开源数据构造 Chinese-Jev 数据集

## 1. 固定来源

先记录原始地址、revision、许可和原始 split。pipeline 不替用户判断数据是否可
再分发，也不会把代码仓库许可证自动当成数据许可证。

建议目录：

```text
my-dataset/
  config.json
  adapter.py
  raw/
    train.jsonl
    dev.jsonl
    test.jsonl
```

可以先运行 `chinese-jev-data init my-dataset` 获得可执行模板。注意模板是最小
平铺布局——单个 `raw.jsonl` 与 `config.json`、`adapter.py` 同级；上面的 `raw/`
子目录是多文件来源时的整理建议，采用时把 source 的 `paths` 指向对应文件即可
（路径相对配置目录，不能越出）。

## 2. 确定监督语义

只转换原标注能够证明的目标：

| 原任务 | 推荐类型 | 约束 |
|---|---|---|
| 单标签分类、单选题 | `choice` | 候选必须互斥，完整保留答键 |
| 相关性、星级等有序标签 | `score` | 每一级必须有不同且可排序的定义 |
| 二元属性或多标签逐项判断 | `noul` | false 必须有明确监督依据 |

数字类别编号不是 Score；未标注候选不是天然负例；多答案题不能直接压成一个
互斥单选标签。

## 3. 编写 adapter

adapter 读取本地固定文件，每次产出一个只有一个 decision 的 case：

```python
{
  "id": "stable-id",
  "state": {"text": "待判断内容"},
  "questions": {
    "decision": {
      "type": "choice",
      "instructions": "选择对应类别。",
      "criteria": {"A": "类别甲", "B": "类别乙"}
    }
  },
  "gold": {
    "decision": {
      "type": "choice",
      "probabilities": {"A": 1.0, "B": 0.0}
    }
  },
  "_meta": {
    "source": "source-name",
    "source_id": "original-row-id",
    "original_id": "original-supervision-id",
    "group_key": "document:123",
    "supervision": "source_hard_label"
  }
}
```

解析不确定的记录调用 `audit(source, id, reason)`。需要单独复核的材料可以写入
`review`，但写入复查日志不等于进入最终 cases，是否产出 case 由 adapter 决定。
例如 SQuAD 的默认模式只写复查材料；如果所有来源都没有产出有效 case，build 会失败。

## 4. 定义分组

`group_key` 是最重要的划分边界。同一母问题、病例、案件、完整对话、共享题干或
文档的所有派生 decision 必须使用同一个 key。

当一条记录连接多个母单位时：

```python
case["_meta"]["integration"] = {
    "link_keys": ["patient:abc", "document:xyz"]
}
```

如果 A 连接 B、B 连接 C，三者会形成同一个 component。官方 test/validation
所在 component 拥有更高优先级，训练侧的关联记录会被排除。

## 5. 配置划分和输出

在 `init` 生成的 `config.json` 中修改以下设置，并保留已有非空 `sources`
及 `adapter_files`。这是配置片段，不是独立配置文件：

```json
{
  "format_version": 1,
  "seed": 42,
  "calibration_fraction": 0.01,
  "validation_fraction": 0.02,
  "test_fraction": 0.02,
  "compression": "gzip",
  "write_notebooks": false
}
```

来源已有官方 dev/test 时，为它们分别建立 `split=validation` 和 `split=test` 的
source。内部比例只作用于 `split=train` 的 component。同一 component 不会被拆分，
采用固定 seed 下的哈希划分，而非标签分层。实际行数比例可能明显偏离组件比例，
尤其有少量大组件时；三种题型也不会自动均衡。标记 `prepartitioned` 的来源要求
这三个内部留出比例全部为零。

所有输入和插件路径相对于配置目录，不能使用绝对路径或 `..` 越出目录。

## 6. 构造与验收

```bash
chinese-jev-data build --config config.json --output release --workers 4
```

并发仅用于独立来源的解析。全局去重、冲突检查、传递分组和 split 所有权仍执行
一次，单个来源不会自动拆分。输出目录必须不存在。`build` 已自动调用全量复读
验收；之后需要单独复查时，再运行 `chinese-jev-data verify --release release`。

重点查看：

- `manifest.json`：输入和实现文件哈希、按 `original_id` 去重的来源单位及 group/decision 计数；
- `verification.json`：全量复读后的 schema、重复、冲突和泄漏检查；
- `rejected.jsonl`：自动拒收原因；
- `review.jsonl`：adapter 标出的复查项。

并行时，worker 的转换日志保存在 `conversion-rejected.jsonl` 和
`conversion-review.jsonl`，另有 `parallel.json`。根目录的 `rejected.jsonl` 和
`review.jsonl` 此时对应全局合并阶段，不能用它们单独统计转换损耗。

## 7. 用真实 BERT tokenizer 编译

对所有准备使用的划分分别运行 `chinese-jev-data prepare`，包括 calibration（如需）。训练配置中的
`max_len`、`head_max_len` 和 tokenizer 必须与这里完全一致。推荐先使用
`state-truncation=reject` 获取真实拒收率，再决定某个任务是否允许截断。

Tokenizer 必须在本地目录或缓存中；prepare 不在线下载。输出为 token items、
拒收日志、编译 manifest 和 tokenizer。它不会改写原 cases，也不会回填原始
manifest 的数量。以编译接受量计训练规模，并检查逐选项任务是否仍有完整母题。
