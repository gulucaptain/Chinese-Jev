# Chinese-Jev Data Pipeline

## English

**English** | [中文](#中文)

Build Chinese-Jev decision datasets from local, labeled sources. Source adapters map
records and supervision to a shared format; the common engine handles validation,
grouped splits, exact deduplication, and release files. General and domain-specific
tasks use the same build command, with source-specific label and grouping rules.

Optional BERT compilation is included in the package. It turns decisions into
encoder inputs, candidate markers, and target distributions using a local tokenizer.

### Quick start

Python 3.9 or later. From this `data-pipeline/` directory (not the repository root —
that installs the fine-tuning package instead and leaves you without the
`chinese-jev-data` command):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
```

For BERT compilation, also install `python -m pip install '.[prepare]'` from this
directory. The conversion and verification commands use the standard library.

```bash
chinese-jev-data init my-chinese-jev-dataset
cd my-chinese-jev-dataset
chinese-jev-data build --config config.json --output release --workers 1
```

`init` creates `adapter.py`, `config.json`, and `raw.jsonl`. The runnable example
produces three training decisions, one per type. Its holdout fractions are zero,
so calibration, validation, and test files are empty. Replace the example inputs
and configure splits before building a real dataset — the fine-tuning pipeline's
`chinese-jev prepare` refuses a split with no cases.

`build` automatically runs a full-file verifier. To recheck an existing release:

```bash
chinese-jev-data verify --release release
```

Build and prepare outputs must use new directories.

A verified release is what the fine-tuning pipeline consumes: point a dataset config's
`data_root` at it and run `chinese-jev prepare` (see [`../docs/pipeline.md`](../docs/pipeline.md))
to tokenize it and produce the `dataset.json` a training run references.

### Decision format

Each released JSONL record contains `id`, `state`, `questions`, `gold`, and `_meta`,
with exactly one question. Models read `state` and `questions`; `gold` supplies the
target distribution, and `_meta` holds provenance and grouping information.

| Type | Criteria | Probability keys |
|---|---|---|
| `choice` | Named candidate object | The same candidate keys, in their declared order |
| `noul` | Explicit false/true descriptions | `"false"`, `"true"` |
| `score` | Ordered list of levels | `"0"`, `"1"`, …, indexing that list |

Targets may be hard or soft distributions. Category IDs do not imply ordered
scores, and missing annotations do not establish negative labels. See the
[example records](examples/basic/raw.jsonl) and their
[adapter](examples/basic/adapter.py) for all three types.

### Adapters and configuration

The package registers **48 adapter entries**, including C³, T2Ranking, CBLUE,
medical exams and dialogues, CAIL, JEC-QA, FinFE, and news/review classification.
Each entry expects a particular source schema and may require auxiliary label
files or task options. SQuAD defaults to review-only output; its answerability
mode requires an explicit `is_impossible` label.

```bash
chinese-jev-data list-adapters
chinese-jev-data list-adapters --config config.json
```

For a new source, implement the same interface as the generated adapter:

```python
def convert(source, paths, audit, review):
    # Parse records and yield cases; report rejected records through audit.
    ...

ADAPTERS = {"my_dataset": convert}
```

Register the file in `adapter_files`. A source declares `name`, `adapter`, `split`,
`url`, `revision`, `license`, `source_family`, `task_family`, and `domain`, plus its
input paths. T2Ranking uses `queries`, `qrels`, and `collection` instead of `paths`,
and accepts optional `query_fraction` (hash-sample queries, 0–1, default 1.0) and
`sample_seed` (default 42).
Paths are relative to the config directory and must stay within it. Adapter files
are executable Python plugins. Full configuration: [example](examples/basic/config.json).

### Splits, deduplication, and parallelism

The engine joins shared `group_key` values and optional
`_meta.integration.link_keys` into connected components. A component's highest
declared split takes priority: `test > validation > calibration > train`.
Lower-priority rows are excluded, not moved into the held-out split.

For remaining train components, set `calibration_fraction`, `validation_fraction`,
and `test_fraction`, whose sum must be less than one. Assignment is deterministic
for fixed inputs and seed. It is a component-level hash split, without label
stratification or type balancing; row proportions can differ from these fractions.
Prepartitioned sources require all three fractions to be zero.

Deduplication compares serialized `state` and ordered `questions`. Identical inputs
with conflicting targets are excluded. Candidate reordering and semantic
near-duplicates require additional task-specific handling.

`--workers 2` through `16` convert source entries in separate processes, followed
by one global merge. A single large source is not automatically sharded. Parallel
runs also write temporary, uncompressed case files, so reserve space beyond the
final gzip size.

### Outputs and BERT compilation

The example configuration writes:

```text
release/
  train.cases.jsonl.gz
  calibration.cases.jsonl.gz
  validation.cases.jsonl.gz
  test.cases.jsonl.gz
  manifest.json
  verification.json
  rejected.jsonl
  review.jsonl
```

Parallel builds additionally produce `parallel.json`, `conversion-rejected.jsonl`,
and `conversion-review.jsonl`. Inspect those conversion logs as well as the global
merge logs. Two config keys change the listing above: `compression` defaults to
`"none"` — plain `.cases.jsonl` without the `.gz` suffix; the generated template
sets `"gzip"`, which the fine-tuning side's default
`file_template: "{split}.cases.jsonl.gz"` assumes — and `write_notebooks` defaults
to `true`, adding a human-readable `<split>.notebook.jsonl(.gz)` review companion
per split (the template sets `false`). `manifest.json` records inputs, implementation
versions by hash, counts, and split policy. `verification.json` independently
checks schema, probability targets, IDs, exact inputs, and group isolation; it
does not verify label meaning or tokenizer fit.

With a compatible tokenizer already available locally, compile each split:

```bash
chinese-jev-data prepare \
  --input release/train.cases.jsonl.gz \
  --tokenizer /path/to/bert-tokenizer \
  --max-len 1024 \
  --head-max-len 256 \
  --state-truncation reject \
  --output release/train-tokens
```

Use the tokenizer and limits intended for training. The compiler checks Chinese-Jev's
BERT layout and rejects truncated instructions or candidates. `reject` also
rejects truncated state; `right` explicitly permits it. The output contains
`items.jsonl`, `rejected.jsonl`, a compilation `manifest.json`, and the tokenizer.
`--pt-shard-size 10000` additionally writes lists of items as PyTorch shards.

Compilation filters individual decisions without changing the source cases or
build manifest. Use its accepted counts for training. Tasks requiring complete
option bundles need a separate bundle check after compilation.

### Documentation and scope

Detailed guides are currently in Chinese:
[architecture](docs/architecture.md), [build guide](docs/build-your-dataset.md),
[adapters](docs/adapters.md), [adapter contract](docs/adapter-contract.md), and
[release checks](docs/release-checklist.md).

Data acquisition, annotation policy, redistribution rights, sampling quotas, and
training remain project decisions. The package supplies local conversion,
structural verification, and optional BERT compilation. Code is licensed under
[Apache-2.0](LICENSE); source datasets retain their own terms.

---

## 中文

[English](#english) | **中文**

将本地带监督数据转换为 Chinese-Jev 决策数据。Adapter 负责原始字段与标签映射，
共享核心负责格式校验、分组划分、精确去重及文件交付。通用与垂域任务使用同一
构造入口，按来源配置监督含义和分组规则。

包内提供可选的 BERT 输入编译，使用本地 tokenizer 将决策转换为编码器输入、
候选标记位置和目标分布。

### 快速开始

需要 Python 3.9 或以上版本。在本 `data-pipeline/` 目录执行（不要在上层仓库根目录
执行——那会安装微调包，得不到 `chinese-jev-data` 命令）：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
```

如需 BERT 编译，在此目录额外执行 `python -m pip install '.[prepare]'`。
数据转换与验收仅依赖标准库。

```bash
chinese-jev-data init my-chinese-jev-dataset
cd my-chinese-jev-dataset
chinese-jev-data build --config config.json --output release --workers 1
```

`init` 生成 `adapter.py`、`config.json`、`raw.jsonl`。示例可直接运行，产出三条
训练 decisions，每种类型一条；默认留出比例均为零，其他三个划分文件为空。
构建真实数据前，请替换示例输入并配置划分——微调管线的 `chinese-jev prepare`
会拒绝没有任何 case 的划分。

`build` 已自动完成全量复读验收。需要重新检查已有结果时再执行：

```bash
chinese-jev-data verify --release release
```

`build` 和 `prepare` 均要求输出目录尚不存在。

验收通过的 release 即微调管线的输入：把数据集配置的 `data_root` 指向它，运行
`chinese-jev prepare`（见 [`../docs/pipeline.md`](../docs/pipeline.md)）完成分词并产出
训练配置所引用的 `dataset.json`。

### 决策格式

每行包含 `id / state / questions / gold / _meta`，且只有一个问题。模型读取
`state` 和 `questions`；`gold` 保存目标分布，`_meta` 保存来源与分组信息。

| 类型 | Criteria | 概率键 |
|---|---|---|
| `choice` | 命名候选对象 | 对应候选键，沿用声明顺序 |
| `noul` | 明确的 false/true 描述 | `"false"`、`"true"` |
| `score` | 等级有序列表 | `"0"`、`"1"` 等列表索引 |

支持硬、软概率目标。类别编号不代表有序评分，缺失标注也不代表负例。三种类型的
完整转换见[原始示例](examples/basic/raw.jsonl)与 [adapter](examples/basic/adapter.py)。

### Adapter 与配置

包内注册 **48 个 adapter 入口**，包括 C³、T2Ranking、CBLUE、医学考试与对话、
CAIL、JEC-QA、FinFE 及新闻、评论分类。各入口对应具体原始格式，部分还需标签表
或任务参数。SQuAD 默认只输出复查材料；可回答性模式要求显式的 `is_impossible` 标签。

```bash
chinese-jev-data list-adapters
chinese-jev-data list-adapters --config config.json
```

新来源沿用生成的 adapter 接口：

```python
def convert(source, paths, audit, review):
    # 解析原记录并逐条 yield case；拒收记录通过 audit 报告。
    ...

ADAPTERS = {"my_dataset": convert}
```

在 `adapter_files` 注册文件。每个来源声明 `name / adapter / split / url /
revision / license / source_family / task_family / domain` 及输入路径。
T2Ranking 使用 `queries / qrels / collection` 三个路径替代 `paths`，另支持可选的
`query_fraction`（按 query 哈希抽样，0 到 1，默认 1.0）与 `sample_seed`（默认 42）。
路径相对于配置目录，不能越出该目录；插件文件会作为 Python 代码执行。
完整配置见[示例](examples/basic/config.json)。

### 划分、去重与并行

共享 `group_key` 或 `_meta.integration.link_keys` 的记录会形成传递关联组件。
组件采用最高的已声明划分优先级：`test > validation > calibration > train`。
低优先级的关联记录被排除，不会挪入留出集。

其余 train 组件按 `calibration_fraction / validation_fraction / test_fraction`
确定性分配，三者之和须小于一。这是组件级哈希划分，不是标签分层或题型均衡抽样；
实际行数比例可能偏离配置值。使用 `prepartitioned` 来源时，三项比例必须全部为零。

精确去重比较序列化的 `state` 和有序 `questions`；相同输入的冲突目标全部排除。
候选换序和语义近重复需按任务另行处理。

`--workers 2` 至 `16` 按来源条目多进程转换，随后进行一次全局合并。
单个大来源不会自动拆分；并行过程会写出未压缩临时 cases，需要为此预留磁盘。

### 输出与 BERT 编译

示例配置输出四个 `*.cases.jsonl.gz` 划分文件，以及 `manifest.json`、
`verification.json`、`rejected.jsonl` 和 `review.jsonl`。
并行构造另有 `parallel.json`、`conversion-rejected.jsonl`、
`conversion-review.jsonl`，应同时检查转换日志与全局合并日志。
两个配置键会改变上述清单：`compression` 默认为 `"none"`（输出无 `.gz` 后缀的
`.cases.jsonl`；模板设为 `"gzip"`，微调侧默认的
`file_template: "{split}.cases.jsonl.gz"` 也以此为前提）；`write_notebooks`
默认为 `true`，会为每个划分额外产出便于人工复查的 `<split>.notebook.jsonl(.gz)`
（模板设为 `false`）。

`manifest.json` 记录输入、实现文件哈希、数量和划分策略；`verification.json`
独立检查格式、概率、ID、精确输入及分组隔离，不判断标签语义或分词后是否超长。

Tokenizer 在本地准备好后，对每个划分单独编译：

```bash
chinese-jev-data prepare \
  --input release/train.cases.jsonl.gz \
  --tokenizer /path/to/bert-tokenizer \
  --max-len 1024 \
  --head-max-len 256 \
  --state-truncation reject \
  --output release/train-tokens
```

使用与训练一致的 tokenizer 和长度上限。编译器核对 Chinese-Jev BERT 布局，问题或候选
会被截断时拒收；`reject` 同时拒收超长 state，`right` 则显式允许右截断。
输出包含 `items.jsonl`、`rejected.jsonl`、编译阶段的 `manifest.json` 及 tokenizer。
增加 `--pt-shard-size 10000` 可额外输出保存 item 列表的 PyTorch 分片。

编译逐 decision 筛选，不会回写原 cases 或构造 manifest。训练数量应以编译后的
接受量为准；要求完整选项包的任务，还需在编译后核对母题完整性。

### 文档与范围

[通用架构](docs/architecture.md) · [构造指南](docs/build-your-dataset.md) ·
[Adapter 清单](docs/adapters.md) · [Adapter 契约](docs/adapter-contract.md) ·
[发布验收](docs/release-checklist.md)

来源获取、标注规则、再分发权限、采样配额和模型训练由具体项目安排。本包提供
本地转换、结构验收和可选 BERT 编译。代码使用 [Apache-2.0](LICENSE)，源数据
保留各自的许可条件。
