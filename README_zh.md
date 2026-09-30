# Chinese-Jev：让 System One 模型走向中文任务

[![arXiv](https://img.shields.io/badge/arXiv-2609.36965-B31B1B?logo=arxiv&logoColor=white&style=flat-square)](https://arxiv.org/abs/2609.36965)
[![GitHub](https://img.shields.io/badge/GitHub-Code-181717?logo=github&logoColor=white&style=flat-square)](https://github.com/gulucaptain/Chinese-Jev)
[![Project Page](https://img.shields.io/badge/Project-Page-222222?logo=githubpages&logoColor=white&style=flat-square)](https://gulucaptain.github.io/Chinese-Jev/)
[![CJ-Bench](https://img.shields.io/badge/HF-CJ--Bench-FFD21E?logo=huggingface&logoColor=black&labelColor=FFF4BC&style=flat-square)](https://huggingface.co/datasets/bbldCVer-hf/CJ-Bench)
[![CJ-Dataset](https://img.shields.io/badge/HF-CJ--Dataset-FFD21E?logo=huggingface&logoColor=black&labelColor=FFF4BC&style=flat-square)](https://huggingface.co/datasets/bbldCVer-hf/CJ-Dataset#english)

[English](README.md) | **中文**

Chinese-Jev 提供类型化决策数据构建、编码器模型微调、概率校准与留出集评估代码。
仓库包含序列格式、模型实现和训练目标；**不包含预训练权重或真实数据集**。
运行训练流水线需要另行准备兼容的模型包和数据集。

这是独立的社区项目，与 TypeSafe AI 无关联，也未获其背书。“Jev”和“System One Model”
用于指称 TypeSafe AI 的产品及术语；本仓库不包含其专有模型、权重或训练数据。

```
原始数据 ──(data-pipeline)──> 发布数据集 ──(prepare)──> token 缓存 + dataset.json
                                                                  │
                                            训练配置 ──(run)──> 微调后的模型包
```

## 模型家族

[论文](https://arxiv.org/abs/2609.36965)介绍了一个通用模型，以及面向三个专业领域分别微调的版本。
模型权重计划开源；Hugging Face 模型链接将在发布后补充。

| 版本 | 适用范围 | 🤗 Hugging Face |
|---|---|---|
| 🌐 通用 | 中文通用决策任务 | ⏳ 待开源 |
| 🩺 医疗 | 医疗领域微调版本 | ⏳ 待开源 |
| ⚖️ 法律 | 法律领域微调版本 | ⏳ 待开源 |
| 💼 金融 | 金融领域微调版本 | ⏳ 待开源 |

## 安装

```bash
python -m pip install -e .                 # 微调流水线（torch、transformers）
python -m pip install -e ./data-pipeline   # 可选：数据集构建命令行工具
```

## 预训练模型与检查点格式

训练从一个模型包（bundle）开始，即包含 `encoder/`、`tokenizer/`、
`model.safetensors` 和 `rl_agent_config.json` 的目录。中文及其他多语言数据推荐使用
**laya-multilingual**：基于 mmBERT-base 的 laya 模型，具有 3.22 亿（322M）参数，
支持 100 多种语言，最大位置长度为 8,192。

```bash
hf download convaiinnovations/laya-multilingual --local-dir models/multilingual
```

模型卡：<https://huggingface.co/convaiinnovations/laya-multilingual>。
处理英文语料时，也可以用同样的方式使用仅面向英文的 `convaiinnovations/laya`（ModernBERT-large）。
之前微调生成的模型包（`<work_dir>/bundle`）也可以作为训练起点，其目录结构与初始模型包一致。

## 快速开始

需要先准备一个已发布的数据集（JSONL 格式的决策样本，可通过 `data-pipeline/` 构建，
或参阅 [docs/new-dataset.md](docs/new-dataset.md)），以及上文所述的模型包。

```bash
# 1. 用一份配置描述整个任务：复制 configs/template.json，设置数据集相关字段
#    （dataset_params、cache_dir），并将 bundle 指向模型包；再设置运行相关字段
#    （work_dir、max_len、epochs 等）。将 head_max_len 保持为 null，
#    prepare 会根据数据测量所需长度。

# 2. 准备数据：测量序列长度相关信息，对所有划分进行分词，在缓存根目录写入
#    dataset.json，并将测量值回填到配置中。
chinese-jev prepare --config configs/my-dataset.json

# 3. 如有需要，在同一份配置中调整 max_len 和其他超参数。prepare 输出的长度表
#    可用于选择 max_len。随后用一条命令完成分词（复用缓存）、训练、校准和评估。
chinese-jev run --config configs/my-dataset.json

# 多 GPU 运行，并创建带时间戳的运行目录：
bash scripts/train.sh --config configs/my-dataset.json --gpus 0,1,2,3
```

微调后的模型包保存在 `<work_dir>/bundle`，目录结构与输入模型包相同，
并包含针对各问题类型拟合的温度参数。

配置中的相对路径以**项目根目录**为基准解析，即从配置文件所在位置向上查找时，
最近的包含 `pyproject.toml` 或 `.git` 的祖先目录。
例如，`configs/` 中的配置将路径写为 `runs/cache/medical` 时，
实际指向 `<repo>/runs/cache/medical`。

## 仓库结构

| 路径 | 说明 |
|---|---|
| [`src/chinese_jev/`](src/chinese_jev/) | 微调流水线：数据准备 → 训练 → 校准 → 评估 |
| [`data-pipeline/`](data-pipeline/) | 数据集构建：通过数据源适配器生成经过校验、去重和划分的发布数据集 |
| [`configs/`](configs/) | 每份配置描述数据集和一次运行，驱动所有阶段（格式见 `template.json`） |
| [`docs/`](docs/) | [流水线指南](docs/pipeline.md)、[添加数据集](docs/new-dataset.md) |
| [`tests/`](tests/) | 基于断言的检查，无需 pytest 即可运行 |
| [`scripts/train.sh`](scripts/train.sh) | 一条命令启动训练，支持多 GPU，并为每次运行创建独立目录 |

## 评估指定模型

`evaluate` 默认评估当前运行训练得到的模型包。使用 `--bundle` 可以在同一套缓存和指标上
评估任何符合模型包结构的目录，例如另一次运行的输出、某个 epoch 的检查点，
或用于微调前后对比的基础模型：

```bash
# 尚未微调的基础模型：
chinese-jev evaluate --config configs/my.json --bundle models/multilingual
# 当前运行的指定检查点：
chinese-jev evaluate --config configs/my.json --bundle runs/my-run/checkpoints/epoch_00000002
# 在当前运行的数据划分上评估另一次运行的最终模型包：
chinese-jev evaluate --config configs/my.json --bundle runs/other-run/bundle
```

结果写入 `<work_dir>/evaluation.json`，包含准确率（accuracy）、期望校准误差（ECE）、
对数评分（log score），以及按问题类型统计的结果。

## 详细命令

常用入口为 `prepare`（测量并构建缓存）、`run`（训练、校准、评估）和 `evaluate`
（评估指定模型包）。独立阶段、配置字段、分布式训练及启动脚本的参数见
[训练流水线文档](docs/pipeline.md)；CLI 也提供 `--help`。

## 数据集格式

JSONL 的每一行对应一个决策，包含 `state`（材料正文）、一个问题（`choice`、`noul` 或 `score`）、
问题选项上的目标概率分布，以及记录来源信息的 `_meta`。
格式、校验规则和数据构建流水线详见
[`data-pipeline/README.md`](data-pipeline/README.md)。

## 验证与当前边界

先安装项目依赖，再运行训练流水线检查：

```bash
python -m pip install -e .
python tests/test_pipeline.py
CHINESE_JEV_TEST_BUNDLE=/path/to/bundle python tests/test_pipeline.py
cd data-pipeline
python tests/test_pipeline.py
python tests/test_bert_compiler.py
```

可选的 `CHINESE_JEV_TEST_BUNDLE` 用于启用分词器往返检查。
仓库不包含 Chinese-Jev 微调权重，也尚未列出已验证的平台矩阵。
[论文](https://arxiv.org/abs/2609.36965)报告了
[CJ-Bench](https://huggingface.co/datasets/bbldCVer-hf/CJ-Bench)结果；这些结果依赖论文中的评测设置。
在自己的留出数据上仍需验证准确率与校准水平。数据发布还需完成
[发布检查表](data-pipeline/docs/release-checklist.md)中结构校验以外的检查。

## 许可证与来源边界

本仓库的原创代码采用 [Apache-2.0](LICENSE) 许可。
[推荐的基础模型](https://huggingface.co/convaiinnovations/laya-multilingual)
是独立发布的资源，有其自身的许可和署名要求。原始数据集各自保留原有条款；
允许用于训练不等于允许重新分发原始记录或派生数据。仓库代码许可证不改变
外部模型权重或数据的许可。

## 论文引用

如果您在研究中使用 Chinese-Jev、CJ-Bench 或 CJ-Dataset，请引用：

```bibtex
@misc{wang2026chinesejev,
  title         = {Chinese-Jev: Bringing System One Model to Chinese-Language Tasks},
  author        = {Zexiao Wang and Zihao Zhang and Xudong Wang and Pan Wang and Ziyi Ye and Haoyu Zhao and Zuxuan Wu and Shuicheng Yan},
  year          = {2026},
  eprint        = {2609.36965},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2609.36965}
}
```
