# Chinese-Jev: Bringing System One Model to Chinese-Language Tasks

[![arXiv](https://img.shields.io/badge/arXiv-2609.36965-B31B1B?logo=arxiv&logoColor=white&style=flat-square)](https://arxiv.org/abs/2609.36965)
[![GitHub](https://img.shields.io/badge/GitHub-Code-181717?logo=github&logoColor=white&style=flat-square)](https://github.com/gulucaptain/Chinese-Jev)
[![Project Page](https://img.shields.io/badge/Project-Page-222222?logo=githubpages&logoColor=white&style=flat-square)](https://gulucaptain.github.io/Chinese-Jev/)
[![CJ-Bench](https://img.shields.io/badge/HF-CJ--Bench-FFD21E?logo=huggingface&logoColor=black&labelColor=FFF4BC&style=flat-square)](https://huggingface.co/datasets/bbldCVer-hf/CJ-Bench)
[![CJ-Dataset](https://img.shields.io/badge/HF-CJ--Dataset-FFD21E?logo=huggingface&logoColor=black&labelColor=FFF4BC&style=flat-square)](https://huggingface.co/datasets/bbldCVer-hf/CJ-Dataset#english)

**English** | [中文](README_zh.md)

Chinese-Jev provides code to construct typed-decision datasets, fine-tune an
encoder-based model, calibrate its probabilities, and evaluate it on held-out data.
The sequence format, model implementation, and training objective are included here;
**pretrained weights and real-world datasets are not included**. Bring a compatible
model bundle and a dataset to run the training pipeline.

This is an independent community project, not affiliated with or endorsed by
TypeSafe AI. “Jev” and “System One Model” refer to TypeSafe AI's product and terminology;
this repository does not contain its proprietary model, weights, or training data.

```
raw sources ──(data-pipeline)──> released dataset ──(prepare)──> token cache + dataset.json
                                                                        │
                                              training config ──(run)──> fine-tuned bundle
```

## Model Family

The [paper](https://arxiv.org/abs/2609.36965) describes a general model followed by
separate fine-tunes for three specialized domains. Model weights are planned for open
release; Hugging Face model links will be added when available.

| Variant | Focus | 🤗 Hugging Face |
|---|---|---|
| 🌐 General | General-purpose Chinese decision tasks | ⏳ To be open-sourced |
| 🩺 Medical | Medical-domain fine-tune | ⏳ To be open-sourced |
| ⚖️ Legal | Legal-domain fine-tune | ⏳ To be open-sourced |
| 💼 Financial | Financial-domain fine-tune | ⏳ To be open-sourced |

## Install

```bash
python -m pip install -e .                 # the fine-tuning pipeline (torch, transformers)
python -m pip install -e ./data-pipeline   # optional: the dataset construction CLI
```

## Pretrained Model and Checkpoint Format

Training starts from a *bundle*: a directory with `encoder/`, `tokenizer/`,
`model.safetensors` and `rl_agent_config.json`. The recommended base for Chinese (and any
multilingual) data is **laya-multilingual** — the laya build of mmBERT-base (322M
parameters, 100+ languages, positions up to 8,192):

```bash
hf download convaiinnovations/laya-multilingual --local-dir models/multilingual
```

Model card: <https://huggingface.co/convaiinnovations/laya-multilingual>. The English-only
`convaiinnovations/laya` (ModernBERT-large) works the same way for English corpora. A
bundle produced by a previous fine-tune (`<work_dir>/bundle`) is also a valid starting
point — the layout is identical by construction.

## Quickstart

You need a released dataset (JSONL cases; build one with `data-pipeline/` or see
[docs/new-dataset.md](docs/new-dataset.md)) and a bundle (above).

```bash
# 1. Describe the whole job once: copy configs/template.json, point the dataset half
#    (dataset_params, cache_dir) at your release and the bundle at your model, and pick
#    the run half (work_dir, max_len, epochs, ...). Leave head_max_len as null —
#    prepare measures it from the data.

# 2. Prepare: measure geometry, tokenize every split, write dataset.json at the cache
#    root. The measured values are filled back into the config.
chinese-jev prepare --config configs/my-dataset.json

# 3. Adjust max_len / hyperparameters in the same file if needed (prepare printed a
#    length table to pick max_len from), then run: tokenize (reusing the cache),
#    train, calibrate, evaluate — in one command.
chinese-jev run --config configs/my-dataset.json

# Multi-GPU, timestamped run directory:
bash scripts/train.sh --config configs/my-dataset.json --gpus 0,1,2,3
```

The fine-tuned bundle lands at `<work_dir>/bundle`, with the same layout as the input
bundle, plus fitted per-question-type temperatures.

Relative paths in a config resolve against the **project root** (the nearest ancestor of
the config file with a `pyproject.toml` or `.git`), so a config in `configs/` saying
`runs/cache/medical` means `<repo>/runs/cache/medical`.

## Repository layout

| path | what it is |
|---|---|
| [`src/chinese_jev/`](src/chinese_jev/) | the fine-tuning pipeline: prepare → train → calibrate → evaluate |
| [`data-pipeline/`](data-pipeline/) | dataset construction: source adapters → validated, deduplicated, split releases |
| [`configs/`](configs/) | one config per dataset+run, driving every stage (`template.json` documents the shape) |
| [`docs/`](docs/) | [pipeline guide](docs/pipeline.md), [adding a dataset](docs/new-dataset.md) |
| [`tests/`](tests/) | assert-based checks, runnable without pytest |
| [`scripts/train.sh`](scripts/train.sh) | one-command (multi-GPU) launcher with per-run directories |

## Evaluating a specific model

`evaluate` scores the run's own trained bundle by default. `--bundle` scores any
bundle-shaped directory against the same cache and metrics — another run's output, an
epoch checkpoint, or the base model for a before/after comparison:

```bash
# the base model, before any fine-tuning:
chinese-jev evaluate --config configs/my.json --bundle models/multilingual
# a specific checkpoint of this run:
chinese-jev evaluate --config configs/my.json --bundle runs/my-run/checkpoints/epoch_00000002
# another run's final bundle, on this run's splits:
chinese-jev evaluate --config configs/my.json --bundle runs/other-run/bundle
```

Results go to `<work_dir>/evaluation.json` (accuracy, ECE, log score, per question type).

## Detailed commands

The main entry points are `prepare` (measure and build the cache), `run` (train,
calibrate, and evaluate), and `evaluate` (score a chosen bundle). For individual stages,
configuration keys, distributed training, and launcher flags, see the
[training pipeline guide](docs/pipeline.md). The CLI also exposes `--help` for each stage.

## Dataset format

One decision per JSONL line: `state` (the material), one question (`choice`, `noul`, or
`score`), a gold probability distribution over its options, and `_meta` provenance. The
format, its validation rules, and the construction pipeline are documented in
[`data-pipeline/README.md`](data-pipeline/README.md).

## Verification and current limits

Install the project dependencies before running the training-pipeline checks:

```bash
python -m pip install -e .
python tests/test_pipeline.py
CHINESE_JEV_TEST_BUNDLE=/path/to/bundle python tests/test_pipeline.py
cd data-pipeline
python tests/test_pipeline.py
python tests/test_bert_compiler.py
```

The optional bundle variable enables tokenizer round-trip checks. The repository does
not bundle trained Chinese-Jev weights or a tested platform matrix.
The [paper](https://arxiv.org/abs/2609.36965) reports results on
[CJ-Bench](https://huggingface.co/datasets/bbldCVer-hf/CJ-Bench); those results depend on
the paper's evaluation setup. Validate accuracy and calibration on your own held-out data.
See the [data release checklist](data-pipeline/docs/release-checklist.md) for
source-specific checks beyond structural verification.

## License and source boundaries

Original code in this repository is released under [Apache-2.0](LICENSE). The
[recommended base model](https://huggingface.co/convaiinnovations/laya-multilingual)
is a separate release with its own license and attribution. Source datasets retain
their own terms; permission to train on a dataset does not automatically permit
redistribution of its records or derived releases. The repository's code license does
not relicense external weights or data.

## Citation

If you use Chinese-Jev, CJ-Bench, or CJ-Dataset in research, please cite:

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
