# The fine-tuning pipeline

The pipeline turns a released decision dataset (see [`../data-pipeline/`](../data-pipeline/))
into a fine-tuned typed decision model. One config drives it — the dataset fields
(`dataset`, `dataset_params`, `cache_dir`, `head_max_len`) feed `prepare`, the run
fields (`work_dir`, `max_len`, hyperparameters) feed `run`:

```
config  ──(prepare)──>  token cache + dataset.json   (measured values filled back)
   │
   └────(run)──>  fine-tuned bundle
```

`prepare` is paid once per dataset; every training run — every `max_len`, every
hyperparameter — reuses its cache. Several run configs can share one cache by naming
the same `cache_dir` (or by pointing `dataset_info` at the prepared `dataset.json`,
which replaces the inline dataset fields). The lower-level stages
(`stats`, `tokenize`, `train`, `calibrate`, `evaluate`) remain independently runnable.

## Paths in configs

Relative paths resolve against the **project root**: the nearest ancestor of the config
file containing a `pyproject.toml` or `.git`. A config in `configs/` can therefore say
`runs/cache/my-dataset` and mean `<repo>/runs/cache/my-dataset` — not `configs/runs/...`. A
config outside any project resolves against its own directory. CLI flags (`--cache-dir`,
`--work-dir`) resolve against the shell's cwd, where they were typed.

## Stages

### `prepare` — the whole data side, once per dataset

```bash
chinese-jev prepare --config configs/my.json
```

Does four things:

1. If the config's `head_max_len` is `null`, measures the head budget from the data
   (see `stats` below) and uses the measured value.
2. Tokenizes every split into the binary cache (see `tokenize` below).
3. Writes **`dataset.json`** at the cache root: the dataset's identity (adapter and
   parameters), the measured `head_max_len`, per-split item counts and length
   percentiles, the tokenizer fingerprint, and a recommended `max_len`.
4. Fills the measured values back into the config file (a null `head_max_len`, and
   `max_len` when the key is absent), so the same file is ready for `run`.

A separate training config can instead reference the prepared dataset with a single key:

```json
{ "dataset_info": "runs/cache/my-dataset/dataset.json",
  "bundle": "models/multilingual", "work_dir": "runs/my-run", "max_len": 4096 }
```

`dataset_info` supplies `dataset`, `dataset_params`, `cache_dir` (the file's own
directory) and `head_max_len`; a config that sets both `dataset_info` and
`dataset`/`dataset_params` is refused, so there is exactly one source of truth for what
the data is. Moving the cache directory moves the dataset — the reference is the file,
not a convention.

### `stats` — measure the geometry without writing a cache

```bash
chinese-jev stats --config configs/my.json [--sample 200000]
```

Walks the corpus and reports the *head budget*: the prompt width the widest option set in
the data actually needs. If the config left `head_max_len` as `null`, the measured value
is **written back into the config file**; an explicit value is compared against the
measurement and warned about, never overwritten.

This number must be measured, never guessed. `build_sequence` does not fail when the
options exceed `head_max_len`; it silently re-truncates every option to a few tokens, and
the model trains on options it cannot tell apart. The tokenize stage replicates the budget
check and *skips* such questions loudly instead — but only a measured `head_max_len` keeps
that skip count at zero. `prepare` and `tokenize` run the same measurement automatically
when `head_max_len` is `null` (an existing cache's measured value is adopted instead of
re-measuring).

### `tokenize` — build the token cache

```bash
chinese-jev tokenize --config configs/my.json
```

Tokenizes every split into a memory-mappable binary cache and prints the length
distribution table used to pick `max_len`. Work is sharded by case range across processes
and merged in order, so the cache is byte-identical regardless of worker count.
(`prepare` = `tokenize` + `dataset.json`; use `tokenize` alone when rebuilding a subset
of splits with `--splits`.)

Memory stays bounded no matter the corpus size: each worker writes its shard to disk and
returns only a summary, and the merge pre-allocates the output arrays from those summaries
and streams the shards into memory-mapped windows one at a time. Tokenizing a 10M-item
split needs the same peak RAM as a 10k-item one. The merge logs progress every 500k items;
on a filesystem without sparse-file support the pre-allocation costs the full padded array
size in disk (still zero RAM), so make sure the cache volume has room for the arrays.

The cache stores the prompt and the document state **separately**, with the state kept at
full length. A training run reassembles the exact sequence its `max_len` calls for at read
time, so changing `max_len` between runs is a filter over stored lengths, not a rebuild.
See the format documentation at the top of `src/chinese_jev/cache.py`.

### `train`

```bash
chinese-jev train --config configs/my.json
# or multi-GPU:
torchrun --standalone --nproc-per-node=4 -m chinese_jev train --config configs/my.json
```

GRPO policy gradient over a strictly proper scoring rule reward, plus soft cross-entropy
on the target distribution. Length-bucketed batches, gradient accumulation, cosine
schedule, step/epoch checkpoints with `--resume latest`. The final bundle lands at
`<work_dir>/bundle` and has the same layout as the input bundle, so it is a drop-in
replacement wherever the base model was used.

### `calibrate`

Fits one temperature per question type on a held-out split by maximum likelihood, and
writes it into the bundle's config. Without this step the reported confidences mean
nothing; with it, "confidence 0.9" is right about 90% of the time. A question type with no
items in the calibration split keeps temperature 1.0 — the `items per type` line in the
output says when that happened, which small or unstratified splits can make routine.

### `evaluate`

```bash
chinese-jev evaluate --config configs/my.json --splits validation,test
# score a specific model instead of this run's trained bundle:
chinese-jev evaluate --config configs/my.json --bundle runs/other-run/bundle
chinese-jev evaluate --config configs/my.json --bundle runs/my-run/checkpoints/epoch_00000002
# multi-GPU: shards the split across ranks, gathers, then computes metrics
torchrun --standalone --nproc-per-node=8 -m chinese_jev evaluate --config configs/my.json
```

Accuracy, expected calibration error, and mean log score, overall and per question type,
written to `<work_dir>/evaluation.json`. By default the run's own `<work_dir>/bundle` is
scored (falling back to the config's base `bundle` when nothing was trained); `--bundle`
scores any bundle-shaped directory — another run's output, an epoch checkpoint, or the
base model for a before/after comparison. Under torchrun each rank scores a round-robin
shard of the split on its own GPU; the per-item predictions are gathered before any metric
is computed (ECE bins over a shard are not the ECE bins over the split), so the reported
numbers are identical to a single-process run. `eval_max_items` caps the items per split
when a quick signal is enough.

## The cache directory

`cache_dir` decides where the token cache lives:

- unset → `<work_dir>/cache`, private to the run;
- set (config key or `--cache-dir`) → any directory, shared between runs.

With the single-config flow this is automatic: the config sets `cache_dir` once,
`prepare` fills the cache, and every run reading the same config finds it. A sweep
is then just `--work-dir`/`--max-len` overrides (or copies of the config) naming
the same `cache_dir`:

```bash
chinese-jev prepare --config configs/my.json
chinese-jev run --config configs/my.json --work-dir runs/len2048 --max-len 2048
chinese-jev run --config configs/my.json --work-dir runs/len4096 --max-len 4096
```

Reuse is validated, not assumed: `tokenize` compares the cache's own manifest (adapter
parameters, `head_max_len`) against the config and rebuilds when they disagree, and
`train` refuses a cache whose tokenizer fingerprint does not match the bundle's tokenizer.

## Config reference

Every key, its default, and what it does is documented inline in
`src/chinese_jev/config.py` (`DEFAULTS`).

The **dataset half** (drives `prepare`):

| key | meaning |
|---|---|
| `dataset` | adapter name in the registry (`chinese-jev`, `jsonl`, or your own) |
| `dataset_params` | adapter arguments: `data_root`, `file_template`, `splits` |
| `bundle` | the model whose tokenizer the cache is built with |
| `cache_dir` | where the cache and `dataset.json` land |
| `head_max_len` | `null` = measure automatically; a number = use it (validated) |

The **run half** (drives `run`/`train`/`calibrate`/`evaluate`):

| key | meaning |
|---|---|
| `bundle` | model directory: `encoder/`, `tokenizer/`, `model.safetensors`, `rl_agent_config.json` |
| `work_dir` | run directory: checkpoints, metrics, calibration, evaluation |
| `max_len` | training sequence length; sweepable without re-tokenizing |
| `overlong` | `filter` drops items over `max_len`; `truncate` cuts their state instead |

(The two-config style — a run config whose `dataset_info` points at a prepared
`dataset.json` instead of carrying the dataset fields inline — still works, and is
handy when many run configs share one prepared dataset.)

Any key can be overridden from the command line with `--set KEY=VALUE`.
