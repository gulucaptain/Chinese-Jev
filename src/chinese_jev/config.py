"""Configuration for the pipeline.

One config file describes one dataset plus the run. Dataset-specific knowledge lives in
the adapter's params, everything else is shared, so pointing the pipeline at a new corpus
means writing a new config rather than editing the pipeline.

Relative paths in a config resolve against the enclosing *project root* (the nearest
ancestor of the config file with a `pyproject.toml` or `.git`), so `configs/my.json` can
say `runs/...` and mean `<repo>/runs/...`.

Two directories matter and they are deliberately separate:

- `work_dir` is the run: checkpoints, metrics, calibration, evaluation. One per run.
- `cache_dir` is the token cache: expensive to build, independent of `max_len`, and
  shareable. It defaults to `<work_dir>/cache`, but pointing several runs at one explicit
  `cache_dir` lets a length or hyperparameter sweep tokenize the corpus exactly once.
  Staleness is checked against the cache's own manifest (data path, prompt geometry,
  tokenizer fingerprint), not against where the cache happens to live.

The usual split of labour is: one *dataset* config drives `chinese-jev prepare`, which
tokenizes and writes `dataset.json` at the cache root; each *training* config then only
says `dataset_info: <path to that file>` plus run hyperparameters.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

# The run's own defaults. Anything a dataset has to override is empty here, so a config
# that forgets a required field fails loudly in `validate` instead of training on nothing.
DEFAULTS = dict(
    # --- what to train on -------------------------------------------------------------
    dataset=None,          # registry name; see chinese_jev.adapters.REGISTRY
    dataset_params={},     # adapter-specific: paths, split names, file templates, ...
    # A `dataset.json` written by `chinese-jev prepare`: it supplies dataset,
    # dataset_params, cache_dir and the measured head_max_len, so a training config
    # only names the prepared dataset instead of repeating how it was built.
    dataset_info=None,
    bundle=None,           # model bundle: encoder/, tokenizer/, model.safetensors
    work_dir="runs/default",
    cache_dir=None,        # token cache location; None means <work_dir>/cache

    # --- sequence geometry ------------------------------------------------------------
    # max_len is a *training-time* choice: the cache stores the prompt and the state
    # separately and reassembles at read time, so changing it needs no re-tokenization.
    # It is capped by the encoder's position limit (mmBERT: 8192).
    max_len=4096,
    # head_max_len caps the prompt. It must clear the widest option set the data contains,
    # or `build_sequence` falls back to a fixed per-option budget that makes options
    # indistinguishable. `chinese-jev stats` measures what a dataset actually needs; None
    # here means "use the measured value from the cache's manifest".
    head_max_len=None,
    head_layers=2,
    head_dropout=0.1,
    act_costs={"escalate": 0.5},
    attention="eager",
    gradient_checkpointing=False,

    # --- optimisation -----------------------------------------------------------------
    epochs=4,
    batch_size=8,
    accumulation=4,
    seed=42,
    encoder_lr=2.5e-5,
    head_lr=1e-4,
    weight_decay=0.01,
    max_grad_norm=1.0,
    group_size=4,
    sigma_start=0.4,
    sigma_end=0.1,
    precision="bf16",

    # --- length policy ----------------------------------------------------------------
    # What to do with an item that does not fit in max_len. "filter" drops it (the hot
    # knob: lower max_len, fewer items); "truncate" keeps it by cutting the state, which
    # is lossless for the decision because the state is rendered last.
    overlong="filter",
    # Batch items of similar length together to cut padding to the longest item.
    length_bucketed_batches=True,
    max_tokens_per_batch=None,   # optional token cap per forward; None means batch_size only

    # --- data loading / sampling ------------------------------------------------------
    num_workers=4,               # DataLoader workers; the cache is random-access so this is I/O only
    max_items=None,              # cap items per split (smoke runs); None means everything
    sample_weights={},           # repeat rarer question types or sources; empty = natural frequencies

    # --- checkpoints ------------------------------------------------------------------
    save_every_steps=2000,
    keep_last_checkpoints=3,
    keep_epoch_checkpoints=True,
    resume=None,                 # path to a checkpoint dir, or "latest"
    calibration_max_items=2000,
    eval_max_items=None,         # cap items per split during evaluation; None means everything

    # --- tokenization -----------------------------------------------------------------
    tokenize_workers=0,          # 0 = auto (min(cpu//4, 32))
    tokenize_chunk=2000,         # cases per shard task

    log_every=50,
    max_steps=None,
)

_BOOL = {"gradient_checkpointing", "length_bucketed_batches", "keep_epoch_checkpoints"}
_INT = {"max_len", "head_max_len", "head_layers", "epochs", "batch_size", "accumulation",
        "seed", "group_size", "num_workers", "save_every_steps", "keep_last_checkpoints",
        "calibration_max_items", "eval_max_items", "tokenize_workers", "tokenize_chunk",
        "log_every", "max_steps", "max_items", "max_tokens_per_batch"}
_FLOAT = {"head_dropout", "encoder_lr", "head_lr", "weight_decay", "max_grad_norm",
          "sigma_start", "sigma_end"}
_STR = {"dataset", "bundle", "work_dir", "cache_dir", "precision", "overlong", "attention",
        "resume", "dataset_info"}


class ConfigError(ValueError):
    """A config that would train the wrong thing, or not train at all."""


def read_config(path=None, **overrides):
    """Load a JSON config over the defaults, rejecting unknown or mistyped fields.

    Relative paths resolve against the *project root*: the nearest ancestor of the config
    file that contains a `pyproject.toml` or `.git` (the config's own directory if there is
    none). A config in `<repo>/configs/` can therefore say `datasets/...` or `runs/...`
    and mean the repo-rooted path, instead of paths relative to `configs/` itself. CLI
    `overrides` resolve against the cwd, which is where the caller typed them.
    """
    cfg = copy.deepcopy(DEFAULTS)
    base = None
    raw = {}
    if path is not None:
        path = Path(path).expanduser().resolve()
        base = _project_root(path.parent)
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ConfigError("config must be a JSON object")
        unknown = set(raw) - set(DEFAULTS)
        if unknown:
            raise ConfigError("unknown config keys: %s" % ", ".join(sorted(unknown)))
        for key, value in raw.items():
            expected = _expected_type(key)
            if expected is float and isinstance(value, int):
                value = float(value)
            if value is not None and not isinstance(value, expected):
                raise ConfigError("config key %r must be %s, got %s"
                                  % (key, expected.__name__, type(value).__name__))
            cfg[key] = value
    for key, value in overrides.items():
        if key not in DEFAULTS:
            raise ConfigError("unknown override %r" % key)
        if value is not None:
            cfg[key] = value
    if raw.get("dataset_info") and (raw.get("dataset") or raw.get("dataset_params")):
        # One source of truth: a prepared dataset already records how it was built, and a
        # config that also says so can silently disagree with the cache it points at.
        raise ConfigError("give either 'dataset_info' or 'dataset'/'dataset_params', not both")
    for key in ("bundle", "work_dir", "cache_dir", "dataset_info"):
        cfg[key] = _resolve(cfg[key], base)
    if cfg["dataset_info"]:
        _apply_dataset_info(cfg)
    if cfg["dataset"] is None:
        raise ConfigError("config needs a 'dataset' (or a 'dataset_info' from `chinese-jev prepare`)")
    if cfg["bundle"] is None:
        raise ConfigError("config needs a 'bundle' (the model directory to fine-tune)")
    # Dataset paths resolve too, so a config keeps working when the pipeline is invoked
    # from another directory — and so the path recorded in a cache manifest is stable
    # enough to compare against later.
    params = dict(cfg["dataset_params"] or {})
    for key, value in list(params.items()):
        if isinstance(value, str) and key.endswith(("_root", "_dir", "_path")):
            params[key] = _resolve(value, base)
    cfg["dataset_params"] = params
    if cfg["resume"] not in (None, "latest"):
        cfg["resume"] = _resolve(cfg["resume"], base)
    validate(cfg)
    return cfg


def _project_root(start):
    """The directory a config's relative paths mean: the enclosing project's root.

    Walks up from the config file's directory to the nearest `pyproject.toml` or `.git`.
    A config outside any project falls back to its own directory, which keeps a config
    that travels with its run self-contained.
    """
    start = Path(start)
    for d in (start, *start.parents):
        if (d / "pyproject.toml").exists() or (d / ".git").exists():
            return d
    return start


def _apply_dataset_info(cfg):
    """Fill the dataset half of the config from a `dataset.json` written by `prepare`.

    The cache is wherever the file is: `dataset.json` lives at the cache root, so moving
    the prepared directory moves the cache reference with it. An explicit `head_max_len`
    in the config still wins (train validates it against the cache manifest); a null one
    takes the measured value the dataset was tokenized at.
    """
    info_path = Path(cfg["dataset_info"])
    if not info_path.exists():
        raise ConfigError("dataset_info %s does not exist; run `chinese-jev prepare` first"
                          % info_path)
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if info.get("format_version") != 1:
        raise ConfigError("%s was written by a different pipeline version (format %r)"
                          % (info_path, info.get("format_version")))
    cfg["dataset"] = info["dataset"]
    cfg["dataset_params"] = dict(info.get("dataset_params") or {})
    cfg["cache_dir"] = str(info_path.parent)
    if cfg["head_max_len"] is None:
        cfg["head_max_len"] = info.get("head_max_len")
    if cfg["max_items"] is None and info.get("max_items") is not None:
        # A dataset prepared with a cap (a smoke slice) caps what a run can train on.
        cfg["max_items"] = info["max_items"]


def cache_root(cfg):
    """Where this run reads and writes its token cache.

    An explicit `cache_dir` decouples the cache from the run so several runs can share
    one tokenization; the default keeps everything under the run directory.
    """
    if cfg.get("cache_dir"):
        return Path(cfg["cache_dir"])
    return Path(cfg["work_dir"]) / "cache"


def split_cache_dir(cfg, split):
    """One split's cache directory: `<cache_root>/<split>`."""
    return cache_root(cfg) / split


def _expected_type(key):
    if key in _BOOL:
        return bool
    if key in _INT:
        return int
    if key in _FLOAT:
        return float
    if key in _STR:
        return str
    return (dict, list)


def _resolve(value, base):
    """Make a path absolute against the config's directory, leaving the rest alone."""
    if base is None or value is None:
        return value
    p = Path(value).expanduser()
    return str(p if p.is_absolute() else (base / p))


def validate(cfg):
    if cfg["overlong"] not in ("filter", "truncate"):
        raise ConfigError("overlong must be 'filter' or 'truncate', got %r" % cfg["overlong"])
    if cfg["max_len"] is not None and cfg["max_len"] < 16:
        raise ConfigError("max_len=%r is too small to hold a prompt" % cfg["max_len"])
    if cfg["head_max_len"] is not None and cfg["head_max_len"] < 32:
        raise ConfigError("head_max_len=%r is too small to hold a prompt" % cfg["head_max_len"])
    for key in ("batch_size", "accumulation", "epochs", "group_size"):
        if cfg[key] < 1:
            raise ConfigError("%s must be >= 1, got %r" % (key, cfg[key]))
    if cfg["group_size"] < 2:
        # The GRPO advantage subtracts a group mean; a group of one has no baseline and
        # the normaliser divides by ~0.
        raise ConfigError("group_size must be >= 2 for a GRPO baseline, got %r" % cfg["group_size"])
    if cfg["dataset_params"] and not isinstance(cfg["dataset_params"], dict):
        raise ConfigError("dataset_params must be an object")
    return cfg


def describe(cfg):
    """A compact provenance record stored in the run directory."""
    keep = ("dataset", "bundle", "max_len", "head_max_len", "epochs", "batch_size",
            "accumulation", "encoder_lr", "head_lr", "seed", "group_size", "overlong",
            "precision", "gradient_checkpointing")
    return {k: cfg[k] for k in keep}
