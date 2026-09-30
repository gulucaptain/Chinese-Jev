"""Command-line entry point: `chinese-jev <stage>` (or `python -m chinese_jev <stage>`).

The blessed flow is one config per dataset+run:

    chinese-jev prepare --config configs/my.json     # once per dataset
    chinese-jev run --config configs/my.json         # train, calibrate, evaluate

The config holds both halves: the dataset fields (`dataset`, `dataset_params`,
`cache_dir`, `head_max_len`) that drive `prepare`, and the run hyperparameters that
drive `run`. `prepare` measures the head budget (when `head_max_len` is null),
tokenizes every split, writes `dataset.json` at the cache root, and fills the measured
values back into the config. The lower-level
stages (`stats`, `tokenize`, `train`, `calibrate`, `evaluate`) remain independently
runnable, so a long tokenization is paid once and reused by several training lengths.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from . import adapters as adapters_mod
from . import config as cfgmod
from .config import cache_root, split_cache_dir


def _adapter(cfg):
    return adapters_mod.get(cfg["dataset"]).from_params(cfg["dataset_params"])


def _log(msg):
    print(msg, flush=True)


def _ensure_head_max_len(cfg, adapter, splits, sample=None, verbose=True):
    """Fill a null `head_max_len`: adopt an existing cache's measured value, else measure.

    The measurement walks the split once with the tokenizer (options only, not the
    states), so it is much cheaper than tokenization itself — but on a very large corpus
    `--sample` bounds it.
    """
    if cfg["head_max_len"] is not None:
        return
    for split in splits:
        manifest_path = split_cache_dir(cfg, split) / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("adapter_params") == _adapter_params(cfg):
                cfg["head_max_len"] = manifest["head_max_len"]
                _log("  head_max_len=%d adopted from the existing %s cache"
                     % (cfg["head_max_len"], split))
                return
    from .encoding import measure_head_budget
    from .tokenizer import load_tokenizer

    split = "train" if "train" in splits else splits[0]
    _log("  head_max_len is not set; measuring it from split %r..." % split)
    report = measure_head_budget(adapter, split, load_tokenizer(cfg["bundle"]),
                                 sample=sample, verbose=verbose)
    cfg["head_max_len"] = report["recommended_head_max_len"]
    _log("  measured head_max_len=%d (widest option set: %d options)"
         % (cfg["head_max_len"], report["widest_option_set"]))


def cmd_tokenize(cfg, splits, force, sample=None, verbose=True):
    from .prepare import print_length_report, tokenize_split, write_manifest_summary

    cache_root(cfg).mkdir(parents=True, exist_ok=True)
    adapter = _adapter(cfg)
    splits = splits or list(adapter.splits)
    _ensure_head_max_len(cfg, adapter, splits, sample=sample, verbose=verbose)
    manifests = {}
    for split in splits:
        out_dir = split_cache_dir(cfg, split)
        if (out_dir / "manifest.json").exists() and not force:
            existing = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
            if _cache_matches(existing, cfg):
                _log("  %s: reusing cache at %s (%s items)"
                     % (split, out_dir, f"{existing['n_items']:,}"))
                manifests[split] = existing
                continue
            _log("  %s: cache is stale, rebuilding" % split)
        manifests[split] = tokenize_split(cfg, adapter, split, out_dir, cfg["bundle"],
                                          verbose=verbose, log=_log)
    write_manifest_summary(cfg["work_dir"], manifests)
    print_length_report(manifests, log=_log)
    return manifests


def cmd_prepare(cfg, splits, force, sample=None, verbose=True, config_path=None):
    """The whole data side: measure geometry, build every cache, describe the dataset.

    After this, the dataset is a single artifact — the cache directory with its
    `dataset.json` — and a training config references it by that one path.
    """
    from .prepare import fill_measured_values, write_dataset_info

    manifests = cmd_tokenize(cfg, splits, force, sample=sample, verbose=verbose)
    info_path = write_dataset_info(cfg, manifests)
    _log("Dataset prepared -> %s" % info_path)
    if config_path:
        filled = fill_measured_values(cfg, info_path, config_path)
        if filled:
            _log("Measured values written back into %s: %s"
                 % (config_path, ", ".join("%s=%s" % kv for kv in filled.items())))
        _log("The same config now trains: chinese-jev run --config %s" % config_path)
        _log("(edit max_len and the hyperparameters in it as needed)")
    return manifests


def _cache_matches(manifest, cfg):
    """A cache is reusable only if the data path and prompt geometry are unchanged.

    The check is against the cache's own manifest, never against where the cache lives, so
    a cache shared between runs via an explicit `cache_dir` stays exactly as trustworthy
    as a private one.
    """
    if manifest.get("head_max_len") != cfg["head_max_len"]:
        return False
    recorded = manifest.get("adapter_params")
    return recorded is None or recorded == _adapter_params(cfg)


def _adapter_params(cfg):
    params = dict(cfg["dataset_params"])
    params["max_items"] = cfg["max_items"]
    return params


def cmd_stats(cfg, split=None, sample=None, config_path=None, verbose=True):
    """Measure the head budget a dataset needs, without writing a cache.

    A null `head_max_len` in the config is filled in place with the measured value, so
    "measure, then copy the number over" is one command. An explicit value is compared
    against the measurement instead — never overwritten.
    """
    from .encoding import measure_head_budget
    from .tokenizer import load_tokenizer

    adapter = _adapter(cfg)
    split = split or (list(adapter.splits)[0] if adapter.splits else "train")
    tok = load_tokenizer(cfg["bundle"])
    _log("Measuring prompt geometry for dataset=%s split=%s" % (cfg["dataset"], split))
    report = measure_head_budget(adapter, split, tok, sample=sample, verbose=verbose)
    report["encoder_max_position"] = _encoder_limit(cfg["bundle"])
    recommended = report["recommended_head_max_len"]
    _log("")
    if cfg["head_max_len"] is None and config_path:
        raw = json.loads(Path(config_path).read_text(encoding="utf-8"))
        raw["head_max_len"] = recommended
        Path(config_path).write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n",
                                     encoding="utf-8")
        _log("head_max_len was null -> wrote the measured %d into %s" % (recommended, config_path))
    elif cfg["head_max_len"] is not None:
        if cfg["head_max_len"] < recommended:
            _log("WARNING: configured head_max_len=%d is below the measured %d; the widest "
                 "option sets will be skipped at tokenize time"
                 % (cfg["head_max_len"], recommended))
        else:
            _log("configured head_max_len=%d covers the measured %d — ok"
                 % (cfg["head_max_len"], recommended))
    _log("Suggested config values:")
    _log("  \"head_max_len\": %d" % recommended)
    _log("  \"max_len\": %s     # hard ceiling for this encoder"
         % (report["encoder_max_position"] or "?"))
    return report


def _encoder_limit(bundle):
    path = Path(bundle) / "encoder" / "config.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("max_position_embeddings")


def cmd_run(cfg, splits, skip_tokenize, verbose=True):
    """Tokenize, train, calibrate, evaluate — the whole thing.

    Works both single-process and under torchrun. Tokenization and calibration run on
    rank 0 only (they are cheap next to training, and several ranks would fight over the
    same output files); training and evaluation use every rank.
    """
    from . import distributed as dist_utils
    from .calibrate import calibrate
    from .evaluate import evaluate
    from .train import train

    rank, _, _, _ = dist_utils.init_process_group()
    main = dist_utils.is_main()
    run_dir = Path(cfg["work_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    if not skip_tokenize:
        if main:
            cmd_tokenize(cfg, splits, force=False, verbose=verbose)
        dist_utils.barrier()
    elif main:
        _log("Skipping tokenization; using the cache at %s" % cache_root(cfg))
    train(cfg, verbose=verbose, log=_log)
    if main:
        calibrate(cfg, split="validation", verbose=verbose, log=_log)
    dist_utils.barrier()
    evaluate(cfg, splits=("validation", "test"), verbose=verbose, log=_log)
    if main:
        _log("\nRun finished in %.1f min -> %s" % ((time.time() - started) / 60, run_dir))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="chinese-jev",
                                     description="chinese-jev fine-tuning pipeline")
    parser.add_argument("stage", choices=("prepare", "tokenize", "stats", "train",
                                          "calibrate", "evaluate", "run"))
    parser.add_argument("--config", required=True, help="path to a run config JSON")
    parser.add_argument("--work-dir", default=None, help="override the run directory")
    parser.add_argument("--cache-dir", default=None,
                        help="override the token cache directory (share it across runs)")
    parser.add_argument("--splits", default=None, help="comma-separated splits (tokenize/run/evaluate)")
    parser.add_argument("--split", default="validation", help="split for calibrate")
    parser.add_argument("--sample", type=int, default=None,
                        help="cases to sample when measuring the head budget (stats/prepare/tokenize)")
    parser.add_argument("--bundle", default=None,
                        help="evaluate this bundle instead of <work_dir>/bundle (evaluate)")
    parser.add_argument("--max-len", type=int, default=None, help="override max_len (train)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-items", type=int, default=None, help="cap items per split (smoke runs)")
    parser.add_argument("--head-max-len", type=int, default=None)
    parser.add_argument("--resume", default=None, help="checkpoint dir, or 'latest'")
    parser.add_argument("--force", action="store_true", help="rebuild an existing token cache")
    parser.add_argument("--skip-tokenize", action="store_true", help="reuse the existing cache (run)")
    parser.add_argument("--overlong", choices=("filter", "truncate"), default=None)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="override any config key; repeatable")
    args = parser.parse_args(argv)
    if args.bundle and args.stage != "evaluate":
        parser.error("--bundle only applies to evaluate; set the config's 'bundle' key "
                     "(or --set bundle=...) to change what other stages use")

    overrides = {}
    for key in ("max_len", "epochs", "max_steps", "max_items", "head_max_len", "overlong",
                "work_dir", "cache_dir", "resume"):
        value = getattr(args, key)
        if value is not None:
            overrides[key] = value
    for item in args.set:
        if "=" not in item:
            parser.error("--set expects KEY=VALUE, got %r" % item)
        key, raw = item.split("=", 1)
        try:
            overrides[key] = json.loads(raw)
        except json.JSONDecodeError:
            overrides[key] = raw

    cfg = cfgmod.read_config(args.config, **overrides)
    splits = args.splits.split(",") if args.splits else None
    run_dir = Path(cfg["work_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    # Under torchrun every rank enters here; the run record and the banner belong to one.
    if int(os.environ.get("RANK", "0")) == 0:
        (run_dir / "run_config.json").write_text(
            json.dumps({"config_path": str(Path(args.config).resolve()), "effective": cfg},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        _log("Run directory: %s" % run_dir)
        _log("Cache directory: %s" % cache_root(cfg))
        _log("  dataset=%s max_len=%s head_max_len=%s epochs=%s overlong=%s"
             % (cfg["dataset"], cfg["max_len"], cfg["head_max_len"], cfg["epochs"], cfg["overlong"]))

    if args.stage == "prepare":
        cmd_prepare(cfg, splits, force=args.force, sample=args.sample, config_path=args.config)
    elif args.stage == "tokenize":
        cmd_tokenize(cfg, splits, force=args.force, sample=args.sample)
    elif args.stage == "stats":
        cmd_stats(cfg, split=args.split if args.split != "validation" else None,
                  sample=args.sample, config_path=args.config)
    elif args.stage == "train":
        from .train import train

        train(cfg, log=_log)
    elif args.stage == "calibrate":
        from .calibrate import calibrate

        calibrate(cfg, split=args.split, log=_log)
    elif args.stage == "evaluate":
        from .evaluate import evaluate

        evaluate(cfg, splits=splits or ("validation", "test"), log=_log, bundle=args.bundle)
    elif args.stage == "run":
        cmd_run(cfg, splits, args.skip_tokenize)
    return 0
