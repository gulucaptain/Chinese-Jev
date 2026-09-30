"""Score a trained bundle on the held-out splits.

Reports accuracy, expected calibration error and mean log score per split and per question
type. Accuracy alone hides the failure this pipeline is most likely to produce — a model
that is confident and wrong — so the calibration numbers are reported next to it.

Evaluation is distributed-aware: under torchrun, each rank scores a round-robin shard of
the split on its own device and the per-item predictions are gathered before any metric is
computed, so the reported numbers are identical to a single-process run — a large test
split just finishes `world_size` times sooner.

    torchrun --standalone --nproc-per-node=8 -m chinese_jev evaluate --config <cfg>
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import distributed as dist_utils
from .cache import CacheError
from .calibrate import QTYPE_ORDER, collect_logits
from .config import split_cache_dir
from .modeling import answer_confidence, ece_score
from .sequence import QTYPES
from .tokenizer import load_tokenizer, tokenizer_fingerprint
from .train import build_and_load


def score_split(cfg, split, model, device, rank=0, world_size=1, log=print):
    directory = split_cache_dir(cfg, split)
    if not (directory / "manifest.json").exists():
        return None
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    local = dict(cfg)
    local["head_max_len"] = max(cfg["head_max_len"] or 0, manifest["head_max_len"])
    temperatures = cfg.get("temperature") or [1.0, 1.0, 1.0]

    preds = collect_logits(model, device, local, directory, local["max_len"],
                           cfg.get("eval_max_items"), log=log, rank=rank, world_size=world_size)
    # Reassemble the whole split before computing anything: ECE bins over a shard are not
    # the ECE bins over the split, so metrics must never be averaged across ranks.
    preds = [p for shard in dist_utils.all_gather_object(preds) for p in shard]
    if not preds:
        return None

    correct, conf, logscore = [], [], []
    by_type = {name: {"n": 0, "correct": 0, "conf": [], "correct_flags": []} for name in QTYPE_ORDER}
    for qname, logits, target in preds:
        temp = temperatures[QTYPES[qname]] if len(temperatures) > QTYPES[qname] else 1.0
        z = np.asarray(logits, dtype=np.float64) / max(1e-6, temp)
        z = z - z.max()
        p = np.exp(z)
        p = p / p.sum()
        k = len(p)
        label = int(np.argmax(target))
        is_correct = int(np.argmax(p) == label)
        confidence = answer_confidence(p, k)
        correct.append(is_correct)
        conf.append(confidence)
        logscore.append(float(np.log(max(1e-12, p[label]))))
        entry = by_type[qname]
        entry["n"] += 1
        entry["correct"] += is_correct
        entry["conf"].append(confidence)
        entry["correct_flags"].append(is_correct)

    correct = np.asarray(correct)
    conf = np.asarray(conf)
    report = {
        "split": split,
        "items": len(correct),
        "accuracy": float(correct.mean()),
        "ece": float(ece_score(conf, correct)),
        "mean_logscore": float(np.mean(logscore)),
        "mean_confidence": float(conf.mean()),
        "by_type": {},
    }
    for name, entry in by_type.items():
        if not entry["n"]:
            continue
        c = np.asarray(entry["correct_flags"])
        report["by_type"][name] = {
            "items": entry["n"],
            "accuracy": float(c.mean()),
            "ece": float(ece_score(np.asarray(entry["conf"]), c)),
        }
    return report


def evaluate(cfg, splits=("validation", "test"), verbose=True, log=print, bundle=None):
    """Score a bundle on `splits`.

    By default the run's own trained bundle (`<work_dir>/bundle`) is scored, falling back
    to the config's base bundle when no training has happened. An explicit `bundle`
    (CLI `--bundle`) overrides both — that is how an arbitrary checkpoint, an epoch
    checkpoint, or someone else's bundle gets evaluated against this cache.
    """
    rank, local_rank, world_size, _ = dist_utils.init_process_group()
    device = dist_utils.setup_device(local_rank, world_size)
    main = dist_utils.is_main()
    run_dir = Path(cfg["work_dir"])
    local = dict(cfg)
    chosen = Path(bundle) if bundle else run_dir / "bundle"
    if (chosen / "model.safetensors").exists():
        local["bundle"] = str(chosen)
        saved = chosen / "rl_agent_config.json"
        if saved.exists():
            local["temperature"] = json.loads(saved.read_text(encoding="utf-8")).get("temperature")
    elif bundle:
        raise FileNotFoundError("no model.safetensors under --bundle %s" % chosen)
    # Refuse a bundle whose tokenizer disagrees with the cache before touching the GPU:
    # mismatched ids would otherwise surface as an inscrutable device-side assert.
    fp = tokenizer_fingerprint(load_tokenizer(local["bundle"]))
    for split in splits:
        mpath = split_cache_dir(local, split) / "manifest.json"
        if not mpath.exists():
            continue
        cached = json.loads(mpath.read_text(encoding="utf-8")).get("tokenizer_fingerprint")
        if cached is not None and cached != fp:
            raise CacheError("the cache at %s was built with a different tokenizer than "
                             "bundle %s ships; retokenize or evaluate a matching bundle"
                             % (mpath.parent, local["bundle"]))
    model, _ = build_and_load(local, device, verbose and main)
    if main:
        log("Evaluating bundle %s" % local["bundle"])
        if world_size > 1:
            log("  across %d ranks" % world_size)

    results = {}
    for split in splits:
        report = score_split(local, split, model, device, rank=rank, world_size=world_size, log=log)
        if report is None:
            if main:
                log("  %-12s (no cache, skipped)" % split)
            continue
        results[split] = report
        if main:
            log("  %-12s items=%8s acc=%.4f ece=%.4f logscore=%.4f conf=%.4f"
                % (split, f"{report['items']:,}", report["accuracy"], report["ece"],
                   report["mean_logscore"], report["mean_confidence"]))
            for name, entry in report["by_type"].items():
                log("      %-8s items=%7s acc=%.4f ece=%.4f"
                    % (name, f"{entry['items']:,}", entry["accuracy"], entry["ece"]))
    if results and main:
        (run_dir / "evaluation.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    dist_utils.barrier()
    return results
