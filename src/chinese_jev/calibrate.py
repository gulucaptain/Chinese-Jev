"""Fit per-question-type temperatures on held-out items.

The temperature is the scale a decision distribution is divided by before its probabilities
are reported. It is fitted here on a split the run never trained on: fitting it on training
items measures the fit rather than the calibration, because the model is near-certain and
near-correct on them and the optimiser has nothing to soften.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from . import distributed as dist_utils
from .cache import TokenCache, filter_items
from .config import split_cache_dir
from .data import collate_sequences
from .train import autocast_ctx, build_and_load

QTYPE_ORDER = ("choice", "score", "noul")


def fit_one_temp(selected):
    """One temperature per question type, by maximum likelihood under a LBFGS search.

    `selected` is a list of (logits, target) pairs. The objective is the mean negative log
    likelihood of the target under softmax(logits / T), which is proper for a distribution
    target and does not collapse to a one-hot fit the way accuracy-based fitting would.
    """
    if len(selected) < 10:
        return 1.0
    k_max = max(len(z) for z, _ in selected)
    z = torch.full((len(selected), k_max), -1e4)
    target = torch.zeros((len(selected), k_max))
    for i, (zi, ti) in enumerate(selected):
        z[i, :len(zi)] = torch.tensor(zi, dtype=torch.float32)
        target[i, :len(ti)] = torch.tensor(ti, dtype=torch.float32)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = -(target * torch.log_softmax(z / log_t.exp(), -1)).sum(-1).mean()
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.clamp(log_t.exp(), 0.1, 10.0).item())


def collect_logits(model, device, cfg, directory, max_len, limit, batch_size=16, log=print,
                   rank=0, world_size=1):
    """Run the model over a split and return [(qtype_name, logits, target), ...].

    `rank`/`world_size` shard the items round-robin, after `limit` is applied, so a
    distributed caller scores each item exactly once and the union over ranks is the same
    item set a single process would score.
    """
    cache = TokenCache(directory)
    keep = filter_items(cache, max_len, cfg["overlong"])
    if limit is not None and len(keep) > limit:
        keep = keep[:limit]
    keep = keep[rank::world_size]
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(keep), batch_size):
            idx = keep[start:start + batch_size]
            items = [cache.load_sequence(int(i), max_len) for i in idx]
            batch = collate_sequences(items, cache.manifest["pad_token_id"])
            for key in ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"):
                batch[key] = batch[key].to(device)
            with autocast_ctx(device, cfg["precision"]):
                logits, _ = model(batch["input_ids"], batch["attention_mask"], batch["marker_pos"],
                                  batch["marker_mask"], batch["qtype"])
            logits = logits.float().cpu().numpy()
            for r, it in enumerate(items):
                k = len(it["markers"])
                out.append((cache.item_meta(int(idx[r]))["qtype_name"], logits[r, :k], it["target"]))
    model.train()
    return out


def calibrate(cfg, split="validation", verbose=True, log=print):
    """Fit and record per-type temperatures from `split`."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(cfg["work_dir"])
    directory = split_cache_dir(cfg, split)
    if not (directory / "manifest.json").exists():
        raise FileNotFoundError("no cache for split %r at %s" % (split, directory))
    bundle = run_dir / "bundle"
    if not (bundle / "model.safetensors").exists():
        bundle = Path(cfg["bundle"])
    cfg = dict(cfg)
    cfg["bundle"] = str(bundle)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    cfg["head_max_len"] = max(cfg["head_max_len"] or 0, manifest["head_max_len"])

    model, _ = build_and_load(cfg, device, verbose)
    log("Fitting temperatures on %s (up to %s items)..."
        % (split, f"{cfg['calibration_max_items']:,}" if cfg["calibration_max_items"] else "all"))
    preds = collect_logits(model, device, cfg, directory, cfg["max_len"],
                           cfg["calibration_max_items"], log=log)

    temperatures = [1.0, 1.0, 1.0]
    per_type = {}
    for qi, name in enumerate(QTYPE_ORDER):
        selected = [(z, t) for qn, z, t in preds if qn == name]
        if selected:
            temperatures[qi] = fit_one_temp(selected)
        per_type[name] = {"temperature": round(temperatures[qi], 4), "items": len(selected)}
    result = {"split": split, "items": len(preds), "temperature": temperatures, "by_type": per_type}
    if dist_utils.is_main():
        (run_dir / "calibration.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        cfg_path = bundle / "rl_agent_config.json"
        if cfg_path.exists():
            saved = json.loads(cfg_path.read_text(encoding="utf-8"))
            saved["temperature"] = [float(t) for t in temperatures]
            saved["calibration_status"] = "fitted"
            saved["calibration_split"] = split
            # Per-bucket overrides inherited from the base checkpoint would shadow the new
            # per-type temperatures on exactly the option counts they cover.
            saved.pop("temperature_by_options", None)
            cfg_path.write_text(json.dumps(saved, ensure_ascii=False, indent=2), encoding="utf-8")
    log("  temperatures (choice, score, noul): [%s]"
        % ", ".join("%.3f" % t for t in temperatures))
    log("  items per type: %s" % {k: v["items"] for k, v in per_type.items()})
    return result
