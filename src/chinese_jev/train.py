"""Training loop.

The objective is a GRPO policy-gradient term over a proper-scoring-rule reward
(`modeling.proper_reward`), plus a soft cross-entropy term on the full target
distribution. The model and the reward live in `modeling.py`; this module is the
operational shell around them: a step/epoch checkpoint schedule with resume,
length-filtered batches, length-bucketed ordering, and a per-run metrics file. None of it
changes the math.
"""
from __future__ import annotations

import json
import math
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from . import distributed as dist_utils
from .cache import CacheError
from .config import split_cache_dir
from .data import BatchPlan, CachedSplit, collate_sequences, plan_epoch
from .modeling import build_model, proper_reward
from .tokenizer import load_tokenizer, tokenizer_fingerprint


def to_device(batch, device):
    """Move every tensor in a batch.

    Moving a chosen subset was how `target` and `qtype` stayed on the host while the model
    ran on the card, which surfaces much later as a device mismatch inside the reward.
    """
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()}


def build_and_load(cfg, device, verbose=True):
    """The bundle's encoder and heads, restored from its own weights.

    `pretrained=False` with an explicit encoder_dir keeps `build_model` from reaching for
    the hub: every parameter arrives from the bundle on disk.
    """
    bundle = Path(cfg["bundle"])
    model_cfg = json.loads((bundle / "rl_agent_config.json").read_text(encoding="utf-8"))
    model_cfg.update(max_len=cfg["max_len"], head_max_len=cfg["head_max_len"],
                     act_costs=cfg["act_costs"], gradient_checkpointing=cfg["gradient_checkpointing"],
                     amp_dtype=cfg["precision"])
    model = build_model(model_cfg, encoder_dir=str(bundle / "encoder"), pretrained=False)
    weights = load_file(str(bundle / "model.safetensors"))
    missing = [k for k in model.state_dict() if k not in weights]
    if missing:
        raise CacheError("bundle is missing %d parameters (e.g. %s)" % (len(missing), missing[:3]))
    model.load_state_dict(weights, strict=True)
    if cfg["gradient_checkpointing"]:
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.head_checkpointing = True
    model.to(device)
    model.train()
    return model, model_cfg


def decision_loss(logits, act, batch, *, sigma, group_size, w_sph=0.75, w_rps=1.0):
    """One batch's loss: GRPO policy gradient over proper_reward + soft cross-entropy.

    `z` and the reward are detached, so the only gradient path is the squared-error
    surrogate `(z - logits)^2`. The `act * 0.0` term gives the action head a zero-valued
    but real gradient path, which DDP's reducer needs when `find_unused_parameters=True`.
    """
    logits = logits.float()
    if not torch.isfinite(logits).all() or not torch.isfinite(act).all():
        raise FloatingPointError("non-finite model outputs")
    mask = batch["marker_mask"]
    target = batch["target"].float()
    k = mask.sum(-1, keepdim=True).clamp(min=1).float()

    # Zero-mean perturbation over the options, so the sampled distributions stay on the
    # simplex and the baseline compares like with like.
    eps = torch.randn((group_size,) + logits.shape, device=logits.device) * sigma * mask
    eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
    z = logits.detach().unsqueeze(0) + eps
    q = torch.softmax(z.masked_fill(~mask, -1e4), -1)

    with torch.no_grad():
        reward = proper_reward(q, target.unsqueeze(0), batch["qtype"], mask, w_sph=w_sph, w_rps=w_rps)
        advantage = reward - reward.mean(0, keepdim=True)
        advantage = advantage / (advantage.std() + 1e-6)

    logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma ** 2)
    loss_rl = -(advantage * logp).mean()
    loss_ce = -(target * F.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
    loss = loss_rl + loss_ce + 0.0 * act.sum()
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite loss")
    return loss, {"loss": float(loss.detach()), "ce": float(loss_ce.detach()),
                  "rl": float(loss_rl.detach()), "reward": float(reward.mean())}


def autocast_ctx(device, precision):
    if precision == "fp32" or device.type != "cuda":
        from contextlib import nullcontext
        return nullcontext()
    return torch.autocast("cuda", dtype=torch.bfloat16 if precision == "bf16" else torch.float16)


def accumulation_window_size(batch_index, n_batches, accumulation):
    """Use one denominator throughout each full or partial accumulation window."""
    start = ((batch_index - 1) // accumulation) * accumulation
    return min(accumulation, n_batches - start)


# ----------------------------------------------------------------------------- checkpoints


def checkpoint_dir(run_dir, kind, step):
    return Path(run_dir) / "checkpoints" / ("%s_%08d" % (kind, step))


def save_checkpoint(model, model_cfg, tok, path, cfg, *, step, epoch, global_step, history,
                    epochs_completed=None, partial_epoch=False):
    """Write a loadable bundle plus the state needed to resume.

    `epochs_completed` is how many epochs finished, which is what a resume iterates from.
    `epoch` is the 1-based epoch the weights belong to, for a human reading the directory.
    The two differ at a mid-epoch step checkpoint, where the epoch in progress has to run
    again from its start because the batch plan is not replayed.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    bare = model.module if hasattr(model, "module") else model
    state = {k: v.half().contiguous().cpu() for k, v in bare.state_dict().items()}
    save_file(state, str(path / "model.safetensors"))
    bare.encoder.config.save_pretrained(str(path / "encoder"))
    tok.save_pretrained(str(path / "tokenizer"))
    if epochs_completed is None:
        epochs_completed = 0 if partial_epoch else epoch
    saved = dict(model_cfg)
    saved.update(
        max_len=cfg["max_len"], head_max_len=cfg["head_max_len"], act_costs=cfg["act_costs"],
        head_layers=cfg["head_layers"], gradient_checkpointing=cfg["gradient_checkpointing"],
        fine_tuned=True, model_name="chinese-jev:%s" % cfg["dataset"],
        training=dict(epoch=epoch, epochs_completed=epochs_completed, partial_epoch=partial_epoch,
                      step=step, global_step=global_step, epochs=cfg["epochs"],
                      batch_size=cfg["batch_size"], accumulation=cfg["accumulation"],
                      encoder_lr=cfg["encoder_lr"], head_lr=cfg["head_lr"], seed=cfg["seed"]),
    )
    (path / "rl_agent_config.json").write_text(json.dumps(saved, ensure_ascii=False, indent=2),
                                               encoding="utf-8")
    (path / "training_state.json").write_text(json.dumps({
        "epoch": epoch, "epochs_completed": epochs_completed, "partial_epoch": partial_epoch,
        "step": step, "global_step": global_step,
        "history_tail": history[-20:] if history else [],
        "config": {k: v for k, v in cfg.items() if isinstance(v, (int, float, str, bool, type(None)))},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def prune_checkpoints(run_dir, kind, keep):
    """Delete older step checkpoints, keeping the newest `keep`."""
    if not keep or keep < 1:
        return
    dirs = sorted((Path(run_dir) / "checkpoints").glob("%s_*" % kind))
    for d in dirs[:-keep]:
        shutil.rmtree(d, ignore_errors=True)


def find_resume(run_dir, resume):
    """The checkpoint to resume from, or None."""
    if resume is None:
        return None
    if resume == "latest":
        ckpt_root = Path(run_dir) / "checkpoints"
        if not ckpt_root.exists():
            return None
        candidates = sorted(ckpt_root.glob("step_*")) + sorted(ckpt_root.glob("epoch_*"))
        if not candidates:
            return None
        # Whichever checkpoint was written last, by the global step it records.
        best, best_step = None, -1
        for d in candidates:
            state_path = d / "training_state.json"
            if not state_path.exists():
                continue
            step = json.loads(state_path.read_text(encoding="utf-8")).get("global_step", -1)
            if step > best_step:
                best, best_step = d, step
        return best
    path = Path(resume)
    return path if path.exists() else None


def load_resume_weights(model, path, cfg, device):
    """Restore weights and report where in the schedule to pick up.

    Returns a start state whose `epoch` is the first epoch still to run. A mid-epoch step
    checkpoint restarts its epoch from the beginning, because the batch order it was part
    way through is not replayed; a completed epoch resumes at the next one.
    """
    weights = load_file(str(Path(path) / "model.safetensors"))
    bare = model.module if hasattr(model, "module") else model
    state = bare.state_dict()
    compatible = {k: v for k, v in weights.items() if k in state and state[k].shape == v.shape}
    dropped = [k for k in weights if k not in compatible]
    state.update(compatible)
    bare.load_state_dict(state, strict=True)
    start = {"epoch": 0, "step": 0, "global_step": 0}
    state_path = Path(path) / "training_state.json"
    if state_path.exists():
        recorded = json.loads(state_path.read_text(encoding="utf-8"))
        completed = recorded.get("epochs_completed")
        if completed is None:
            # A checkpoint from before this field existed recorded the in-progress epoch.
            completed = recorded.get("epoch", 0)
        start = {
            "epoch": int(completed),
            "step": int(recorded.get("step", 0)),
            "global_step": int(recorded.get("global_step", 0)),
        }
    return start, dropped


# ------------------------------------------------------------------------------------ train


def train(cfg, verbose=True, log=print):
    """Train on the prepared train split and return the final bundle directory."""
    rank, local_rank, world_size, is_distributed = dist_utils.init_process_group()
    device = dist_utils.setup_device(local_rank, world_size)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    main = dist_utils.is_main()
    verbose = verbose and main

    run_dir = Path(cfg["work_dir"])
    train_dir = split_cache_dir(cfg, "train")
    if not (train_dir / "manifest.json").exists():
        raise CacheError("no token cache at %s; run `chinese-jev tokenize` first" % train_dir)
    tok = load_tokenizer(cfg["bundle"])
    manifest = json.loads((train_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("tokenizer_fingerprint") != tokenizer_fingerprint(tok):
        raise CacheError("the cache was built with a different tokenizer than the bundle ships; retokenize")
    if cfg["head_max_len"] is None:
        # The measured width is the cache's own, so a config that omits head_max_len still
        # trains at the geometry the data was tokenized at.
        cfg["head_max_len"] = manifest["head_max_len"]
    elif cfg["head_max_len"] < manifest["head_max_len"]:
        raise CacheError("head_max_len=%d is narrower than the %d the cache was built at; "
                         "the widest option sets would lose options"
                         % (cfg["head_max_len"], manifest["head_max_len"]))

    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg["seed"])

    model, model_cfg = build_and_load(cfg, device, verbose)

    dataset = CachedSplit(train_dir, cfg["max_len"], cfg["overlong"], cfg["max_items"], cfg["seed"])
    if len(dataset) == 0:
        raise CacheError("no items survive max_len=%s with overlong=%r; raise max_len or set "
                         "overlong=truncate" % (cfg["max_len"], cfg["overlong"]))
    lengths = np.asarray(dataset.cache.prompt_len, dtype=np.int64) \
        + np.asarray(dataset.cache.state_len, dtype=np.int64) + 1

    enc_params = [p for n, p in model.named_parameters() if n.startswith("encoder.")]
    head_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    optimizer = torch.optim.AdamW(
        [{"params": enc_params, "lr": cfg["encoder_lr"]},
         {"params": head_params, "lr": cfg["head_lr"]}],
        weight_decay=cfg["weight_decay"])

    # The schedule has to be known before the first step, and it depends on how many
    # batches a rank gets, so one rank plans the whole epoch and shares it.
    plan = None
    n_rank_batches = 0
    if main:
        plan = BatchPlan(plan_epoch(dataset.indices, cfg["batch_size"],
                                    length_bucketed=cfg["length_bucketed_batches"], seed=cfg["seed"],
                                    lengths=lengths, max_tokens_per_batch=cfg["max_tokens_per_batch"]))
        n_rank_batches = len(plan.shard(0, world_size))
    plan_json = dist_utils.broadcast_object(plan.to_json() if main else None, src=0)
    plan = BatchPlan.from_json(plan_json)
    mine = plan.shard(rank, world_size)
    n_rank_batches = dist_utils.truncate_to_shortest(len(mine))
    if n_rank_batches == 0:
        raise CacheError("not enough batches for %d ranks" % world_size)
    steps_per_epoch = math.ceil(n_rank_batches / cfg["accumulation"])
    total_updates = cfg["epochs"] * steps_per_epoch
    if cfg["max_steps"] is not None:
        total_updates = min(total_updates, cfg["max_steps"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_updates),
                                                           eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg["precision"] == "fp16" and device.type == "cuda"))

    resume_path = find_resume(run_dir, cfg["resume"])
    start_epoch, global_step, updates_done = 0, 0, 0
    if resume_path is not None:
        start_state, dropped = load_resume_weights(model, resume_path, cfg, device)
        start_epoch = start_state["epoch"]
        updates_done = start_state.get("step", 0)
        global_step = start_state.get("global_step", 0)
        # Fast-forward the schedule rather than restarting the cosine, or a resumed run
        # would train at a fresh high LR for the rest of its life. The initial LR is
        # captured first: stepping before any optimizer.step() warns, and the warning says
        # nothing useful about whether the fast-forward is right.
        for _ in range(updates_done):
            optimizer.step()
            scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        if verbose:
            log("Resuming from %s (epoch %d, step %d)" % (resume_path, start_epoch, updates_done))
        if dropped:
            log("  dropped %d parameters absent from the checkpoint (e.g. %s)"
                % (len(dropped), dropped[:3]))

    if is_distributed:
        from torch.nn.parallel import DistributedDataParallel

        model = DistributedDataParallel(model, device_ids=[local_rank] if device.type == "cuda" else None,
                                        find_unused_parameters=True)

    if verbose:
        log("Training")
        log("  run       : %s" % run_dir)
        log("  cache     : %s" % train_dir)
        log("  device    : %s (%s) | world_size=%d" % (device, cfg["precision"], world_size))
        log("  bundle    : %s" % cfg["bundle"])
        log("  items     : %s kept of %s at max_len=%s (%s)"
            % (f"{len(dataset):,}", f"{manifest['n_items']:,}", cfg["max_len"], cfg["overlong"]))
        log("  geometry  : head_max_len=%s | %s batches/rank, batch=%d accum=%d -> %d updates/epoch"
            % (cfg["head_max_len"], f"{n_rank_batches:,}", cfg["batch_size"], cfg["accumulation"],
               steps_per_epoch))
        log("  lr        : encoder=%g head=%g | sigma %.2f -> %.2f"
            % (cfg["encoder_lr"], cfg["head_lr"], cfg["sigma_start"], cfg["sigma_end"]))

    out = run_dir / "bundle"
    if main:
        out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    history = []
    stop = False

    for epoch in range(start_epoch, cfg["epochs"]):
        if stop:
            break
        # The plan is rebuilt per epoch from a per-epoch seed, so each epoch sees a
        # different shuffle while every rank still agrees on the plan.
        if dist_utils.is_main():
            epoch_plan = BatchPlan(plan_epoch(dataset.indices, cfg["batch_size"],
                                              length_bucketed=cfg["length_bucketed_batches"],
                                              seed=cfg["seed"] + epoch, lengths=lengths,
                                              max_tokens_per_batch=cfg["max_tokens_per_batch"]))
        else:
            epoch_plan = None
        epoch_json = dist_utils.broadcast_object(epoch_plan.to_json() if epoch_plan else None, src=0)
        epoch_batches = BatchPlan.from_json(epoch_json).shard(rank, world_size)
        epoch_batches = epoch_batches[:n_rank_batches]

        progress = epoch / max(1, cfg["epochs"] - 1)
        sigma = cfg["sigma_start"] + (cfg["sigma_end"] - cfg["sigma_start"]) * progress
        n_batches = len(epoch_batches)

        optimizer.zero_grad(set_to_none=True)
        micro = 0
        epoch_loss = 0.0
        epoch_updates = 0
        epoch_started = time.time()

        for batch_index, batch_indices in enumerate(epoch_batches, start=1):
            items = [dataset.item(i) for i in batch_indices]
            batch = to_device(collate_sequences(items, tok.pad_token_id), device)

            with autocast_ctx(device, cfg["precision"]):
                logits, act = model(batch["input_ids"], batch["attention_mask"],
                                    batch["marker_pos"], batch["marker_mask"], batch["qtype"])
            loss, report = decision_loss(logits, act, batch, sigma=sigma, group_size=cfg["group_size"])
            window = accumulation_window_size(batch_index, n_batches, cfg["accumulation"])
            scaler.scale(loss / max(1, window)).backward()
            micro += 1

            if micro % cfg["accumulation"] == 0 or batch_index == n_batches:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                updates_done += 1
                global_step += 1
                epoch_updates += 1
                epoch_loss += dist_utils.reduce_metric(report["loss"], device, world_size)
                history.append(dict(epoch=epoch, step=updates_done, global_step=global_step,
                                    sigma=sigma, **report))
                if verbose and cfg["log_every"] and updates_done % cfg["log_every"] == 0:
                    log("    epoch %d/%d step %d/%d loss=%.4f ce=%.4f rl=%.4f reward=%.3f lr=%.2e"
                        % (epoch + 1, cfg["epochs"], updates_done, total_updates, report["loss"],
                           report["ce"], report["rl"], report["reward"],
                           scheduler.get_last_lr()[0]))

                if cfg["save_every_steps"] and updates_done % cfg["save_every_steps"] == 0:
                    dist_utils.barrier()
                    if dist_utils.is_main():
                        path = checkpoint_dir(run_dir, "step", updates_done)
                        save_checkpoint(model, model_cfg, tok, path, cfg, step=updates_done,
                                        epoch=epoch + 1, global_step=global_step, history=history,
                                        epochs_completed=epoch, partial_epoch=True)
                        prune_checkpoints(run_dir, "step", cfg["keep_last_checkpoints"])
                        log("      checkpoint -> %s" % path)
                    dist_utils.barrier()

                if cfg["max_steps"] is not None and updates_done >= cfg["max_steps"]:
                    stop = True
                    break

        if verbose:
            log("  epoch %d/%d done in %.1fs | mean loss %.4f | %d updates"
                % (epoch + 1, cfg["epochs"], time.time() - epoch_started,
                   epoch_loss / max(1, epoch_updates), epoch_updates))
        dist_utils.barrier()
        if dist_utils.is_main() and cfg["keep_epoch_checkpoints"]:
            path = checkpoint_dir(run_dir, "epoch", epoch + 1)
            save_checkpoint(model, model_cfg, tok, path, cfg,
                            step=updates_done, epoch=epoch + 1, global_step=global_step, history=history,
                            epochs_completed=epoch + 1, partial_epoch=False)
            log("      epoch checkpoint -> %s" % path)
        dist_utils.barrier()
        if stop:
            break

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.time() - t0
    metrics = dict(rank=rank, world_size=world_size, device=str(device), updates=len(history),
                   train_seconds=elapsed, seconds_per_update=elapsed / max(1, len(history)),
                   peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                   peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
                   max_len=cfg["max_len"], head_max_len=cfg["head_max_len"],
                   items=len(dataset), batches_per_rank=n_rank_batches,
                   batch_size_per_rank=cfg["batch_size"],
                   effective_batch=cfg["batch_size"] * cfg["accumulation"] * world_size)
    (run_dir / f"training_metrics.rank{rank}.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8")

    if dist_utils.is_main():
        (run_dir / "training_history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        bare = model.module if is_distributed else model
        save_checkpoint(bare, model_cfg, tok, out, cfg, step=updates_done, epoch=cfg["epochs"],
                        global_step=global_step, history=history)
        log("  trained in %.1f min | final bundle -> %s" % (elapsed / 60, out))
    dist_utils.barrier()
    return out
