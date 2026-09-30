"""The decision model, its reward, and the metrics a run is judged by.

`DecisionModel` is a bidirectional encoder with a typed decision head: the encoder reads
the whole sequence, the head re-attends over it with the question type embedded in, and
one logit is scored per `[MASK]` marker (see `sequence.py` for the layout). Training
optimizes `proper_reward`, a strictly proper scoring rule, so the model is rewarded for
reporting its actual belief distribution rather than for sharpening toward the argmax.
"""
from __future__ import annotations

import math
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .sequence import QTYPES


class DecisionModel(nn.Module):
    """Bidirectional transformer encoder backbone + typed decision head."""

    def __init__(self, encoder, head_layers=2, n_act=2, dropout=0.1):
        super().__init__()
        self.encoder = encoder
        d = encoder.config.hidden_size
        nhead = max(1, d // 64)
        layer = nn.TransformerEncoderLayer(d, nhead, 4 * d, dropout, batch_first=True, norm_first=True)
        self.head = nn.TransformerEncoder(layer, head_layers, enable_nested_tensor=False) if head_layers > 0 else None
        self.type_emb = nn.Embedding(3, d)
        self.scorer = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.act_head = nn.Sequential(nn.Linear(d + 4, 256), nn.GELU(), nn.Linear(256, n_act))
        self.register_buffer("temperature", torch.ones(3))
        self.head_checkpointing = False

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype, detach_encoder=False):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        if detach_encoder:
            h = h.detach()
        h = h + self.type_emb(qtype)[:, None, :]
        if self.head is not None:
            pad = ~attention_mask.bool()
            for layer in self.head.layers:
                if self.head_checkpointing and self.training and torch.is_grad_enabled():
                    # Non-reentrant checkpointing also trains the head when its input
                    # is frozen. Default RNG preservation keeps dropout consistent.
                    h = checkpoint(layer, h, src_key_padding_mask=pad, use_reentrant=False)
                else:
                    h = layer(h, src_key_padding_mask=pad)
        idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
        m = torch.gather(h, 1, idx)
        logits = self.scorer(m).squeeze(-1).float()
        logits = logits.masked_fill(~marker_mask, -1e4)

        p = torch.softmax(logits.detach(), -1)
        k = marker_mask.sum(-1).clamp(min=2).float()
        ent = -(p * torch.log(p.clamp_min(1e-9))).sum(-1) / torch.log(k)
        if p.size(-1) >= 2:
            top2 = p.topk(2, -1).values
        else:
            # A single-option question has exactly one marker, so p.topk(2, ...) has
            # nothing to select for the second slot and raises. The answer is still
            # well-defined: softmax over one logit is 1.0 regardless of its value, so pad
            # the missing second entry with 0.0 — that gives the act head
            # top1 - top2 == 1.0, the same "fully decided" signal it would see for any
            # other unambiguous top-1-vs-rest gap.
            top1 = p.topk(1, -1).values
            top2 = torch.cat([top1, torch.zeros_like(top1)], dim=-1)
        feats = torch.stack([top2[:, 0], top2[:, 0] - top2[:, 1], ent, k / 255.0], -1)
        pooled = h[:, 0].float()
        act_logits = self.act_head(torch.cat([pooled, feats], -1))
        return logits, act_logits


def _apply_rope_config(ecfg):
    """Carry transformers>=5 per-layer RoPE settings over to the attributes 4.x reads.

    A checkpoint re-saved by transformers 5 stores RoPE as
    `rope_parameters = {"full_attention": {"rope_theta": ...}, "sliding_attention": {...}}`.
    transformers 4.x does not know that key, so it keeps its own defaults and any
    checkpoint whose sliding-attention theta differs silently runs the wrong RoPE base —
    mmBERT is exactly that case, both of its thetas are 160000. Map the values onto
    `global_rope_theta` / `local_rope_theta`, which 4.x does read. On transformers 5 this
    is a no-op beyond re-setting the same numbers.
    """
    rope = getattr(ecfg, "rope_parameters", None)
    if not isinstance(rope, dict):
        return
    flat = rope.get("rope_theta")
    for layer_type, attr in (("full_attention", "global_rope_theta"),
                             ("sliding_attention", "local_rope_theta")):
        params = rope.get(layer_type)
        theta = params.get("rope_theta") if isinstance(params, dict) else flat
        if theta is not None and hasattr(ecfg, attr):
            setattr(ecfg, attr, float(theta))


def build_model(cfg, encoder_dir=None, pretrained=True):
    """The model a bundle's `model_config` describes, optionally with fresh encoder weights.

    `pretrained=False` with an explicit `encoder_dir` builds the architecture from the
    local encoder config without downloading anything; the caller then loads the bundle's
    own weights over it.
    """
    from transformers import AutoConfig, AutoModel

    if not pretrained or (encoder_dir and os.path.exists(encoder_dir)):
        ecfg = AutoConfig.from_pretrained(encoder_dir or cfg["encoder"])
        _apply_rope_config(ecfg)
        enc = AutoModel.from_config(ecfg, attn_implementation="sdpa")
    else:
        enc = AutoModel.from_pretrained(cfg["encoder"], attn_implementation="sdpa")
    return DecisionModel(enc, cfg.get("head_layers", 2), len(cfg.get("act_costs", {})) + 1)


def proper_reward(q, target, qtype, mask, w_sph=0.5, w_rps=1.0, log_floor=-9.21):
    """Strictly proper scoring rule reward: log score + spherical score, minus a ranked
    probability score penalty for ordered (`score`) questions.

    q: [..., N, K] reported distributions; target: [N, K] one-hot or soft targets.
    """
    q = q * mask
    logq = torch.log(q.clamp_min(1e-12)).clamp_min(log_floor)
    log_score = (target * logq).sum(-1)
    sph = (target * q).sum(-1) / q.norm(dim=-1).clamp_min(1e-9)
    r = log_score + w_sph * sph
    is_score = (qtype == QTYPES["score"]).float()
    if is_score.any():
        k = mask.sum(-1).clamp(min=2).float()
        cdf_q = torch.cumsum(q, -1)
        cdf_t = torch.cumsum(target, -1)
        rps = (((cdf_q - cdf_t) ** 2) * mask).sum(-1) / (k - 1)
        r = r - w_rps * rps * is_score
    return r


def ece_score(conf, correct, bins=15):
    """Expected Calibration Error across confidence bins."""
    if len(conf) == 0:
        return float("nan")
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        sel = (conf >= lo if i == 0 else conf > lo) & (conf <= hi)
        if sel.any():
            e += sel.mean() * abs(conf[sel].mean() - correct[sel].mean())
    return float(e)


def answer_confidence(p, k):
    """Probability mass on the answer being reported: max(p).

    This is the quantity temperature scaling fits and the quantity `ece_score` is computed
    on, so it carries the property calibration promises: of the answers returned at
    confidence c, about c of them are right.
    """
    if k < 1:
        return 1.0
    return float(np.clip(np.max(p[:k]), 0.0, 1.0))


def confidence_from_probs(p, k):
    """Normalized Shannon entropy confidence: 1 - H(p) / log(k).

    How concentrated the whole distribution is. Useful, but not calibrated: it is not what
    temperature scaling fits and not what the reported ECE measures. See `answer_confidence`.
    """
    if k < 2:
        return 1.0
    p = p[:k]
    ent = -(p * np.log(np.clip(p, 1e-12, 1.0))).sum()
    return float(np.clip(1.0 - ent / math.log(k), 0.0, 1.0))
