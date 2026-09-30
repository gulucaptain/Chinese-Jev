"""Batching over the token cache.

Two things keep this cheap on a long-document corpus.

Batches are assembled by **length bucket** rather than at random. With `max_len=4096` and a
median item of ~200 tokens, a random batch of 8 pads every row to its longest member, and
the encoder pays for the padded width. Sorting into buckets first groups similar lengths,
which cuts the batch's width to roughly the longest member's *own* length instead of the
corpus maximum.

Reconstruction is deferred to the point of use. `__getitem__` returns an index, not a
sequence, and `collate_sequences` rebuilds the batch's sequences at the run's `max_len`. A
cache therefore survives a change of `max_len`, and the decode cost is paid once per visit
instead of once per tokenization.
"""
from __future__ import annotations

import json

import numpy as np
import torch
from torch.utils.data import Dataset

from .cache import TokenCache, filter_items


class CachedSplit(Dataset):
    """A split's item indices, filtered to what the run's max_len can hold."""

    def __init__(self, directory, max_len, overlong="filter", max_items=None, seed=42):
        self.cache = TokenCache(directory)
        keep = filter_items(self.cache, max_len, overlong)
        if max_items is not None and len(keep) > max_items:
            rng = np.random.default_rng(seed)
            keep = np.sort(rng.choice(keep, size=max_items, replace=False))
        self.indices = keep
        self.max_len = max_len
        self.overlong = overlong

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return int(self.indices[i])

    def item(self, index):
        return self.cache.load_sequence(index, self.max_len)

    def stats(self):
        m = self.cache.manifest
        return {"dir": str(self.cache.dir), "total_items": m["n_items"],
                "kept_items": len(self.indices), "max_len": self.max_len,
                "overlong": self.overlong}


def collate_sequences(item_dicts, pad_id):
    """Pad a list of rebuilt items into one batch."""
    n = len(item_dicts)
    L = max(len(it["ids"]) for it in item_dicts)
    k_max = max(len(it["markers"]) for it in item_dicts)
    ids = torch.full((n, L), pad_id, dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.long)
    mpos = torch.zeros((n, k_max), dtype=torch.long)
    mmask = torch.zeros((n, k_max), dtype=torch.bool)
    target = torch.zeros((n, k_max), dtype=torch.float32)
    for i, it in enumerate(item_dicts):
        L_i = len(it["ids"])
        ids[i, :L_i] = torch.tensor(it["ids"], dtype=torch.long)
        att[i, :L_i] = 1
        k = len(it["markers"])
        mpos[i, :k] = torch.tensor(it["markers"], dtype=torch.long)
        mmask[i, :k] = True
        target[i, :k] = torch.tensor(it["target"], dtype=torch.float32)
    return {
        "input_ids": ids,
        "attention_mask": att,
        "marker_pos": mpos,
        "marker_mask": mmask,
        "target": target,
        "qtype": torch.tensor([it["qtype"] for it in item_dicts], dtype=torch.long),
        "label": torch.tensor([it["label"] for it in item_dicts], dtype=torch.long),
    }


class BatchPlan:
    """The epoch's batches, ordered by length bucket and divided across ranks.

    Built by the main process and broadcast as a plain nested list, so every rank agrees on
    the plan without each of them recomputing it from a shared seed. The plan is a list of
    lists of item indices; nothing about it is model-specific.
    """

    def __init__(self, batches):
        self.batches = batches

    def __len__(self):
        return len(self.batches)

    def shard(self, rank, world_size):
        """Round-robin, so every rank sees the same mix of sources and lengths.

        Contiguous blocks would give rank 0 the shortest buckets and the last rank the
        longest, which turns a length-balanced plan into a load imbalance across ranks.
        """
        return self.batches[rank::world_size]

    def to_json(self):
        return json.dumps(self.batches)

    @classmethod
    def from_json(cls, text):
        return cls(json.loads(text))


def plan_epoch(indices, batch_size, *, length_bucketed=True, seed=0, lengths=None,
               max_tokens_per_batch=None):
    """Order the epoch's items into batches.

    Length-bucketed grouping is done in two passes: shuffle, then sort by length, then
    chunk, then shuffle the chunks. Sorting alone would make every epoch visit the corpus
    in length order, which correlates with source and question type and would give the
    optimiser a systematically biased gradient sequence.
    """
    indices = list(indices)
    rng = np.random.default_rng(seed)
    if not length_bucketed or lengths is None:
        rng.shuffle(indices)
        return [[int(i) for i in indices[s:s + batch_size]]
                for s in range(0, len(indices), batch_size)]

    order = rng.permutation(len(indices))
    # Sort within large chunks rather than globally: a global sort would put every long
    # document in the last few batches, so the epoch's tail would be all long items.
    chunk = max(batch_size * 64, 1024)
    grouped = []
    for start in range(0, len(order), chunk):
        window = order[start:start + chunk]
        window = window[np.argsort(np.asarray([lengths[indices[i]] for i in window]))]
        grouped.extend(window.tolist())

    batches = [int(i) for i in grouped]
    out = []
    if max_tokens_per_batch is None:
        for s in range(0, len(batches), batch_size):
            out.append(batches[s:s + batch_size])
    else:
        # Greedy packing under a token cap: long items get smaller batches, short ones
        # larger, so a step costs roughly the same whichever bucket it landed in.
        current, width = [], 0
        for i in batches:
            n = int(lengths[i])
            if current and (width + n) > max_tokens_per_batch:
                out.append(current)
                current, width = [], 0
            current.append(i)
            width += n
        if current:
            out.append(current)
    # Shuffle the batch order, not the contents: contents are the length bucket.
    rng.shuffle(out)
    return out
