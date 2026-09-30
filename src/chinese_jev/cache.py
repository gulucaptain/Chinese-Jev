"""The on-disk token cache, and the read path that turns it back into sequences.

Layout of `<cache_root>/<split>/`:

    tokens.i32.npy      padded prompt ids, uint32, (n_items, prompt_width)
    prompt_len.i32.npy  real prompt length, (n_items,)
    state.i32.npy       padded state ids, uint32, (n_items, state_width)
    state_len.i32.npy   real state length, (n_items,)
    markers.i32.npy     marker positions *relative to the start of the sequence*, (n, k_max)
    n_options.i16.npy   real option count, (n_items,)
    targets.f32.npy     target distribution, (n_items, k_max)
    meta.jsonl          one JSON object per item, line-aligned with the arrays
    manifest.json       widths, geometry, tokenizer fingerprint, and the length histogram

Why prompt and state are separate arrays is the whole point of this format.
`sequence.build_sequence` lays the sequence out as

    [CLS] <type> instructions [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] <state> [SEP]

so the state sits last and takes whatever room is left. Storing the prompt once and
keeping the state's *full* tokenization lets `load_sequence` rebuild the exact sequence
`build_sequence` would build, for any `max_len`, without re-tokenizing anything. Changing
`max_len` between runs is therefore a filter on `prompt_len + state_len`, not a rebuild —
which is also why the cache lives in its own configurable directory (`cache_dir`) rather
than being owned by any single run.

The stored marker positions are the ones a full-length build produced; they are clipped to
`max_len` at read time, which reproduces `build_sequence`'s own `[m for m in markers if m
< max_len]`.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

MASK_POSITION_SENTINEL = -1


class CacheError(ValueError):
    """A cache that cannot be trusted to describe the run it is being used for."""


def _load(path, dtype, ndim):
    arr = np.load(path, mmap_mode="r")
    if arr.dtype != np.dtype(dtype):
        raise CacheError("%s has dtype %s, expected %s" % (path, arr.dtype, dtype))
    if arr.ndim != ndim:
        raise CacheError("%s has %d dims, expected %d" % (path, arr.ndim, ndim))
    return arr


class TokenCache:
    """Random-access reader over one split's arrays.

    Arrays are memory-mapped, so a split larger than RAM is fine and process start-up does
    not pay for reading it. This is also why the DataLoader can use workers: no file
    handle or seek state is shared between processes.
    """

    def __init__(self, directory):
        self.dir = Path(directory)
        manifest_path = self.dir / "manifest.json"
        if not manifest_path.exists():
            raise CacheError("no cache at %s (run `chinese-jev tokenize` first)" % self.dir)
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("format_version") != 1:
            raise CacheError("%s was written by a different pipeline version (format %r)"
                             % (manifest_path, self.manifest.get("format_version")))
        self.tokens = _load(self.dir / "tokens.i32.npy", np.uint32, 2)
        self.prompt_len = _load(self.dir / "prompt_len.i32.npy", np.int32, 1)
        self.state = _load(self.dir / "state.i32.npy", np.uint32, 2)
        self.state_len = _load(self.dir / "state_len.i32.npy", np.int32, 1)
        self.markers = _load(self.dir / "markers.i32.npy", np.int32, 2)
        self.n_options = _load(self.dir / "n_options.i16.npy", np.int16, 1)
        self.targets = _load(self.dir / "targets.f32.npy", np.float32, 2)
        n = len(self.prompt_len)
        for name, arr in (("tokens", self.tokens), ("state", self.state), ("markers", self.markers),
                          ("n_options", self.n_options), ("targets", self.targets),
                          ("state_len", self.state_len)):
            if len(arr) != n:
                raise CacheError("cache arrays disagree on item count: %s has %d, prompt_len has %d"
                                 % (name, len(arr), n))
        self._meta = None

    def __len__(self):
        return len(self.prompt_len)

    @property
    def meta(self):
        """Item metadata, loaded lazily: training does not need it, reporting does."""
        if self._meta is None:
            path = self.dir / "meta.jsonl"
            rows = []
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    rows.append(json.loads(line))
            if len(rows) != len(self):
                raise CacheError("meta.jsonl has %d rows for %d items" % (len(rows), len(self)))
            self._meta = rows
        return self._meta

    @property
    def k_max(self):
        return int(self.manifest["k_max"])

    @property
    def head_max_len(self):
        return int(self.manifest["head_max_len"])

    def prompt_tokens(self, index):
        k = int(self.prompt_len[index])
        return self.tokens[index, :k].tolist()

    def state_tokens(self, index):
        k = int(self.state_len[index])
        return self.state[index, :k].tolist()

    def full_len(self, index):
        """The length this item would have at an unbounded max_len.

        The stored prompt already carries its own closing separator, so the only token
        added back is the one that closes the sequence after the state.
        """
        return int(self.prompt_len[index]) + int(self.state_len[index]) + 1

    def n_markers(self, index):
        return int(self.n_options[index])

    def load_sequence(self, index, max_len, pad_id=0):
        """Rebuild one training item at `max_len`.

        Mirrors `sequence.build_sequence` exactly for a full-length build and applies the
        same right-truncation of the state when the item is over budget, so an item
        trained from this cache carries the same sequence inference would build.
        """
        prompt = self.prompt_tokens(index)
        state = self.state_tokens(index)
        k = self.n_markers(index)
        markers = self.markers[index, :k].tolist()

        # The stored prompt already ends in its own [SEP], so it is exactly the `ids`
        # prefix build_sequence measures: `room = max_len - len(ids) - 1`, where the -1
        # reserves the separator that closes the sequence after the state.
        room = max(0, max_len - len(prompt) - 1)
        state = state[:room]

        ids = prompt + state + [pad_id]
        # The trailing pad token above is a placeholder for the closing separator; the
        # caller's pad id is not the separator, so fix it up with the real one.
        ids[-1] = self.manifest["sep_token_id"]
        markers = [m for m in markers if m < len(ids)]
        if len(markers) != k:
            # build_sequence drops just the same markers. It is only reachable when the
            # caller picked a max_len below the prompt's own width, which `filter_items`
            # refuses to hand out in "filter" mode.
            raise CacheError("item %d: max_len=%d cannot hold the prompt (%d tokens + %d markers)"
                             % (index, max_len, len(prompt), k))
        target = self.targets[index, :k].tolist()
        meta = self.meta[index]
        return {
            "ids": ids,
            "markers": markers,
            "qtype": int(meta["qtype"]),
            "target": target,
            "label": int(meta["label"]),
            "length": len(ids),
        }

    def item_meta(self, index):
        return self.meta[index]

    def fingerprint(self):
        return self.manifest.get("tokenizer_fingerprint")

    def summary(self):
        return self.manifest


def filter_items(cache, max_len, overlong="filter"):
    """The item indices this run should train on at `max_len`.

    `filter` keeps only items that fit; `truncate` keeps everything, at the cost of
    training a shortened state. Returning indices rather than a materialised list of items
    keeps the decision reversible and cheap: the caller can count them, sample them, or
    shard them without loading a single sequence.
    """
    n = len(cache)
    prompt_len = np.asarray(cache.prompt_len)
    state_len = np.asarray(cache.state_len)
    # The stored prompt is the ids prefix, so the sequence is prompt + state + one closing
    # separator: `build_sequence`'s own `room = max_len - len(ids) - 1`.
    full = prompt_len + state_len + 1
    if overlong == "filter":
        keep = np.flatnonzero(full <= max_len)
    elif overlong == "truncate":
        # An item can only be salvaged if the prompt itself fits; past that the options
        # no longer all survive, and dropping an option makes the target a description of
        # something the model cannot see.
        head_only = prompt_len + 1
        keep = np.flatnonzero(head_only <= max_len)
    else:
        raise CacheError("overlong must be 'filter' or 'truncate', got %r" % overlong)
    return keep.astype(np.int64)
