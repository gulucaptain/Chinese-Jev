"""Parallel tokenization into the binary cache, and the length report that sizes a run.

Work is split by *case range*, not by file: the corpus is one big JSONL stream, so a worker
takes `[start, end)` of the case ordinals and writes a shard. The merge then concatenates
shards in index order, which makes the cache byte-identical regardless of how many workers
ran. That determinism is what lets a cache built on four cores be trusted by a run on
thirty-two — and what makes a shared `cache_dir` safe to reuse across runs.

Memory is bounded at every stage. Each worker holds one shard's items; the merge never
holds more than one shard either: array shapes are known up front from the per-shard
summaries, the `.npy` files are pre-allocated sparsely (untouched padding reads as zero
without ever being written), and items stream shard by shard into memory-mapped windows.
A 10M-item corpus therefore costs the same peak RAM as a 10k-item one.

A worker needs random access to the stream to take a range without reading from the start,
so gzipped input is decompressed once into a plain JSONL plus a byte-offset index. The
index is the same trick the training reader uses, and it costs one pass over the corpus.
"""
from __future__ import annotations

import json
import os
import shutil
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .cache import CacheError
from .encoding import encode_case
from .tokenizer import load_tokenizer, tokenizer_fingerprint

FORMAT_VERSION = 1
# Octuple-word aligned rows keep the numpy writes amenable to SIMD copies; the row width is
# the prompt ceiling, and items shorter than it are zero-padded.
_ROW_ALIGN = 8
_LENGTH_BINS = (128, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096,
                6144, 8192, 16384, 32768, 65536)


class _Worker:
    """Per-process tokenizer state.

    The tokenizer is loaded once per worker process and reused, because loading it per
    shard dominated the cost of a shard.
    """

    _tok = None
    _bundle = None

    @classmethod
    def tok(cls, bundle):
        if cls._tok is None or cls._bundle != bundle:
            cls._tok = load_tokenizer(bundle)
            cls._bundle = bundle
        return cls._tok


def _shard_task(args):
    """Tokenize one case range into a shard file, returning the shard's summary.

    The summary carries everything the merge needs to size the arrays and build the
    manifest (widths, counts, histograms), so the items themselves only ever travel
    through the shard file — never through process-pool pickling or a merged list.
    """
    (shard_id, bundle, start, end, head_max_len, flat_path, offsets_path, out_dir) = args
    tok = _Worker.tok(bundle)

    summary = {"shard": shard_id, "n_items": 0, "failed_cases": 0, "n_skipped": 0,
               "prompt_width": 0, "state_width": 0, "k_max": 0, "state_truncated": 0,
               "lengths": Counter(), "qtype_counts": Counter(), "source_counts": Counter(),
               "n_options_histogram": Counter()}
    offsets = np.load(offsets_path, mmap_mode="r")
    # One JSON object per line, not a pickled blob: the shard is plain text, which keeps
    # the worker's output inspectable and cheap to stream at merge time.
    items_path = Path(out_dir) / ("shard_%05d_items.jsonl" % shard_id)
    skipped_path = Path(out_dir) / ("shard_%05d_skipped.jsonl" % shard_id)
    with open(flat_path, "rb") as f, \
            items_path.open("w", encoding="utf-8") as items_f, \
            skipped_path.open("w", encoding="utf-8") as skipped_f:
        f.seek(int(offsets[start]))

        def skip(cid, reason):
            summary["n_skipped"] += 1
            skipped_f.write(json.dumps({"id": cid, "reason": reason}, ensure_ascii=False) + "\n")

        for _ in range(start, end):
            line = f.readline()
            if not line:
                break
            try:
                case = json.loads(line)
            except json.JSONDecodeError:
                summary["failed_cases"] += 1
                continue
            try:
                items, why = encode_case(case, tok, head_max_len)
            except Exception as e:  # a malformed case must not kill a 32-worker run
                summary["failed_cases"] += 1
                skip(case.get("id", "?"), "%s: %s" % (type(e).__name__, e))
                continue
            for qid, reason in why:
                skip("%s/%s" % (case.get("id", "?"), qid), reason)
            for it in items:
                items_f.write(json.dumps(it, ensure_ascii=False))
                items_f.write("\n")
                p, s, k = len(it["prompt_ids"]), len(it["state_ids"]), len(it["markers"])
                summary["n_items"] += 1
                summary["prompt_width"] = max(summary["prompt_width"], p)
                summary["state_width"] = max(summary["state_width"], s)
                summary["k_max"] = max(summary["k_max"], k)
                summary["state_truncated"] += int(it["state_truncated"])
                summary["lengths"][p + s + 1] += 1
                summary["qtype_counts"][it["qtype_name"]] += 1
                summary["source_counts"][it["source"]] += 1
                summary["n_options_histogram"][str(k)] += 1
    return summary


def _flatten_stream(adapter, split, limit, flat_path, offsets_path, verbose=True):
    """Decompress one pass into a seekable JSONL and record each line's offset."""
    offsets = [0]
    n = 0
    t0 = time.time()
    with open(flat_path, "w", encoding="utf-8") as out:
        for case in adapter.iter_cases(split, limit=limit):
            out.write(json.dumps(case, ensure_ascii=False))
            out.write("\n")
            n += 1
            offsets.append(out.tell())
            if verbose and n % 500000 == 0:
                print("    streamed %s cases (%.0fs)" % (f"{n:,}", time.time() - t0), flush=True)
    # The final offset is the end of the last line, so it doubles as "one past the end".
    np.save(offsets_path, np.array(offsets[:-1], dtype=np.int64))
    return n


def tokenize_split(cfg, adapter, split, out_dir, bundle, verbose=True, log=print):
    """Build one split's cache at `out_dir`. Returns the manifest."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    staging = out_dir / "_shards"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    flat = staging / "stream.jsonl"
    offsets_path = staging / "offsets.npy"
    log("  streaming %s to disk..." % split)
    n_cases = _flatten_stream(adapter, split, cfg["max_items"], flat, offsets_path, verbose)
    if n_cases == 0:
        # Distinct from the every-question-skipped error below: here the source itself is
        # empty. The usual way to get here is a freshly `init`ed data-pipeline project,
        # whose template builds validation/test with zero holdout fractions.
        try:
            where = " (%s)" % adapter.split_path(split)
        except NotImplementedError:
            where = ""
        raise CacheError("split %r contains no cases%s; fill the split, or remove it from "
                         "dataset_params.splits if it genuinely has no data" % (split, where))

    tok = load_tokenizer(bundle)
    head_max_len = cfg["head_max_len"]
    if head_max_len is None:
        raise CacheError("head_max_len is required for tokenization; run `chinese-jev stats` first")

    workers = cfg["tokenize_workers"] or max(1, min(os.cpu_count() // 4, 32))
    chunk = max(64, cfg["tokenize_chunk"])
    ranges = [(s, min(s + chunk, n_cases)) for s in range(0, n_cases, chunk)]
    log("  tokenizing %s %s cases across %d workers (%d tasks)..."
        % (f"{n_cases:,}", split, workers, len(ranges)))

    tasks = [(i, str(bundle), s, e, head_max_len, str(flat), str(offsets_path), str(staging))
             for i, (s, e) in enumerate(ranges)]

    t0 = time.time()
    if workers == 1:
        summaries = [_shard_task(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            summaries = []
            for i, s in enumerate(pool.map(_shard_task, tasks, chunksize=1)):
                summaries.append(s)
                if verbose and (i + 1) % max(1, len(tasks) // 10) == 0:
                    log("    %d/%d shards (%.0fs)" % (i + 1, len(tasks), time.time() - t0))

    total = _combine_summaries(summaries)
    if verbose and total["n_skipped"]:
        _log_skip_preview(staging, summaries, log)
    log("    %s items from %s cases in %.0fs (%d cases failed to parse, %d questions skipped)"
        % (f"{total['n_items']:,}", f"{n_cases:,}", time.time() - t0,
           total["failed_cases"], total["n_skipped"]))
    if total["n_items"] == 0:
        raise CacheError("no usable items in split %r; every question was skipped" % split)

    manifest = _write_cache(staging, summaries, total, out_dir, split, tok, head_max_len, cfg,
                            n_cases=n_cases, adapter_name=adapter.registry_name,
                            adapter_params={**adapter.params, "max_items": cfg["max_items"]},
                            verbose=verbose, log=log)
    shutil.rmtree(staging, ignore_errors=True)
    return manifest


def _combine_summaries(summaries):
    total = {"n_items": 0, "failed_cases": 0, "n_skipped": 0, "state_truncated": 0,
             "prompt_width": 0, "state_width": 0, "k_max": 0,
             "lengths": Counter(), "qtype_counts": Counter(), "source_counts": Counter(),
             "n_options_histogram": Counter()}
    for s in summaries:
        for key in ("n_items", "failed_cases", "n_skipped", "state_truncated"):
            total[key] += s[key]
        for key in ("prompt_width", "state_width", "k_max"):
            total[key] = max(total[key], s[key])
        for key in ("lengths", "qtype_counts", "source_counts", "n_options_histogram"):
            total[key].update(s[key])
    return total


def _log_skip_preview(staging, summaries, log, limit=20):
    shown = 0
    for s in summaries:
        if shown >= limit:
            break
        path = Path(staging) / ("shard_%05d_skipped.jsonl" % s["shard"])
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    log("      skipped %s" % line.strip()[:160])
                    shown += 1
                    if shown >= limit:
                        break


def _allocate_npy(path, dtype, shape):
    """A fresh sparse `.npy` file: a real numpy header, then a hole of zeros.

    Rows are filled in place through memory-mapped windows; padding beyond each row's real
    length is never written at all, and sparse-file semantics guarantee it reads as zero.
    This is what lets the merge build a multi-hundred-GiB cache without ever allocating it.
    """
    dtype = np.dtype(dtype)
    with open(path, "wb") as f:
        np.lib.format.write_array_header_2_0(
            f, dict(descr=dtype.str, fortran_order=False, shape=tuple(shape)))
        header = f.tell()
        f.truncate(header + int(np.prod(shape)) * dtype.itemsize)
    return header


def _write_cache(staging, summaries, total, out_dir, split, tok, head_max_len, cfg, *,
                 n_cases, adapter_name, adapter_params, verbose, log):
    """Stream the shards, in order, into the memory-mappable arrays."""
    n = total["n_items"]
    prompt_width = int(np.ceil(total["prompt_width"] / _ROW_ALIGN) * _ROW_ALIGN)
    state_width = int(np.ceil(total["state_width"] / _ROW_ALIGN) * _ROW_ALIGN)
    k_max = total["k_max"]

    layout = {
        "tokens.i32.npy": (np.uint32, (n, prompt_width)),
        "prompt_len.i32.npy": (np.int32, (n,)),
        "state.i32.npy": (np.uint32, (n, state_width)),
        "state_len.i32.npy": (np.int32, (n,)),
        "markers.i32.npy": (np.int32, (n, k_max)),
        "n_options.i16.npy": (np.int16, (n,)),
        "targets.f32.npy": (np.float32, (n, k_max)),
    }
    headers = {name: _allocate_npy(out_dir / name, dtype, shape)
               for name, (dtype, shape) in layout.items()}

    t0 = time.time()
    written = 0
    next_report = 500000
    with (out_dir / "meta.jsonl").open("w", encoding="utf-8") as meta_f:
        for s in sorted(summaries, key=lambda s: s["shard"]):
            count = s["n_items"]
            shard = Path(staging) / ("shard_%05d_items.jsonl" % s["shard"])
            if count:
                windows = {}
                for name, (dtype, shape) in layout.items():
                    row = shape[1:]
                    row_bytes = int(np.prod(row)) * np.dtype(dtype).itemsize
                    if row_bytes:
                        windows[name] = np.memmap(out_dir / name, dtype=dtype, mode="r+",
                                                  offset=headers[name] + written * row_bytes,
                                                  shape=(count, *row))
                    else:
                        # A zero-width array (every state empty) has no bytes to map;
                        # writes land in a throwaway buffer and the file stays header-only.
                        windows[name] = np.empty((count, *row), dtype=dtype)
                i = 0
                with shard.open("r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        it = json.loads(line)
                        p, st, k = len(it["prompt_ids"]), len(it["state_ids"]), len(it["markers"])
                        windows["tokens.i32.npy"][i, :p] = it["prompt_ids"]
                        windows["prompt_len.i32.npy"][i] = p
                        windows["state.i32.npy"][i, :st] = it["state_ids"]
                        windows["state_len.i32.npy"][i] = st
                        windows["markers.i32.npy"][i, :k] = it["markers"]
                        windows["n_options.i16.npy"][i] = k
                        windows["targets.f32.npy"][i, :k] = it["target"]
                        meta_f.write(json.dumps({
                            "case_id": it["case_id"], "question_id": it["question_id"],
                            "source": it["source"], "qtype": it["qtype"],
                            "qtype_name": it["qtype_name"], "label": it["label"],
                            "n_options": it["n_options"],
                            "state_truncated": it["state_truncated"], **it["meta"],
                        }, ensure_ascii=False) + "\n")
                        i += 1
                if i != count:
                    raise CacheError("shard %d holds %d items, its summary says %d"
                                     % (s["shard"], i, count))
                for w in windows.values():
                    if isinstance(w, np.memmap):
                        w.flush()
                windows.clear()
                written += count
            shard.unlink()
            if verbose and written >= next_report:
                log("    merged %s/%s items (%.0fs)" % (f"{written:,}", f"{n:,}", time.time() - t0))
                next_report += 500000
    if written != n:
        raise CacheError("merged %d items, summaries promised %d" % (written, n))

    manifest = {
        "format_version": FORMAT_VERSION,
        "split": split,
        "adapter": adapter_name,
        # Recorded so a later `tokenize` can tell whether this cache still describes the
        # configured data path instead of silently training on a stale one.
        "adapter_params": adapter_params,
        "n_items": int(n),
        "n_cases": int(n_cases),
        "n_questions_skipped": int(total["n_skipped"]),
        "n_cases_failed": int(total["failed_cases"]),
        "prompt_width": int(prompt_width),
        "state_width": int(state_width),
        "k_max": int(k_max),
        "head_max_len": int(head_max_len),
        "max_len_ceiling": int(cfg["max_len"]) if cfg.get("max_len") else None,
        "cls_token_id": int(tok.cls_token_id),
        "sep_token_id": int(tok.sep_token_id),
        "mask_token_id": int(tok.mask_token_id),
        "pad_token_id": int(tok.pad_token_id),
        "tokenizer_fingerprint": tokenizer_fingerprint(tok),
        "length_histogram": _histogram(total["lengths"]),
        "qtype_counts": _sorted_counts(total["qtype_counts"]),
        "source_counts": _sorted_counts(total["source_counts"]),
        "n_options_histogram": _sorted_counts(total["n_options_histogram"]),
        "state_truncated": int(total["state_truncated"]),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    return manifest


def _histogram(length_counts, bins=_LENGTH_BINS):
    """The length report, computed from counts so no N-item array is ever materialised.

    Percentiles use numpy's linear interpolation over the sorted values, so the numbers
    match what `np.percentile` would say about the expanded array.
    """
    values = np.array(sorted(length_counts), dtype=np.int64)
    weights = np.array([length_counts[v] for v in values], dtype=np.int64)
    cumulative = weights.cumsum()
    n = int(cumulative[-1])
    out = {f"<= {b}": int(weights[values <= b].sum()) for b in bins}
    out["max"] = int(values[-1])
    for q in (50, 90, 95, 99, 99.9):
        rank = (n - 1) * q / 100.0
        lo = values[np.searchsorted(cumulative, int(np.floor(rank)), side="right")]
        hi = values[np.searchsorted(cumulative, int(np.ceil(rank)), side="right")]
        out["p%s" % q] = int(lo + (hi - lo) * (rank - np.floor(rank)))
    out["mean"] = float(np.dot(values, weights) / n)
    return out


def _sorted_counts(counter):
    return dict(sorted(counter.items(), key=lambda kv: -kv[1]))


def write_manifest_summary(work_dir, manifests):
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    (Path(work_dir) / "prepare_summary.json").write_text(
        json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "length_histogram"}
                    for k, v in manifests.items()}, ensure_ascii=False, indent=2), encoding="utf-8")


def write_dataset_info(cfg, manifests):
    """The dataset descriptor a training config points at via `dataset_info`.

    Written at the cache root, next to the split directories it describes, so the file's
    own location *is* the cache location — a training config that names this file needs no
    other dataset knowledge. Everything here restates the split manifests; the file exists
    so that one artifact, not a config convention, carries "what this dataset is".
    """
    from .config import cache_root

    root = cache_root(cfg)
    ref = manifests.get("train") or next(iter(manifests.values()))
    h = ref["length_histogram"]
    recommended = next((cap for cap in (1024, 2048, 4096, 6144, 8192)
                        if h.get(f"<= {cap}", 0) >= 0.995 * ref["n_items"]), None)
    info = {
        "format_version": FORMAT_VERSION,
        "dataset": cfg["dataset"],
        "dataset_params": cfg["dataset_params"],
        "max_items": cfg["max_items"],
        "head_max_len": int(cfg["head_max_len"]),
        "recommended_max_len": recommended,
        "tokenized_with_bundle": cfg["bundle"],
        "tokenizer_fingerprint": ref.get("tokenizer_fingerprint"),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "splits": {
            split: {
                "n_items": m["n_items"],
                "n_cases": m["n_cases"],
                "n_questions_skipped": m["n_questions_skipped"],
                "prompt_width": m["prompt_width"],
                "state_width": m["state_width"],
                "k_max": m["k_max"],
                "length": {k: m["length_histogram"][k]
                           for k in ("p50", "p90", "p99", "p99.9", "max", "mean")},
            }
            for split, m in manifests.items()
        },
    }
    path = root / "dataset.json"
    path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def fill_measured_values(cfg, info_path, config_path):
    """Write the values `prepare` measured back into the config file itself.

    One config drives both `prepare` and `run`; the data-derived fields the user left
    null (`head_max_len`, and `max_len` when the key is absent) are filled from the
    measurement so the file becomes self-contained. Nothing the user wrote is touched.
    Configs that use `dataset_info` instead of inline dataset fields are left alone —
    their dataset.json already carries the measurement.

    Returns the dict of keys actually written, empty if none.
    """
    path = Path(config_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("dataset_info"):
        return {}
    info = json.loads(Path(info_path).read_text(encoding="utf-8"))
    filled = {}
    if raw.get("head_max_len") is None:
        filled["head_max_len"] = info["head_max_len"]
    if "max_len" not in raw and info.get("recommended_max_len"):
        filled["max_len"] = info["recommended_max_len"]
    if filled:
        raw.update(filled)
        path.write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return filled


def print_length_report(manifests, log=print):
    """The table a human reads to pick max_len."""
    log("")
    log("Length distribution (prompt + state + 1), by split:")
    log("  %-12s %10s %8s %8s %8s %8s %8s %8s" %
        ("split", "items", "p50", "p90", "p99", "p99.9", "max", "mean"))
    for split, m in manifests.items():
        h = m["length_histogram"]
        log("  %-12s %10s %8d %8d %8d %8d %8d %8.1f" %
            (split, f"{m['n_items']:,}", h["p50"], h["p90"], h["p99"], h["p99.9"], h["max"], h["mean"]))
    log("")
    log("Items retained as a function of max_len:")
    ref = manifests.get("train") or next(iter(manifests.values()))
    total = ref["n_items"]
    h = ref["length_histogram"]
    for cap in (1024, 2048, 4096, 6144, 8192):
        key = f"<= {cap}"
        if key in h:
            log("  max_len=%-6d keeps %10s / %s (%5.1f%%)"
                % (cap, f"{h[key]:,}", f"{total:,}", 100.0 * h[key] / max(1, total)))
    log("")
