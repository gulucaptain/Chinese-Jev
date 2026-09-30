"""Bounded source-parallel conversion followed by one global Chinese-Jev build.

Workers only parse independent source entries into temporary case spools.  The
parent process then performs global deduplication, conflict removal and grouped
splitting once, so parallelism cannot create per-worker dataset boundaries.
"""
from __future__ import annotations

import concurrent.futures
import json
import multiprocessing
import os
import tempfile
from pathlib import Path

from . import core
from .registry import registry
from .runner import _config, decorate_case

CORE_ADAPTERS = {
    "c3": core.convert_c3,
    "exam_csv": core.convert_exam_csv,
    "dureader_yesno": core.convert_yesno,
    "squad": core.convert_squad,
    "chinese_jev": core.convert_chinese_jev,
}


def _emit_audit(stream, source, sid, reason):
    core.emit(stream, {"source": source["name"], "source_id": str(sid), "reason": str(reason)})


def _convert_one(config_path, index, spool_dir):
    config_path = Path(config_path)
    config = _config(config_path)
    source = config["sources"][index]
    base = config_path.parent
    output = Path(spool_dir) / f"{index:05d}.cases.jsonl"
    rejected = Path(spool_dir) / f"{index:05d}.rejected.jsonl"
    review_path = Path(spool_dir) / f"{index:05d}.review.jsonl"
    adapters, _ = registry(config_path, config)
    all_adapters = {**CORE_ADAPTERS, **adapters}
    paths = [base / path for path in source.get("paths", [])]
    auxiliary = {key: base / value for key, value in source.get("auxiliary_paths", {}).items()}
    converter_source = dict(source, _resolved_auxiliary_paths=auxiliary)
    raw_files = list(dict.fromkeys([*paths, *auxiliary.values()]))
    if source["adapter"] == "t2ranking":
        raw_files = list(dict.fromkeys([
            *raw_files, *(base / source[key] for key in ("queries", "qrels", "collection"))
        ]))
    for path in raw_files:
        if not path.is_file():
            raise FileNotFoundError(path)

    converted = 0
    t2_work = None
    with output.open("w", encoding="utf-8") as cases, \
         rejected.open("w", encoding="utf-8") as audit_stream, \
         review_path.open("w", encoding="utf-8") as review:
        def audit(src, sid, reason):
            _emit_audit(audit_stream, src, sid, reason)
        if source["adapter"] == "t2ranking":
            t2_work = tempfile.TemporaryDirectory(prefix="t2-worker-")
            rows = core.convert_t2(source, base, Path(t2_work.name))
        else:
            if source["adapter"] not in all_adapters:
                raise ValueError(f"unsupported adapter {source['adapter']!r}")
            if not paths:
                raise ValueError(f"source {source['name']!r} has no paths")
            rows = all_adapters[source["adapter"]](converter_source, paths, audit, review)
        for case in rows:
            case = decorate_case(case, source)
            core.validate_case(case)
            core.emit(cases, case)
            converted += 1
    if t2_work is not None:
        t2_work.cleanup()
    return {
        "index": index,
        "source": source["name"],
        "converted": converted,
        "spool": str(output),
        "rejected": str(rejected),
        "review": str(review_path),
        "inputs": [{"path": os.path.relpath(path, base), "bytes": path.stat().st_size,
                    "sha256": core.file_sha256(path)} for path in raw_files],
    }


def _native_cases(source, paths, audit, review):
    for path in paths:
        for line, case in core.jsonl(path):
            try:
                yield core.validate_case(case)
            except (KeyError, TypeError, ValueError) as exc:
                audit(source, f"{path.name}:{line}", str(exc))


def _append_files(paths, target):
    with Path(target).open("w", encoding="utf-8") as output:
        for path in paths:
            with Path(path).open(encoding="utf-8") as source:
                for line in source:
                    output.write(line)


def build(config_path, output, *, workers=2):
    """Convert source entries concurrently, then run one global build pass."""
    if type(workers) is not int or not 1 <= workers <= 16:
        raise ValueError("workers must be an integer in 1..16")
    config_path = Path(config_path).resolve()
    config = _config(config_path)
    _, implementation_files = registry(config_path, config)
    # TemporaryDirectory may return a relative path on supported Python versions.
    # Absolute spool paths prevent core.build from resolving them against temp twice.
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="chinese-jev-parallel-", dir=output.parent) as temp:
        spool = Path(temp) / "spool"
        spool.mkdir()
        jobs = [(config_path, index, spool) for index in range(len(config["sources"]))]
        if workers == 1:
            results = [_convert_one(*job) for job in jobs]
        else:
            context = multiprocessing.get_context("spawn")
            with concurrent.futures.ProcessPoolExecutor(
                    max_workers=workers, mp_context=context) as pool:
                futures = [pool.submit(_convert_one, *job) for job in jobs]
                results = [future.result() for future in futures]
        results.sort(key=lambda row: row["index"])

        merged = dict(config)
        merged.pop("adapter_files", None)
        merged.pop("adapter_modules", None)
        merged["sources"] = []
        for result, original in zip(results, config["sources"]):
            source = dict(original)
            source["original_adapter"] = source["adapter"]
            source["adapter"] = "parallel_cases"
            source["paths"] = [result["spool"]]
            source.pop("auxiliary_paths", None)
            for key in ("queries", "qrels", "collection"):
                source.pop(key, None)
            merged["sources"].append(source)
        merge_config = Path(temp) / "merge.json"
        merge_config.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
        core.build(merge_config, output, extra_adapters={"parallel_cases": _native_cases},
                   adapter_files=[__file__])

        _append_files([row["rejected"] for row in results], output / "conversion-rejected.jsonl")
        _append_files([row["review"] for row in results], output / "conversion-review.jsonl")
        public_results = [{key: value for key, value in row.items()
                           if key not in {"spool", "rejected", "review"}}
                          for row in results]
        parallel = {"backend": "source_parallel_global_merge", "workers": workers,
                    "sources": public_results}
        (output / "parallel.json").write_text(
            json.dumps(parallel, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest_path = output / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["config"] = config
        manifest["execution"] = {"backend": parallel["backend"], "workers": workers}
        manifest["inputs"] = [{"source": row["source"], "files": row["inputs"]}
                              for row in public_results]
        manifest["adapter_sha256"] = core.implementation_hashes(implementation_files)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return manifest
