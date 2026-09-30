"""Build and independently verify a Chinese-Jev cases release."""
from __future__ import annotations

import collections
import json
import sqlite3
import tempfile
from pathlib import Path, PurePosixPath, PureWindowsPath
from urllib.parse import parse_qsl, urlsplit

from . import core
from .registry import registry


def _config(path):
    path = Path(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("format_version", 1) != 1:
        raise ValueError("unsupported format_version")
    if not isinstance(value.get("sources"), list) or not value["sources"]:
        raise ValueError("sources must be a nonempty list")

    sensitive = {"token", "access_token", "auth_token", "hf_token", "api_key", "apikey",
                 "password", "passwd", "secret", "authorization", "private_key"}
    credential_queries = sensitive | {
        "auth", "key", "signature", "sig", "credential",
        "x_amz_credential", "x_amz_signature", "x_goog_credential", "x_goog_signature",
    }

    def portable_path(raw, location):
        if not isinstance(raw, str) or not raw:
            raise ValueError(f"{location} must be a nonempty relative path")
        posix = PurePosixPath(raw)
        windows = PureWindowsPath(raw)
        if posix.is_absolute() or windows.is_absolute() or ".." in posix.parts or ".." in windows.parts:
            raise ValueError(f"{location} must stay within the config directory")

    def inspect(item, location="config"):
        if isinstance(item, dict):
            for key, child in item.items():
                normalized = str(key).lower().replace("-", "_")
                if normalized in sensitive:
                    raise ValueError(f"sensitive field {location}.{key} must come from the runtime environment")
                inspect(child, f"{location}.{key}")
        elif isinstance(item, list):
            for index, child in enumerate(item):
                inspect(child, f"{location}[{index}]")

    inspect(value)
    for source in value["sources"]:
        if not isinstance(source, dict):
            raise ValueError("each source must be an object")
        for key in ("name", "adapter", "split", "url", "revision", "license",
                    "source_family", "task_family", "domain"):
            if not isinstance(source.get(key), str) or not source[key].strip():
                raise ValueError(f"source {source.get('name', '<unknown>')!r} needs nonempty {key}")
        parsed = urlsplit(source["url"])
        if parsed.username or parsed.password:
            raise ValueError(f"source {source['name']!r} URL must not contain credentials")
        query_keys = {key.lower().replace("-", "_") for key, _ in parse_qsl(parsed.query)}
        if query_keys & credential_queries:
            raise ValueError(f"source {source['name']!r} URL query contains credentials")
        paths = source.get("paths", [])
        auxiliary = source.get("auxiliary_paths", {})
        if not isinstance(paths, list) or not all(isinstance(item, str) and item for item in paths):
            raise ValueError(f"source {source['name']!r} paths must be a list of relative paths")
        if not isinstance(auxiliary, dict) or not all(isinstance(key, str) and key and
                isinstance(item, str) and item for key, item in auxiliary.items()):
            raise ValueError(f"source {source['name']!r} auxiliary_paths must map names to relative paths")
        path_fields = list(paths)
        path_fields += list(auxiliary.values())
        path_fields += [source[key] for key in ("queries", "qrels", "collection") if key in source]
        for item in path_fields:
            portable_path(item, f"source {source['name']!r} file path")
    adapter_files = value.get("adapter_files", [])
    if not isinstance(adapter_files, list) or not all(isinstance(item, str) and item for item in adapter_files):
        raise ValueError("adapter_files must be a list of relative Python paths")
    for item in adapter_files:
        portable_path(item, "adapter file")
    return value


def decorate_case(case, source):
    """Fill portable provenance from config and reject contradictory adapter metadata."""
    meta = case.setdefault("_meta", {})
    configured = {"source_family": source["source_family"],
                  "task_family": source["task_family"], "domain": source["domain"],
                  "source": source["name"], "source_url": source["url"],
                  "revision": source["revision"], "license": source["license"],
                  "source_split": source["split"]}
    for key, expected in configured.items():
        existing = meta.get(key)
        if existing is not None and existing != expected:
            raise ValueError(
                f"adapter metadata {key}={existing!r} conflicts with source config {expected!r}"
            )
        meta[key] = expected
    return case


def _decorate(fn):
    """Wrap an adapter with portable provenance enforcement."""
    def adapter(source, paths, audit, review):
        for case in fn(source, paths, audit, review):
            yield decorate_case(case, source)
    return adapter


def build(config_path, output):
    """Run the common build engine with built-in and project adapter plugins."""
    config_path = Path(config_path)
    config = _config(config_path)
    adapters, files = registry(config_path, config)
    decorated = {name: _decorate(fn) for name, fn in adapters.items()}
    return core.build(config_path, output, extra_adapters=decorated, adapter_files=files)


def _case_files(release):
    release = Path(release)
    found = []
    for split in core.SPLITS:
        plain = release / f"{split}.cases.jsonl"
        compressed = release / f"{split}.cases.jsonl.gz"
        if plain.exists() and compressed.exists():
            raise ValueError(f"both compressed and uncompressed {split} files exist")
        path = compressed if compressed.exists() else plain
        if path.exists():
            found.append((split, path))
    if not found:
        raise FileNotFoundError(f"no *.cases.jsonl[.gz] files in {release}")
    return found


def verify(release, *, write_report=True):
    """Re-read the whole release and check its portable data invariants.

    SQLite keeps verification memory bounded for large local releases.  This
    verifier intentionally does not trust the build manifest's row counts.
    """
    release = Path(release)
    counts = collections.Counter()
    files = []
    with tempfile.TemporaryDirectory(prefix="chinese-jev-data-verify-") as tmp:
        db = sqlite3.connect(Path(tmp) / "audit.sqlite")
        db.executescript(
            "CREATE TABLE ids(id TEXT PRIMARY KEY);"
            "CREATE TABLE groups(grp TEXT PRIMARY KEY, split TEXT);"
            "CREATE TABLE inputs(fp TEXT PRIMARY KEY, gold TEXT, split TEXT);"
        )
        for split, path in _case_files(release):
            files.append({"split": split, "path": path.name, "bytes": path.stat().st_size,
                          "sha256": core.file_sha256(path)})
            for line, case in core.jsonl(path):
                try:
                    core.validate_case(case)
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"{path}:{line}: {exc}") from exc
                if len(case["questions"]) != 1:
                    raise ValueError(f"{path}:{line}: baseline requires one decision per case")
                meta = case.get("_meta")
                if not isinstance(meta, dict):
                    raise ValueError(f"{path}:{line}: missing _meta")
                for key in ("source", "source_id", "original_id", "group_key", "source_family",
                            "task_family", "license", "revision", "supervision"):
                    if not isinstance(meta.get(key), str) or not meta[key].strip():
                        raise ValueError(f"{path}:{line}: missing metadata {key}")
                if meta.get("split") != split:
                    raise ValueError(f"{path}:{line}: metadata split differs from filename")
                try:
                    db.execute("INSERT INTO ids VALUES(?)", (case["id"],))
                except sqlite3.IntegrityError as exc:
                    raise ValueError(f"duplicate case id: {case['id']}") from exc
                old = db.execute("SELECT split FROM groups WHERE grp=?", (meta["group_key"],)).fetchone()
                if old and old[0] != split:
                    raise ValueError(f"group leakage: {meta['group_key']} in {old[0]} and {split}")
                db.execute("INSERT OR IGNORE INTO groups VALUES(?,?)", (meta["group_key"], split))
                fp = core.digest([case["state"], case["questions"]])
                gold = core.digest(case["gold"])
                old = db.execute("SELECT gold FROM inputs WHERE fp=?", (fp,)).fetchone()
                if old and old[0] != gold:
                    raise ValueError(f"conflicting gold for identical input at {path}:{line}")
                if old:
                    raise ValueError(f"duplicate input at {path}:{line}")
                db.execute("INSERT INTO inputs VALUES(?,?,?)", (fp, gold, split))
                q = next(iter(case["questions"].values()))
                counts[f"split:{split}"] += 1
                counts[f"type:{split}:{q['type']}"] += 1
                counts[f"source:{split}:{meta['source_family']}"] += 1
                counts[f"task:{split}:{meta['task_family']}"] += 1
        counts["unique_ids"] = db.execute("SELECT COUNT(*) FROM ids").fetchone()[0]
        counts["unique_groups"] = db.execute("SELECT COUNT(*) FROM groups").fetchone()[0]
        db.close()
    report = {
        "format_version": 1,
        "status": "passed",
        "files": files,
        "counts": dict(sorted(counts.items())),
        "checks": [
            "schema_and_probability_contract",
            "one_decision_per_case",
            "required_provenance",
            "case_id_uniqueness",
            "group_split_isolation",
            "exact_input_uniqueness_and_gold_consistency",
            "full_file_sha256",
        ],
        "boundary": "Tokenizer completeness and semantic label correctness require compile audit and adapter tests.",
    }
    if write_report:
        target = release / "verification.json"
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
