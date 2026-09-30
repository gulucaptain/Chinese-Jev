"""Unified local baseline for building Chinese-Jev datasets."""
from __future__ import annotations

import argparse
import json
from importlib import resources
from pathlib import Path

from . import core
from .registry import registry
from .runner import _config, build, verify


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    adapters = commands.add_parser("list-adapters")
    adapters.add_argument("--config", type=Path, help="optional config whose project plugins should also be listed")
    init = commands.add_parser("init", help="create a minimal adapter project")
    init.add_argument("directory", type=Path)
    run = commands.add_parser("build", help="build through the shared serial or source-parallel engine")
    run.add_argument("--config", required=True, type=Path)
    run.add_argument("--output", required=True, type=Path)
    run.add_argument("--workers", type=int, default=1)
    check = commands.add_parser("verify")
    check.add_argument("--release", required=True, type=Path)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--input", required=True, type=Path)
    prepare.add_argument("--tokenizer", required=True)
    prepare.add_argument("--output", required=True, type=Path)
    prepare.add_argument("--max-len", type=int, default=1024)
    prepare.add_argument("--head-max-len", type=int, default=256)
    prepare.add_argument("--state-truncation", choices=("reject", "right"), default="reject")
    prepare.add_argument("--pt-shard-size", type=int, default=0)
    args = parser.parse_args(argv)

    if args.command == "init":
        if args.directory.exists():
            raise FileExistsError(args.directory)
        args.directory.mkdir(parents=True)
        template = resources.files("chinese_jev_data_pipeline").joinpath("templates/basic")
        for name in ("adapter.py", "config.json", "raw.jsonl"):
            (args.directory / name).write_bytes(template.joinpath(name).read_bytes())
        print(json.dumps({"project": str(args.directory), "next": [
            "edit raw.jsonl and adapter.py", "edit config.json",
            f"chinese-jev-data build --config {args.directory}/config.json --output {args.directory}/release"
        ]}, ensure_ascii=False, indent=2))
        return
    if args.command == "list-adapters":
        config = _config(args.config) if args.config else {"sources": []}
        config_path = args.config or (Path.cwd() / "config.json")
        names = set(registry(config_path, config)[0])
        names.update({"c3", "exam_csv", "dureader_yesno", "squad", "chinese_jev", "t2ranking"})
        print("\n".join(sorted(names)))
        return
    if args.command == "build":
        if not 1 <= args.workers <= 16:
            raise ValueError("workers must be in 1..16")
        if args.workers == 1:
            report = build(args.config, args.output)
        else:
            from .parallel_build import build as build_parallel
            report = build_parallel(args.config, args.output, workers=args.workers)
        verification = verify(args.output)
        result = {"output": str(args.output), "counts": report["counts"],
                  "workers": args.workers, "verification": verification["status"]}
    elif args.command == "verify":
        result = verify(args.release)
    elif args.command == "prepare":
        report = core.prepare(args.input, args.tokenizer, args.output,
                              args.max_len, args.head_max_len, args.state_truncation,
                              args.pt_shard_size)
        result = {"output": str(args.output), "counts": report["counts"]}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
