"""Assert-based checks for the fine-tuning pipeline.

Run: CHINESE_JEV_TEST_BUNDLE=/path/to/bundle python tests/test_pipeline.py

Covers the properties that are silent when they break: that a cached sequence rebuilt at
any max_len equals what `sequence.build_sequence` builds directly, that the head budget
measurement catches an option set the default would collapse, that the config refuses the
mistakes that train the wrong thing, and that the cache lands where `cache_dir` says.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from chinese_jev import config as cfgmod                              # noqa: E402
from chinese_jev.adapters import JsonlAdapter, get, source_of        # noqa: E402
from chinese_jev.cache import TokenCache, filter_items               # noqa: E402
from chinese_jev.config import cache_root, split_cache_dir           # noqa: E402
from chinese_jev.data import collate_sequences, plan_epoch           # noqa: E402
from chinese_jev.encoding import encode_case                         # noqa: E402
from chinese_jev.prepare import tokenize_split                       # noqa: E402
from chinese_jev.sequence import build_sequence, normalize_question, serialize_state  # noqa: E402
from chinese_jev.tokenizer import fix_tokenizer_config, load_tokenizer  # noqa: E402

# Path("") is Path("."), which exists and is truthy — keep the raw value around so the
# no-bundle guard in main() can tell "unset" from "set to a real bundle".
_BUNDLE_ENV = os.environ.get("CHINESE_JEV_TEST_BUNDLE", "")
BUNDLE = Path(_BUNDLE_ENV)


def _make_corpus(root, n=200):
    """A small JSONL corpus with the canonical schema, plus a wide option set."""
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n):
        if i % 7 == 0:
            # A choice question with many options, to exercise the head budget.
            crit = {f"opt{j}": "description %d" % j for j in range(60)}
            rows.append({
                "id": "wide-%d" % i,
                "state": "患者主诉：" + "胸痛三天，" * (i % 5 + 1),
                "questions": {"q": {"type": "choice", "instructions": "选择最可能的诊断",
                                    "criteria": crit}},
                "gold": {"q": {"type": "choice",
                               "probabilities": {k: (1.0 if j == 3 else 0.0)
                                                 for j, k in enumerate(crit)}}},
                "_meta": {"source": "corpus_train", "source_split": "train"},
            })
        else:
            rows.append({
                "id": "noul-%d" % i,
                "state": "问题：感冒了怎么办？\n候选回答：" + "多喝水,注意休息。" * (i % 40),
                "questions": {"q": {"type": "noul", "instructions": "候选是否匹配问题？",
                                    "criteria": {"false": "不匹配", "true": "匹配"}}},
                "gold": {"q": {"type": "noul", "probabilities": {"false": 1.0, "true": 0.0}}},
                "_meta": {"source": "corpus_validation" if i % 3 else "corpus_train"},
            })
    # A case with a score question and one with no gold, both of which must not crash.
    rows.append({
        "id": "score-1",
        "state": "患者诉头痛。",
        "questions": {"q": {"type": "score", "instructions": "严重程度", "criteria": ["轻", "中", "重", "危"]}},
        "gold": {"q": {"type": "score", "probabilities": {"0": 0.1, "1": 0.7, "2": 0.2, "3": 0.0}}},
        "_meta": {"source": "corpus_test"},
    })
    rows.append({
        "id": "no-gold",
        "state": "患者诉头痛。",
        "questions": {"q": {"type": "noul", "instructions": "x", "criteria": {"false": "f", "true": "t"}}},
        "gold": {},
        "_meta": {},
    })
    with (root / "train.cases.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return root


def _config(td, **overrides):
    defaults = dict(dataset="chinese-jev", work_dir=str(Path(td) / "run"), bundle=str(BUNDLE),
                    head_max_len=1024, max_len=512)
    defaults.update(overrides)
    return cfgmod.read_config(None, **defaults)


def test_config_rejects_bad_values():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "c.json"
        path.write_text(json.dumps({"dataset": "chinese-jev", "bundle": "b", "group_size": 1}),
                        encoding="utf-8")
        try:
            cfgmod.read_config(str(path))
        except cfgmod.ConfigError as e:
            assert "group_size" in str(e), e
        else:
            raise AssertionError("group_size=1 must be refused: a group with no baseline has no advantage")

        path.write_text(json.dumps({"dataset": "chinese-jev", "bundle": "b", "nonsense": 1}),
                        encoding="utf-8")
        try:
            cfgmod.read_config(str(path))
        except cfgmod.ConfigError as e:
            assert "nonsense" in str(e), e
        else:
            raise AssertionError("an unknown key must be refused rather than ignored")

        path.write_text(json.dumps({"max_len": 4096}), encoding="utf-8")
        try:
            cfgmod.read_config(str(path))
        except cfgmod.ConfigError as e:
            assert "dataset" in str(e), e
        else:
            raise AssertionError("a config with no dataset must be refused")

        path.write_text(json.dumps({"dataset": "chinese-jev"}), encoding="utf-8")
        try:
            cfgmod.read_config(str(path))
        except cfgmod.ConfigError as e:
            assert "bundle" in str(e), e
        else:
            raise AssertionError("a config with no bundle must be refused")
    print("  config validation                     ok")


def test_cache_dir_is_configurable():
    """The cache location must follow the config; relative paths resolve against the
    enclosing project root when there is one, else the config's own directory."""
    with tempfile.TemporaryDirectory() as td:
        cfg = _config(td)
        assert cache_root(cfg) == Path(td) / "run" / "cache", cache_root(cfg)

        shared = Path(td) / "shared_cache"
        cfg = _config(td, cache_dir=str(shared))
        assert cache_root(cfg) == shared
        assert split_cache_dir(cfg, "train") == shared / "train"

        # No project marker above the config: paths resolve against the config's dir.
        path = Path(td) / "conf" / "c.json"
        path.parent.mkdir()
        path.write_text(json.dumps({"dataset": "chinese-jev", "bundle": "bundle",
                                    "cache_dir": "../my_cache"}), encoding="utf-8")
        cfg = cfgmod.read_config(str(path))
        assert cache_root(cfg).resolve() == (Path(td) / "my_cache").resolve(), cache_root(cfg)

        # With a pyproject.toml at td, a config in td/configs/ resolves against td itself,
        # so "runs/x" means <project>/runs/x — not <project>/configs/runs/x.
        (Path(td) / "pyproject.toml").touch()
        proj_cfg = Path(td) / "configs" / "p.json"
        proj_cfg.parent.mkdir()
        proj_cfg.write_text(json.dumps({"dataset": "chinese-jev", "bundle": "models/b",
                                        "work_dir": "runs/x", "cache_dir": "runs/cache"}),
                            encoding="utf-8")
        cfg = cfgmod.read_config(str(proj_cfg))
        assert cfg["work_dir"] == str(Path(td) / "runs" / "x"), cfg["work_dir"]
        assert cfg["bundle"] == str(Path(td) / "models" / "b"), cfg["bundle"]
        assert cache_root(cfg) == Path(td) / "runs" / "cache", cache_root(cfg)
    print("  cache_dir and project-root paths      ok")


def test_config_rejects_dataset_info_ambiguity():
    """dataset_info and dataset/dataset_params in one config must be refused."""
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "c.json"
        path.write_text(json.dumps({"dataset": "chinese-jev", "dataset_info": "x/dataset.json",
                                    "bundle": "b"}), encoding="utf-8")
        try:
            cfgmod.read_config(str(path))
        except cfgmod.ConfigError as e:
            assert "dataset_info" in str(e), e
        else:
            raise AssertionError("dataset plus dataset_info must be refused, not merged")

        path.write_text(json.dumps({"dataset_info": str(Path(td) / "missing" / "dataset.json"),
                                    "bundle": "b"}), encoding="utf-8")
        try:
            cfgmod.read_config(str(path))
        except cfgmod.ConfigError as e:
            assert "prepare" in str(e), e
        else:
            raise AssertionError("a dangling dataset_info must be refused")
    print("  dataset_info config validation        ok")


def test_cache_rebuild_matches_build_sequence():
    """The load path must reproduce `build_sequence` exactly, at several max_len values."""
    tok = load_tokenizer(BUNDLE)
    with tempfile.TemporaryDirectory() as td:
        corpus = _make_corpus(Path(td) / "corpus")
        cfg = _config(td)
        adapter = JsonlAdapter(data_root=str(corpus), file_template="{split}.cases.jsonl")
        out = split_cache_dir(cfg, "train")
        tokenize_split(cfg, adapter, "train", out, cfg["bundle"], verbose=False, log=lambda *_: None)
        cache = TokenCache(out)
        assert len(cache) >= 190, len(cache)

        # Feed the adapter the same cases and rebuild each one with build_sequence directly.
        cases = {c["id"]: c for c in adapter.iter_cases("train")}
        checked = 0
        for index in range(len(cache)):
            meta = cache.item_meta(index)
            case = cases[meta["case_id"]]
            qdef = case["questions"][meta["question_id"]]
            internal = normalize_question(qdef)
            state_ids = tok(serialize_state(case["state"]).replace(tok.mask_token, " "),
                            add_special_tokens=False)["input_ids"]
            for max_len in (256, 512, 1024):
                _ids, _markers = build_sequence(tok, case["state"], internal, max_len,
                                                cache.head_max_len, state_ids=list(state_ids))
                if len(_markers) != meta["n_options"]:
                    # build_sequence itself dropped markers that fell past this max_len;
                    # there is no cache-side counterpart to compare against.
                    continue
                item = cache.load_sequence(index, max_len)
                assert item["ids"] == _ids, (
                    "item %d at max_len=%d differs from build_sequence\n cache: %s\n direct: %s"
                    % (index, max_len, item["ids"][:40], _ids[:40]))
                assert item["markers"] == _markers, (index, max_len, item["markers"], _markers)
                checked += 1
        assert checked > 200, "too few comparisons ran (%d)" % checked
        print("  cache rebuild == build_sequence       ok (%d comparisons)" % checked)


def test_filter_is_a_pure_length_knob():
    with tempfile.TemporaryDirectory() as td:
        corpus = _make_corpus(Path(td) / "corpus")
        cfg = _config(td)
        adapter = JsonlAdapter(data_root=str(corpus), file_template="{split}.cases.jsonl")
        out = split_cache_dir(cfg, "train")
        tokenize_split(cfg, adapter, "train", out, cfg["bundle"], verbose=False, log=lambda *_: None)
        cache = TokenCache(out)
        counts = [len(filter_items(cache, m, "filter")) for m in (256, 1024, 4096)]
        assert counts == sorted(counts), counts
        assert counts[-1] >= counts[0], counts
        assert counts[2] == len(cache), "max_len far above the data must keep everything"
        # Truncate must keep at least as much as filter at the same budget.
        assert len(filter_items(cache, 256, "truncate")) >= counts[0]
        print("  max_len filtering is monotone         ok %s" % counts)


def test_prepare_writes_dataset_info_a_training_config_can_use():
    """`prepare` must leave a dataset.json that a dataset_info config trains from,
    measuring head_max_len itself when the dataset config leaves it null."""
    from chinese_jev.cli import cmd_prepare

    with tempfile.TemporaryDirectory() as td:
        corpus = _make_corpus(Path(td) / "corpus")
        cache_dir = Path(td) / "cache"
        cfg = cfgmod.read_config(None, dataset="jsonl",
                                 dataset_params={"data_root": str(corpus),
                                                 "file_template": "{split}.cases.jsonl",
                                                 "splits": ["train"]},
                                 bundle=str(BUNDLE), work_dir=str(Path(td) / "prep"),
                                 cache_dir=str(cache_dir), tokenize_workers=1)
        assert cfg["head_max_len"] is None
        cmd_prepare(cfg, None, force=False, verbose=False)
        info_path = cache_dir / "dataset.json"
        assert info_path.exists(), "prepare must write dataset.json at the cache root"
        info = json.loads(info_path.read_text(encoding="utf-8"))
        assert info["head_max_len"] == cfg["head_max_len"] and info["head_max_len"] >= 64
        assert info["splits"]["train"]["n_items"] >= 190

        run_cfg_path = Path(td) / "run.json"
        run_cfg_path.write_text(json.dumps({"dataset_info": str(info_path),
                                            "bundle": str(BUNDLE),
                                            "work_dir": str(Path(td) / "run"),
                                            "max_len": 1024}), encoding="utf-8")
        run_cfg = cfgmod.read_config(str(run_cfg_path))
        assert run_cfg["dataset"] == "jsonl"
        assert run_cfg["dataset_params"]["data_root"] == str(corpus)
        assert run_cfg["head_max_len"] == info["head_max_len"]
        assert cache_root(run_cfg) == cache_dir, cache_root(run_cfg)
        cache = TokenCache(split_cache_dir(run_cfg, "train"))
        assert len(cache) == info["splits"]["train"]["n_items"]
        # An explicit head_max_len in the training config must win over the recorded one.
        wider = cfgmod.read_config(str(run_cfg_path), head_max_len=info["head_max_len"] + 64)
        assert wider["head_max_len"] == info["head_max_len"] + 64
        print("  prepare -> dataset.json -> training   ok (head_max_len=%d)" % info["head_max_len"])


def test_stats_autofills_a_null_head_max_len():
    """`stats` must write the measured head_max_len into a config that left it null,
    and only warn — never overwrite — when the config committed to a value."""
    from chinese_jev.cli import cmd_stats

    with tempfile.TemporaryDirectory() as td:
        corpus = _make_corpus(Path(td) / "corpus")
        cfg_path = Path(td) / "c.json"
        body = {"dataset": "jsonl",
                "dataset_params": {"data_root": str(corpus),
                                   "file_template": "{split}.cases.jsonl",
                                   "splits": ["train"]},
                "bundle": str(BUNDLE), "head_max_len": None}
        cfg_path.write_text(json.dumps(body), encoding="utf-8")
        cfg = cfgmod.read_config(str(cfg_path))
        report = cmd_stats(cfg, split="train", config_path=str(cfg_path), verbose=False)
        written = json.loads(cfg_path.read_text(encoding="utf-8"))
        assert written["head_max_len"] == report["recommended_head_max_len"], written

        body["head_max_len"] = 32768  # deliberately generous, must be left alone
        cfg_path.write_text(json.dumps(body), encoding="utf-8")
        cfg = cfgmod.read_config(str(cfg_path))
        cmd_stats(cfg, split="train", config_path=str(cfg_path), verbose=False)
        assert json.loads(cfg_path.read_text(encoding="utf-8"))["head_max_len"] == 32768
        print("  stats autofills null head_max_len     ok")


def test_head_budget_detects_collapse():
    """A too-small head_max_len must be visibly wrong, not silently lossy."""
    tok = load_tokenizer(BUNDLE)
    crit = {f"opt{j}": "description %d" % j for j in range(60)}
    case = {"id": "wide", "state": "材料",
            "questions": {"q": {"type": "choice", "instructions": "选择", "criteria": crit}},
            "gold": {"q": {"type": "choice", "probabilities": {k: 1.0 if j == 0 else 0.0
                                                               for j, k in enumerate(crit)}}},
            "_meta": {}}
    wide, _ = encode_case(case, tok, head_max_len=4096)
    narrow, skipped = encode_case(case, tok, head_max_len=64)
    assert len(wide) == 1, "a 60-option question must encode at a wide budget"
    assert not narrow and skipped, "a 60-option question at head_max_len=64 must be refused, not collapsed"
    print("  head budget refuses collapse          ok")


def test_collate_and_plan():
    """Batching must preserve per-item option counts, and bucketing must not change the set."""
    with tempfile.TemporaryDirectory() as td:
        corpus = _make_corpus(Path(td) / "corpus")
        cfg = _config(td, max_len=1024)
        adapter = JsonlAdapter(data_root=str(corpus), file_template="{split}.cases.jsonl")
        out = split_cache_dir(cfg, "train")
        tokenize_split(cfg, adapter, "train", out, cfg["bundle"], verbose=False, log=lambda *_: None)
        cache = TokenCache(out)
        keep = filter_items(cache, 1024, "filter")
        items = [cache.load_sequence(int(i), 1024) for i in keep[:16]]
        batch = collate_sequences(items, cache.manifest["pad_token_id"])
        assert batch["input_ids"].shape[0] == 16
        for r, it in enumerate(items):
            k = len(it["markers"])
            assert int(batch["marker_mask"][r].sum()) == k
            assert abs(float(batch["target"][r, :k].sum()) - 1.0) < 1e-5

        lengths = cache.prompt_len + cache.state_len + 1
        plan = plan_epoch(keep, 8, length_bucketed=True, seed=1, lengths=lengths)
        flat = [i for b in plan for i in b]
        assert sorted(flat) == sorted(int(i) for i in keep), "a plan must visit every item once"
        unbucketed = plan_epoch(keep, 8, length_bucketed=False, seed=1, lengths=lengths)
        assert sorted(i for b in unbucketed for i in b) == sorted(int(i) for i in keep)
        capped = plan_epoch(keep, 8, length_bucketed=True, seed=1, lengths=lengths,
                            max_tokens_per_batch=600)
        assert sorted(i for b in capped for i in b) == sorted(int(i) for i in keep)
        assert all(len(b) >= 1 for b in capped)
        # Length bucketing must actually reduce the padding a batch would carry.
        def padded(plan_batches):
            return sum(max(int(lengths[i]) for i in b) * len(b) for b in plan_batches)
        assert padded(plan) <= padded(unbucketed) + 1, (padded(plan), padded(unbucketed))
        print("  collation and batch planning          ok")


def test_source_of_strips_split_suffix():
    assert source_of({"_meta": {"source": "cmedqa2_train"}}) == "cmedqa2"
    assert source_of({"_meta": {"source": "dialmed"}}) == "dialmed"
    assert source_of({}) == "unknown"
    assert get("chinese-jev") is not None
    print("  adapter registry and source names     ok")


def test_eval_sharding_partitions_items():
    """Round-robin eval shards must cover every item exactly once, for any world size."""
    keep = list(range(103))
    for world_size in (1, 2, 3, 8):
        shards = [keep[rank::world_size] for rank in range(world_size)]
        merged = sorted(i for shard in shards for i in shard)
        assert merged == keep, (world_size, merged[:10])
    print("  eval sharding partitions items        ok")


def test_distributed_evaluate_matches_single_process():
    """torchrun-based evaluation must report the same metrics as one process.

    Runs the evaluate stage twice over the same tiny cache — single process, then two
    gloo ranks on CPU — and compares evaluation.json. Uses the base bundle directly (no
    training), which is the evaluate stage's own fallback.
    """
    import shutil
    import subprocess

    if shutil.which("torchrun") is None:
        print("SKIP distributed evaluate: torchrun not on PATH")
        return
    with tempfile.TemporaryDirectory() as td:
        corpus = _make_corpus(Path(td) / "corpus", n=40)
        config = {
            "dataset": "jsonl",
            "dataset_params": {"data_root": str(corpus), "file_template": "{split}.cases.jsonl",
                               "splits": ["train"]},
            "bundle": str(BUNDLE),
            "work_dir": str(Path(td) / "run"),
            "cache_dir": str(Path(td) / "cache"),
            "max_len": 512, "head_max_len": 1024, "group_size": 2,
            "precision": "fp32", "num_workers": 0, "tokenize_workers": 2, "tokenize_chunk": 50,
        }
        cfg_path = Path(td) / "config.json"
        cfg_path.write_text(json.dumps(config), encoding="utf-8")

        env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parent.parent / "src"),
                   CUDA_VISIBLE_DEVICES="", CHINESE_JEV_BACKEND="gloo")

        def run(*cmd):
            proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=560)
            assert proc.returncode == 0, "%s failed:\n%s\n%s" % (cmd, proc.stdout[-2000:],
                                                                 proc.stderr[-2000:])

        run(sys.executable, "-m", "chinese_jev", "tokenize", "--config", str(cfg_path))
        result_path = Path(td) / "run" / "evaluation.json"

        run(sys.executable, "-m", "chinese_jev", "evaluate", "--config", str(cfg_path),
            "--splits", "train")
        single = json.loads(result_path.read_text(encoding="utf-8"))["train"]
        result_path.unlink()

        run("torchrun", "--standalone", "--nproc-per-node=2", "-m", "chinese_jev",
            "evaluate", "--config", str(cfg_path), "--splits", "train")
        double = json.loads(result_path.read_text(encoding="utf-8"))["train"]

        assert double["items"] == single["items"], (double["items"], single["items"])
        for key in ("accuracy", "ece", "mean_logscore", "mean_confidence"):
            assert abs(double[key] - single[key]) < 1e-4, (key, single[key], double[key])
        print("  distributed evaluate == single        ok (%d items, acc %.4f)"
              % (single["items"], single["accuracy"]))


def main():
    print("chinese-jev pipeline checks")
    test_config_rejects_bad_values()
    test_cache_dir_is_configurable()
    test_config_rejects_dataset_info_ambiguity()
    test_source_of_strips_split_suffix()
    test_eval_sharding_partitions_items()
    if not _BUNDLE_ENV or not BUNDLE.exists():
        print("SKIP tokenizer-dependent checks: set CHINESE_JEV_TEST_BUNDLE to a model bundle")
        return 0
    fix_tokenizer_config(str(BUNDLE))
    test_head_budget_detects_collapse()
    test_cache_rebuild_matches_build_sequence()
    test_filter_is_a_pure_length_knob()
    test_collate_and_plan()
    test_prepare_writes_dataset_info_a_training_config_can_use()
    test_stats_autofills_a_null_head_max_len()
    test_distributed_evaluate_matches_single_process()
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
