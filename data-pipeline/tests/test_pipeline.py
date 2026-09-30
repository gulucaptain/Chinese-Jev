import copy
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chinese_jev_data_pipeline import core, parallel_build, runner  # noqa: E402


class PipelineTests(unittest.TestCase):
    def test_native_decision_adapter(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = {"id": "two-questions", "state": "一加一等于二。",
                   "questions": {
                       "q1": {"type": "choice", "instructions": "结果是多少？",
                              "criteria": {"A": "二", "B": "三"}},
                       "q2": {"type": "noul", "instructions": "结果是二吗？",
                              "criteria": {"false": "不是", "true": "是"}}},
                   "gold": {"q1": {"type": "choice", "probabilities": {"A": 1, "B": 0}},
                            "q2": {"type": "noul", "probabilities": {"false": 0, "true": 1}}},
                   "_meta": {"group_key": "math:one-plus-one"}}
            (root / "raw.jsonl").write_text(core.dumps(raw) + "\n", encoding="utf-8")
            config = json.loads((ROOT / "examples/basic/config.json").read_text(encoding="utf-8"))
            config.pop("adapter_files")
            config["sources"][0]["adapter"] = "chinese_jev"
            (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
            for backend, name in [(runner.build, "serial"), (parallel_build.build, "parallel")]:
                output = root / name
                backend(root / "config.json", output)
                report = runner.verify(output)
                self.assertEqual(report["counts"]["unique_ids"], 2)
                self.assertEqual(report["counts"]["unique_groups"], 1)
            self.assertEqual((root / "serial/train.cases.jsonl.gz").read_bytes(),
                             (root / "parallel/train.cases.jsonl.gz").read_bytes())

    def test_custom_adapter_builds_all_types(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "release"
            manifest = runner.build(ROOT / "examples/basic/config.json", output)
            report = runner.verify(output)
            rows = [row for _, row in core.jsonl(output / "train.cases.jsonl.gz")]
            types = {next(iter(row["questions"].values()))["type"] for row in rows}
            self.assertEqual(types, {"choice", "score", "noul"})
            self.assertEqual(report["counts"]["unique_ids"], 3)
            self.assertTrue(manifest["adapter_sha256"])
            self.assertFalse((output / "train.notebook.jsonl.gz").exists())

    def test_gzip_release_is_byte_reproducible(self):
        with tempfile.TemporaryDirectory() as tmp:
            first, second = Path(tmp) / "first", Path(tmp) / "second"
            runner.build(ROOT / "examples/basic/config.json", first)
            runner.build(ROOT / "examples/basic/config.json", second)
            for split in core.SPLITS:
                name = f"{split}.cases.jsonl.gz"
                self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())

    def test_group_leakage_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            release = Path(tmp)
            source = {"name": "fixture", "split": "train", "url": "synthetic:test",
                      "revision": "v1", "license": "synthetic"}
            question = {"type": "noul", "instructions": "是否成立？",
                        "criteria": {"false": "不成立", "true": "成立"}}
            train = core.make_case(source, "1", "same-group", "材料一", question,
                                   {"false": 0, "true": 1}, "hard_label")
            train["_meta"].update(source_family="fixture", task_family="binary", split="train")
            test = copy.deepcopy(train)
            test["id"], test["state"], test["_meta"]["split"] = "test-id", "材料二", "test"
            (release / "train.cases.jsonl").write_text(core.dumps(train) + "\n", encoding="utf-8")
            (release / "test.cases.jsonl").write_text(core.dumps(test) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "group leakage"):
                runner.verify(release, write_report=False)

    def test_required_source_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = json.loads((ROOT / "examples/basic/config.json").read_text(encoding="utf-8"))
            del config["sources"][0]["task_family"]
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "task_family"):
                runner.build(path, Path(tmp) / "release")

    def test_generic_parallel_backend_matches_shared_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_example = ROOT / "examples/basic"
            raw = root / "raw.jsonl"
            raw.write_bytes((source_example / "raw.jsonl").read_bytes())
            adapter = root / "adapter.py"
            adapter.write_bytes((source_example / "adapter.py").read_bytes())
            config = json.loads((source_example / "config.json").read_text(encoding="utf-8"))
            config["adapter_files"] = ["adapter.py"]
            config["sources"][0]["paths"] = ["raw.jsonl"]
            (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
            manifest = parallel_build.build(root / "config.json", root / "release", workers=2)
            report = runner.verify(root / "release")
            self.assertEqual(report["counts"]["unique_ids"], 3)
            self.assertEqual(manifest["execution"]["workers"], 2)
            self.assertTrue((root / "release/parallel.json").is_file())
            self.assertEqual(manifest["config"]["sources"][0]["adapter"], "project_jsonl")
            public = (root / "release/parallel.json").read_text(encoding="utf-8")
            self.assertNotIn(str(root), public)
            self.assertNotIn("spool", public)

    def test_parallel_build_accepts_relative_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(os.path.relpath(Path(tmp) / "release"))
            parallel_build.build(ROOT / "examples/basic/config.json", output, workers=2)
            report = runner.verify(output)
            self.assertEqual(report["counts"]["unique_ids"], 3)

    def test_parallel_t2ranking_records_all_input_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = {"queries.tsv": "q1\t如何保存食品？\n",
                      "qrels.tsv": "q1 0 p1 3\n",
                      "collection.tsv": "p1\t按食品包装上的要求冷藏保存。\n"}
            for name, content in inputs.items():
                (root / name).write_text(content, encoding="utf-8")
            config = {"format_version": 1, "seed": 42, "compression": "gzip",
                      "calibration_fraction": 0, "validation_fraction": 0,
                      "test_fraction": 0, "write_notebooks": False, "sources": [{
                          "name": "t2-fixture", "adapter": "t2ranking", "split": "train",
                          "url": "synthetic:test", "revision": "v1", "license": "synthetic",
                          "source_family": "t2ranking", "task_family": "relevance",
                          "domain": "general", "queries": "queries.tsv",
                          "qrels": "qrels.tsv", "collection": "collection.tsv"}]}
            path = root / "config.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            serial = runner.build(path, root / "serial")
            parallel = parallel_build.build(path, root / "parallel", workers=2)
            self.assertEqual(parallel["inputs"][0]["files"], serial["inputs"][0]["files"])
            self.assertEqual({item["path"] for item in parallel["inputs"][0]["files"]}, set(inputs))
            for split in core.SPLITS:
                name = f"{split}.cases.jsonl.gz"
                self.assertEqual((root / "serial" / name).read_bytes(),
                                 (root / "parallel" / name).read_bytes())

    def test_sensitive_config_fields_and_absolute_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = json.loads((ROOT / "examples/basic/config.json").read_text(encoding="utf-8"))
            config["hf_token"] = "must-not-be-published"
            path = root / "secret.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "sensitive field"):
                runner.build(path, root / "release")

            del config["hf_token"]
            config["sources"][0]["paths"] = [str((root / "raw.jsonl").resolve())]
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "config directory"):
                runner.build(path, root / "release")

            config["sources"][0]["paths"] = ["../raw.jsonl"]
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "config directory"):
                runner.build(path, root / "release")

            config["sources"][0]["paths"] = [r"C:\\private\\raw.jsonl"]
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "config directory"):
                runner.build(path, root / "release")

    def test_transitive_links_share_one_split_for_any_domain(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            question = {"type": "noul", "instructions": "是否成立？",
                        "criteria": {"false": "不成立", "true": "成立"}}

            def case(ident, links):
                return {"id": ident, "state": "材料 " + ident,
                        "questions": {"decision": question},
                        "gold": {"decision": {"type": "noul", "probabilities": {"false": 0.0, "true": 1.0}}},
                        "_meta": {"group_key": "group:" + ident, "original_id": ident,
                                  "source_id": ident, "supervision": "synthetic",
                                  "integration": {"link_keys": links}}}

            (root / "train.jsonl").write_text("".join(core.dumps(row) + "\n" for row in [
                case("left", ["entity:left"]),
                case("bridge", ["entity:left", "entity:right"]),
                case("safe", ["entity:safe"]),
            ]), encoding="utf-8")
            (root / "test.jsonl").write_text(core.dumps(case("right", ["entity:right"])) + "\n", encoding="utf-8")
            (root / "adapter.py").write_text(
                "from chinese_jev_data_pipeline.core import jsonl\n"
                "def convert(source, paths, audit, review):\n"
                "    for path in paths:\n"
                "        for _, row in jsonl(path): yield row\n"
                "ADAPTERS={'native': convert}\n", encoding="utf-8")

            def source(name, split):
                return {"name": name, "adapter": "native", "paths": [name + ".jsonl"],
                        "split": split, "url": "synthetic:test", "revision": "v1", "license": "synthetic",
                        "source_family": name, "task_family": "linked_binary", "domain": "example"}

            config = {"format_version": 1, "seed": 42, "calibration_fraction": 0,
                      "validation_fraction": 0, "adapter_files": ["adapter.py"],
                      "sources": [source("train", "train"), source("test", "test")]}
            path = root / "config.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            runner.build(path, root / "release")
            train = [row for _, row in core.jsonl(root / "release/train.cases.jsonl")]
            test = [row for _, row in core.jsonl(root / "release/test.cases.jsonl")]
            self.assertEqual([row["_meta"]["original_id"] for row in train], ["safe"])
            self.assertEqual([row["_meta"]["original_id"] for row in test], ["right"])


if __name__ == "__main__":
    unittest.main()
