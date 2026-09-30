import copy
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chinese_jev_data_pipeline import core  # noqa: E402


class CharacterTokenizer:
    """Deterministic tokenizer for format and length tests, with no model download."""

    cls_token_id, sep_token_id, mask_token_id, pad_token_id = 1, 2, 3, 0
    mask_token = "[MASK]"

    def __call__(self, value, add_special_tokens=False):
        return {"input_ids": [ord(char) + 10 for char in value]}


class BertCompilerTests(unittest.TestCase):
    def test_sequences_match_fixed_format_fixtures(self):
        fixtures = json.loads((ROOT / "tests/fixtures/bert_sequences.json").read_text(encoding="utf-8"))
        for fixture in fixtures:
            with self.subTest(fixture=fixture["name"]):
                ids, markers, diagnostics = core.checked_sequence(
                    CharacterTokenizer(), fixture["state"], fixture["question"],
                    fixture["max_len"], fixture["head_max_len"], fixture["state_truncation"])
                self.assertEqual({"ids": ids, "markers": markers, "diagnostics": diagnostics},
                                 fixture["expected"])

    def test_incomplete_question_or_candidates_are_rejected(self):
        fixtures = [
            ({"t": "choice", "ins": "选择。", "crit": {"A": "甲" * 50, "B": "乙"}},
             "option_text_truncated"),
            ({"t": "choice", "ins": "长" * 200, "crit": {"A": "甲", "B": "乙"}},
             "question_instruction_truncated"),
        ]
        for question, reason in fixtures:
            with self.subTest(reason=reason), self.assertRaisesRegex(ValueError, reason):
                core.checked_sequence(CharacterTokenizer(), "材料", question, 512, 192)

    def test_state_truncation_requires_explicit_permission(self):
        question = {"t": "noul", "ins": "成立吗？", "crit": {"false": "否", "true": "是"}}
        with self.assertRaisesRegex(ValueError, "state_truncated"):
            core.checked_sequence(CharacterTokenizer(), "长" * 200, question, 96, 64)
        ids, markers, diagnostics = core.checked_sequence(
            CharacterTokenizer(), "长" * 200, question, 96, 64, "right")
        self.assertEqual(len(ids), 96)
        self.assertEqual(len(markers), 2)
        self.assertTrue(diagnostics["state_truncated"])

    def test_targets_follow_candidate_order_for_each_type(self):
        fixtures = [
            ("choice", {"B": "乙", "A": "甲"}, {"A": 0.2, "B": 0.8},
             ["B", "A"], [0.8, 0.2]),
            ("score", ["低", "中", "高"], {"2": 0.1, "0": 0.2, "1": 0.7},
             ["0", "1", "2"], [0.2, 0.7, 0.1]),
            ("noul", {"true": "成立", "false": "不成立"}, {"true": 0.8, "false": 0.2},
             ["false", "true"], [0.2, 0.8]),
        ]
        for kind, criteria, probabilities, keys, expected in fixtures:
            with self.subTest(kind=kind):
                case = {"id": "fixture", "state": "材料",
                        "questions": {"q": {"type": kind, "instructions": "判断。", "criteria": criteria}},
                        "gold": {"q": {"type": kind, "probabilities": probabilities}}, "_meta": {}}
                item = core.sequence_item(CharacterTokenizer(), case, "q", 512, 192)
                self.assertEqual(item["_meta"]["candidate_keys"], keys)
                self.assertEqual(item["target"], expected)
                self.assertEqual(len(item["markers"]), len(expected))
                altered = copy.deepcopy(case)
                altered["gold"]["q"]["probabilities"] = dict(zip(keys, reversed(expected)))
                other = core.sequence_item(CharacterTokenizer(), altered, "q", 512, 192)
                self.assertEqual(other["ids"], item["ids"])
                self.assertEqual(other["markers"], item["markers"])
                self.assertNotEqual(other["target"], item["target"])


if __name__ == "__main__":
    unittest.main()
