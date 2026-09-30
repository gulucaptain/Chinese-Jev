"""Verified medical dataset schemas -> existing Chinese-Jev decisions (stdlib only).

See docs/adapters.md for supported source schemas and task boundaries.
These converters read local files; they never fetch data or generate labels.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from .core import group_key, jsonl, make_case, one_hot, text

QIC_LABELS = (
    "疾病表述", "指标解读", "医疗费用", "治疗方案", "功效作用", "病情诊断",
    "其他", "注意事项", "病因分析", "就医建议", "后果表述",
)
QTR_LEVELS = ("不相关", "相关程度较低", "相关", "高度相关")
# CBLUE v1 ordinal labels, not PromptCBLUE's four directional relation classes.
QQR_LEVELS = ("相关程度最低", "相关程度中等", "相关程度最高")


def _case(source, sid, grouping, state, question, probabilities, *, link_texts=()):
    result = make_case(source, sid, grouping, state, question, probabilities, "source_hard_label")
    result["_meta"].update(original_id=str(sid),
                           original_split=source.get("original_split", source["split"]))
    # The shared builder computes the transitive closure of these anchors before splitting.
    # Keep both ends of a pair together, including matches across source families.
    result["_meta"]["integration"] = {
        "link_keys": sorted({group_key("medical_text", value) for value in link_texts}),
    }
    return result


def _rows(paths):
    """CBLUE releases are whole JSON arrays, not JSONL or PromptCBLUE records."""
    for path in paths:
        path = Path(path)
        rows = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(rows, list):
            raise ValueError(f"{path}: expected original CBLUE JSON array")
        for index, row in enumerate(rows, 1):
            sid = f"{path.name}:{row.get('id', index) if isinstance(row, dict) else index}"
            yield sid, row


def _label(row, allowed):
    if not isinstance(row, dict):
        raise ValueError("record must be an object")
    value = row.get("label")
    # In particular, do not coerce bool/None/NA or map missing labels to zero.
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(f"missing/unknown explicit label {value!r}; expected {list(allowed)!r}")
    return value


def convert_qic(source, paths, audit, review):
    criteria = {label: label for label in QIC_LABELS}
    question = dict(type="choice", instructions="判定材料中医疗查询的主要意图。", criteria=criteria)
    for sid, row in _rows(paths):
        try:
            label = _label(row, criteria)
            query = text(row["query"], "query")
            yield _case(source, sid, group_key("question", query), query, question,
                        one_hot(list(criteria), label), link_texts=(query,))
        except (ValueError, KeyError, TypeError) as exc:
            audit(source, sid, str(exc))


def convert_sts(source, paths, audit, review):
    question = dict(type="noul", instructions="两段医疗问句的语义是否相同或相近？",
                    criteria={"false": "语义不相同或不相近", "true": "语义相同或相近"})
    for sid, row in _rows(paths):
        try:
            label = _label(row, ("0", "1"))
            left, right = text(row["text1"], "text1"), text(row["text2"], "text2")
            yield _case(source, sid, group_key("question", left), f"问句一：{left}\n问句二：{right}",
                        question, one_hot(["false", "true"], "true" if label == "1" else "false"),
                        link_texts=(left, right))
        except (ValueError, KeyError, TypeError) as exc:
            audit(source, sid, str(exc))


def _relevance(source, paths, audit, fields, levels, *, ordinal=True):
    keys = [str(i) for i in range(len(levels))]
    question = (dict(type="score", instructions="判定材料中两段文本的相关程度，由低到高选择等级。",
                     criteria=list(levels)) if ordinal else
                dict(type="choice", instructions="判定材料中两个查询的官方相关性类别，保留前后顺序。",
                     criteria=dict(zip(keys, levels))))
    for sid, row in _rows(paths):
        try:
            label = _label(row, keys)
            left, right = (text(row[field], field) for field in fields)
            state = f"{fields[0]}：{left}\n{fields[1]}：{right}"
            yield _case(source, sid, group_key("question", left), state, question, one_hot(keys, label),
                        link_texts=(left, right))
        except (ValueError, KeyError, TypeError) as exc:
            audit(source, sid, str(exc))


def convert_qtr(source, paths, audit, review):
    yield from _relevance(source, paths, audit, ("query", "title"), QTR_LEVELS)


def convert_qqr(source, paths, audit, review):
    # Keep the original classification task: numeric IDs alone do not justify RPS.
    yield from _relevance(source, paths, audit, ("query1", "query2"), QQR_LEVELS, ordinal=False)


def convert_ctc(source, paths, audit, review):
    """Requires the release's full 44-category vocabulary, independently supplied."""
    auxiliary = source.get("_resolved_auxiliary_paths", {})
    if "labels" not in auxiliary:
        raise ValueError("cblue_ctc requires auxiliary_paths.labels: category.xlsx exported as a label->description JSON object")
    criteria = json.loads(Path(auxiliary["labels"]).read_text(encoding="utf-8-sig"))
    if not isinstance(criteria, dict) or len(criteria) != 44:
        raise ValueError("CHIP-CTC labels must contain the full 44-category label->description mapping")
    for label, description in criteria.items():
        text(label, "CTC label")
        text(description, "CTC description")
    if len(set(criteria.values())) != 44:
        raise ValueError("CHIP-CTC descriptions must be distinct")
    question = dict(type="choice", instructions="判定材料中临床试验筛选条件的类别。", criteria=criteria)
    for sid, row in _rows(paths):
        try:
            label = _label(row, criteria)
            content = text(row["text"], "text")
            yield _case(source, sid, group_key("context", content), content, question,
                        one_hot(list(criteria), label), link_texts=(content,))
        except (ValueError, KeyError, TypeError) as exc:
            audit(source, sid, str(exc))


def _exam(source, sid, question_text, options, answer):
    question_text = text(question_text, "question")
    if not isinstance(options, dict) or list(options) not in [list("ABCD"), list("ABCDE")]:
        raise ValueError("medical exam options must have four or five ordered A..D/E keys")
    for option in options.values():
        text(option, "option")
    if len(set(options.values())) != len(options):
        raise ValueError("duplicate exam options need review")
    if not isinstance(answer, str) or len(answer) != 1 or answer not in options:
        raise ValueError("missing or non-single-choice answer; multi-answer items are unsupported")
    question = dict(type="choice", instructions="根据材料中的医学试题，选择一个正确答案。", criteria=options)
    # Only the question and candidates become input. Explanation/answer fields do not.
    return _case(source, sid, group_key("question", question_text), question_text, question,
                 one_hot(list(options), answer), link_texts=(question_text,))


def convert_cmexam(source, paths, audit, review):
    for path in paths:
        path = Path(path)
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not {"Question", "Options", "Answer"} <= set(reader.fieldnames or []):
                raise ValueError(f"{path}: expected original CMExam Question/Options/Answer CSV columns")
            for index, row in enumerate(reader, 2):
                sid = f"{path.name}:{index}"
                try:
                    options = {}
                    for line in text(row["Options"], "Options").splitlines():
                        match = re.fullmatch(r"([A-E])\s+(.+)", line.strip())
                        if match is None or match[1] in options:
                            raise ValueError("invalid CMExam option lines; expected unique 'A text' lines")
                        options[match[1]] = match[2]
                    yield _exam(source, sid, row["Question"], options, row["Answer"].strip())
                except (ValueError, KeyError, TypeError, AttributeError) as exc:
                    audit(source, sid, str(exc))


def convert_medqa(source, paths, audit, review):
    """Original MedQA Mainland China JSONL, either official four/five options."""
    for path in paths:
        path = Path(path)
        for line, row in jsonl(path):
            sid = f"{path.name}:{line}"
            try:
                if not isinstance(row, dict):
                    raise ValueError("MedQA record must be an object")
                if any(row.get(key) for key in ("image", "images", "qimage", "image_path")):
                    raise ValueError("image-dependent questions are outside the text-only MedQA adapter")
                options = row["options"]
                if not isinstance(options, dict):
                    raise ValueError("MedQA options must be an A..D/E object")
                # The original reader constructs candidates by letter, independent of JSON key order.
                ordered = {key: options[key] for key in sorted(options)}
                answer = row.get("answer_idx")
                case = _exam(source, sid, row["question"], ordered, answer)
                if "answer" in row and row["answer"] != ordered[answer]:
                    raise ValueError("answer and answer_idx disagree")
                yield case
            except (ValueError, KeyError, TypeError) as exc:
                audit(source, sid, str(exc))


ADAPTERS = {
    "cblue_qic": convert_qic, "cblue_sts": convert_sts,
    "cblue_qtr": convert_qtr, "cblue_qqr": convert_qqr, "cblue_ctc": convert_ctc,
    "cmexam": convert_cmexam, "medqa_cn": convert_medqa,
}
