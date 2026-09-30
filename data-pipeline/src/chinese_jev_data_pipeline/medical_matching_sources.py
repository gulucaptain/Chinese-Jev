"""Original cMedQA2 candidate labels; query-disjoint matching, not medical truth."""
from __future__ import annotations

import csv
from pathlib import Path

from .core import group_key, make_case, one_hot, text


def _table(path, key, required):
    result = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not set(required) <= set(reader.fieldnames or []):
            raise ValueError(f"missing table fields: {required}")
        for row in reader:
            identity = text(row[key], key)
            if identity in result:
                raise ValueError(f"duplicate {key} in source table")
            result[identity] = row
    return result


def convert_cmedqa2(source, paths, audit, review):
    aux = source.get("_resolved_auxiliary_paths", {})
    if not {"questions", "answers"} <= set(aux):
        raise ValueError("cmedqa2 requires auxiliary_paths.questions and answers")
    questions = _table(aux["questions"], "question_id", ("question_id", "content"))
    answers = _table(aux["answers"], "ans_id", ("ans_id", "content"))
    question = {"type": "noul", "instructions": "根据原问答匹配任务，该候选回答是否与问题匹配？",
                "criteria": {"false": "原数据中的不匹配候选", "true": "原数据中的匹配回答"}}

    def convert_group(qid, pairs, filename):
        original = f"{filename}:{qid}"
        try:
            query = text(questions[qid]["content"], "question content")
        except (KeyError, ValueError) as exc:
            audit(source, original, str(exc))
            return
        # Repeated positive IDs in training triples are one supervised pair.
        # Do not label all answers from another corpus as negative: only use
        # the original train negative IDs or released dev/test labels.
        positives = sum(labels == {"1"} for labels in pairs.values())
        for aid, labels in sorted(pairs.items()):
            sid = f"{original}:{aid}"
            try:
                if len(labels) != 1:
                    raise ValueError("conflicting original candidate labels")
                label = next(iter(labels))
                if label not in ("0", "1"):
                    raise ValueError("unknown original candidate label")
                answer = text(answers[aid]["content"], "answer content")
                result = make_case(source, sid, group_key("medical_text", query),
                    f"问题：{query}\n候选回答：{answer}", question,
                    one_hot(["false", "true"], "true" if label == "1" else "false"),
                    "source_candidate_matching_label", original_id=original)
                result["_meta"]["integration"] = {
                    "task_family": "medical_answer_matching", "medical_field": "general_medicine",
                    "semantic_label": "true" if label == "1" else "false",
                    "original_option_count": len(pairs), "original_answer_count": positives,
                    "link_keys": [group_key("medical_text", query)],
                    "link_exact_state": False,
                    "matching_split_protocol": "query_disjoint_shared_answer_corpus",
                    "candidate_id": aid,
                }
                yield result
            except (KeyError, ValueError, TypeError) as exc:
                audit(source, sid, str(exc))

    for path in paths:
        path = Path(path)
        seen, current, pairs = set(), None, {}
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            fields = set(reader.fieldnames or [])
            training = fields == {"question_id", "pos_ans_id", "neg_ans_id"}
            if not training and not {"question_id", "ans_id", "label"} <= fields:
                raise ValueError("unknown cMedQA2 candidates schema")
            for row in reader:
                qid = text(row["question_id"], "question_id")
                if qid != current:
                    if qid in seen:
                        raise ValueError("cMedQA2 input must keep each original query contiguous")
                    if current is not None:
                        yield from convert_group(current, pairs, path.name)
                    seen.add(qid)
                    current, pairs = qid, {}
                values = ((row["pos_ans_id"], "1"), (row["neg_ans_id"], "0")) if training else ((row["ans_id"], row["label"]),)
                for aid, label in values:
                    pairs.setdefault(text(aid, "answer id"), set()).add(label)
        if current is not None:
            yield from convert_group(current, pairs, path.name)


ADAPTERS = {"cmedqa2_matching": convert_cmedqa2}
