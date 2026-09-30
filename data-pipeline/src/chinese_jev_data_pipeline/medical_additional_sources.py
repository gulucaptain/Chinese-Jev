"""Conditional entity/relation typing and verified diagnosis normalization.

These adapters classify supplied annotated spans/pairs. They do not manufacture
unannotated negatives or claim to train a complete entity/relation extractor.
CDN pairs use PaddleNLP's published retrieval candidates and are checked against
the corresponding original CHIP-CDN gold file; hidden test placeholders fail.
"""
from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

from .core import group_key, make_case, one_hot, text
from .medical_extended import _records

ENTITY_TYPES = {"dis": "疾病", "sym": "临床表现", "dru": "药物", "equ": "医疗设备",
                "pro": "医疗程序", "bod": "身体", "ite": "医学检验项目",
                "mic": "微生物类", "dep": "科室"}
TEXT2DT_RELATIONS = ("临床表现", "治疗药物", "治疗方案", "用法用量", "禁用药物", "基本情况")


def _case(source, sid, document, state, question, label, family, *, original_id,
          supervision="source_hard_label"):
    group = group_key("medical_text", document)
    keys = question["criteria"] if question["type"] == "choice" else ("false", "true")
    result = make_case(source, sid, group, state, question, one_hot(keys, label),
                       supervision, original_id=original_id)
    result["_meta"]["integration"] = {
        "task_family": family, "semantic_label": label,
        "medical_field": "general_medicine", "original_option_count": len(keys),
        "original_answer_count": 1, "link_keys": [group],
    }
    return result


def convert_cmeee_typing(source, paths, audit, review):
    question = {"type": "choice", "instructions": "判断材料中指定实体片段的医学实体类型。",
                "criteria": ENTITY_TYPES}
    for path, index, row in _records(paths):
        sid = f"{path.name}:{index}"
        try:
            document = text(row["text"], "text")
            annotations = row["entities"]
            if not isinstance(annotations, list):
                raise ValueError("entities must be a list; hidden test has no gold")
            spans = defaultdict(set)
            for entity in annotations:
                a, b, label = entity["start_idx"], entity["end_idx"], entity["type"]
                if type(a) is not int or type(b) is not int or not 0 <= a <= b < len(document):
                    raise ValueError("invalid inclusive entity offsets")
                if label not in ENTITY_TYPES:
                    raise ValueError("unknown entity type")
                if "entity" in entity and entity["entity"] != document[a:b + 1]:
                    raise ValueError("entity text does not match source offsets")
                spans[a, b].add(label)
            for (a, b), labels in sorted(spans.items()):
                if len(labels) != 1:
                    audit(source, f"{sid}:{a}:{b}", "same span has multiple entity types")
                    continue
                state = f"材料：{document}\n指定实体：{document[a:b + 1]}\n字符位置（从0开始，含两端）：{a}–{b}"
                yield _case(source, f"{sid}:{a}:{b}", document, state, question,
                            next(iter(labels)), "entity_typing_given_span", original_id=sid)
        except (KeyError, TypeError, ValueError) as exc:
            audit(source, sid, str(exc))


def _relation_cases(source, sid, document, pairs, labels, family, audit):
    question = {"type": "choice", "instructions": "根据材料判断指定主体到客体的关系类型。主体和客体已给定。",
                "criteria": {label: label for label in labels}}
    for index, ((subject, obj), gold) in enumerate(sorted(pairs.items())):
        if len(gold) != 1:
            audit(source, f"{sid}:{index}", "annotated entity pair has multiple relations; not forced into single choice")
            continue
        label = next(iter(gold))
        if label not in labels:
            audit(source, f"{sid}:{index}", "relation absent from independent schema")
            continue
        state = f"材料：{document}\n主体：{subject}\n客体：{obj}"
        yield _case(source, f"{sid}:{index}", document, state, question, label, family, original_id=sid)


def convert_cmeie_typing(source, paths, audit, review):
    schema_path = source.get("_resolved_auxiliary_paths", source.get("auxiliary_paths", {})).get("schema")
    if not schema_path:
        raise ValueError("CMeIE requires its independent 53_schemas.json")
    labels = []
    for _, _, row in _records([schema_path]):
        label = text(row["predicate"], "predicate")
        if label not in labels:
            labels.append(label)
    if len(labels) < 2:
        raise ValueError("CMeIE schema needs multiple predicates")
    # 53 subject/object schemas contain 44 distinct predicates; synonymous
    # relations are not split by the gold entity type (which is not an input).
    for path, index, row in _records(paths):
        sid = f"{path.name}:{index}"
        try:
            document = text(row["text"], "text")
            annotations = row["spo_list"]
            if not isinstance(annotations, list):
                raise ValueError("spo_list must be a list")
            pairs = defaultdict(set)
            for spo in annotations:
                subject, obj = text(spo["subject"], "subject"), text(spo["object"]["@value"], "object")
                if subject not in document or obj not in document:
                    raise ValueError("annotated entity missing from source text")
                pairs[subject, obj].add(text(spo["predicate"], "predicate"))
            yield from _relation_cases(source, sid, document, pairs, labels,
                                       "relation_typing_given_pair", audit)
        except (KeyError, TypeError, ValueError) as exc:
            audit(source, sid, str(exc))


def convert_text2dt_typing(source, paths, audit, review):
    for path, index, row in _records(paths):
        sid = f"{path.name}:{index}"
        try:
            document = text(row["text"], "text")
            tree = row["tree"]
            if not isinstance(tree, list):
                raise ValueError("tree must be a list")
            pairs = defaultdict(set)
            for node in tree:
                for triple in node["triples"]:
                    if not isinstance(triple, list) or len(triple) != 3:
                        raise ValueError("triple must have subject, predicate, object")
                    subject, label, obj = (text(v, "triple field") for v in triple)
                    pairs[subject, obj].add(label)
            yield from _relation_cases(source, sid, document, pairs, TEXT2DT_RELATIONS,
                                       "decision_rule_relation_typing_given_pair", audit)
        except (KeyError, TypeError, ValueError) as exc:
            audit(source, sid, str(exc))


def convert_cdn_pairs(source, paths, audit, review):
    gold_path = source.get("_resolved_auxiliary_paths", source.get("auxiliary_paths", {})).get("original_gold")
    if not gold_path:
        raise ValueError("CDN candidates require original labeled CHIP-CDN file")
    gold = defaultdict(set)
    for _, _, row in _records([gold_path]):
        diagnosis = text(row["text"], "diagnosis")
        normalized = text(row["normalized_result"], "normalized_result")
        gold[diagnosis].update(text(x, "normalized term") for x in normalized.split("##"))
    question = {"type": "noul", "instructions": "依据原始诊断词的规范化标注，候选标准诊断术语是否是其对应术语之一？",
                "criteria": {"false": "不是对应术语", "true": "是对应术语之一"}}
    for value in paths:
        path = Path(value)
        if source.get("original_split", "").lower() in ("test", "official_test"):
            raise ValueError("Paddle CDN test labels are hidden-test zero placeholders")
        with path.open(encoding="utf-8", newline="") as stream:
            # The release is raw tab-separated text: literal quotes belong to
            # clinical terms, not CSV quoting. Standard quote handling corrupts
            # 28,210 training pairs in the fixed public version.
            reader = csv.DictReader(stream, delimiter="\t", quoting=csv.QUOTE_NONE)
            if reader.fieldnames != ["text_a", "text_b", "label"]:
                raise ValueError("expected Paddle CDN columns text_a/text_b/label")
            for index, row in enumerate(reader, 2):
                sid = f"{path.name}:{index}"
                try:
                    diagnosis, candidate = text(row["text_a"], "text_a"), text(row["text_b"], "text_b")
                    if diagnosis not in gold or row["label"] not in ("0", "1"):
                        raise ValueError("candidate has no original gold diagnosis or invalid label")
                    if int(row["label"]) != int(candidate in gold[diagnosis]):
                        raise ValueError("candidate label contradicts original normalized_result")
                    state = f"原始诊断词：{diagnosis}\n候选标准术语：{candidate}"
                    label = "true" if row["label"] == "1" else "false"
                    yield _case(source, sid, diagnosis, state, question, label,
                                "diagnosis_normalization_candidate_match",
                                original_id=group_key("medical_text", diagnosis),
                                supervision="source_gold_verified_retrieved_candidates")
                except (KeyError, TypeError, ValueError) as exc:
                    audit(source, sid, str(exc))


ADAPTERS = {"cmeee_typing": convert_cmeee_typing, "cmeie_typing": convert_cmeie_typing,
            "text2dt_typing": convert_text2dt_typing, "cdn_retrieved_pairs": convert_cdn_pairs}


def _template_patterns(source, task):
    path = source.get("_resolved_auxiliary_paths", source.get("auxiliary_paths", {})).get("templates")
    if not path:
        raise ValueError("AlternateCD adapter requires the author prompt templates")
    templates = json.loads(Path(path).read_text(encoding="utf-8"))[task]
    patterns = []
    for template in templates:
        template = template.replace("\\\\n", "\n").replace("\\n", "\n").strip()
        if template.endswith("\n答："):
            template = template[:-3].strip()
        pattern = re.escape(template).replace(re.escape("[INPUT_TEXT]"), "(?P<document>.+)")
        pattern = pattern.replace(re.escape("[LIST_LABELS]"), ".+?")
        patterns.append(re.compile(pattern, re.S))
    return patterns


def _unprompt(prompt, patterns):
    matches = {m.group("document").strip() for p in patterns if (m := p.fullmatch(prompt.strip()))}
    if len(matches) != 1:
        raise ValueError("cannot uniquely recover document from fixed author prompt templates")
    return text(matches.pop(), "document")


def convert_alternatecd_mdcf(source, paths, audit, review):
    patterns = _template_patterns(source, "CHIP-MDCFNPC")
    criteria = {
        "阳性": "按原任务，已有症状疾病或假设未来可能发生的疾病等",
        "阴性": "未患有该症状疾病",
        "其他": "没有回答、不知道、回答不明确或模棱两可，无法推断",
        "不标注": "无实际意义，或与病人当前状态独立的提及",
    }
    question = {"type": "choice", "instructions": "按给定医患对话判断指定临床发现的原任务状态标签。",
                "criteria": criteria}
    for path, index, row in _records(paths):
        sid = f"{path.name}:{index}"
        try:
            if row["task_dataset"] != "CHIP-MDCFNPC" or set(row["answer_choices"]) != set(criteria):
                raise ValueError("wrong AlternateCD task or label vocabulary")
            document = _unprompt(text(row["input"]), patterns)
            findings = defaultdict(set)
            for line in text(row["target"]).splitlines():
                if "：" not in line:
                    audit(source, sid, "finding target line missing entity/label separator")
                    continue
                finding, label = line.rsplit("：", 1)
                if not finding.strip() or label not in criteria or finding.strip() not in document:
                    audit(source, sid, "finding target invalid or mention absent from dialogue")
                    continue
                findings[finding.strip()].add(label)
            for i, (finding, labels) in enumerate(sorted(findings.items())):
                if len(labels) != 1:
                    audit(source, f"{sid}:{i}", "same mention has different states without occurrence offsets")
                    continue
                state = f"医患对话：{document}\n指定临床发现：{finding}"
                yield _case(source, f"{sid}:{i}", document, state, question, next(iter(labels)),
                            "clinical_finding_status_given_mention", original_id=text(row["sample_id"]),
                            supervision="source_hard_label_research_reformatted")
        except (KeyError, TypeError, ValueError) as exc:
            audit(source, sid, str(exc))


def convert_alternatecd_causal(source, paths, audit, review):
    patterns = _template_patterns(source, "CMedCausal")
    labels = ("因果关系", "上下位关系")
    for path, index, row in _records(paths):
        sid = f"{path.name}:{index}"
        try:
            if row["task_dataset"] != "CMedCausal" or set(row["answer_choices"]) != set(labels) | {"条件关系"}:
                raise ValueError("wrong AlternateCD task or label vocabulary")
            document = _unprompt(text(row["input"]), patterns)
            pairs = defaultdict(set)
            for line in text(row["target"]).splitlines():
                if "：" not in line:
                    audit(source, sid, "causal target has no labeled entity pair")
                    continue
                pair, label = line.rsplit("：", 1)
                if label == "条件关系":
                    audit(source, sid, "nested conditional triple needs separate task; not included in pair typing")
                    continue
                # Strict delimiter parsing avoids silently changing entity
                # boundaries in the flattened author target representation.
                if label not in labels or pair.count("，") != 1:
                    audit(source, sid, "ambiguous flattened causal entity pair")
                    continue
                subject, obj = (x.strip() for x in pair.split("，"))
                if not subject or not obj or subject not in document or obj not in document:
                    audit(source, sid, "causal entity mention absent from document")
                    continue
                pairs[subject, obj].add(label)
            for case in _relation_cases(source, sid, document, pairs, labels,
                                        "causal_or_hierarchical_typing_given_pair", audit):
                case["_meta"]["original_id"] = text(row["sample_id"])
                case["_meta"]["supervision"] = "source_hard_label_research_reformatted"
                yield case
        except (KeyError, TypeError, ValueError) as exc:
            audit(source, sid, str(exc))


ADAPTERS.update(alternatecd_mdcf=convert_alternatecd_mdcf,
                alternatecd_causal_pair=convert_alternatecd_causal)


PSYQA_STRATEGIES = {
    "Restatement": "重述来访者描述",
    "Approval and Reassurance": "认可与安慰",
    "Interpretation": "解释来访者的情况或心理体验",
    "Direct Guidance": "直接提供行动建议",
    "Self-disclosure": "回答者自我披露",
    "Information": "提供客观知识或信息",
    "Others": "其他支持策略或表达",
}


def convert_psyqa_strategy(source, paths, audit, review):
    question = {"type": "choice", "instructions": "根据上下文，判断指定回答片段采用的心理支持策略。",
                "criteria": PSYQA_STRATEGIES}
    for path, index, row in _records(paths):
        sid = f"{path.name}:{index}"
        try:
            query = text(row["question"], "question")
            description = row["description"]
            if not isinstance(description, str):
                raise ValueError("description must be a string")
            document = f"问题：{query}\n问题描述：{description}"
            original_id = str(row["questionID"])
            answers = row["answers"]
            if not isinstance(answers, list):
                raise ValueError("answers must be a list")
            for ai, answer in enumerate(answers):
                if answer.get("has_label") is not True:
                    audit(source, f"{sid}:{ai}", "answer has no strategy annotations")
                    continue
                response = text(answer["answer_text"], "answer_text")
                for si, span in enumerate(answer["labels_sequence"]):
                    a, b, label = span["start"], span["end"], span["type"]
                    if type(a) is not int or type(b) is not int or not 0 <= a < b <= len(response):
                        audit(source, f"{sid}:{ai}:{si}", "invalid half-open strategy offsets")
                        continue
                    if label not in PSYQA_STRATEGIES:
                        audit(source, f"{sid}:{ai}:{si}", "unknown strategy label")
                        continue
                    state = f"{document}\n此前回答：{response[:a]}\n指定回答片段：{response[a:b]}"
                    case = _case(source, f"{sid}:{ai}:{si}", document, state, question, label,
                                 "mental_health_support_strategy_given_span", original_id=original_id)
                    case["_meta"]["integration"]["medical_field"] = "mental_health_support"
                    case["_meta"]["integration"]["link_keys"].append(group_key("medical_text", query))
                    yield case
        except (KeyError, TypeError, ValueError) as exc:
            audit(source, sid, str(exc))


ADAPTERS["psyqa_strategy"] = convert_psyqa_strategy
