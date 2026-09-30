"""Legal decision adapters; standard library only, with no implicit negative sampling.

Source schemas and task boundaries are listed in docs/adapters.md. Unknown annotations are
rejected, never converted into negative targets. Auxiliary label files are resolved
and hashed by the shared build engine before these generators are called.
"""
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from .core import dumps, group_key, jsonl, make_case, one_hot, text

CONFLICT_TYPES = (
    "无冲突", "职权或责任划分不符", "概念或定义范围不符",
    "处罚幅度或范围不符", "增设或变更适用条件",
)
ELEMENT_DOMAINS = {"divorce": "婚姻家庭", "labor": "劳动争议", "loan": "借款纠纷"}


def _lines(source, key):
    auxiliary = source.get("_resolved_auxiliary_paths", {})
    if key not in auxiliary:
        raise ValueError(f"auxiliary_paths.{key} is required")
    values = Path(auxiliary[key]).read_text(encoding="utf-8-sig").splitlines()
    # Ignore only blank trailing lines; an interior blank would shift the mapping.
    while values and not values[-1].strip():
        values.pop()
    values = [text(value.strip(), key) for value in values]
    if not values or len(values) != len(set(values)):
        raise ValueError(f"{key} must contain a nonempty, unique label list")
    return values


def _source_check(source):
    if "negative_sampling" in source:
        raise ValueError("negative_sampling is not implemented; all candidates are retained")


def _row_id(path, line, row):
    value = row.get("id", line) if isinstance(row, dict) else line
    if not isinstance(value, (str, int)) or isinstance(value, bool) or not str(value).strip():
        raise ValueError("id must be a nonempty string or integer")
    return f"{path.name}:{value}"


def _case(source, original_id, suffix, group, state, question, selected,
          supervision="source_hard_label", **metadata):
    row = make_case(source, original_id + suffix, group, state, question,
                    one_hot(list(question["criteria"]), selected), supervision)
    row["_meta"].update(original_id=original_id,
                         original_split=source.get("original_split", source["split"]),
                         **metadata)
    return row


def _binary(source, original_id, suffix, group, state, instruction, positive, **meta):
    question = {"type": "noul", "instructions": instruction,
                "criteria": {"false": "否", "true": "是"}}
    return _case(source, original_id, suffix, group, state, question,
                 "true" if positive else "false", **meta)


def _labels(value, universe, *, empty=False):
    if not isinstance(value, list) or (not empty and not value):
        raise ValueError("explicit nonempty label list required" if not empty
                         else "explicit labels list required")
    if any(not isinstance(label, str) or label not in universe for label in value):
        raise ValueError("unknown label or invalid label type")
    if len(value) != len(set(value)):
        raise ValueError("duplicate labels")
    return set(value)


def convert_elements(source, paths, audit, review):
    """CAIL2019: each JSONL line is one document's list of annotated sentences."""
    _source_check(source)
    if source.get("labels_available") is not True:
        raise ValueError("labels_available=true required: official inference inputs also use labels=[]")
    domain = source.get("domain")
    if domain not in ELEMENT_DOMAINS:
        raise ValueError("domain must be divorce, labor or loan")
    tags, names = _lines(source, "tags"), _lines(source, "tag_names")
    if len(tags) != len(names):
        raise ValueError("tags and tag_names must have equal length and matching line order")
    prefix = {"divorce": "DV", "labor": "LB", "loan": "LN"}[domain]
    if set(tags) != {prefix + str(index) for index in range(1, 21)}:
        raise ValueError("tags must contain the complete 20-label universe for this domain")
    for path in paths:
        for line, document in jsonl(path):
            sid = f"{path.name}:{line}"
            try:
                if not isinstance(document, list) or not document:
                    raise ValueError("each JSONL row must be a nonempty document sentence list")
                sentences = [text(item["sentence"], "sentence") for item in document]
                # Validate the entire document first: never keep half a corrupt document.
                labels = [_labels(item["labels"], tags, empty=True) for item in document]
                group = group_key("cail2019_document", dumps(sentences))
                for index, (sentence, selected) in enumerate(zip(sentences, labels)):
                    state = {"领域": ELEMENT_DOMAINS[domain], "句子": sentence}
                    for tag, name in zip(tags, names):
                        yield _binary(source, sid, f":{index}:{tag}", group, state,
                                      "该句是否包含下列案件要素？\n" + name, tag in selected,
                                      sentence_index=index, target_label=tag, target_name=name,
                                      supervision="source_complete_multilabel_annotation")
            except (ValueError, KeyError, TypeError) as exc:
                audit(source, sid, str(exc))


def convert_jec_qa(source, paths, audit, review):
    """JEC-QA JSONL statement/option_list/answer; choose the task before reading gold."""
    _source_check(source)
    mode = source.get("mode", "multiple")
    if mode not in ("single", "multiple"):
        raise ValueError("jec_qa mode must be single or multiple")
    for path in paths:
        for line, row in jsonl(path):
            sid = f"{path.name}:{line}"
            try:
                sid = _row_id(path, line, row)
                statement = text(row["statement"], "statement")
                options = row["option_list"]
                if not isinstance(options, dict) or set(options) != set("ABCD"):
                    raise ValueError("option_list must contain exactly A/B/C/D")
                options = {key: text(options[key], "option") for key in "ABCD"}
                if len(set(options.values())) != 4:
                    raise ValueError("duplicate option texts")
                answer = row["answer"]
                selected = _labels(list(answer) if isinstance(answer, str) else answer, options)
                # Keeping options in state avoids the per-candidate 48-token limit.
                state = {"题目": statement, "选项": options}
                group = group_key("jec_question", statement)
                if mode == "single":
                    if len(selected) != 1:
                        raise ValueError("single mode requires exactly one answer; no first-label fallback")
                    question = {"type": "choice", "instructions": "选择该题的正确选项。",
                                "criteria": {key: "选项 " + key for key in "ABCD"}}
                    yield _case(source, sid, ":single", group, state, question, next(iter(selected)))
                else:
                    for key in "ABCD":
                        yield _binary(source, sid, ":" + key, group, state,
                                      f"选项 {key} 是否属于该题的正确答案集合？", key in selected,
                                      target_label=key, supervision="source_complete_answer_set")
            except (ValueError, KeyError, TypeError) as exc:
                audit(source, sid, str(exc))


def convert_cail2018(source, paths, audit, review):
    """CAIL2018 accusation/article labels; original fact is the only model input."""
    _source_check(source)
    task, mode = source.get("task", "accusation"), source.get("mode", "noul")
    if task not in ("accusation", "articles") or mode not in ("noul", "single_choice"):
        raise ValueError("cail2018 requires task=accusation|articles and mode=noul|single_choice")
    labels = _lines(source, "labels")
    if len(labels) < 2:
        raise ValueError("at least two labels required")
    if task == "articles" and any(not label.isascii() or not label.isdecimal() for label in labels):
        raise ValueError("article universe must contain decimal article numbers")
    criteria = {label: ("刑法第" + label + "条" if task == "articles" else label)
                for label in labels}
    for path in paths:
        for line, row in jsonl(path):
            sid = f"{path.name}:{line}"
            try:
                sid = _row_id(path, line, row)
                fact = text(row["fact"], "fact")
                raw_labels = row["meta"]["accusation" if task == "accusation" else "relevant_articles"]
                if task == "articles":
                    if not isinstance(raw_labels, list) or any(
                            not (type(value) is int or isinstance(value, str) and value.isascii()
                                 and value.isdecimal()) for value in raw_labels):
                        raise ValueError("relevant_articles must be an explicit list of integer article numbers")
                    raw_labels = [str(value) for value in raw_labels]
                selected = _labels(raw_labels, criteria)
                group = group_key("cail2018_fact", fact)
                if mode == "single_choice":
                    if len(selected) != 1:
                        raise ValueError("single_choice excludes multilabel cases; no first-label fallback")
                    question = {"type": "choice", "instructions": "选择案件对应的" +
                                ("罪名。" if task == "accusation" else "刑法条文。"), "criteria": criteria}
                    yield _case(source, sid, ":" + task, group, fact, question, next(iter(selected)),
                                "source_hard_label_single_label_subset", task=task)
                else:
                    for label, description in criteria.items():
                        instruction = ("案件是否涉及罪名：" + description + "？" if task == "accusation"
                                       else "案件是否适用" + description + "？")
                        yield _binary(source, sid, f":{task}:{label}", group, fact, instruction,
                                      label in selected, task=task, target_label=label,
                                      supervision="source_complete_multilabel_annotation")
            except (ValueError, KeyError, TypeError) as exc:
                audit(source, sid, str(exc))


def convert_scm(source, paths, audit, review):
    """Released CAIL2019-SCM zip has explicit B/C labels in all three splits."""
    _source_check(source)
    for path in paths:
        for line, row in jsonl(path):
            sid = f"{path.name}:{line}"
            try:
                sid = _row_id(path, line, row)
                documents = {key: text(row[key], key) for key in "ABC"}
                if len(set(documents.values())) != 3:
                    raise ValueError("A/B/C must contain distinct case texts")
                label = row["label"]
                if label not in ("B", "C"):
                    raise ValueError("SCM requires an explicit B/C label")
                question = {"type": "choice", "instructions": "案件 B 与案件 C 中，哪一个与案件 A 更相似？",
                            "criteria": {"B": "案件 B", "C": "案件 C"}}
                yield _case(source, sid, "", group_key("scm_query", documents["A"]),
                            documents, question, label)
            except (ValueError, KeyError, TypeError) as exc:
                audit(source, sid, str(exc))


def convert_lcr_cn(source, paths, audit, review):
    """LCR-CN v4 conflict classification, with gold reference laws supplied as input."""
    _source_check(source)
    for path in paths:
        for line, row in jsonl(path):
            sid = f"{path.name}:{line}"
            try:
                sid = _row_id(path, line, row)
                content, title = text(row["content"], "content"), text(row["title"], "title")
                laws = row["high_level_laws"]
                if not isinstance(laws, list) or not laws:
                    raise ValueError("high_level_laws must be a nonempty list of provision texts")
                laws = [text(law, "high_level_laws item") for law in laws]
                label = row["conflict_type"]
                if label not in CONFLICT_TYPES:
                    raise ValueError("unknown conflict_type")
                url = text(row["url"], "source document url")
                parsed = urlsplit(url)
                document_ids = parse_qs(parsed.query).get("id")
                document = parsed.netloc + parsed.path + "?id=" + document_ids[0] if document_ids else url
                state = {"下位法标题": title, "下位法条文": content, "上位法条文": laws}
                question = {"type": "choice", "instructions": "依据给定上位法，判断下位法条文的冲突类型。",
                            "criteria": {str(index): value for index, value in enumerate(CONFLICT_TYPES)}}
                yield _case(source, sid, "", group_key("lcr_document", document), state, question,
                            str(CONFLICT_TYPES.index(label)), task="conflict_classification_gold_references")
            except (ValueError, KeyError, TypeError) as exc:
                audit(source, sid, str(exc))


ADAPTERS = {"cail2019_elements": convert_elements, "jec_qa": convert_jec_qa,
            "cail2018": convert_cail2018, "cail2019_scm": convert_scm, "lcr_cn": convert_lcr_cn}
