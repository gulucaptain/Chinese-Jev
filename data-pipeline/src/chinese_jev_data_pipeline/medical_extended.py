"""Additional native medical adapters for the shared Chinese-Jev data pipeline.

Local input only. JSON framing is detected from content, including cached .raw
files. Source labels and explanations never become model inputs. Full-label
TCM-SD conversion does not imply its 148 candidates fit the tokenizer budget.
"""
from __future__ import annotations

import csv
import json
import re
from collections import OrderedDict
from pathlib import Path

from .core import digest, group_key, make_case, one_hot, text

CMID_COARSE = ("病症", "药物", "其他", "治疗方案")
_NON_TEXT = re.compile(r"(?:如|见|下|上|该|此)图|图示|图中|图片|照片|如下表|见下表|见表\d|!\[[^\]]*\]\(|<img\b", re.I)


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result


def _records(paths):
    for value in paths:
        path = Path(value)
        with path.open(encoding="utf-8-sig") as stream:
            first = stream.read(1)
            while first and first.isspace():
                first = stream.read(1)
            stream.seek(0)
            if first == "[":
                rows = json.load(stream, object_pairs_hook=_object)
                for index, row in enumerate(rows, 1):
                    yield path, index, row
            elif first == "{":
                for index, line in enumerate(stream, 1):
                    if line.strip():
                        yield path, index, json.loads(line, object_pairs_hook=_object)
            else:
                raise ValueError(f"{path}: expected JSON array or JSONL objects")


def _medical_links(*parts):
    return sorted({group_key("medical_text", value) for value in parts if value.strip()})


def _case(source, sid, group, state, question, probabilities, *, task_family,
          semantic_label, medical_field, option_count=None, answer_count=None,
          links=(), original_id=None):
    result = make_case(source, sid, group, state, question, probabilities,
                       "source_hard_label", original_id=original_id)
    result["_meta"]["integration"] = {
        "task_family": task_family, "semantic_label": semantic_label,
        "medical_field": medical_field,
        "original_option_count": option_count, "original_answer_count": answer_count,
        "link_keys": sorted(set(links) | {group}),
    }
    return result


def convert_cmid_coarse(source, paths, audit, review):
    criteria = {label: label for label in CMID_COARSE}
    question = dict(type="choice", instructions="判定医疗问句所属的粗粒度意图类别。", criteria=criteria)
    for path, index, row in _records(paths):
        sid = f"{path.name}:{index}"
        try:
            if not isinstance(row, dict):
                raise ValueError("CMID record must be an object")
            labels = row["label_4class"]
            if not isinstance(labels, list) or len(labels) != 1 or not isinstance(labels[0], str):
                raise ValueError("CMID label_4class must contain exactly one string")
            label = labels[0].strip()
            # The release mixes 病症 and '病症'; remove exactly one paired wrapper.
            if len(label) >= 2 and label[0] == label[-1] == "'":
                label = label[1:-1]
            if label not in criteria:
                raise ValueError("unknown CMID coarse label")
            query = text(row["originalText"], "originalText")
            group = group_key("medical_text", query)
            yield _case(source, sid, group, query, question, one_hot(criteria, label),
                        task_family="medical_intent", semantic_label=label,
                        medical_field="general_medicine", option_count=4, answer_count=1,
                        links=_medical_links(query))
        except (ValueError, TypeError, KeyError) as exc:
            audit(source, sid, str(exc))


def _exam_input(row, options, answer):
    query = text(row["question"], "question")
    if any(row.get(key) for key in ("image", "images", "qimage", "image_path", "picture", "table")):
        raise ValueError("image/table-dependent exam record needs separate review")
    if not isinstance(options, dict) or not 2 <= len(options) <= 26:
        raise ValueError("exam options must contain 2..26 letter-keyed candidates")
    keys = [chr(ord("A") + index) for index in range(len(options))]
    if set(options) != set(keys):
        raise ValueError("exam options must have contiguous A..Z keys")
    ordered = {key: text(options[key], "option") for key in keys}
    if len({value.strip() for value in ordered.values()}) != len(ordered):
        raise ValueError("duplicate exam candidates need review")
    if _NON_TEXT.search(query) or any(_NON_TEXT.search(value) for value in ordered.values()):
        raise ValueError("explicit image/table reference needs separate review")
    if not isinstance(answer, str):
        raise ValueError("exam answer must be explicit letter string")
    answer = answer.strip()
    if not re.fullmatch(r"[A-Z](?:[A-Z]|[,，、\s]+[A-Z])*", answer):
        raise ValueError("invalid exam answer format")
    selected = re.findall("[A-Z]", answer)
    if len(selected) != len(set(selected)) or not set(selected) <= set(keys):
        raise ValueError("duplicate or out-of-range answer labels")
    return query, ordered, selected


def convert_explain_cpe(source, paths, audit, review):
    for path, index, row in _records(paths):
        sid = f"{path.name}:{index}"
        try:
            if not isinstance(row, dict):
                raise ValueError("ExplainCPE record must be an object")
            if type(row.get("id")) not in (str, int):
                raise ValueError("ExplainCPE needs a string/integer id")
            sid = f"{path.name}:{row['id']}"
            options = row["options"]
            if not isinstance(options, list) or len(options) != 5:
                raise ValueError("ExplainCPE needs the original five options")
            query, criteria, answers = _exam_input(row, dict(zip("ABCDE", options)), row["answer"])
            if len(answers) != 1:
                raise ValueError("ExplainCPE needs one answer")
            question = dict(type="choice", instructions="根据药师考试题选择一个正确答案。", criteria=criteria)
            group = group_key("medical_text", query)
            yield _case(source, sid, group, query, question, one_hot(criteria, answers[0]),
                        task_family="medical_exam_single", semantic_label="single_answer",
                        medical_field="pharmacy", option_count=5, answer_count=1,
                        links=_medical_links(query))
        except (ValueError, TypeError, KeyError) as exc:
            audit(source, sid, str(exc))


def convert_cmb_exam(source, paths, audit, review):
    for path, index, row in _records(paths):
        sid = f"{path.name}:{index}"
        try:
            if not isinstance(row, dict):
                raise ValueError("CMB exam record must be an object")
            kind = row["question_type"]
            if kind not in ("单项选择题", "多项选择题"):
                raise ValueError("unsupported CMB question_type")
            query, options, answers = _exam_input(row, row["option"], row["answer"])
            if (kind == "单项选择题" and len(answers) != 1) or (kind == "多项选择题" and len(answers) < 2):
                raise ValueError("CMB question_type and number of answers disagree")
            field = text(row["exam_subject"], "exam_subject")
            for key in ("exam_type", "exam_class"):
                text(row[key], key)
            group = group_key("medical_text", query)
            common = dict(medical_field=field, option_count=len(options),
                          answer_count=len(answers), links=_medical_links(query), original_id=sid)
            if kind == "单项选择题":
                question = dict(type="choice", instructions="根据医学试题选择一个正确答案。", criteria=options)
                cases = [_case(source, sid, group, query, question, one_hot(options, answers[0]),
                               task_family="medical_exam_single", semantic_label="single_answer", **common)]
            else:
                # Full original candidates remain visible, including comparative/all-of-the-above options.
                context = query + "\n\n原题全部选项：\n" + "\n".join(f"{key}：{value}" for key, value in options.items())
                cases = []
                for key in options:
                    question = dict(type="noul", instructions=f"这是多项选择题。依据原题完整选项，选项 {key} 是否应被选中？",
                                    criteria={"false": "原答案未选择此选项", "true": "原答案选择此选项"})
                    label = "true" if key in answers else "false"
                    cases.append(_case(source, f"{sid}:{key}", group, context, question,
                                       one_hot(["false", "true"], label), task_family="medical_exam_multi",
                                       semantic_label=label, **common))
            for case in cases:
                case["_meta"]["integration"].update(exam_type=row["exam_type"], exam_class=row["exam_class"])
                yield case
        except (ValueError, TypeError, KeyError) as exc:
            audit(source, sid, str(exc))


def convert_webmedqa(source, paths, audit, review):
    mode = source.get("mode", "choice")
    if mode not in ("choice", "noul"):
        raise ValueError("webmedqa mode must be choice or noul")
    answer_links = source.get("link_shared_answers", "all")
    if answer_links not in ("all", "positive", "none"):
        raise ValueError("link_shared_answers must be all/positive/none")
    for value in paths:
        path = Path(value)
        groups = OrderedDict()
        with path.open(encoding="utf-8-sig", newline="") as stream:
            for index, fields in enumerate(csv.reader(stream, delimiter="\t", quoting=csv.QUOTE_NONE), 1):
                if not fields:
                    continue
                if len(fields) != 5:
                    raise ValueError(f"{path}:{index}: expected five TSV fields; cannot safely recover the question group")
                department, label, qid, query, answer = fields
                groups.setdefault(qid, []).append((department, label, query, answer))
        for qid, rows in groups.items():
            sid = f"{path.name}:{qid}"
            try:
                text(qid, "question_id")
                if len(rows) != 5:
                    raise ValueError("webMedQA needs five original candidates per question")
                if len({(row[0], row[2]) for row in rows}) != 1:
                    raise ValueError("inconsistent department/question within question_id")
                department, _, query, _ = rows[0]
                text(department, "department")
                text(query, "question")
                for _, label, _, answer in rows:
                    text(answer, "answer")
                    if label not in ("0", "1"):
                        raise ValueError("unknown webMedQA match label")
                if sum(row[1] == "1" for row in rows) != 1:
                    raise ValueError("webMedQA needs one positive and four negative matches")
                if len({row[3].strip() for row in rows}) != 5:
                    raise ValueError("duplicate webMedQA answer candidates need review")
                # Native files put the positive first. Sort without reading labels.
                rows = sorted(rows, key=lambda row: digest([query, row[3]]))
                options = dict(zip("ABCDE", (row[3] for row in rows)))
                winner = "ABCDE"[next(i for i, row in enumerate(rows) if row[1] == "1")]
                group = group_key("medical_text", query)
                linked_answers = (list(options.values()) if answer_links == "all" else
                                  [options[winner]] if answer_links == "positive" else [])
                common = dict(medical_field=department, option_count=5, answer_count=1,
                              links=_medical_links(query, *linked_answers), original_id=sid)
                if mode == "choice":
                    state = f"问题：{query}\n\n候选回答：\n" + "\n\n".join(f"{key}：{answer}" for key, answer in options.items())
                    question = dict(type="choice", instructions="选择与问题匹配的原始回答。", criteria={key: f"回答 {key}" for key in options})
                    yield _case(source, sid, group, state, question, one_hot(options, winner),
                                task_family="medical_answer_matching", semantic_label="matched_answer", **common)
                else:
                    question = dict(type="noul", instructions="该回答是否与问题匹配？",
                                    criteria={"false": "原数据中不匹配", "true": "原数据中匹配"})
                    for key, answer in options.items():
                        label = "true" if key == winner else "false"
                        yield _case(source, f"{sid}:{key}", group, f"问题：{query}\n回答：{answer}", question,
                                    one_hot(["false", "true"], label), task_family="medical_answer_matching",
                                    semantic_label=label, **common)
            except (ValueError, TypeError, KeyError) as exc:
                audit(source, sid, str(exc))


def convert_tcm_sd(source, paths, audit, review):
    auxiliary = source.get("_resolved_auxiliary_paths", {})
    if "labels" not in auxiliary:
        raise ValueError("tcm_sd requires auxiliary_paths.labels: independent full 148-syndrome label->description JSON")
    criteria = json.loads(Path(auxiliary["labels"]).read_text(encoding="utf-8-sig"), object_pairs_hook=_object)
    if not isinstance(criteria, dict) or len(criteria) != 148:
        raise ValueError("TCM-SD needs the full independent 148-syndrome vocabulary")
    for key, value in criteria.items():
        text(key, "syndrome label")
        text(value, "syndrome description")
    if len(set(criteria.values())) != 148:
        raise ValueError("TCM-SD descriptions must be distinct")
    question = dict(type="choice", instructions="依据病历材料，在完整证型集合中判断原病历的规范证型。", criteria=criteria)
    for path, index, row in _records(paths):
        sid = f"{path.name}:{index}"
        try:
            if not isinstance(row, dict):
                raise ValueError("TCM-SD record must be an object")
            label = row["norm_syndrome"]
            if not isinstance(label, str) or label not in criteria:
                raise ValueError("unknown TCM-SD norm_syndrome")
            uid = row["user_id"]
            if type(uid) not in (str, int) or not str(uid).strip():
                raise ValueError("TCM-SD requires original user_id for grouping")
            parts = []
            for key, display in (("chief_complaint", "主诉"), ("description", "病史"), ("detection", "四诊")):
                value = row[key]
                if not isinstance(value, str):
                    raise ValueError(f"{key} must be a string")
                if value.strip():
                    parts.append((display, value))
            if not parts:
                raise ValueError("empty TCM-SD clinical material")
            state = "\n".join(f"{display}：{value}" for display, value in parts)
            group = group_key(source.get("source_family", source["name"]) + ":case", str(uid))
            yield _case(source, sid, group, state, question, one_hot(criteria, label),
                        task_family="syndrome_classification", semantic_label=label,
                        medical_field="traditional_chinese_medicine", option_count=148, answer_count=1,
                        links=_medical_links(state), original_id=sid)
        except (ValueError, TypeError, KeyError) as exc:
            audit(source, sid, str(exc))


ADAPTERS = {
    "cmid_coarse": convert_cmid_coarse, "explain_cpe": convert_explain_cpe,
    "cmb_exam": convert_cmb_exam, "webmedqa": convert_webmedqa,
    "tcm_sd": convert_tcm_sd,
}
