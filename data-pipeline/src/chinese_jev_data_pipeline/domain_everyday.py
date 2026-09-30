"""Verified native financial, news-topic and hotel-review formats (stdlib only).

Converters plug into the shared build engine; no download or teacher labels.
See docs/adapters.md for supported source schemas and task boundaries.
"""
from __future__ import annotations

import csv
import json

from . import core as base

FINFE_LABELS = {0: "negative", 1: "neutral", 2: "positive"}
FINFE_CRITERIA = {"negative": "消极", "neutral": "中性", "positive": "积极"}
# CLUE's native IDs, not a Hugging Face remapping to contiguous 0..14.
# Chinese names agree with the publisher's FewCLUE label_index2en2zh.json.
TNEWS_LABELS = {
    "100": ("news_story", "故事"), "101": ("news_culture", "文化"),
    "102": ("news_entertainment", "娱乐"), "103": ("news_sports", "体育"),
    "104": ("news_finance", "财经"), "106": ("news_house", "房产"),
    "107": ("news_car", "汽车"), "108": ("news_edu", "教育"),
    "109": ("news_tech", "科技"), "110": ("news_military", "军事"),
    "112": ("news_travel", "旅游"), "113": ("news_world", "国际"),
    "114": ("news_stock", "股票"), "115": ("news_agriculture", "农业"),
    "116": ("news_game", "电竞"),
}


def _native_id(path, value):
    if isinstance(value, bool) or not isinstance(value, (str, int)) or not str(value).strip():
        raise ValueError("original record ID must be a nonempty string or integer")
    return f"{path.name}:{value}"


def _metadata(case, source, original_id, label, domain, variant):
    case["_meta"].update(original_id=original_id, original_label=str(label), domain=domain,
                         original_split=source.get("original_split", source["split"]),
                         task_variant=variant)
    return case


def _array(path):
    rows = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(rows, list):
        raise ValueError(f"{path.name}: expected a JSON array of native records")
    return enumerate(rows, 1)


def convert_finfe(source, paths, audit, review):
    """BBT native [text, integer sentiment] rows -> one three-way choice."""
    for path in paths:
        for number, row in _array(path):
            sid = _native_id(path, number)
            try:
                if not isinstance(row, list) or len(row) != 2:
                    raise ValueError("FinFE requires [text, label]; unlabeled problem_list is unsupported")
                content, label = row
                base.text(content, "FinFE text")
                if type(label) is not int or label not in FINFE_LABELS:
                    raise ValueError("FinFE label must be integer 0/1/2 (negative/neutral/positive)")
                q = {"type": "choice", "instructions": "判断这段金融评论的整体情感倾向。",
                     "criteria": dict(FINFE_CRITERIA)}
                case = base.make_case(source, sid, base.group_key("text", content), content, q,
                                      base.one_hot(FINFE_CRITERIA, FINFE_LABELS[label]), "source_hard_label")
                _metadata(case, source, sid, label, "finance", "native_sentiment")
            except (ValueError, KeyError, TypeError) as exc:
                audit(source, sid, str(exc))
                continue
            yield case


def convert_finnsp(source, paths, audit, review):
    """BBT negative-news flag only; gold entity extraction is not model input."""
    header = ["id", "title", "text", "entity", "negative", "key_entity"]
    for path in paths:
        for number, row in _array(path):
            sid = _native_id(path, number)
            if isinstance(row, list) and len(row) == 6 and [str(x).lstrip("\ufeff") for x in row] == header:
                audit(source, sid, "FinNSP embedded header excluded")
                continue
            try:
                if not isinstance(row, list) or len(row) != 6:
                    raise ValueError("FinNSP requires [id,title,text,entity,negative,key_entity]")
                raw_id, title, content, entities, label, _gold_entities = row
                sid = _native_id(path, raw_id)
                base.text(content, "FinNSP text")
                if not isinstance(title, str) or not isinstance(entities, str):
                    raise ValueError("FinNSP title/entity must be strings (empty is allowed)")
                if not isinstance(label, str) or label not in {"0", "1"}:
                    raise ValueError("FinNSP negative flag must be string '0' or '1'")
                state = {"title": title, "text": content, "entities": entities}
                q = {"type": "noul", "instructions": "材料中是否包含与实体相关的负面消息？",
                     "criteria": {"false": "不包含负面消息", "true": "包含负面消息"}}
                # Variants with different entity lists from the same article stay together.
                case = base.make_case(source, sid, base.group_key("text", content), state, q,
                                      base.one_hot(["false", "true"], "true" if label == "1" else "false"),
                                      "source_hard_label")
                _metadata(case, source, sid, label, "finance", "negative_news_flag_only")
            except (ValueError, KeyError, TypeError) as exc:
                audit(source, sid, str(exc))
                continue
            yield case


def _tnews_inventory(source):
    path = source.get("_resolved_auxiliary_paths", {}).get("labels")
    if path is None:
        raise ValueError("TNEWS requires auxiliary_paths.labels pointing to official labels.json (JSONL)")
    observed = {}
    for _, row in base.jsonl(path):
        if not isinstance(row, dict) or type(row.get("label")) not in (str, int):
            raise ValueError("invalid TNEWS label inventory row")
        label = str(row["label"])
        if label in observed:
            raise ValueError("duplicate TNEWS label inventory ID")
        observed[label] = row.get("label_desc")
    if observed != {key: value[0] for key, value in TNEWS_LABELS.items()}:
        raise ValueError("TNEWS inventory must match the 15 official native IDs and label_desc names")


def convert_tnews(source, paths, audit, review):
    """Native 15-way classification and/or taxonomy-specific sports binary task."""
    _tnews_inventory(source)
    mode = source.get("mode", "choice")
    if mode not in {"choice", "sports_noul", "both"}:
        raise ValueError("TNEWS mode must be choice, sports_noul, or both")
    criteria = {key: value[1] for key, value in TNEWS_LABELS.items()}
    for path in paths:
        for number, row in base.jsonl(path):
            sid = _native_id(path, number)
            try:
                if not isinstance(row, dict):
                    raise ValueError("TNEWS records must be JSON objects")
                sid = _native_id(path, row.get("id", number))
                content = base.text(row.get("sentence"), "TNEWS sentence")
                if type(row.get("label")) not in (str, int) or str(row["label"]) not in TNEWS_LABELS:
                    raise ValueError("TNEWS requires a known native label (not unlabeled test or remapped 0..14)")
                label = str(row["label"])
                if "label_desc" in row and row["label_desc"] != TNEWS_LABELS[label][0]:
                    raise ValueError("TNEWS label_desc disagrees with native label")
                group = base.group_key("text", content)
                cases = []
                if mode in {"choice", "both"}:
                    q = {"type": "choice", "instructions": "按照 TNEWS 分类规范判断这则新闻标题的类别。",
                         "criteria": dict(criteria)}
                    case = base.make_case(source, sid + ":choice", group, content, q,
                                          base.one_hot(criteria, label), "source_hard_label")
                    cases.append(_metadata(case, source, sid, label, "news_topics", "native_topic"))
                if mode in {"sports_noul", "both"}:
                    q = {"type": "noul", "instructions": "按照 TNEWS 分类规范，这则标题是否属于 news_sports（体育）类别？",
                         "criteria": {"false": "其他新闻类别", "true": "体育新闻类别"}}
                    case = base.make_case(source, sid + ":sports_noul", group, content, q,
                                          base.one_hot(["false", "true"], "true" if label == "103" else "false"),
                                          "derived_from_source_topic_label")
                    cases.append(_metadata(case, source, sid, label, "sports", "derived_sports_topic"))
            except (ValueError, KeyError, TypeError) as exc:
                audit(source, sid, str(exc))
                continue
            yield from cases


def convert_chnsenticorp_htl(source, paths, audit, review):
    """Publisher's ChnSentiCorp_htl_all.csv: label 1 positive, 0 negative."""
    for path in paths:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if (reader.fieldnames is None or not {"label", "review"} <= set(reader.fieldnames)
                    or len(reader.fieldnames) != len(set(reader.fieldnames))):
                raise ValueError("ChnSentiCorp_htl_all requires CSV columns label,review")
            for number, row in enumerate(reader, 2):
                sid = _native_id(path, number)
                try:
                    if None in row or row.get("label") not in {"0", "1"}:
                        raise ValueError("hotel review requires label 0/1 and a well-formed CSV row")
                    content = base.text(row.get("review"), "hotel review")
                    label = row["label"]
                    q = {"type": "noul", "instructions": "这段酒店评论的整体情感是否为正面？",
                         "criteria": {"false": "负面评论", "true": "正面评论"}}
                    case = base.make_case(source, sid, base.group_key("text", content), content, q,
                                          base.one_hot(["false", "true"], "true" if label == "1" else "false"),
                                          "source_hard_label")
                    _metadata(case, source, sid, label, "lifestyle", "native_hotel_sentiment")
                except (ValueError, KeyError, TypeError) as exc:
                    audit(source, sid, str(exc))
                    continue
                yield case


ADAPTERS = {"finfe": convert_finfe, "finnsp": convert_finnsp, "tnews": convert_tnews,
            "chnsenticorp_htl": convert_chnsenticorp_htl}
