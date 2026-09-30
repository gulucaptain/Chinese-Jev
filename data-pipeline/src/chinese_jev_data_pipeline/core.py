"""Local datasets -> validated Chinese-Jev typed decisions.

Conversion uses only the Python standard library and never downloads data or calls an LLM.
Tokenization additionally needs transformers and a local tokenizer; PyTorch shards need torch.
All output directories must be new; a failed run never replaces an existing dataset.
"""
from __future__ import annotations

import collections
import contextlib
import csv
import gzip
import hashlib
import importlib.metadata
import io
import json
import math
import os
import sqlite3
import tempfile
import unicodedata
from array import array
from pathlib import Path

from . import __version__

QTYPES = {"choice": 0, "score": 1, "noul": 2}
SPLITS = ("train", "calibration", "validation", "test")
RELEVANCE_LEVELS = [
    "材料与问题不相关。",
    "材料与问题相关，但不能满足问题的信息需求。",
    "材料部分满足问题的信息需求。",
    "材料准确包含问题的答案，充分满足信息需求。",
]


class UnionFind:
    """Compact transitive grouping for adapter-supplied lineage anchors."""

    def __init__(self):
        self.parent = array("I")
        self.size = array("I")

    def add(self):
        index = len(self.parent)
        self.parent.append(index)
        self.size.append(1)
        return index

    def find(self, index):
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, left, right):
        left, right = self.find(left), self.find(right)
        if left == right:
            return
        if self.size[left] < self.size[right]:
            left, right = right, left
        self.parent[right] = left
        self.size[left] += self.size[right]


def dumps(obj):
    # Preserve insertion order: choice criteria order is the target index order.
    return json.dumps(obj, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def digest(obj):
    return hashlib.sha256(dumps(obj).encode("utf-8")).hexdigest()


def text(value, name="text"):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value  # Do not modify span offsets, punctuation, or Traditional Chinese.


def group_key(kind, value):
    normalized = " ".join(unicodedata.normalize("NFC", text(value)).split())
    return kind + ":" + digest(normalized)


def fraction(key, seed):
    return int(digest([str(seed), key])[:16], 16) / 2**64


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def implementation_hashes(paths):
    """Hash implementation files without publishing host-specific paths.

    A digest suffix disambiguates plugins that happen to share a filename.
    """
    entries = [(Path(path).name, file_sha256(path)) for path in paths]
    frequencies = collections.Counter(name for name, _ in entries)
    result = {}
    for name, checksum in entries:
        key = name if frequencies[name] == 1 else f"{name}:{checksum[:12]}"
        result[key] = checksum
    return dict(sorted(result.items()))


def jsonl(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8-sig") as f:
        for number, line in enumerate(f, 1):
            if line.strip():
                try:
                    yield number, json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{number}: invalid JSON") from exc


def emit(f, row):
    f.write(dumps(row) + "\n")


def one_hot(keys, selected):
    if selected not in keys:
        raise ValueError(f"label {selected!r} not in {keys!r}")
    return {k: float(k == selected) for k in keys}


def question_keys(q):
    t, crit = q["type"], q.get("criteria")
    text(q["instructions"], "instructions")
    if "labels" in q:
        raise ValueError("custom labels are not supported by the Chinese-Jev decision schema")
    if t == "choice":
        if not isinstance(crit, dict) or len(crit) < 2:
            raise ValueError("choice needs at least two criteria in a dict")
        keys = list(crit)
        for k in keys:
            text(k, "choice key")
        descriptions = [dumps(v) for v in crit.values() if v not in (None, "")]
        if len(descriptions) != len(set(descriptions)):
            raise ValueError("duplicate choice descriptions need review")
        return keys
    if t == "score":
        if not isinstance(crit, list) or len(crit) < 2:
            raise ValueError("score needs an ordered list with at least two levels")
        if any(not isinstance(v, str) or not v.strip() for v in crit):
            raise ValueError("score level descriptions must be nonempty strings")
        if len(set(crit)) != len(crit):
            raise ValueError("score level descriptions must differ")
        return [str(i) for i in range(len(crit))]
    if t == "noul":
        if not isinstance(crit, dict) or set(crit) != {"false", "true"}:
            raise ValueError("noul criteria must explicitly describe false and true")
        return ["false", "true"]
    raise ValueError(f"unsupported type: {t!r}")


def validate_case(case):
    if not isinstance(case["state"], (str, dict, list)):
        raise ValueError("state must be str, dict or list")
    if not case["state"]:
        raise ValueError("state must not be empty")
    qs, gold = case["questions"], case["gold"]
    if not isinstance(qs, dict) or not qs or set(qs) != set(gold):
        raise ValueError("questions and gold must have the same nonempty question-id set")
    for qid, q in qs.items():
        keys = question_keys(q)
        g = gold[qid]
        if g.get("type", q["type"]) != q["type"]:
            raise ValueError("gold type differs from question type")
        probs = g["probabilities"]
        if not isinstance(probs, dict) or set(probs) != set(keys):
            raise ValueError("probabilities must contain exactly all candidate keys")
        vals = [probs[k] for k in keys]
        if any(isinstance(v, bool) or not isinstance(v, (int, float))
               or not math.isfinite(v) or v < 0 or v > 1 for v in vals):
            raise ValueError("probabilities must be finite numbers between 0 and 1")
        total = sum(vals)
        if not math.isclose(total, 1.0, abs_tol=1e-4, rel_tol=0):
            raise ValueError(f"probability sum {total} differs from 1")
        g["probabilities"] = {k: float(probs[k] / total) for k in keys}
    return case


def make_case(source, source_id, group, state, q, probabilities, supervision, *, original_id=None):
    return validate_case({
        "id": source["name"] + ":" + digest([str(source_id), state, q])[:24],
        "state": state,
        "questions": {"decision": q},
        "gold": {"decision": {"type": q["type"], "probabilities": probabilities}},
        "_meta": {
            "source": source["name"], "source_id": str(source_id),
            "original_id": str(source_id if original_id is None else original_id),
            "original_split": source.get("original_split", source["split"]),
            "source_split": source["split"], "group_key": group,
            "source_url": source["url"], "revision": source["revision"],
            "license": source["license"], "supervision": supervision,
        },
    })


def choice_case(source, source_id, group, context, question, options, answer, *, original_id=None):
    if not isinstance(options, list) or len(options) < 2 or len(set(options)) != len(options):
        raise ValueError("choice options must be distinct and contain at least two entries")
    if type(answer) is not int or not 0 <= answer < len(options):
        raise ValueError("answer index out of range")
    for opt in options:
        text(opt, "option")
    keys = [chr(ord("A") + i) if i < 26 else f"option_{i}" for i in range(len(options))]
    q = {"type": "choice", "instructions": "根据材料选择问题的正确答案。\n" + text(question),
         "criteria": dict(zip(keys, options))}
    return make_case(source, source_id, group, context, q,
                     one_hot(keys, keys[answer]), "source_hard_label", original_id=original_id)


def convert_c3(source, paths, audit, review):
    for path in paths:
        docs = json.loads(path.read_text(encoding="utf-8-sig"))
        for doc in docs:
            if not isinstance(doc, list) or len(doc) != 3:
                raise ValueError(f"{path}: expected original C3 [document, questions, id]")
            parts, questions, docid = doc
            context = "\n".join(text(p, "document part") for p in parts)
            group = group_key("context", context)
            for i, q in enumerate(questions):
                source_id = f"{path.name}:{docid}:{i}"
                try:
                    opts, answer = q["choice"], q.get("answer")
                    if not isinstance(answer, str) or opts.count(answer) != 1:
                        raise ValueError("answer must match exactly one option; missing/ambiguous label")
                    yield choice_case(source, source_id, group, context,
                                      q["question"], opts, opts.index(answer), original_id=f"{path.name}:{docid}")
                except (ValueError, KeyError, TypeError) as exc:
                    audit(source, source_id, str(exc))


def convert_exam_csv(source, paths, audit, review):
    # CMMLU: Question/A/B/C/D/Answer; C-Eval: question/A/B/C/D/answer.
    # A split is always supplied explicitly; there is no automatic train/test reassignment.
    for path in paths:
        with path.open(encoding="utf-8-sig", newline="") as f:
            for i, row in enumerate(csv.DictReader(f), 2):
                sid = f"{path.name}:{i}"
                try:
                    question = text(row.get("Question", row.get("question")), "question")
                    answer = row.get("Answer", row.get("answer", ""))
                    if answer not in ("A", "B", "C", "D"):
                        raise ValueError("missing or non-single-choice answer")
                    yield choice_case(source, sid, group_key("question", question),
                                      "请依据题目作答。", question, [row[k] for k in "ABCD"],
                                      "ABCD".index(answer))
                except (ValueError, KeyError, TypeError) as exc:
                    audit(source, sid, str(exc))


def convert_yesno(source, paths, audit, review):
    mode = source.get("mode", "noul")
    if mode not in ("noul", "choice", "both"):
        raise ValueError("dureader_yesno mode must be noul, choice or both")
    for path in paths:
        for line, row in jsonl(path):
            sid = f"{path.name}:{row.get('id', line)}"
            try:
                label = row.get("yesno_answer")
                if label not in ("Yes", "No", "Depends"):
                    raise ValueError("missing/unknown yesno_answer; unlabeled test data cannot train")
                question, answer = text(row["question"]), text(row["answer"])
                state = "问题：" + question + "\n待判定的回答：" + answer
                if source.get("include_documents", False):
                    state += "\n参考材料：" + dumps(row["documents"])
                group = group_key("question", question)
                if mode in ("choice", "both"):
                    crit = {"Yes": "回答持明确肯定态度。", "No": "回答持明确否定态度。",
                            "Depends": "回答表示分情况或无法确定。"}
                    q = {"type": "choice", "instructions": "判断给定回答对于问题的观点极性。",
                         "criteria": crit}
                    yield make_case(source, sid + ":choice", group, state, q,
                                    one_hot(list(crit), label), "source_hard_label", original_id=sid)
                if mode in ("noul", "both"):
                    if label == "Depends":
                        audit(source, sid + ":noul", "Depends excluded from binary Yes/No subset")
                    else:
                        q = {"type": "noul", "instructions": "给定回答是否对问题持肯定态度？",
                             "criteria": {"false": "回答持否定态度。", "true": "回答持肯定态度。"}}
                        yield make_case(source, sid + ":noul", group, state, q,
                                        one_hot(["false", "true"], "true" if label == "Yes" else "false"),
                                        "source_hard_label_binary_subset", original_id=sid)
            except (ValueError, KeyError, TypeError) as exc:
                audit(source, sid, str(exc))


def convert_squad(source, paths, audit, review):
    mode = source.get("mode", "review")
    if mode not in ("review", "answerability"):
        raise ValueError("squad mode must be review or answerability")
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        for article in data["data"]:
            for para in article["paragraphs"]:
                context = text(para["context"])
                group = group_key("context", context)
                for item in para["qas"]:
                    sid = str(item["id"])
                    try:
                        question = text(item["question"])
                        answers = item.get("answers", [])
                        for a in answers:
                            start, value = a["answer_start"], text(a["text"])
                            if not isinstance(start, int) or start < 0 or context[start:start + len(value)] != value:
                                raise ValueError("reference answer offset does not match context")
                        if mode == "review":
                            emit(review, {"source": source["name"], "source_id": sid,
                                          "source_split": source["split"], "group_key": group,
                                          "context": context, "question": question,
                                          "reference_answers": answers,
                                          "status": "needs_candidates_and_reviewed_labels"})
                            continue
                        impossible = item.get("is_impossible")
                        if type(impossible) is not bool:
                            raise ValueError("answerability requires an explicit boolean is_impossible label")
                        if impossible and answers or not impossible and not answers:
                            raise ValueError("is_impossible and answers disagree")
                        q = {"type": "noul", "instructions": "材料是否足以回答下列问题？\n" + question,
                             "criteria": {"false": "材料中没有足够信息。", "true": "材料中有足够信息。"}}
                        yield make_case(source, sid, group, context, q,
                                        one_hot(["false", "true"], "false" if impossible else "true"),
                                        "source_answerability_label", original_id=f"{path.name}:{sid}")
                    except (ValueError, KeyError, TypeError) as exc:
                        audit(source, sid, str(exc))


def convert_chinese_jev(source, paths, audit, review):
    for path in paths:
        for line, raw in jsonl(path):
            sid = str(raw.get("id", f"{path.name}:{line}"))
            try:
                row = dict(raw)
                for field in ("state", "questions", "gold"):
                    # Native cases have a plain string state; notebook rows encode all three as JSON.
                    if field != "state" or isinstance(raw.get("questions"), str):
                        if isinstance(row[field], str):
                            row[field] = json.loads(row[field])
                row = validate_case(row)
                group = row.get("_meta", {}).get("group_key")
                if not group:
                    group = group_key("state", dumps(row["state"]))
                text(group, "group_key")
                for qid, q in row["questions"].items():
                    yield make_case(source, sid + ":" + qid, group, row["state"], q,
                                    row["gold"][qid]["probabilities"],
                                    source.get("supervision", "provided_probabilities"),
                                    original_id=f"{path.name}:{sid}")
            except (ValueError, KeyError, TypeError) as exc:
                audit(source, sid, str(exc))


def convert_t2(source, base, work):
    """Join only human graded qrels with query/passages; never label unjudged pairs as zero."""
    keep = source.get("query_fraction", 1.0)
    if not isinstance(keep, (int, float)) or not 0 < keep <= 1:
        raise ValueError("query_fraction must be in (0, 1]")
    with sqlite3.connect(work / "t2_join.sqlite") as db:
        db.executescript("DROP TABLE IF EXISTS queries; DROP TABLE IF EXISTS pairs; DROP TABLE IF EXISTS passages;"
                         "CREATE TABLE queries(qid TEXT PRIMARY KEY, query TEXT, grp TEXT);"
                         "CREATE TABLE pairs(qid TEXT, pid TEXT, grade INTEGER, PRIMARY KEY(qid,pid));"
                         "CREATE INDEX by_pid ON pairs(pid);"
                         "CREATE TABLE passages(pid TEXT PRIMARY KEY, passage TEXT);")
        with (base / source["queries"]).open(encoding="utf-8-sig") as f:
            for n, line in enumerate(f, 1):
                qid, query = line.rstrip("\r\n").split("\t", 1)
                if n == 1 and (qid, query) == ("qid", "text"):
                    continue
                text(qid, "qid")
                text(query, "query")
                db.execute("INSERT INTO queries VALUES(?,?,?)", (qid, query, group_key("question", query)))
        with (base / source["qrels"]).open(encoding="utf-8-sig") as f:
            for n, line in enumerate(f, 1):
                fields = line.split()
                if n == 1 and fields == ["qid", "-", "pid", "rel"]:
                    continue
                if len(fields) != 4:
                    raise ValueError(f"qrels line {n}: expected qid, iteration, pid, human grade (4 columns)")
                qid, _, pid, value = fields
                if value not in ("0", "1", "2", "3"):
                    raise ValueError(f"qrels line {n}: human grade must be 0..3")
                found = db.execute("SELECT grp FROM queries WHERE qid=?", (qid,)).fetchone()
                if found is None:
                    raise ValueError(f"qrels has unknown query {qid!r}")
                if fraction(found[0], source.get("sample_seed", 42)) >= keep:
                    continue
                previous = db.execute("SELECT grade FROM pairs WHERE qid=? AND pid=?", (qid, pid)).fetchone()
                if previous and previous[0] != int(value):
                    raise ValueError(f"conflicting qrels grades for {qid}/{pid}")
                db.execute("INSERT OR IGNORE INTO pairs VALUES(?,?,?)", (qid, pid, int(value)))
        with (base / source["collection"]).open(encoding="utf-8-sig") as f:
            for n, line in enumerate(f, 1):
                pid, passage = line.rstrip("\r\n").split("\t", 1)
                if n == 1 and (pid, passage) == ("pid", "text"):
                    continue
                if db.execute("SELECT 1 FROM pairs WHERE pid=? LIMIT 1", (pid,)).fetchone():
                    db.execute("INSERT INTO passages VALUES(?,?)", (pid, text(passage, "passage")))
        missing = db.execute("SELECT pairs.pid FROM pairs LEFT JOIN passages USING(pid) "
                             "WHERE passages.pid IS NULL LIMIT 1").fetchone()
        if missing:
            raise ValueError(f"collection missing labeled passage {missing[0]}")
        rows = db.execute("SELECT pairs.qid,pairs.pid,grade,query,grp,passage FROM pairs "
                          "JOIN queries USING(qid) JOIN passages USING(pid) ORDER BY pairs.qid,pairs.pid")
        for qid, pid, grade, query, group, passage in rows:
            q = {"type": "score", "instructions": "评价材料满足问题信息需求的程度。\n问题：" + query,
                 "criteria": RELEVANCE_LEVELS.copy()}
            yield make_case(source, f"{qid}:{pid}", group, passage, q,
                            one_hot(["0", "1", "2", "3"], str(grade)), "source_human_ordinal_label")
    (work / "t2_join.sqlite").unlink()


@contextlib.contextmanager
def new_output(path):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Output already exists: {path}. Choose a new directory.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="." + path.name + "-", dir=path.parent) as tmp:
        work = Path(tmp) / "result"
        work.mkdir()
        yield work
        if path.exists():
            raise FileExistsError(path)
        os.rename(work, path)


def build(config_path, output, *, extra_adapters=None, adapter_files=()):
    """Build common cases; optional converters share splitting and auditing.

    Auxiliary inputs (label inventories, etc.) are resolved beside the config
    and hashed exactly like raw data. Their resolved paths are converter-only.
    """
    config_path = Path(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    base = config_path.parent
    seed = config.get("seed", 42)
    compression = config.get("compression", "none")
    if compression not in ("none", "gzip"):
        raise ValueError("compression must be 'none' or 'gzip'")
    write_notebooks = config.get("write_notebooks", True)
    if type(write_notebooks) is not bool:
        raise ValueError("write_notebooks must be boolean")
    calib = config.get("calibration_fraction", 0.0)
    val = config.get("validation_fraction", 0.0)
    test = config.get("test_fraction", 0.0)
    if not (0 <= calib < 1 and 0 <= val < 1 and 0 <= test < 1 and calib + val + test < 1):
        raise ValueError("holdout fractions must be nonnegative and sum to less than 1")
    if any(s.get("prepartitioned") is True for s in config["sources"]) and (calib or val or test):
        raise ValueError("prepartitioned sources require calibration/validation/test fractions of zero; "
                         "do not randomly split document-partitioned pairs again")
    adapters = {"c3": convert_c3, "exam_csv": convert_exam_csv, "dureader_yesno": convert_yesno,
                "squad": convert_squad, "chinese_jev": convert_chinese_jev}
    if extra_adapters:
        if (set(adapters) | {"t2ranking"}) & set(extra_adapters):
            raise ValueError("extra adapters must not replace an existing converter")
        adapters.update(extra_adapters)
    names, inputs, counts = set(), [], collections.Counter()
    links, group_nodes, union = {}, {}, UnionFind()

    def connect_group(case):
        meta = case["_meta"]
        group = text(meta["group_key"], "group_key")
        integration = meta.get("integration") or {}
        if not isinstance(integration, dict):
            raise ValueError("_meta.integration must be an object")
        extra = integration.get("link_keys", meta.get("link_keys", []))
        if not isinstance(extra, list) or any(not isinstance(v, str) or not v.strip() for v in extra):
            raise ValueError("link_keys must be a list of nonempty strings")
        node = group_nodes.get(group)
        if node is None:
            node = union.add()
            group_nodes[group] = node
        for anchor in ["group:" + group, *("link:" + value for value in extra)]:
            previous = links.get(anchor)
            if previous is None:
                links[anchor] = node
            else:
                union.union(node, previous)
    with new_output(output) as work, contextlib.ExitStack() as stack:
        audit_file = stack.enter_context((work / "rejected.jsonl").open("w", encoding="utf-8"))
        review = stack.enter_context((work / "review.jsonl").open("w", encoding="utf-8"))
        db = sqlite3.connect(work / "staging.sqlite")
        stack.callback(db.close)
        db.executescript("CREATE TABLE groups(grp TEXT PRIMARY KEY, priority INTEGER);"
                         "CREATE TABLE cases(input_key TEXT PRIMARY KEY, gold_key TEXT, grp TEXT, "
                         "priority INTEGER, payload TEXT, conflict INTEGER DEFAULT 0);"
                         "CREATE TABLE units(source TEXT, original_id TEXT, grp TEXT, split TEXT, stage TEXT, "
                         "PRIMARY KEY(source,original_id,grp,split,stage));")

        def record_unit(case, split, stage):
            meta = case["_meta"]
            original_id = str(meta.get("original_id", meta["source_id"]))
            db.execute("INSERT OR IGNORE INTO units VALUES(?,?,?,?,?)",
                       (meta["source"], original_id, meta["group_key"], split, stage))
            counts[f"decisions:{stage}:{split}:{meta['source']}"] += len(case["questions"])

        def audit(source, sid, reason):
            counts["rejected:" + source["name"] + ":" + reason] += 1
            emit(audit_file, {"source": source["name"], "source_id": str(sid), "reason": reason})

        for source in config["sources"]:
            for key in ("name", "adapter", "split", "url", "revision", "license"):
                text(source[key], key)
            if source["name"] in names or source["split"] not in SPLITS:
                raise ValueError("source names must be unique; split must be train/calibration/validation/test")
            names.add(source["name"])
            paths = [base / p for p in source.get("paths", [])]
            if len({p.name for p in paths}) != len(paths):
                raise ValueError("paths within one source must have unique filenames to keep original IDs unambiguous; "
                                 "use separate named sources for same-named shards")
            auxiliary = source.get("auxiliary_paths", {})
            if not isinstance(auxiliary, dict) or any(not isinstance(k, str) or not isinstance(v, str)
                                                     or not k or not v for k, v in auxiliary.items()):
                raise ValueError("auxiliary_paths must map names to local file paths")
            resolved_auxiliary = {k: base / v for k, v in auxiliary.items()}
            converter_source = dict(source, _resolved_auxiliary_paths=resolved_auxiliary)
            if source["adapter"] == "t2ranking":
                files = [base / source[k] for k in ("queries", "qrels", "collection")]
                rows = convert_t2(source, base, work)
            elif source["adapter"] in adapters:
                if not paths:
                    raise ValueError("paths must not be empty")
                files, rows = paths, adapters[source["adapter"]](converter_source, paths, audit, review)
            else:
                raise ValueError(f"unsupported adapter {source['adapter']!r}")
            files = list(dict.fromkeys([*files, *resolved_auxiliary.values()]))
            inputs.append({"source": source, "files": [
                {"path": os.path.relpath(p, base), "bytes": p.stat().st_size, "sha256": file_sha256(p)}
                for p in files]})
            priority = SPLITS.index(source["split"])
            for case in rows:
                validate_case(case)
                if len(case["questions"]) != 1:
                    raise ValueError("adapters must emit exactly one decision per case")
                counts["converted:" + source["name"]] += 1
                # Portable releases identify semantic families independently of
                # adapter names.  Older configs need not provide these fields;
                # the unified chinese_jev_data_pipeline entry point requires them.
                for key in ("source_family", "task_family", "domain"):
                    if key not in source:
                        continue
                    existing = case["_meta"].get(key)
                    if existing is not None and existing != source[key]:
                        raise ValueError(f"adapter metadata {key}={existing!r} conflicts with "
                                         f"source config {source[key]!r}")
                    case["_meta"][key] = source[key]
                case["_meta"].setdefault("original_split", source.get("original_split", source["split"]))
                connect_group(case)
                record_unit(case, source["split"], "converted")
                group = case["_meta"]["group_key"]
                db.execute("INSERT INTO groups VALUES(?,?) ON CONFLICT(grp) DO UPDATE SET "
                           "priority=MAX(priority,excluded.priority)", (group, priority))
                ikey, gkey = digest([case["state"], case["questions"]]), digest(case["gold"])
                previous = db.execute("SELECT gold_key,priority FROM cases WHERE input_key=?", (ikey,)).fetchone()
                if previous:
                    if previous[0] != gkey:
                        db.execute("UPDATE cases SET conflict=1 WHERE input_key=?", (ikey,))
                        audit(source, case["id"], "identical input has conflicting gold; all copies excluded")
                    elif priority > previous[1]:
                        db.execute("UPDATE cases SET grp=?,priority=?,payload=? WHERE input_key=?",
                                   (group, priority, dumps(case), ikey))
                        counts["duplicate_inputs"] += 1
                    else:
                        counts["duplicate_inputs"] += 1
                    continue
                db.execute("INSERT INTO cases VALUES(?,?,?,?,?,0)", (ikey, gkey, group, priority, dumps(case)))
        db.commit()
        if not group_nodes:
            raise ValueError("no valid cases were produced")
        component_members = collections.defaultdict(list)
        for group, node in group_nodes.items():
            component_members[union.find(node)].append(group)
        component_ids = {root: "component:" + digest(sorted(members))
                         for root, members in component_members.items()}
        priorities = {group: priority for group, priority in db.execute("SELECT grp,priority FROM groups")}
        component_priority = collections.defaultdict(int)
        for group, node in group_nodes.items():
            root = union.find(node)
            component_priority[root] = max(component_priority[root], priorities[group])
        db.execute("CREATE TABLE group_components(grp TEXT PRIMARY KEY, component TEXT, priority INTEGER)")
        db.executemany("INSERT INTO group_components VALUES(?,?,?)", (
            (group, component_ids[union.find(node)], component_priority[union.find(node)])
            for group, node in group_nodes.items()
        ))
        counts["connected_components"] = len(component_members)
        counts["declared_link_keys"] = sum(key.startswith("link:") for key in links)
        db.commit()

        @contextlib.contextmanager
        def output_stream(stem):
            if compression == "gzip":
                with (work / (stem + ".gz")).open("wb") as raw:
                    with gzip.GzipFile(filename="", mode="wb", fileobj=raw,
                                       compresslevel=6, mtime=0) as compressed:
                        with io.TextIOWrapper(compressed, encoding="utf-8") as stream:
                            yield stream
            else:
                with (work / stem).open("w", encoding="utf-8") as stream:
                    yield stream

        outputs = {s: stack.enter_context(output_stream(f"{s}.cases.jsonl")) for s in SPLITS}
        notebooks = ({s: stack.enter_context(output_stream(f"{s}.notebook.jsonl")) for s in SPLITS}
                     if write_notebooks else {})
        query = ("SELECT payload,cases.priority,group_components.priority,conflict,component "
                 "FROM cases JOIN group_components USING(grp) ORDER BY input_key")
        for payload, original, highest, conflict, component in db.execute(query):
            if conflict:
                counts["conflicting_inputs_excluded"] += 1
                continue
            if original < highest:
                counts["cross_split_group_rows_excluded"] += 1
                continue  # Keep the official held-out rows; do not move train variants into test.
            case = json.loads(payload)
            split = SPLITS[original]
            if split == "train":
                u = fraction(component, seed)
                if u < calib:
                    split = "calibration"
                elif u < calib + val:
                    split = "validation"
                elif u < calib + val + test:
                    split = "test"
                else:
                    split = "train"
            source_group = case["_meta"]["group_key"]
            if source_group != component:
                case["_meta"]["source_group_key"] = source_group
            case["_meta"]["group_key"] = component
            case["_meta"]["split"] = split
            record_unit(case, split, "output")
            emit(outputs[split], case)
            if write_notebooks:
                emit(notebooks[split], {"id": case["id"], "state": dumps(case["state"]),
                                        "questions": dumps(case["questions"]), "gold": dumps(case["gold"])})
            for qid, q in case["questions"].items():
                probs = case["gold"][qid]["probabilities"]
                label = max(probs, key=probs.get)
                counts[f"output:{split}:{q['type']}"] += 1
                counts[f"labels:{split}:{case['_meta']['source']}:{q['type']}:{label}"] += 1
        review.flush()
        with (work / "review.jsonl").open(encoding="utf-8") as f:
            counts["review_rows"] = sum(1 for line in f if line.strip())
        manifest = {"format_version": 1, "pipeline_sha256": file_sha256(__file__),
                    "config": config, "inputs": inputs, "counts": dict(sorted(counts.items())),
                    "split_policy": "transitive linked-component hash; official test > validation > calibration > train; exclude lower-priority overlaps",
                    "dedup_policy": "exact state+ordered questions; exclude contradictory targets",
                    "output": {"compression": compression, "write_notebooks": write_notebooks},
                    "note": "No tokenizer, model training, generated negatives or action labels in build."}
        manifest["adapter_sha256"] = implementation_hashes(adapter_files)
        manifest["source_units"] = [dict(source=s, stage=stage, split=split,
                                        original_records=records, groups=groups,
                                        decisions=counts[f"decisions:{stage}:{split}:{s}"])
                                    for s, stage, split, records, groups in db.execute(
                                        "SELECT source,stage,split,COUNT(DISTINCT original_id),COUNT(DISTINCT grp) "
                                        "FROM units GROUP BY source,stage,split ORDER BY source,stage,split")]
        (work / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        stack.close()
        (work / "staging.sqlite").unlink()
    return manifest


def _criterion_text(value):
    """Use stable, readable JSON for structured candidate descriptions."""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(", ", ": "), default=str)


def _option_texts(question):
    """Render candidates in the same order as their probability targets."""
    kind, criteria = question["t"], question["crit"]
    if kind == "choice":
        return [key if value is None or value == "" else f"{key}: {_criterion_text(value)}"
                for key, value in criteria.items()]
    if kind == "score":
        return [f"level {index}: {_criterion_text(value)}" for index, value in enumerate(criteria)]
    if kind == "noul":
        defaults = {"false": "no, the statement does not hold", "true": "yes, the statement holds"}
        return [f"{key}: " + (defaults[key] if criteria.get(key) in (None, "")
                              else _criterion_text(criteria[key])) for key in ("false", "true")]
    raise ValueError(f"unsupported type: {kind!r}")


def checked_sequence(tok, state, internal, max_len, head_max_len, state_truncation="reject"):
    """Compile the Chinese-Jev BERT layout, rejecting incomplete heads.

    Layout: [CLS] type + instruction [SEP] [MASK] candidate ... [SEP] state [SEP].
    The state and candidate text are formatted without consulting gold labels.
    """
    if state_truncation not in ("reject", "right"):
        raise ValueError("state_truncation must be reject or right")
    mask = tok.mask_token
    def encode(value):
        return tok(value, add_special_tokens=False)["input_ids"]
    head = encode(f"{internal['t']} question: {internal['ins'].replace(mask, ' ')}")
    options = [[tok.mask_token_id] + encode(" " + o.replace(mask, " ")) for o in _option_texts(internal)]
    if not options:
        raise ValueError("no_candidate_options")
    clipped = [o[:49] for o in options]  # 1 marker + at most 48 description tokens.
    budget = head_max_len - sum(map(len, clipped))
    if budget < 16:
        per = max(4, (head_max_len - 16) // max(1, len(clipped)))
        clipped = [o[:per] for o in clipped]
        budget = head_max_len - sum(map(len, clipped))
    kept_head = head[:max(8, budget)]
    if clipped != options:
        raise ValueError("option_text_truncated")
    if kept_head != head:
        raise ValueError("question_instruction_truncated")
    prefix = [tok.cls_token_id] + kept_head + [tok.sep_token_id]
    markers = []
    for opt in clipped:
        markers.append(len(prefix))
        prefix.extend(opt)
    prefix.append(tok.sep_token_id)
    state_text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
    body = encode(state_text.replace(mask, " "))
    room = max(0, max_len - len(prefix) - 1)
    truncated = len(body) > room
    if truncated and state_truncation == "reject":
        raise ValueError("state_truncated")
    if not body or room == 0:
        raise ValueError("empty_state_tokens")
    ids = prefix + body[:room] + [tok.sep_token_id]
    if len(markers) != len(options) or any(ids[m] != tok.mask_token_id for m in markers):
        raise ValueError("candidate_marker_missing")
    return ids, markers, {"state_tokens": len(body), "state_tokens_kept": min(len(body), room),
                          "state_truncated": truncated}


def sequence_item(tok, case, qid, max_len, head_max_len, state_truncation="reject"):
    q = case["questions"][qid]
    internal = {"t": q["type"], "ins": q["instructions"], "crit": q["criteria"]}
    ids, markers, diagnostics = checked_sequence(
        tok, case["state"], internal, max_len, head_max_len, state_truncation)
    keys = question_keys(q)
    probs = case["gold"][qid]["probabilities"]
    target = [probs[k] for k in keys]
    return {"ids": ids, "markers": markers, "qtype": QTYPES[q["type"]], "target": target,
            "label": max(range(len(target)), key=target.__getitem__),
            "_meta": {**case["_meta"], "case_id": case["id"], "question_id": qid,
                      "candidate_keys": keys, **diagnostics}}


def collate_items(items, pad_id):
    """Pad encoder inputs, candidate markers, targets and decision types for a batch."""
    import torch
    if not items:
        raise ValueError("empty batch")
    b, length, k = len(items), max(len(x["ids"]) for x in items), max(len(x["markers"]) for x in items)
    result = {"input_ids": torch.full((b, length), pad_id, dtype=torch.long),
              "attention_mask": torch.zeros((b, length), dtype=torch.long),
              "marker_pos": torch.zeros((b, k), dtype=torch.long),
              "marker_mask": torch.zeros((b, k), dtype=torch.bool),
              "target": torch.zeros((b, k), dtype=torch.float32),
              "qtype": torch.tensor([x["qtype"] for x in items], dtype=torch.long),
              "label": torch.tensor([x["label"] for x in items], dtype=torch.long)}
    for i, item in enumerate(items):
        n, m = len(item["ids"]), len(item["markers"])
        if m != len(item["target"]):
            raise ValueError("target/marker count mismatch")
        result["input_ids"][i, :n] = torch.tensor(item["ids"])
        result["attention_mask"][i, :n] = 1
        result["marker_pos"][i, :m] = torch.tensor(item["markers"])
        result["marker_mask"][i, :m] = True
        result["target"][i, :m] = torch.tensor(item["target"])
    return result


def prepare(input_path, tokenizer_path, output, max_len, head_max_len,
            state_truncation="reject", pt_shard_size=0):
    from transformers import AutoTokenizer
    if max_len < 16 or head_max_len < 16 or head_max_len >= max_len or pt_shard_size < 0:
        raise ValueError("need 16 <= head_max_len < max_len and pt_shard_size >= 0")
    tok = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True, trust_remote_code=False)
    tok.padding_side = "right"
    if any(getattr(tok, k + "_token_id") is None for k in ("cls", "mask", "sep", "pad")):
        raise ValueError("tokenizer needs cls/mask/sep/pad tokens")
    if len({tok.cls_token_id, tok.mask_token_id, tok.sep_token_id, tok.pad_token_id}) != 4:
        raise ValueError("cls/mask/sep/pad must use distinct IDs")
    if max_len > tok.model_max_length:
        raise ValueError("max_len exceeds tokenizer.model_max_length; check the encoder's real limit")
    if pt_shard_size:
        import torch
    counts, shard, shard_index = collections.Counter(), [], 0
    with new_output(output) as work:
        with (work / "items.jsonl").open("w", encoding="utf-8") as out, \
             (work / "rejected.jsonl").open("w", encoding="utf-8") as rejected:
            for _, case in jsonl(input_path):
                validate_case(case)
                for qid in case["questions"]:
                    try:
                        item = sequence_item(tok, case, qid, max_len, head_max_len, state_truncation)
                    except ValueError as exc:
                        counts["rejected:" + str(exc)] += 1
                        emit(rejected, {"case_id": case["id"], "question_id": qid, "reason": str(exc)})
                        continue
                    emit(out, item)
                    counts["accepted:" + str(item["qtype"])] += 1
                    counts["state_truncated"] += int(item["_meta"]["state_truncated"])
                    counts["length_bucket:" + str((len(item["ids"]) - 1) // 128 * 128 + 128)] += 1
                    if pt_shard_size:
                        shard.append(item)
                        if len(shard) == pt_shard_size:
                            torch.save(shard, work / f"items-{shard_index:05d}.pt")
                            shard, shard_index = [], shard_index + 1
        if shard:
            torch.save(shard, work / f"items-{shard_index:05d}.pt")
        tok.save_pretrained(work / "tokenizer")
        report = {"input_sha256": file_sha256(input_path), "pipeline_sha256": file_sha256(__file__),
                  "encoder_kind": "bert", "max_len": max_len, "head_max_len": head_max_len,
                  "state_truncation": state_truncation, "padding_side": "right", "counts": dict(counts),
                  "tokenizer_files": {p.name: file_sha256(p) for p in sorted((work / "tokenizer").iterdir()) if p.is_file()},
                  "special_tokens": {k: {"token": getattr(tok, k + "_token"), "id": getattr(tok, k + "_token_id")}
                                     for k in ("cls", "mask", "sep", "pad")},
                  "versions": {"chinese-jev-data-pipeline": __version__,
                               "transformers": importlib.metadata.version("transformers")}}
        (work / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
