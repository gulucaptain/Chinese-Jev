"""Minimal project adapter; copy this file and replace the raw parsing only."""
from chinese_jev_data_pipeline import core as p


def convert(source, paths, audit, review):
    for path in paths:
        for line, row in p.jsonl(path):
            source_id = row.get("id", f"{path.name}:{line}")
            try:
                qtype = row["type"]
                q = {"type": qtype, "instructions": row["instruction"],
                     "criteria": row["criteria"]}
                keys = p.question_keys(q)
                label = str(row["label"])
                probabilities = p.one_hot(keys, label)
                group = p.group_key("raw_parent", row["group"])
                yield p.make_case(source, source_id, group, row["state"], q,
                                  probabilities, "source_hard_label")
            except (KeyError, TypeError, ValueError) as exc:
                audit(source, source_id, f"invalid raw row: {exc}")


ADAPTERS = {"project_jsonl": convert}
