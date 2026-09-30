"""Turn canonical cases into cache-ready token items.

Two things here are worth more than the code they sit in.

**The head budget is measured, not assumed.** `build_sequence` falls back to a fixed
per-option budget when the prompt does not fit `head_max_len`, and that fallback makes
options indistinguishable instead of failing. A wrong `head_max_len` therefore trains a
model that cannot tell the choices apart, silently. `measure_head_budget` walks the data
until it has seen the widest option set and returns the width that set actually needs, so
the run is sized from evidence and `tokenize` refuses rather than degrades.

**Items are built once, at full width.** `encode_case` builds each item at an effectively
unbounded `max_len`, and the cache stores the prompt and the state separately; every
narrower `max_len` is then a read-time prefix (see `cache.py`).
"""
from __future__ import annotations

import numpy as np

from . import adapters as adapters_mod
from .cache import CacheError
from .sequence import QTYPES, build_sequence, normalize_question, render_options, serialize_state

# A state is stored in full so any max_len can be reconstructed later. The cap is only
# there to bound the cache on a corpus with a pathological outlier, and every item that
# hits it is reported, because it caps what `max_len` can ever be raised to.
STATE_TOKEN_CAP = 65536


def build_target(qdef, gold):
    """Target distribution over the question's options, in `render_options` order.

    choice reads the gold probabilities by criterion key, score by ascending level index,
    noul by the fixed [false, true] order. Weights are normalised so a soft target is a
    distribution even when the source stores unnormalised weights.
    """
    internal = normalize_question(qdef)
    t = internal["t"]
    probs = gold.get("probabilities") or {}
    if t == "choice":
        target = [float(probs.get(key, 0.0)) for key in internal["crit"].keys()]
    elif t == "noul":
        target = [float(probs.get("false", 0.0)), float(probs.get("true", 0.0))]
    elif t == "score":
        target = [float(probs.get(str(i), 0.0)) for i in range(len(internal["crit"]))]
    else:
        raise ValueError("unknown question type %r" % t)
    total = sum(target)
    if total <= 0:
        # An all-zero target would train toward nothing. That is a data defect, not a
        # decision, so refuse it rather than invent a uniform target the reward would
        # then score as a real answer.
        raise ValueError("gold probabilities carry no mass")
    target = [v / total for v in target]
    return internal, target, target.index(max(target))


def prompt_fits_head(tok, internal, head_max_len):
    """Whether `build_sequence` will lay this question out without collapsing its options.

    `build_sequence` computes `opt_budget = head_max_len - sum(len(option_tokens))` and,
    when that drops below 16, silently re-truncates every option to a small fixed width
    instead of failing. The option *count* is unchanged, so the markers still line up with
    the target and nothing downstream notices — the model is simply handed options it
    cannot tell apart.

    Replicating the budget check here is what turns that silent loss into a skip.
    """
    opts = render_options(internal)
    total = 0
    for text in opts:
        # Same truncation build_sequence applies: 48 tokens per option, plus one [MASK].
        tokens = tok(" " + str(text).replace(tok.mask_token, " "), add_special_tokens=False,
                     truncation=True, max_length=48)["input_ids"]
        total += 1 + len(tokens)
    return (head_max_len - total) >= 16


def encode_case(case, tok, head_max_len, state_cap=STATE_TOKEN_CAP):
    """Every training item a case yields, plus a reason for each question that was skipped."""
    state = case["state"]
    mask_tok = tok.mask_token
    # Tokenize the shared state once per case and let build_sequence slice it per question.
    state_ids = tok(serialize_state(state).replace(mask_tok, " "), add_special_tokens=False)["input_ids"]
    state_id_list = list(state_ids)
    truncated_state = len(state_id_list) > state_cap
    if truncated_state:
        state_id_list = state_id_list[:state_cap]

    items, skipped = [], []
    for qid, qdef in (case.get("questions") or {}).items():
        gold = (case.get("gold") or {}).get(qid)
        if gold is None:
            skipped.append((qid, "no gold"))
            continue
        if gold.get("type") not in (None, qdef["type"]):
            skipped.append((qid, "gold type %r != question type %r" % (gold.get("type"), qdef["type"])))
            continue
        try:
            internal, target, label = build_target(qdef, gold)
        except ValueError as e:
            skipped.append((qid, str(e)))
            continue
        expected = len(render_options(internal))
        if not prompt_fits_head(tok, internal, head_max_len):
            skipped.append((qid, "head_max_len=%d too small for %d options; build_sequence "
                                 "would collapse them" % (head_max_len, expected)))
            continue
        # Build at an unbounded max_len. The prompt is bounded by head_max_len and the
        # state by state_cap, so this is the widest the item can ever be and every
        # narrower max_len is a prefix of it.
        seq, markers = build_sequence(tok, state, internal, state_cap + head_max_len + 8,
                                      head_max_len, state_ids=state_id_list)
        if len(markers) != expected or len(markers) != len(target):
            # build_sequence drops markers that fall past the budget. A truncated prompt
            # means the question lost options, so its target no longer describes what the
            # model can see; training on it would teach the wrong answer.
            skipped.append((qid, "markers=%d options=%d target=%d" % (len(markers), expected, len(target))))
            continue
        # The prompt ends where the state begins; build_sequence puts the state last and
        # closes the sequence with one more separator. Stored prompt = prefix + its own
        # closing [SEP]; the sequence is stored_prompt + state + [SEP], so the trailing
        # separator past the state is the one token the length formula adds back.
        prompt_end = len(seq) - len(state_id_list) - 1
        prompt = seq[:prompt_end]
        if len(state_id_list) > len(seq) - prompt_end:
            state_id_list = seq[prompt_end:]  # state was right-truncated by state_cap
        meta = case.get("_meta") or {}
        items.append({
            "case_id": case["id"],
            "question_id": qid,
            "source": adapters_mod.source_of(case),
            "qtype": QTYPES[internal["t"]],
            "qtype_name": internal["t"],
            "label": label,
            "n_options": expected,
            "target": target,
            "prompt_ids": prompt,
            "markers": markers,
            "state_ids": state_id_list,
            "state_truncated": truncated_state,
            "meta": {
                "source": meta.get("source", ""),
                "source_split": meta.get("source_split", ""),
                "group_key": meta.get("group_key", ""),
                "semantic_label": (meta.get("integration") or {}).get("semantic_label", ""),
            },
        })
        # Restore the full state for the next question in this case.
        state_id_list = list(state_ids[:state_cap])
    return items, skipped


def measure_head_budget(adapter, split, tok, quantile=0.999, sample=None, verbose=True):
    """The head budget a dataset actually needs, measured from its widest option sets.

    Returns a dict with the recommended `head_max_len` and the evidence behind it. The
    recommendation clears the widest set seen with a margin, because the fallback this
    guards against is silent and expensive.
    """
    widths = []
    n_cases = 0
    for case in adapter.iter_cases(split, limit=sample):
        n_cases += 1
        for qdef in (case.get("questions") or {}).values():
            try:
                internal = normalize_question(qdef)
            except (KeyError, TypeError):
                continue
            opts = render_options(internal)
            # The prompt is the type + instructions, then a [MASK] and the option text per
            # option, which is what build_sequence lays down before the state.
            head = len(tok("%s question: %s" % (internal["t"], internal["ins"]),
                           add_special_tokens=False)["input_ids"]) + 3
            for text in opts:
                head += 1 + len(tok(" " + str(text).replace(tok.mask_token, " "),
                                    add_special_tokens=False, truncation=True, max_length=48)["input_ids"])
            widths.append((head, len(opts)))
    if not widths:
        raise CacheError("no questions found in split %r; cannot measure a head budget" % split)
    heads = np.array([w for w, _ in widths])
    widest = max(widths, key=lambda w: w[1])
    recommended = int(np.ceil(widest[0] / 64.0) * 64)
    report = {
        "cases": n_cases,
        "questions": len(widths),
        "head_p50": int(np.percentile(heads, 50)),
        "head_p99": int(np.percentile(heads, 99)),
        "head_max": int(heads.max()),
        "widest_option_set": int(widest[1]),
        "head_at_widest": int(widest[0]),
        "recommended_head_max_len": recommended,
    }
    if verbose:
        print("  head budget: %d questions over %d cases | p50=%d p99=%d max=%d"
              % (report["questions"], n_cases, report["head_p50"], report["head_p99"], report["head_max"]))
        print("  widest option set: %d options need %d prompt tokens -> head_max_len=%d"
              % (report["widest_option_set"], report["head_at_widest"], recommended))
    return report
