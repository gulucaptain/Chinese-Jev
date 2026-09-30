"""Question normalization, option rendering, and token sequence construction.

This is the contract every other module builds on: a *question* is normalized into the
internal form ``{"t", "ins", "crit"[, "labels"]}``, its options are rendered into texts in
a fixed order, and ``build_sequence`` lays a decision out as

    [CLS] <type> instructions [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]

The `[MASK]` positions are the *markers*: the model scores one logit per marker, so the
marker list and the target distribution must always have the same length and order. Every
invariant the token cache maintains (see `cache.py`) is an invariant of this layout.
"""
from __future__ import annotations

import json

QTYPES = {"choice": 0, "score": 1, "noul": 2}
QTYPE_NAMES = {v: k for k, v in QTYPES.items()}
_DEFAULT_NOUL_LABELS = {"false": "false", "true": "true"}


def serialize_state(state):
    """The text a decision is made against. Strings pass through, structure becomes JSON."""
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False)


def render_criterion(value):
    """Render one criterion value as text.

    Strings pass through; anything structured (dict, list, number) becomes compact JSON, so
    a rubric reads as JSON rather than a Python repr.
    """
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(", ", ": "), default=str)


def _resolve_noul_labels(labels=None):
    if labels is None:
        labels = _DEFAULT_NOUL_LABELS
    if not isinstance(labels, dict) or set(labels) != {"false", "true"}:
        raise ValueError("noul labels must map exactly 'false' and 'true' to distinct non-empty strings")
    false_label, true_label = labels["false"], labels["true"]
    if not isinstance(false_label, str) or not isinstance(true_label, str):
        raise ValueError("noul labels must map exactly 'false' and 'true' to distinct non-empty strings")
    false_label, true_label = false_label.strip(), true_label.strip()
    if not false_label or not true_label or false_label == true_label:
        raise ValueError("noul labels must map exactly 'false' and 'true' to distinct non-empty strings")
    return false_label, true_label


def normalize_question(qdef):
    """The canonical question schema, normalized to the internal form the pipeline uses.

    Input is ``{"type", "instructions", "criteria"[, "labels"]}``; output is
    ``{"t", "ins", "crit"[, "labels"]}``. choice criteria given as a list become a dict of
    label -> None; noul criteria keys are lowercased so boolean literals normalize to
    "true"/"false"; non-string instructions are serialized with ``ensure_ascii=False`` so
    non-ASCII text reaches the tokenizer as text, not as `\\uXXXX` escapes.
    """
    t = qdef["type"]
    crit = qdef.get("criteria")
    if t == "choice" and isinstance(crit, list):
        crit = {c: None for c in crit}
    elif t == "noul" and isinstance(crit, dict):
        crit = {str(k).lower(): v for k, v in crit.items()}
    ins = qdef["instructions"]
    if not isinstance(ins, str):
        ins = json.dumps(ins, ensure_ascii=False)
    q = {"t": t, "ins": ins, "crit": crit}
    if "labels" in qdef:
        q["labels"] = qdef["labels"]
    return q


def render_options(q):
    """Render option texts in label-index order. Noul semantic order is always [false, true]."""
    t, crit = q["t"], q.get("crit")
    if t != "noul" and "labels" in q:
        raise ValueError("labels is only supported for noul questions")
    if t == "choice":
        # only None/"" mean "no description"; 0 and False are legitimate criterion values
        return [k if v is None or v == "" else "%s: %s" % (k, render_criterion(v)) for k, v in crit.items()]
    if t == "score":
        return ["level %d: %s" % (i, render_criterion(c)) for i, c in enumerate(crit)]
    crit = crit or {}
    false_label, true_label = _resolve_noul_labels(q.get("labels"))
    false_crit, true_crit = crit.get("false"), crit.get("true")
    return [
        false_label + ": "
        + (render_criterion(false_crit) if false_crit not in (None, "") else "no, the statement does not hold"),
        true_label + ": "
        + (render_criterion(true_crit) if true_crit not in (None, "") else "yes, the statement holds"),
    ]


def build_sequence(tok, state, q, max_len=512, head_max_len=192, option_order=None,
                   truncate_left=False, state_ids=None):
    """Format: [CLS] <type> instructions [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP].

    `state_ids` lets a caller tokenize the shared state once and reuse it across every
    question, instead of re-serializing and re-tokenizing the same document per question.

    When the rendered options do not fit `head_max_len`, every option is re-truncated to a
    small fixed width instead of failing: the option *count* is preserved so markers still
    line up with the target, but the options become indistinguishable. `encoding.py`
    replicates this budget check up front precisely so that a dataset build turns this
    silent degradation into an explicit skip.
    """
    mask_tok = tok.mask_token
    opts = render_options(q)
    order = option_order if option_order is not None else list(range(len(opts)))
    ins = str(q["ins"]).replace(mask_tok, " ")
    head_ids = tok("%s question: %s" % (q["t"], ins), add_special_tokens=False)["input_ids"]
    opt_ids = []
    for i in order:
        # Cap at the tokenizer, not after the fact: truncation=True, max_length=48 keeps
        # the first 48 tokens without paying to tokenize a long description in full.
        opt_tokens = tok(
            " " + opts[i].replace(mask_tok, " "),
            add_special_tokens=False,
            truncation=True,
            max_length=48,
        )["input_ids"]
        opt_ids.append([tok.mask_token_id] + opt_tokens)
    opt_budget = head_max_len - sum(len(o) for o in opt_ids)
    if opt_budget < 16:
        per = max(4, (head_max_len - 16) // max(1, len(opt_ids)))
        opt_ids = [o[:per] for o in opt_ids]
        opt_budget = head_max_len - sum(len(o) for o in opt_ids)
    head_ids = head_ids[: max(8, opt_budget)]
    ids = [tok.cls_token_id] + head_ids + [tok.sep_token_id]
    markers = []
    for o in opt_ids:
        markers.append(len(ids))
        ids.extend(o)
    ids.append(tok.sep_token_id)
    room = max(0, max_len - len(ids) - 1)
    if state_ids is None:
        state_ids = tok(serialize_state(state).replace(mask_tok, " "), add_special_tokens=False)["input_ids"]
    # not state_ids[-room:]: with no room left, state_ids[-0:] is the whole state rather than none of it
    st = state_ids[max(0, len(state_ids) - room):] if truncate_left else state_ids[:room]
    ids = ids + st + [tok.sep_token_id]
    return ids[:max_len], [m for m in markers if m < max_len]
