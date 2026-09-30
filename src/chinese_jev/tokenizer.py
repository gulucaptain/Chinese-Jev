"""Tokenizer loading and identity.

A *bundle* is a model directory: `encoder/`, `tokenizer/`, `model.safetensors` and a
`rl_agent_config.json` describing the head geometry. This module owns the two operations
the rest of the pipeline needs from it: loading the tokenizer robustly across transformers
versions, and fingerprinting it so a token cache can prove it still matches the model it
will be trained with.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import warnings
from pathlib import Path


def fix_tokenizer_config(bundle):
    """Ensure `<bundle>/tokenizer/tokenizer_config.json` loads across transformers versions."""
    cfg_file = os.path.join(str(bundle), "tokenizer", "tokenizer_config.json")
    if not os.path.exists(cfg_file):
        return
    try:
        with open(cfg_file) as f:
            tcfg = json.load(f)
        changed = False
        if tcfg.get("tokenizer_class") in (None, "TokenizersBackend"):
            tcfg["tokenizer_class"] = "PreTrainedTokenizerFast"
            tcfg.pop("backend", None)
            tcfg.pop("is_local", None)
            changed = True
        # Checkpoints built on the mmBERT/Gemma tokenizer store extra_special_tokens as a
        # list; transformers expects a mapping and raises "'list' object has no attribute
        # 'keys'", which makes AutoTokenizer — and so the whole model — fail to load.
        extra = tcfg.get("extra_special_tokens")
        if isinstance(extra, list):
            tcfg["extra_special_tokens"] = {"extra_%d" % i: t for i, t in enumerate(extra)}
            changed = True
        if changed:
            # The config may be a symlink into a shared HuggingFace blob store, so writing
            # through it would corrupt a file shared with other revisions and processes.
            # Write a sibling temporary file and `os.replace` it into place: the entry
            # becomes a regular file, swapped atomically, so a concurrent load never sees
            # a missing or half-written config.
            cfg_dir = os.path.dirname(cfg_file)
            fd, tmp_file = tempfile.mkstemp(dir=cfg_dir, prefix=".tokenizer_config.", suffix=".tmp")
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump(tcfg, f, indent=2)
                # mkstemp creates the file 0600; keep the mode the file had so a shared
                # cache stays readable to the same users as before.
                try:
                    os.chmod(tmp_file, os.stat(cfg_file).st_mode & 0o777)
                except OSError:
                    pass
                os.replace(tmp_file, cfg_file)
            except BaseException:
                try:
                    os.unlink(tmp_file)
                except OSError:
                    pass
                raise
    except Exception as e:
        # Do not swallow this silently: if the patch did not apply, AutoTokenizer may fail
        # later with a confusing error and no hint that the config was the cause.
        warnings.warn(
            "chinese-jev: could not patch %s (%s); the tokenizer may fail to load with "
            "this transformers version." % (cfg_file, e),
            RuntimeWarning, stacklevel=2)


def load_tokenizer(bundle):
    """The bundle's tokenizer, patched for the installed transformers version."""
    fix_tokenizer_config(str(bundle))
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(Path(bundle) / "tokenizer"))
    if not tok.pad_token_id:
        tok.pad_token = tok.eos_token
    return tok


def tokenizer_fingerprint(tok):
    """A digest of the tokenizer's semantics: vocabulary and special token ids.

    Two tokenizers that agree here assign the same ids to the same text, which is the
    property a cache built with one and a model loaded with the other actually needs.
    """
    vocab = tok.get_vocab()
    payload = json.dumps({
        "vocab_size": len(vocab),
        "special": {n: getattr(tok, n + "_token_id") for n in ("cls", "sep", "mask", "pad", "unk")},
        "checksum": hashlib.sha256(
            "".join("%s\x00%d\n" % (k, v) for k, v in sorted(vocab.items())).encode("utf-8")
        ).hexdigest(),
    }, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
