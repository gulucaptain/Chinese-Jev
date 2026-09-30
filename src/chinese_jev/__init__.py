"""chinese-jev: a self-contained pipeline for training typed decision models.

The pipeline is built around one idea: the token cache is independent of the training
sequence length. The corpus is tokenized once into a memory-mappable cache that stores the
prompt and the document state separately, and a run reassembles the exact sequence its
`max_len` calls for at read time. Changing `max_len` therefore costs nothing but a filter
over stored lengths, which is what makes the length knob cheap enough to sweep. The cache
lives wherever `cache_dir` points, so several runs can share one tokenization.

See `docs/pipeline.md` for the guide and `docs/new-dataset.md` for adding a corpus.
"""
__all__ = ["adapters", "cache", "calibrate", "cli", "config", "data", "distributed",
           "encoding", "evaluate", "modeling", "prepare", "sequence", "tokenizer", "train"]

__version__ = "0.1.0"
