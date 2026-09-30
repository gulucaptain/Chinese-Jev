"""Dataset adapters: everything that is specific to one corpus.

The rest of the pipeline speaks one canonical case schema, the one the data pipeline in
`data-pipeline/` releases:

    {"id": str,
     "state": str,                      # the material the decision is made against
     "questions": {qid: {"type": "choice"|"score"|"noul",
                         "instructions": str,
                         "criteria": <list|dict>}},
     "gold": {qid: {"type": ..., "probabilities": {label: float}}},
     "_meta": {...}}                    # passed through untouched

An adapter turns raw records into that schema. Registering one is the whole cost of
supporting a new corpus — see `docs/new-dataset.md` for a worked example.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

REGISTRY = {}


def register(name):
    def deco(cls):
        REGISTRY[name] = cls
        return cls
    return deco


def get(name):
    if name not in REGISTRY:
        raise KeyError("unknown dataset %r; registered: %s"
                       % (name, ", ".join(sorted(REGISTRY))))
    return REGISTRY[name]


class Adapter:
    """Base class. Subclasses declare which splits exist and yield canonical cases."""

    #: split name -> whatever `iter_cases` needs to find it
    splits = ()

    def __init__(self, **params):
        self.params = params

    @classmethod
    def from_params(cls, params):
        return cls(**params)

    @property
    def registry_name(self):
        """The name this class was registered under, for dispatching into worker processes.

        Workers are separate processes, so the adapter cannot be pickled across as a method
        bound to an unregistered class; they rebuild it from the registry instead.
        """
        for name, cls in REGISTRY.items():
            if cls is type(self):
                return name
        raise KeyError("%s is not registered; add @register(...) to it"
                       % type(self).__name__)

    def split_path(self, split):
        raise NotImplementedError

    def iter_cases(self, split, limit=None):
        raise NotImplementedError

    def describe(self):
        return {"adapter": type(self).__name__, "params": self.params}


@register("jsonl")
class JsonlAdapter(Adapter):
    """Reads `<data_root>/<file_template>` as JSONL, optionally gzipped.

    `open` picks gzip from the suffix, so a corpus that ships `.jsonl.gz` and one that
    ships plain `.jsonl` are the same adapter with a different path template. This is the
    adapter a new corpus usually needs, and the base class the corpus-specific ones extend.
    """

    splits = ("train", "validation", "test")

    def __init__(self, data_root, file_template="{split}.cases.jsonl.gz", splits=None, **extra):
        super().__init__(data_root=data_root, file_template=file_template, splits=splits, **extra)
        self.data_root = Path(data_root)
        self.file_template = file_template
        self.splits = tuple(splits) if splits else type(self).splits

    def split_path(self, split):
        return self.data_root / self.file_template.format(split=split)

    def open(self, path):
        path = Path(path)
        if path.suffix == ".gz":
            return gzip.open(path, "rt", encoding="utf-8")
        return path.open("r", encoding="utf-8")

    def iter_cases(self, split, limit=None):
        path = self.split_path(split)
        if not path.exists():
            raise FileNotFoundError("split %r not found at %s" % (split, path))
        seen = 0
        with self.open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
                seen += 1
                if limit is not None and seen >= limit:
                    return


@register("chinese-jev")
class ChineseJevAdapter(JsonlAdapter):
    """A chinese-jev release: one decision per line, `_meta.source` per row.

    Rows are the canonical schema already, so no field mapping is needed. A case holds
    exactly one question, which is why downstream counts of "cases" and "items" agree for
    these datasets.
    """

    splits = ("train", "validation", "test")


def source_of(case, default="unknown"):
    """The corpus a case came from, for the per-source histograms.

    Different corpora record this differently and it is only ever used for reporting, so a
    missing value degrades to a single bucket rather than failing the tokenization.
    """
    meta = case.get("_meta") or {}
    for key in ("source", "source_family", "source_split"):
        value = meta.get(key)
        if value:
            # Corpus names carry their split as a suffix (`cmedqa2_train`, `dialmed_train`),
            # which would otherwise split one corpus into per-split buckets.
            for suffix in ("_train", "_validation", "_val", "_test", "_dev"):
                if value.endswith(suffix):
                    return value[: -len(suffix)]
            return value
    return default
