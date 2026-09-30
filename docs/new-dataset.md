# Adding a new dataset

The pipeline speaks one canonical case schema. Supporting a new corpus means providing an
adapter that yields it — nothing in tokenization, training, calibration or evaluation is
dataset-specific.

## The canonical case

```json
{
  "id": "case-0001",
  "state": "the material the decision is made against (string, or JSON-serializable)",
  "questions": {
    "q1": {"type": "choice", "instructions": "...", "criteria": {"label_a": "desc", "label_b": null}}
  },
  "gold": {
    "q1": {"type": "choice", "probabilities": {"label_a": 1.0, "label_b": 0.0}}
  },
  "_meta": {"source": "corpus_name", "group_key": "..."}
}
```

Question types and their `criteria` / `probabilities` shapes:

| type | criteria | probability keys |
|---|---|---|
| `choice` | dict of label → description (or list of labels) | the same labels |
| `noul` | `{"false": desc, "true": desc}` (either optional) | `"false"`, `"true"` |
| `score` | ordered list of level descriptions | `"0"`, `"1"`, … indexing the list |

Targets may be soft; they are normalized to a distribution. A question with no gold, a
gold with no mass, or options too wide for `head_max_len` is skipped with a recorded
reason, never trained on silently.

If the corpus does not exist yet, build it with the release pipeline in
[`../data-pipeline/`](../data-pipeline/) — its output is this schema, one decision per
line, and the built-in `chinese-jev` adapter reads it directly.

## If the data is already JSONL in this schema

No code. Point the generic adapter at it:

```json
{
  "dataset": "jsonl",
  "dataset_params": {
    "data_root": "/data/my-corpus",
    "file_template": "{split}.cases.jsonl.gz",
    "splits": ["train", "validation", "test"]
  }
}
```

Plain `.jsonl` and gzipped `.jsonl.gz` are both fine; the suffix decides.

## If the records need mapping

Subclass `JsonlAdapter` (or `Adapter` for non-JSONL sources), register it, and translate
per record in `iter_cases`:

```python
from chinese_jev.adapters import JsonlAdapter, register

@register("my-corpus")
class MyCorpusAdapter(JsonlAdapter):
    splits = ("train", "validation", "test")

    def iter_cases(self, split, limit=None):
        for i, row in enumerate(super().iter_cases(split, limit=limit)):
            yield {
                "id": row.get("id", f"row-{i}"),
                "state": row["document"],
                "questions": {"q": {"type": "noul",
                                    "instructions": "Does the answer match the question?",
                                    "criteria": {"false": "no", "true": "yes"}}},
                "gold": {"q": {"type": "noul",
                               "probabilities": {"false": 1.0 - row["label"], "true": row["label"]}}},
                "_meta": {"source": row.get("source", "my-corpus")},
            }
```

The adapter must be importable before the CLI runs (put it on `PYTHONPATH` and import it
from a small launcher, or add it to `chinese_jev/adapters.py` in your fork). Workers
rebuild adapters from the registry by name, so `@register` is required, and everything the
adapter needs must arrive through `dataset_params` (it is pickled to worker processes as
plain data).

## Then

```bash
# once per dataset: measure head_max_len (null in the config = automatic),
# build the cache, write dataset.json — read the length table it prints
chinese-jev prepare --config configs/my-corpus.json

# same config: train, calibrate, evaluate
chinese-jev run --config configs/my-corpus.json
```

with the config carrying both the dataset fields (`dataset`, `dataset_params`,
`cache_dir`) and the run fields (`bundle`, `work_dir`, `max_len`, hyperparameters) —
copy `configs/template.json` and fill in the paths.

The length table `prepare` prints (items retained per `max_len`) is the evidence for
choosing `max_len`; nothing needs re-tokenizing when you change it.
