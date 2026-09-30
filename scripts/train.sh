#!/usr/bin/env bash
# Launch a chinese-jev fine-tuning run.
#
#   bash scripts/train.sh --config configs/my.json
#   bash scripts/train.sh --config ... --gpus 0,1,2,3
#   bash scripts/train.sh --config ... --max-len 2048 --epochs 2 --tag len2048
#   bash scripts/train.sh --config ... --smoke
#
# Every run gets its own directory under --runs-root, named `<dataset>_<tag>_<timestamp>`.
# Nothing is overwritten and two runs of the same config stay comparable: a length sweep is
# a series of sibling directories, not a series of edits. Pair the sweep with a shared
# --cache-dir so the corpus is tokenized once.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Relative CLI paths mean "relative to where the user typed them", so resolve them
# against the launch directory before moving to the repo root.
LAUNCH_DIR="$PWD"
cd "$REPO_ROOT"

abspath() { case "$1" in /*|"") printf '%s' "$1" ;; *) printf '%s' "$LAUNCH_DIR/$1" ;; esac }

PYBIN="$(command -v python || command -v python3)"

CONFIG=""
RUNS_ROOT="$REPO_ROOT/runs"
TAG=""
GPUS=""
SMOKE=0
SKIP_TOKENIZE=0
WORK_DIR=""
CACHE_DIR=""
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)        CONFIG="$2"; shift 2 ;;
    --runs-root)     RUNS_ROOT="$2"; shift 2 ;;
    --tag)           TAG="$2"; shift 2 ;;
    --gpus)          GPUS="$2"; shift 2 ;;
    --smoke)         SMOKE=1; shift ;;
    --skip-tokenize) SKIP_TOKENIZE=1; shift ;;
    --work-dir)      WORK_DIR="$2"; shift 2 ;;
    --cache-dir)     CACHE_DIR="$2"; shift 2 ;;
    -h|--help)       sed -n '2,13p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *)               EXTRA+=("$1"); shift ;;
  esac
done

CONFIG="$(abspath "$CONFIG")"
RUNS_ROOT="$(abspath "$RUNS_ROOT")"
WORK_DIR="$(abspath "$WORK_DIR")"
CACHE_DIR="$(abspath "$CACHE_DIR")"

if [[ -z "$CONFIG" || ! -f "$CONFIG" ]]; then
  echo "error: --config <file> is required (see configs/)" >&2
  exit 2
fi

# A dataset_info-style config has no "dataset" key; the config's own name is the next
# best label for the run directory.
DATASET="$("$PYBIN" - "$CONFIG" <<'PY'
import json, os, sys
cfg = json.load(open(sys.argv[1]))
print(cfg.get("dataset") or os.path.splitext(os.path.basename(sys.argv[1]))[0])
PY
)"
STAMP="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="${WORK_DIR:-$RUNS_ROOT/${DATASET}${TAG:+_$TAG}_$STAMP}"
mkdir -p "$RUN_DIR"

# GPU selection: restrict CUDA_VISIBLE_DEVICES and let torchrun see only those cards, so
# `--gpus 2,3` means local ranks 0 and 1 on cards 2 and 3.
if [[ -n "$GPUS" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPUS"
fi
NPROC="${NPROC:-$("$PYBIN" -c "import torch;print(max(1,torch.cuda.device_count()))")}"

ARGS=(run --config "$CONFIG" --work-dir "$RUN_DIR")
[[ -n "$CACHE_DIR" ]] && ARGS+=(--cache-dir "$CACHE_DIR")
[[ $SMOKE -eq 1 ]] && ARGS+=(--max-items 2000 --max-steps 20 --epochs 1)
[[ $SKIP_TOKENIZE -eq 1 ]] && ARGS+=(--skip-tokenize)
if [[ ${#EXTRA[@]} -gt 0 ]]; then
  ARGS+=("${EXTRA[@]}")
fi

LOG="$RUN_DIR/train.log"
export PYTHONUNBUFFERED=1
echo "run directory : $RUN_DIR"
echo "config        : $CONFIG"
echo "procs         : $NPROC"
echo "log           : $LOG"

if [[ "$NPROC" -gt 1 ]]; then
  # One process per GPU. torchrun sets RANK/WORLD_SIZE/LOCAL_RANK, which the pipeline
  # reads to build the process group; tokenize/calibrate stay on rank 0 because they are
  # cheap next to training, while train/evaluate use every rank.
  torchrun --standalone --nproc-per-node="$NPROC" -m chinese_jev "${ARGS[@]}" 2>&1 | tee "$LOG"
else
  "$PYBIN" -m chinese_jev "${ARGS[@]}" 2>&1 | tee "$LOG"
fi

echo "run complete -> $RUN_DIR"
