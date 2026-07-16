#!/usr/bin/env bash
# =============================================================================
# 3D FID for the Flow-Matching Solver x Step Sweep
#
# Thin wrapper around evaluations/eval_fid_sweep.py. It derives the real-data
# root, image size and split directory from the project config, then runs one
# 3D FID pass over every (solver, num_steps) cell produced by
# scripts/solver_sweep_fm.sh. The extractor and real reference are loaded once
# and shared across all cells (see eval_fid_sweep.py).
#
# Prerequisite:
#   Run scripts/solver_sweep_fm.sh first so that
#   <SWEEP_ROOT>/<solver>_steps_<N>/xs/ directories exist.
#
# Usage:
#   bash scripts/fid_solver_sweep.sh
#
# Common overrides (environment variables):
#   PYTHON=/path/to/eval-env/bin/python \
#   DATA_ROOT_REAL=/path/to/LIDC-IDRI \
#   NUM_SAMPLES=500 \
#   bash scripts/fid_solver_sweep.sh
#
#   SOLVERS="rk4" STEPS_LIST="10 20" bash scripts/fid_solver_sweep.sh
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# Project configuration
# -----------------------------------------------------------------------------

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

CONFIG="${CONFIG:-lidc.yaml}"
GLOBAL_CONFIG="${GLOBAL_CONFIG:-configs/global/${CONFIG}}"

# Replace this interpreter placeholder when FID dependencies live in a separate
# environment, or override with PYTHON=...; otherwise `python` is used.
DEFAULT_EVAL_PY="{EVAL_PYTHON}"
if [[ -z "${PYTHON:-}" ]]; then
    if [[ -x "$DEFAULT_EVAL_PY" ]]; then
        PYTHON="$DEFAULT_EVAL_PY"
    else
        PYTHON="python"
    fi
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

# -----------------------------------------------------------------------------
# Sweep / evaluation configuration
# -----------------------------------------------------------------------------

# Matches the extended Euler sweep in solver_sweep_fm.sh. New FID rows are
# merged into any existing fid_summary.csv (keyed by solver+steps) rather than
# replacing it, so prior results are preserved.
SOLVERS="${SOLVERS:-euler}"
STEPS_LIST="${STEPS_LIST:-60 80 100 150 200}"

# Must match scripts/solver_sweep_fm.sh (OUTPUT_ROOT / fake sub-dir).
SWEEP_ROOT="${SWEEP_ROOT:-samples/solver_sweep}"
FAKE_SUBDIR="${FAKE_SUBDIR:-xs}"

# Max volumes per side for FID; clamped to what each cell actually holds.
NUM_SAMPLES="${NUM_SAMPLES:-500}"

EVAL_DATASET="${EVAL_DATASET:-lidc-idri}"
PRETRAIN_PATH="${PRETRAIN_PATH:-evaluations/pretrained/resnet_50_23dataset.pth}"

# Cache dir for the shared real statistics + the output CSV.
ACT_DIR="${ACT_DIR:-${SWEEP_ROOT}/fid_acts}"
OUTPUT_CSV="${OUTPUT_CSV:-${SWEEP_ROOT}/fid_summary.csv}"

# Reuse cached real stats across repeated runs (skip the real preload). The
# extended sweep runs after the initial one, so the real stats in ACT_DIR
# already exist; default to reusing them (no real-data root needed). Set
# REUSE_REAL=0 to force a fresh computation of the real statistics.
REUSE_REAL="${REUSE_REAL:-1}"

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

log() {
    printf '\n\033[1;34m[fid-sweep]\033[0m %s\n' "$*"
}

error() {
    printf '\n\033[1;31m[fid-sweep error]\033[0m %s\n' "$*" >&2
    exit 1
}

read_yaml() {
    # Usage: read_yaml <yaml-file> <dotted.key>
    "$PYTHON" - "$1" "$2" <<'PY'
import sys, yaml
with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f)
val = cfg
for key in sys.argv[2].split("."):
    if not isinstance(val, dict) or key not in val:
        raise KeyError(f"Missing '{sys.argv[2]}' in '{sys.argv[1]}' (at '{key}')")
    val = val[key]
print(val)
PY
}

# -----------------------------------------------------------------------------
# Validate + derive configuration
# -----------------------------------------------------------------------------

[[ -f "$GLOBAL_CONFIG" ]] || error "Global config does not exist: $GLOBAL_CONFIG"
[[ -f "evaluations/eval_fid_sweep.py" ]] || error "evaluations/eval_fid_sweep.py was not found."
[[ -f "$PRETRAIN_PATH" ]] || error "Pretrained FID weights not found: $PRETRAIN_PATH"
[[ -d "$SWEEP_ROOT" ]] || error "Sweep root not found: $SWEEP_ROOT (run scripts/solver_sweep_fm.sh first)."

IMAGE_SIZE="${IMAGE_SIZE:-$(read_yaml "$GLOBAL_CONFIG" data.image_size)}"
DATA_ROOT_REAL="${DATA_ROOT_REAL:-$(read_yaml "$GLOBAL_CONFIG" data.path)}"
TRAIN_LIST="$(read_yaml "$GLOBAL_CONFIG" data.train_list)"
SPLIT_DIR="${SPLIT_DIR:-$(dirname "$TRAIN_LIST")}"

# Real data is only needed when the cached real stats are not being reused.
REUSE_FLAG=()
if [[ "$REUSE_REAL" == "1" ]]; then
    REUSE_FLAG=(--reuse_real_stats)
else
    [[ -d "$DATA_ROOT_REAL" ]] || error "Real-data directory does not exist: $DATA_ROOT_REAL"
fi

# -----------------------------------------------------------------------------
# Print configuration
# -----------------------------------------------------------------------------

log "Project root   : $PROJECT_ROOT"
log "Python         : $PYTHON"
log "Config         : $GLOBAL_CONFIG"
log "Dataset        : $EVAL_DATASET"
log "Image size     : $IMAGE_SIZE"
log "Real-data root : $DATA_ROOT_REAL"
log "Split dir      : $SPLIT_DIR"
log "Pretrain path  : $PRETRAIN_PATH"
log "Sweep root     : $SWEEP_ROOT"
log "Solvers        : $SOLVERS"
log "Steps          : $STEPS_LIST"
log "Num samples    : $NUM_SAMPLES"
log "Reuse real     : $REUSE_REAL"
log "Output CSV     : $OUTPUT_CSV"

# -----------------------------------------------------------------------------
# Run the FID sweep (the driver loops over all cells internally)
# -----------------------------------------------------------------------------

# The driver overwrites its --output_csv with only the cells it just evaluated.
# To preserve prior results, write to a temp file and merge (keyed by
# solver+steps, new rows winning) into the canonical OUTPUT_CSV.
NEW_CSV="$(mktemp "${TMPDIR:-/tmp}/fid_sweep_new.XXXXXX.csv")"
trap 'rm -f "$NEW_CSV"' EXIT

# shellcheck disable=SC2086
"$PYTHON" evaluations/eval_fid_sweep.py \
    --sweep_root "$SWEEP_ROOT" \
    --fake_subdir "$FAKE_SUBDIR" \
    --solvers $SOLVERS \
    --steps $STEPS_LIST \
    --dataset "$EVAL_DATASET" \
    --img_size "$IMAGE_SIZE" \
    --data_root_real "$DATA_ROOT_REAL" \
    --split_dir "$SPLIT_DIR" \
    --pretrain_path "$PRETRAIN_PATH" \
    --path_to_activations "$ACT_DIR" \
    --output_csv "$NEW_CSV" \
    --num_samples "$NUM_SAMPLES" \
    "${REUSE_FLAG[@]}"

# Merge new rows into any existing summary (existing rows kept, matching
# solver+steps overwritten, new step counts appended).
"$PYTHON" - "$OUTPUT_CSV" "$NEW_CSV" <<'PY'
import csv, os, sys

dst, src = sys.argv[1], sys.argv[2]
fields = ["solver", "steps", "nfe", "num_fake", "fid"]
rows, order = {}, []

def load(path):
    if not os.path.exists(path):
        return
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            key = (r["solver"], r["steps"])
            if key not in rows:
                order.append(key)
            rows[key] = r

load(dst)   # existing results first
load(src)   # new results override / extend

os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
with open(dst, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for key in order:
        w.writerow({k: rows[key].get(k, "") for k in fields})
print(f"Merged {len(order)} FID rows into: {dst}")
PY

log "FID sweep complete."
log "Summary CSV: $OUTPUT_CSV"

column -s, -t "$OUTPUT_CSV" 2>/dev/null || cat "$OUTPUT_CSV"
