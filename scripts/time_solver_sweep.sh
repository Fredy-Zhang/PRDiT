#!/usr/bin/env bash
# =============================================================================
# Repeated (median-of-N) runtime timing for the Flow-Matching solver sweep
#
# The main sweep (scripts/solver_sweep_fm.sh) times each cell exactly once,
# while generating the full FID sample set. Single wall-clock runs are noisy
# (see the Euler 60/80 and RK4 runtime inversions), so this script re-measures
# runtime only: it runs sample.py TIMING_REPEATS times per (solver, steps) cell
# on a small throwaway sample count and reports the MEDIAN seconds/sample.
#
# It never touches the FID generation outputs under samples/solver_sweep/<cell>/
# -- every timing run writes to a temporary directory that is deleted right
# after, so this is safe to run before, after, or alongside the FID sweep.
#
# Usage:
#   bash scripts/time_solver_sweep.sh
#
# Matched-NFE example (Euler vs RK4 at 240/300/400 true NFE):
#   SOLVERS="euler" STEPS_LIST="240 300 400" bash scripts/time_solver_sweep.sh
#   SOLVERS="rk4"   STEPS_LIST="60 75 100"  bash scripts/time_solver_sweep.sh
#
# Common overrides (environment variables):
#   CKPT={MODEL_CHECKPOINT}
#   TIMING_REPEATS=3        # runs per cell; median reported
#   TIMING_SAMPLES=20       # samples generated per timing run (throwaway)
#   BATCH_SIZE=4            # must match the production BATCH_SIZE for fair s/sample
#   RETIME_EXISTING=1       # re-time cells already present in the CSV (default: skip)
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# Project configuration
# -----------------------------------------------------------------------------

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

CONFIG="${CONFIG:-lidc.yaml}"
GLOBAL_CONFIG="${GLOBAL_CONFIG:-configs/global/${CONFIG}}"
CKPT="${CKPT:-{MODEL_CHECKPOINT}}"

PYTHON="${PYTHON:-python}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

# -----------------------------------------------------------------------------
# Sweep / timing configuration
# -----------------------------------------------------------------------------

SOLVERS="${SOLVERS:-euler}"
STEPS_LIST="${STEPS_LIST:-240 300 400}"

TIMING_REPEATS="${TIMING_REPEATS:-3}"
TIMING_SAMPLES="${TIMING_SAMPLES:-20}"
BATCH_SIZE="${BATCH_SIZE:-4}"

# Skip cells already present in the median CSV unless RETIME_EXISTING=1.
RETIME_EXISTING="${RETIME_EXISTING:-0}"

OUTPUT_ROOT="${OUTPUT_ROOT:-samples/solver_sweep}"
LOG_ROOT="${LOG_ROOT:-logs/solver_sweep}"
SUMMARY_CSV="${SUMMARY_CSV:-${OUTPUT_ROOT}/runtime_median.csv}"

# Stochastic-corrector knobs (ignored by euler/heun/rk4; kept for parity).
ETA="${ETA:-0.0}"
ERR_TARGET="${ERR_TARGET:-1.0}"
SAMPLE_EXTRA_ARGS="${SAMPLE_EXTRA_ARGS:-}"

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

log()   { printf '\n\033[1;34m[time-sweep]\033[0m %s\n' "$*"; }
error() { printf '\n\033[1;31m[time-sweep error]\033[0m %s\n' "$*" >&2; exit 1; }

nfe_per_step() {
    case "$1" in
        euler)   echo 1 ;;
        heun)    echo 2 ;;
        rk4)     echo 4 ;;
        pc)      echo 1 ;;
        pc_heun) echo 2 ;;
        *)       error "Unknown solver: $1 (use euler|heun|rk4|pc|pc_heun)" ;;
    esac
}

# Median of the numeric arguments (mean of the two middle values when even).
median() {
    printf '%s\n' "$@" | sort -n | awk '
        { a[NR] = $1 }
        END {
            if (NR == 0) { print "nan"; exit }
            if (NR % 2) { printf "%.6f", a[(NR + 1) / 2] }
            else        { printf "%.6f", (a[NR / 2] + a[NR / 2 + 1]) / 2 }
        }'
}

# Does (solver, steps) already have a row in the median CSV?
cell_already_timed() {
    local solver="$1" steps="$2"
    [[ -f "$SUMMARY_CSV" ]] || return 1
    awk -F, -v s="$solver" -v n="$steps" \
        'NR > 1 && $1 == s && $2 == n { found = 1 } END { exit !found }' \
        "$SUMMARY_CSV"
}

# -----------------------------------------------------------------------------
# Validate configuration
# -----------------------------------------------------------------------------

[[ -f "$GLOBAL_CONFIG" ]] || error "Global config does not exist: $GLOBAL_CONFIG"
[[ -f "sample.py" ]]      || error "sample.py was not found under: $PROJECT_ROOT"
[[ -f "$CKPT" ]]          || error "Checkpoint does not exist: $CKPT"
[[ "$TIMING_REPEATS" =~ ^[0-9]+$ && "$TIMING_REPEATS" -ge 1 ]] \
    || error "TIMING_REPEATS must be a positive integer (got: $TIMING_REPEATS)"

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"

log "Project root   : $PROJECT_ROOT"
log "Checkpoint     : $CKPT"
log "Solvers        : $SOLVERS"
log "Steps          : $STEPS_LIST"
log "Repeats/cell   : $TIMING_REPEATS"
log "Samples/run    : $TIMING_SAMPLES (throwaway)"
log "Batch size     : $BATCH_SIZE"
log "Median CSV     : $SUMMARY_CSV"

# Append-safe header (median CSV is merged, keyed by solver+steps, below).
if [[ ! -f "$SUMMARY_CSV" ]]; then
    printf '%s\n' \
        "solver,steps,nfe,timing_samples,repeats,median_seconds_per_sample,seconds_per_sample_runs" \
        > "$SUMMARY_CSV"
fi

# Accumulate this invocation's rows in a temp file, then merge into SUMMARY_CSV.
NEW_CSV="$(mktemp "${TMPDIR:-/tmp}/time_sweep_new.XXXXXX.csv")"
printf '%s\n' \
    "solver,steps,nfe,timing_samples,repeats,median_seconds_per_sample,seconds_per_sample_runs" \
    > "$NEW_CSV"
trap 'rm -f "$NEW_CSV"' EXIT

# -----------------------------------------------------------------------------
# Timing sweep
# -----------------------------------------------------------------------------

for SOLVER in $SOLVERS; do
    NFE_PER_STEP="$(nfe_per_step "$SOLVER")"

    for STEPS in $STEPS_LIST; do
        [[ "$STEPS" =~ ^[0-9]+$ && "$STEPS" -ge 1 ]] \
            || error "Invalid sampling-step value: $STEPS"

        if [[ "$RETIME_EXISTING" != "1" ]] && cell_already_timed "$SOLVER" "$STEPS"; then
            log "Skipping (already timed; RETIME_EXISTING=1 to force): $SOLVER steps=$STEPS"
            continue
        fi

        NFE=$((NFE_PER_STEP * STEPS))
        log "============================================================"
        log "solver=$SOLVER  steps=$STEPS  NFE=$NFE  (x${TIMING_REPEATS})"
        log "============================================================"

        RUNS=()
        for ((r = 1; r <= TIMING_REPEATS; r++)); do
            TMP_OUT="$(mktemp -d "${TMPDIR:-/tmp}/time_${SOLVER}_${STEPS}.XXXXXX")"
            SAMPLE_LOG="${LOG_ROOT}/timing_${SOLVER}_steps_${STEPS}_run${r}.log"

            START="$(date +%s.%N)"
            # shellcheck disable=SC2086
            "$PYTHON" sample.py \
                --config "$CONFIG" \
                --ckpt "$CKPT" \
                --solver "$SOLVER" \
                --eta "$ETA" \
                --err-target "$ERR_TARGET" \
                --num-samples "$BATCH_SIZE" \
                --total-samples "$TIMING_SAMPLES" \
                --num-sampling-steps "$STEPS" \
                --output-dir "$TMP_OUT" \
                $SAMPLE_EXTRA_ARGS \
                > "$SAMPLE_LOG" 2>&1 \
                || { rm -rf "$TMP_OUT"; error "sample.py failed (see $SAMPLE_LOG)"; }
            END="$(date +%s.%N)"

            rm -rf "$TMP_OUT"

            SPS="$(awk -v s="$START" -v e="$END" -v n="$TIMING_SAMPLES" \
                'BEGIN { printf "%.6f", (e - s) / (n > 0 ? n : 1) }')"
            RUNS+=("$SPS")
            log "run $r/$TIMING_REPEATS: ${SPS} s/sample"
        done

        MEDIAN="$(median "${RUNS[@]}")"
        RUNS_JOINED="$(IFS=';'; echo "${RUNS[*]}")"

        printf '%s,%s,%s,%s,%s,%s,%s\n' \
            "$SOLVER" "$STEPS" "$NFE" "$TIMING_SAMPLES" "$TIMING_REPEATS" \
            "$MEDIAN" "$RUNS_JOINED" \
            >> "$NEW_CSV"

        log "median: ${MEDIAN} s/sample  (runs: ${RUNS_JOINED})"
    done
done

# -----------------------------------------------------------------------------
# Merge new rows into the canonical median CSV (existing kept, matching
# solver+steps overwritten, new cells appended) -- same pattern as the FID sweep.
# -----------------------------------------------------------------------------

"$PYTHON" - "$SUMMARY_CSV" "$NEW_CSV" <<'PY'
import csv, os, sys

dst, src = sys.argv[1], sys.argv[2]
fields = ["solver", "steps", "nfe", "timing_samples", "repeats",
          "median_seconds_per_sample", "seconds_per_sample_runs"]
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
load(src)   # this run's results override / extend

os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
with open(dst, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for key in order:
        w.writerow({k: rows[key].get(k, "") for k in fields})
print(f"Merged {len(order)} timing rows into: {dst}")
PY

log "Timing sweep complete."
log "Median CSV: $SUMMARY_CSV"
column -s, -t "$SUMMARY_CSV" 2>/dev/null || cat "$SUMMARY_CSV"
