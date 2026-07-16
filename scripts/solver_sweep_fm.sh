#!/usr/bin/env bash
# =============================================================================
# Flow-Matching Solver x Step Sweep
#
# Generates samples from a fixed trained PRDiT checkpoint for every
# (solver, num_steps) combination in the grid below:
#
#   solvers : euler  (original 1st-order flow-matching solver, 1 NFE/step)
#             heun   (2nd-order predictor-corrector,           2 NFE/step)
#             rk4    (classic 4th-order Runge-Kutta,           4 NFE/step)
#   steps   : configurable (default extends Euler to 60 80 100 150 200)
#
# Each run writes to its own output directory and log, and one row is
# appended to a runtime-summary CSV so runs can be compared fairly by NFE.
#
# Usage:
#   bash scripts/solver_sweep_fm.sh
#
# Common overrides (environment variables):
#   CKPT={MODEL_CHECKPOINT} \
#   CONFIG=lidc.yaml \
#   TOTAL_SAMPLES=1000 \
#   BATCH_SIZE=4 \
#   bash scripts/solver_sweep_fm.sh
#
# Narrow the grid:
#   SOLVERS="rk4" STEPS_LIST="10 20" bash scripts/solver_sweep_fm.sh
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# Project configuration
# -----------------------------------------------------------------------------

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

CONFIG="${CONFIG:-lidc.yaml}"
GLOBAL_CONFIG="${GLOBAL_CONFIG:-configs/global/${CONFIG}}"

# Replace the placeholder or override it with CKPT=... on the CLI.
CKPT="${CKPT:-{MODEL_CHECKPOINT}}"

PYTHON="${PYTHON:-python}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

# -----------------------------------------------------------------------------
# Sweep configuration
# -----------------------------------------------------------------------------

# Extended Euler-only sweep to probe FID at higher step counts. Override with
# SOLVERS=... / STEPS_LIST=... to run other grids. Cells whose output already
# exists are skipped, so re-running never regenerates or clobbers prior results.
SOLVERS="${SOLVERS:-euler}"
STEPS_LIST="${STEPS_LIST:-60 80 100 150 200}"

TOTAL_SAMPLES="${TOTAL_SAMPLES:-500}"
BATCH_SIZE="${BATCH_SIZE:-4}"

OUTPUT_ROOT="${OUTPUT_ROOT:-samples/solver_sweep}"
LOG_ROOT="${LOG_ROOT:-logs/solver_sweep}"

# Stochastic-corrector strength for `pc`/`pc_heun` (ignored by other solvers).
ETA="${ETA:-0.0}"
# dt^2-normalised local-error gate for the `pc_heun` solver.
ERR_TARGET="${ERR_TARGET:-1.0}"

# Extra args forwarded verbatim to sample.py (must be supported by sample.py).
SAMPLE_EXTRA_ARGS="${SAMPLE_EXTRA_ARGS:-}"

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

log() {
    printf '\n\033[1;34m[solver-sweep]\033[0m %s\n' "$*"
}

error() {
    printf '\n\033[1;31m[solver-sweep error]\033[0m %s\n' "$*" >&2
    exit 1
}

nfe_per_step() {
    # Usage: nfe_per_step <solver>  ->  echoes function-evals per step.
    case "$1" in
        euler)   echo 1 ;;
        heun)    echo 2 ;;
        rk4)     echo 4 ;;
        pc)      echo 1 ;;
        pc_heun) echo 2 ;;
        *)       error "Unknown solver: $1 (use euler|heun|rk4|pc|pc_heun)" ;;
    esac
}

# -----------------------------------------------------------------------------
# Validate configuration
# -----------------------------------------------------------------------------

[[ -f "$GLOBAL_CONFIG" ]] || error "Global config does not exist: $GLOBAL_CONFIG"
[[ -f "sample.py" ]]      || error "sample.py was not found under: $PROJECT_ROOT"
[[ -f "$CKPT" ]]          || error "Checkpoint does not exist: $CKPT"

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"

# -----------------------------------------------------------------------------
# Print experiment configuration
# -----------------------------------------------------------------------------

log "Project root  : $PROJECT_ROOT"
log "Config        : $GLOBAL_CONFIG"
log "Checkpoint    : $CKPT"
log "Solvers       : $SOLVERS"
log "Steps         : $STEPS_LIST"
log "Total samples : $TOTAL_SAMPLES"
log "Batch size    : $BATCH_SIZE"
log "Eta (pc*)     : $ETA"
log "Err target    : $ERR_TARGET"
log "Output root   : $OUTPUT_ROOT"

# -----------------------------------------------------------------------------
# Runtime summary file
# -----------------------------------------------------------------------------

# Append-safe: keep any existing rows and only write the header once, so an
# extended sweep adds to the prior runtime_summary.csv instead of erasing it.
SUMMARY_CSV="${OUTPUT_ROOT}/runtime_summary.csv"
if [[ ! -f "$SUMMARY_CSV" ]]; then
    printf '%s\n' \
        "solver,steps,nfe,total_samples,total_seconds,seconds_per_sample,output_dir" \
        > "$SUMMARY_CSV"
fi

# -----------------------------------------------------------------------------
# Sweep
# -----------------------------------------------------------------------------

for SOLVER in $SOLVERS; do
    NFE_PER_STEP="$(nfe_per_step "$SOLVER")"

    for STEPS in $STEPS_LIST; do
        if ! [[ "$STEPS" =~ ^[0-9]+$ ]] || [[ "$STEPS" -lt 1 ]]; then
            error "Invalid sampling-step value: $STEPS"
        fi

        NFE=$((NFE_PER_STEP * STEPS))
        EXPERIMENT_NAME="${SOLVER}_steps_${STEPS}"
        OUTPUT_DIR="${OUTPUT_ROOT}/${EXPERIMENT_NAME}"
        SAMPLE_LOG="${LOG_ROOT}/sample_${EXPERIMENT_NAME}.log"

        mkdir -p "$OUTPUT_DIR"

        # Idempotent: if this cell was already generated, leave it (and its
        # existing summary row) untouched so re-runs only fill in new cells.
        if [[ -d "${OUTPUT_DIR}/xs" ]] && [[ -n "$(ls -A "${OUTPUT_DIR}/xs" 2>/dev/null)" ]]; then
            log "Skipping (already generated): $OUTPUT_DIR"
            continue
        fi

        log "============================================================"
        log "solver=$SOLVER  steps=$STEPS  NFE=$NFE"
        log "Output: $OUTPUT_DIR"
        log "============================================================"

        START_TIME="$(date +%s)"

        # shellcheck disable=SC2086
        "$PYTHON" sample.py \
            --config "$CONFIG" \
            --ckpt "$CKPT" \
            --solver "$SOLVER" \
            --eta "$ETA" \
            --err-target "$ERR_TARGET" \
            --num-samples "$BATCH_SIZE" \
            --total-samples "$TOTAL_SAMPLES" \
            --num-sampling-steps "$STEPS" \
            --output-dir "$OUTPUT_DIR" \
            $SAMPLE_EXTRA_ARGS \
            2>&1 | tee "$SAMPLE_LOG"

        END_TIME="$(date +%s)"
        TOTAL_SECONDS=$((END_TIME - START_TIME))
        SECONDS_PER_SAMPLE="$(
            awk -v t="$TOTAL_SECONDS" -v n="$TOTAL_SAMPLES" \
                'BEGIN { printf "%.6f", t / (n > 0 ? n : 1) }'
        )"

        printf '%s,%s,%s,%s,%s,%s,%s\n' \
            "$SOLVER" "$STEPS" "$NFE" "$TOTAL_SAMPLES" \
            "$TOTAL_SECONDS" "$SECONDS_PER_SAMPLE" "$OUTPUT_DIR" \
            >> "$SUMMARY_CSV"

        [[ -d "${OUTPUT_DIR}/xs" ]] || error \
            "Expected generated-volume directory was not found: ${OUTPUT_DIR}/xs"

        log "Done: solver=$SOLVER steps=$STEPS in ${TOTAL_SECONDS}s"
    done
done

# -----------------------------------------------------------------------------
# Final summary
# -----------------------------------------------------------------------------

log "All solver x step runs are complete."
log "Runtime summary: $SUMMARY_CSV"
log "Logs: $LOG_ROOT"

printf '\nTip: for a fair quality comparison, hold NFE constant across solvers\n'
printf '     (e.g. rk4@10 = 40 NFE  ~=  heun@20 = 40 NFE  ~=  euler@40 = 40 NFE).\n\n'

column -s, -t "$SUMMARY_CSV" 2>/dev/null || cat "$SUMMARY_CSV"
