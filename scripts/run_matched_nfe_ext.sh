#!/usr/bin/env bash
# =============================================================================
# Matched-NFE extension: Euler vs RK4 at true NFE 240 / 300 / 400
#
# Drives the three existing sweep scripts end-to-end for the matched-NFE
# extension, in order:
#
#   1. solver_sweep_fm.sh   -- generate the FID sample sets
#   2. fid_solver_sweep.sh  -- score them (merges into fid_summary.csv)
#   3. time_solver_sweep.sh -- median-of-3 runtimes (throwaway samples)
#
# The matched-NFE mapping (1 NFE/step for Euler, 4 NFE/step for RK4):
#
#   True NFE | Euler steps | RK4 steps
#   ---------+-------------+----------
#      240   |     240     |    60
#      300   |     300     |    75
#      400   |     400     |   100
#
# All three underlying scripts are idempotent (already-generated / already-timed
# cells are skipped) and merge into the existing CSVs, so this is safe to re-run.
#
# This launches GPU sampling jobs -- submit it through Slurm rather than running
# it on the login node.
#
# Usage:
#   bash scripts/run_matched_nfe_ext.sh
#
# Common overrides (forwarded to the underlying scripts via the environment):
#   CKPT={MODEL_CHECKPOINT} \
#   CONFIG=lidc.yaml \
#   TOTAL_SAMPLES=500 \
#   bash scripts/run_matched_nfe_ext.sh
#
# Run only some stages (any non-empty value enables a stage):
#   DO_GENERATE=1 DO_FID= DO_TIME= bash scripts/run_matched_nfe_ext.sh
# =============================================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# ---------------------------------------------------------------------------
# Pin the last visible GPU. Respects an explicit CUDA_VISIBLE_DEVICES if the
# caller (or Slurm) already set one; otherwise selects the highest GPU index
# reported by nvidia-smi so all sub-scripts inherit the same single device.
# Override with GPU_INDEX=<n> to force a specific one.
# ---------------------------------------------------------------------------
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    if [[ -n "${GPU_INDEX:-}" ]]; then
        export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
    elif command -v nvidia-smi >/dev/null 2>&1; then
        LAST_GPU="$(nvidia-smi --query-gpu=index --format=csv,noheader | tail -n 1 | tr -d ' ')"
        [[ -n "$LAST_GPU" ]] && export CUDA_VISIBLE_DEVICES="$LAST_GPU"
    fi
fi

# Matched-NFE step lists.
EULER_STEPS="${EULER_STEPS:-240 300 400}"
RK4_STEPS="${RK4_STEPS:-60 75 100}"

# Stage toggles (default: run all three). Set to empty to skip a stage.
DO_GENERATE="${DO_GENERATE:-1}"
DO_FID="${DO_FID:-1}"
DO_TIME="${DO_TIME:-1}"

log() { printf '\n\033[1;35m[matched-nfe]\033[0m %s\n' "$*"; }

log "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

# ---------------------------------------------------------------------------
# 1. Generate FID sample sets at the new budgets
# ---------------------------------------------------------------------------
if [[ -n "$DO_GENERATE" ]]; then
    log "Stage 1/3: generating sample sets (Euler ${EULER_STEPS} | RK4 ${RK4_STEPS})"
    SOLVERS="euler" STEPS_LIST="$EULER_STEPS" bash scripts/solver_sweep_fm.sh
    SOLVERS="rk4"   STEPS_LIST="$RK4_STEPS"   bash scripts/solver_sweep_fm.sh
fi

# ---------------------------------------------------------------------------
# 2. Score them (merges into fid_summary.csv)
# ---------------------------------------------------------------------------
if [[ -n "$DO_FID" ]]; then
    log "Stage 2/3: computing 3D FID (merges into fid_summary.csv)"
    SOLVERS="euler" STEPS_LIST="$EULER_STEPS" bash scripts/fid_solver_sweep.sh
    SOLVERS="rk4"   STEPS_LIST="$RK4_STEPS"   bash scripts/fid_solver_sweep.sh
fi

# ---------------------------------------------------------------------------
# 3. Median-of-3 runtimes (throwaway samples, won't touch FID outputs)
# ---------------------------------------------------------------------------
if [[ -n "$DO_TIME" ]]; then
    log "Stage 3/3: median-of-3 runtime timing (throwaway samples)"
    SOLVERS="euler" STEPS_LIST="$EULER_STEPS" bash scripts/time_solver_sweep.sh
    SOLVERS="rk4"   STEPS_LIST="$RK4_STEPS"   bash scripts/time_solver_sweep.sh
fi

log "Matched-NFE extension complete."
log "Next: python evaluations/plot_solver_sweep.py --solvers euler rk4 --fit"
