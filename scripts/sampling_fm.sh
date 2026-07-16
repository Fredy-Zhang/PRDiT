#!/usr/bin/env bash
# =============================================================================
# Flow-Matching Sampling-Step Evaluation
#
# Evaluates a fixed trained PRDiT checkpoint using different Euler ODE steps.
#
# Default step settings:
#   10  - fast / lower-quality reference
#   20  - practical fast sampling
#   30  - recommended default
#   50  - higher-quality setting
#   100 - convergence / saturation reference
#
# Usage:
#   bash scripts/evaluate_flow_steps.sh
#
# Common overrides:
#   CKPT=/path/to/model.pt bash scripts/evaluate_flow_steps.sh
#
#   STEPS_LIST="10 20 30 50" \
#   TOTAL_SAMPLES=1000 \
#   BATCH_SIZE=4 \
#   bash scripts/evaluate_flow_steps.sh
#
# Enable FID and MMD:
#   FID_PRETRAIN_PATH=/path/to/resnet_50_23dataset.pth \
#   DATA_ROOT_REAL=/path/to/LIDC-IDRI \
#   bash scripts/evaluate_flow_steps.sh
#
# Use one step setting only:
#   STEPS_LIST="30" bash scripts/evaluate_flow_steps.sh
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# Project configuration
# -----------------------------------------------------------------------------

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

CONFIG="${CONFIG:-lidc.yaml}"
GLOBAL_CONFIG="${GLOBAL_CONFIG:-configs/global/${CONFIG}}"

PYTHON="${PYTHON:-python}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

# -----------------------------------------------------------------------------
# Sampling configuration
# -----------------------------------------------------------------------------

# Recommended Euler flow-matching evaluation grid.
STEPS_LIST="${STEPS_LIST:-10 20 30 50 100}"

TOTAL_SAMPLES="${TOTAL_SAMPLES:-1000}"
BATCH_SIZE="${BATCH_SIZE:-4}"

OUTPUT_ROOT="${OUTPUT_ROOT:-samples/flow_step_ablation}"
LOG_ROOT="${LOG_ROOT:-logs/flow_step_ablation}"

# Optional additional arguments passed directly to sample.py.
#
# Example:
#   SAMPLE_EXTRA_ARGS="--seed 42"
#
# Only use arguments that your sample.py supports.
SAMPLE_EXTRA_ARGS="${SAMPLE_EXTRA_ARGS:-}"

# -----------------------------------------------------------------------------
# Optional metric configuration
# -----------------------------------------------------------------------------

FID_PRETRAIN_PATH="${FID_PRETRAIN_PATH:-}"
DATA_ROOT_REAL="${DATA_ROOT_REAL:-}"

EVAL_DATASET="${EVAL_DATASET:-lidc-idri}"
NUM_METRIC_SAMPLES="${NUM_METRIC_SAMPLES:-$TOTAL_SAMPLES}"

# Set RUN_METRICS=0 to explicitly disable FID/MMD even when paths are supplied.
RUN_METRICS="${RUN_METRICS:-auto}"

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

log() {
    printf '\n\033[1;34m[flow-eval]\033[0m %s\n' "$*"
}

error() {
    printf '\n\033[1;31m[flow-eval error]\033[0m %s\n' "$*" >&2
    exit 1
}

read_yaml() {
    # Usage:
    #   read_yaml <yaml-file> <dotted.key>

    "$PYTHON" - "$1" "$2" <<'PY'
import sys
import yaml

yaml_path = sys.argv[1]
key_path = sys.argv[2]

with open(yaml_path, "r") as f:
    config = yaml.safe_load(f)

value = config

for key in key_path.split("."):
    if not isinstance(value, dict) or key not in value:
        raise KeyError(
            f"Could not find '{key_path}' in '{yaml_path}'. "
            f"Missing component: '{key}'"
        )

    value = value[key]

print(value)
PY
}

find_latest_checkpoint() {
    # Usage:
    #   find_latest_checkpoint <results-dir> <model-tag>

    "$PYTHON" - "$1" "$2" <<'PY'
import glob
import os
import re
import sys

results_dir = sys.argv[1]
model_tag = sys.argv[2]

experiment_dirs = glob.glob(
    os.path.join(results_dir, f"*-{model_tag}")
)

if not experiment_dirs:
    raise SystemExit(
        f"No experiment directory matching '*-{model_tag}' "
        f"was found under '{results_dir}'."
    )


def experiment_index(path):
    name = os.path.basename(path)

    match = re.match(r"^(\d+)-", name)

    if match:
        return int(match.group(1))

    return -1


experiment_dir = max(
    experiment_dirs,
    key=lambda path: (experiment_index(path), os.path.getmtime(path)),
)

checkpoint_dir = os.path.join(experiment_dir, "checkpoints")

best_checkpoints = (
    glob.glob(os.path.join(checkpoint_dir, "best_*.pt"))
    + glob.glob(os.path.join(checkpoint_dir, "best.pt"))
)

if best_checkpoints:
    print(max(best_checkpoints, key=os.path.getmtime))
    raise SystemExit(0)

step_checkpoints = glob.glob(
    os.path.join(checkpoint_dir, "[0-9]*.pt")
)

if not step_checkpoints:
    raise SystemExit(
        f"No checkpoints were found in '{checkpoint_dir}'."
    )


def checkpoint_step(path):
    filename = os.path.basename(path)
    match = re.search(r"(\d+)\.pt$", filename)

    if not match:
        return -1

    return int(match.group(1))


print(max(step_checkpoints, key=checkpoint_step))
PY
}

calculate_runtime_values() {
    # Usage:
    #   calculate_runtime_values <total-seconds> <num-samples>

    "$PYTHON" - "$1" "$2" <<'PY'
import sys

total_seconds = float(sys.argv[1])
num_samples = int(sys.argv[2])

seconds_per_sample = total_seconds / max(num_samples, 1)

print(f"{total_seconds:.4f},{seconds_per_sample:.6f}")
PY
}

# -----------------------------------------------------------------------------
# Validate configuration
# -----------------------------------------------------------------------------

[[ -f "$GLOBAL_CONFIG" ]] || error \
    "Global config does not exist: $GLOBAL_CONFIG"

[[ -f "sample.py" ]] || error \
    "sample.py was not found under: $PROJECT_ROOT"

[[ -f "evaluations/fid.py" ]] || log \
    "Warning: evaluations/fid.py was not found."

[[ -f "evaluations/mmd.py" ]] || log \
    "Warning: evaluations/mmd.py was not found."

mkdir -p "$OUTPUT_ROOT"
mkdir -p "$LOG_ROOT"

# -----------------------------------------------------------------------------
# Derive model information
# -----------------------------------------------------------------------------

RESULTS_DIR="$(read_yaml "$GLOBAL_CONFIG" output.results_dir)"
MODEL_NAME="$(read_yaml "$GLOBAL_CONFIG" model.name)"
IMAGE_SIZE="$(read_yaml "$GLOBAL_CONFIG" data.image_size)"
TRAIN_LIST="$(read_yaml "$GLOBAL_CONFIG" data.train_list)"

MODEL_TAG="${MODEL_NAME//\//-}"
SPLIT_DIR="$(dirname "$TRAIN_LIST")"

# Use CKPT supplied by the user. Otherwise locate the newest fine-stage model.
if [[ -n "${CKPT:-}" ]]; then
    FIXED_CKPT="$CKPT"
else
    log "No CKPT override supplied."
    log "Locating latest checkpoint for model: $MODEL_NAME"

    FIXED_CKPT="$(find_latest_checkpoint "$RESULTS_DIR" "$MODEL_TAG")"
fi

[[ -f "$FIXED_CKPT" ]] || error \
    "Checkpoint does not exist: $FIXED_CKPT"

# Decide whether metric evaluation should run.
if [[ "$RUN_METRICS" == "auto" ]]; then
    if [[ -n "$FID_PRETRAIN_PATH" && -n "$DATA_ROOT_REAL" ]]; then
        RUN_METRICS=1
    else
        RUN_METRICS=0
    fi
fi

if [[ "$RUN_METRICS" == "1" ]]; then
    [[ -f "$FID_PRETRAIN_PATH" ]] || error \
        "FID checkpoint does not exist: $FID_PRETRAIN_PATH"

    [[ -d "$DATA_ROOT_REAL" ]] || error \
        "Real-data directory does not exist: $DATA_ROOT_REAL"
fi

# -----------------------------------------------------------------------------
# Print experiment configuration
# -----------------------------------------------------------------------------

log "Project root       : $PROJECT_ROOT"
log "Config             : $GLOBAL_CONFIG"
log "Model              : $MODEL_NAME"
log "Checkpoint         : $FIXED_CKPT"
log "Image size         : $IMAGE_SIZE"
log "Sampling steps     : $STEPS_LIST"
log "Total samples      : $TOTAL_SAMPLES"
log "Generation batch   : $BATCH_SIZE"
log "Output root        : $OUTPUT_ROOT"
log "Metrics enabled    : $RUN_METRICS"

if [[ "$RUN_METRICS" == "1" ]]; then
    log "Real-data root     : $DATA_ROOT_REAL"
    log "FID model          : $FID_PRETRAIN_PATH"
    log "Metric samples     : $NUM_METRIC_SAMPLES"
fi

# -----------------------------------------------------------------------------
# Runtime summary file
# -----------------------------------------------------------------------------

SUMMARY_CSV="${OUTPUT_ROOT}/runtime_summary.csv"

printf '%s\n' \
    "steps,nfe,total_samples,total_seconds,seconds_per_sample,output_dir" \
    > "$SUMMARY_CSV"

# -----------------------------------------------------------------------------
# Evaluate each Euler step count
# -----------------------------------------------------------------------------

for STEPS in $STEPS_LIST; do
    if ! [[ "$STEPS" =~ ^[0-9]+$ ]] || [[ "$STEPS" -lt 1 ]]; then
        error "Invalid sampling-step value: $STEPS"
    fi

    EXPERIMENT_NAME="steps_${STEPS}"
    OUTPUT_DIR="${OUTPUT_ROOT}/${EXPERIMENT_NAME}"
    SAMPLE_LOG="${LOG_ROOT}/sample_${EXPERIMENT_NAME}.log"

    mkdir -p "$OUTPUT_DIR"

    log "============================================================"
    log "Sampling with $STEPS Euler steps"
    log "NFE: $STEPS"
    log "Output: $OUTPUT_DIR"
    log "============================================================"

    START_TIME="$(date +%s)"

    # shellcheck disable=SC2086
    "$PYTHON" sample.py \
        --config "$CONFIG" \
        --ckpt "$FIXED_CKPT" \
        --num-samples "$BATCH_SIZE" \
        --total-samples "$TOTAL_SAMPLES" \
        --num-sampling-steps "$STEPS" \
        --output-dir "$OUTPUT_DIR" \
        $SAMPLE_EXTRA_ARGS \
        2>&1 | tee "$SAMPLE_LOG"

    END_TIME="$(date +%s)"
    TOTAL_SECONDS=$((END_TIME - START_TIME))

    RUNTIME_VALUES="$(
        calculate_runtime_values "$TOTAL_SECONDS" "$TOTAL_SAMPLES"
    )"

    printf '%s,%s,%s,%s,%s\n' \
        "$STEPS" \
        "$STEPS" \
        "$TOTAL_SAMPLES" \
        "$RUNTIME_VALUES" \
        "$OUTPUT_DIR" \
        >> "$SUMMARY_CSV"

    FAKE_DIR="${OUTPUT_DIR}/xs"

    if [[ ! -d "$FAKE_DIR" ]]; then
        error \
            "Expected generated-volume directory was not found: $FAKE_DIR"
    fi

    log "Generation complete"
    log "Total time: ${TOTAL_SECONDS} seconds"
    log "Generated volumes: $FAKE_DIR"

    # -------------------------------------------------------------------------
    # Optional FID and MMD
    # -------------------------------------------------------------------------

    if [[ "$RUN_METRICS" == "1" ]]; then
        ACTIVATION_DIR="${OUTPUT_DIR}/activations"

        FID_LOG="${LOG_ROOT}/fid_${EXPERIMENT_NAME}.log"
        MMD_LOG="${LOG_ROOT}/mmd_${EXPERIMENT_NAME}.log"

        mkdir -p "$ACTIVATION_DIR"

        log "Calculating 3D FID for $STEPS steps"

        "$PYTHON" evaluations/fid.py \
            --dataset "$EVAL_DATASET" \
            --img_size "$IMAGE_SIZE" \
            --data_root_real "$DATA_ROOT_REAL" \
            --data_root_fake "$FAKE_DIR" \
            --split_dir "$SPLIT_DIR" \
            --pretrain_path "$FID_PRETRAIN_PATH" \
            --path_to_activations "$ACTIVATION_DIR" \
            --num_samples "$NUM_METRIC_SAMPLES" \
            2>&1 | tee "$FID_LOG"

        log "Calculating 3D MMD for $STEPS steps"

        "$PYTHON" evaluations/mmd.py \
            --dataset "$EVAL_DATASET" \
            --img_size "$IMAGE_SIZE" \
            --data_root_real "$DATA_ROOT_REAL" \
            --data_root_fake "$FAKE_DIR" \
            --split_dir "$SPLIT_DIR" \
            --pretrain_path "$FID_PRETRAIN_PATH" \
            --path_to_activations "$ACTIVATION_DIR" \
            --num_samples "$NUM_METRIC_SAMPLES" \
            2>&1 | tee "$MMD_LOG"
    fi
done

# -----------------------------------------------------------------------------
# Final summary
# -----------------------------------------------------------------------------

log "All sampling-step evaluations are complete."
log "Runtime summary: $SUMMARY_CSV"
log "Generation and metric logs: $LOG_ROOT"

printf '\nRecommended interpretation:\n'
printf '  10 steps  : speed-oriented lower bound\n'
printf '  20 steps  : fast practical setting\n'
printf '  30 steps  : recommended default\n'
printf '  50 steps  : higher-quality setting\n'
printf '  100 steps : convergence/saturation reference\n\n'

column -s, -t "$SUMMARY_CSV" 2>/dev/null || cat "$SUMMARY_CSV"
