#!/usr/bin/env bash
# =============================================================================
# PRDiT Flow-Matching end-to-end pipeline
#
#   1. Stage 1 — coarse (local denoiser, depth=0) training  [--from_scratch]
#   2. Locate the latest stage-1 checkpoint and write it into the global config
#      `model.pretrained_path`
#   3. Stage 2 — global residual DiT (depth>0) training
#   4. Evaluation — generate volumes, then (optionally) 3D FID / MMD
#
# NOTE: the instruction referenced `2d/imagenet_t2i_global.yaml`; that file does
# not exist in this repo. This 3D-CT project's analog is
# `configs/global/<CONFIG>` with the `model.pretrained_path` field, which is
# what step 2 fills in.
#
# Usage:
#   bash scripts/run_flow_pipeline.sh
#
# Common overrides (env vars):
#   CONFIG=lidc.yaml NPROC=4 bash scripts/run_flow_pipeline.sh
#   RUN_COARSE=0 RUN_FINE=1 RUN_EVAL=1 bash scripts/run_flow_pipeline.sh   # resume
#   PYTHON=/path/to/env/bin/python bash scripts/run_flow_pipeline.sh
# =============================================================================
set -euo pipefail

# ---- Configuration ----------------------------------------------------------
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

CONFIG="${CONFIG:-lidc.yaml}"                 # config filename (in configs/local & configs/global)
LOCAL_CONFIG="configs/local/${CONFIG}"
GLOBAL_CONFIG="configs/global/${CONFIG}"

PYTHON="${PYTHON:-python}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

# ---- GPU auto-detection -----------------------------------------------------
# Number of visible GPUs, honouring CUDA_VISIBLE_DEVICES; falls back to
# nvidia-smi, then to 1. Override by exporting NPROC explicitly.
detect_gpus() {
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    local IFS=','; local devs=()
    read -ra devs <<< "$CUDA_VISIBLE_DEVICES"
    # Drop empty entries (e.g. trailing comma / unset-but-exported).
    local n=0; for d in "${devs[@]}"; do [[ -n "$d" ]] && n=$((n + 1)); done
    echo "$n"
  elif command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || echo 1
  else
    echo 1
  fi
}

DETECTED_GPUS="$(detect_gpus)"
[[ "$DETECTED_GPUS" -lt 1 ]] && DETECTED_GPUS=1

# Stage toggles (set to 0 to skip a stage, e.g. to resume mid-pipeline).
RUN_COARSE="${RUN_COARSE:-1}"
RUN_FINE="${RUN_FINE:-1}"
RUN_EVAL="${RUN_EVAL:-1}"

# Evaluation knobs.
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-samples/flow_eval}"
EVAL_TOTAL_SAMPLES="${EVAL_TOTAL_SAMPLES:-100}"
EVAL_NUM_SAMPLES="${EVAL_NUM_SAMPLES:-4}"
EVAL_SAMPLING_STEPS="${EVAL_SAMPLING_STEPS:-100}"   # Euler ODE steps (report NFE = steps)
# Optional 3D FID / MMD (need an external 3D ResNet-50 + real-data root).
FID_PRETRAIN_PATH="${FID_PRETRAIN_PATH:-}"
DATA_ROOT_REAL="${DATA_ROOT_REAL:-}"
EVAL_DATASET="${EVAL_DATASET:-lidc-idri}"

log() { printf '\n\033[1;34m[pipeline]\033[0m %s\n' "$*"; }

read_yaml() {  # read_yaml <file> <dotted.key>
  "$PYTHON" - "$1" "$2" <<'PY'
import sys, yaml
with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f)
node = cfg
for k in sys.argv[2].split('.'):
    node = node[k]
print(node)
PY
}

# Choose nproc for a stage: honour an explicit NPROC override, else use the
# detected GPU count, then clamp down to the largest value that still divides
# the stage's batch_size (train.py asserts batch_size % world_size == 0).
effective_nproc() {  # effective_nproc <config_file>
  local batch want
  batch="$(read_yaml "$1" training.batch_size)"
  want="${NPROC:-$DETECTED_GPUS}"
  [[ "$want" -gt "$batch" ]] && want="$batch"
  while [[ "$want" -gt 1 && $((batch % want)) -ne 0 ]]; do
    want=$((want - 1))
  done
  echo "$want"
}

# ---- Derive names from the configs (stay correct if configs are renamed) ----
RESULTS_DIR="$(read_yaml "$LOCAL_CONFIG" output.results_dir)"
COARSE_MODEL="$(read_yaml "$LOCAL_CONFIG" model.name)"      # e.g. PRDiT-B/12/0
FINE_MODEL="$(read_yaml "$GLOBAL_CONFIG" model.name)"       # e.g. PRDiT-B/12/4
IMG_SIZE="$(read_yaml "$GLOBAL_CONFIG" data.image_size)"
COARSE_TAG="${COARSE_MODEL//\//-}"                          # PRDiT-B-12-0
FINE_TAG="${FINE_MODEL//\//-}"                              # PRDiT-B-12-4

NPROC_COARSE="$(effective_nproc "$LOCAL_CONFIG")"
NPROC_FINE="$(effective_nproc "$GLOBAL_CONFIG")"

log "Project        : $PROJECT_ROOT"
log "Config         : $CONFIG  (local=$LOCAL_CONFIG, global=$GLOBAL_CONFIG)"
log "Results dir    : $RESULTS_DIR"
log "Coarse / Fine  : $COARSE_MODEL  ->  $FINE_MODEL"
if [[ -n "${NPROC:-}" ]]; then
  log "GPUs           : NPROC override=$NPROC  ->  coarse=$NPROC_COARSE, fine=$NPROC_FINE"
else
  log "GPUs           : detected=$DETECTED_GPUS  ->  coarse=$NPROC_COARSE, fine=$NPROC_FINE (clamped to divide batch_size)"
fi

# ---- 0. Ensure train/val split files exist ---------------------------------
DATA_PATH="$(read_yaml "$LOCAL_CONFIG" data.path)"
TRAIN_LIST="$(read_yaml "$LOCAL_CONFIG" data.train_list)"   # e.g. lidc_data/train.txt
VAL_LIST="$(read_yaml "$LOCAL_CONFIG" data.val_list)"
VAL_FRAC="$(read_yaml "$LOCAL_CONFIG" data.val_frac)"
SPLIT_SEED="$(read_yaml "$LOCAL_CONFIG" training.seed)"
SPLIT_DATASET="${SPLIT_DATASET:-lidc}"                      # lidc | rad | auto

if [[ -f "$TRAIN_LIST" && -f "$VAL_LIST" ]]; then
  log "Split files present: $TRAIN_LIST ($(wc -l < "$TRAIN_LIST")), $VAL_LIST ($(wc -l < "$VAL_LIST"))"
else
  log "Generating split files from $DATA_PATH (val_frac=$VAL_FRAC, seed=$SPLIT_SEED)"
  "$PYTHON" scripts/split_train_val.py "$DATA_PATH" \
    --output-dir "$(dirname "$TRAIN_LIST")" \
    --val-ratio "$VAL_FRAC" --seed "$SPLIT_SEED" --dataset "$SPLIT_DATASET"
fi

find_latest_ckpt() {  # find_latest_ckpt <results_dir> <model_tag>
  "$PYTHON" - "$1" "$2" <<'PY'
import sys, glob, os, re
results_dir, tag = sys.argv[1], sys.argv[2]
exp_dirs = sorted(glob.glob(os.path.join(results_dir, f"*-{tag}")))
if not exp_dirs:
    sys.exit(f"[find_latest_ckpt] no experiment dir matching *-{tag} under {results_dir}")
exp = exp_dirs[-1]                                  # highest sequential index = newest
ckpt_dir = os.path.join(exp, "checkpoints")
# Prefer the best-by-val checkpoint saved during depth-0 training.
best = glob.glob(os.path.join(ckpt_dir, "best_*.pt")) + glob.glob(os.path.join(ckpt_dir, "best.pt"))
if best:
    print(max(best, key=os.path.getmtime)); sys.exit(0)
steps = glob.glob(os.path.join(ckpt_dir, "[0-9]*.pt"))
if not steps:
    sys.exit(f"[find_latest_ckpt] no checkpoints in {ckpt_dir}")
step_of = lambda p: int(re.search(r'(\d+)\.pt$', os.path.basename(p)).group(1))
print(max(steps, key=step_of))
PY
}

set_pretrained_path() {  # set_pretrained_path <global_config> <ckpt>
  "$PYTHON" - "$1" "$2" <<'PY'
import sys, re
path, ckpt = sys.argv[1], sys.argv[2]
with open(path) as f:
    lines = f.readlines()
done = False
for i, ln in enumerate(lines):
    if re.match(r'^\s*pretrained_path\s*:', ln):
        indent = ln[:len(ln) - len(ln.lstrip())]
        lines[i] = f'{indent}pretrained_path: "{ckpt}"\n'
        done = True
        break
if not done:
    sys.exit("[set_pretrained_path] no `pretrained_path:` key found in " + path)
with open(path, 'w') as f:
    f.writelines(lines)
print(f"[set_pretrained_path] {path} -> pretrained_path: {ckpt}")
PY
}

# ---- 1. Coarse (stage 1) ----------------------------------------------------
if [[ "$RUN_COARSE" == "1" ]]; then
  log "STAGE 1 — coarse training (depth=0, --from_scratch) on $NPROC_COARSE GPU(s)"
  torchrun --nnodes=1 --nproc_per_node="$NPROC_COARSE" train.py --config "$CONFIG" --from_scratch
else
  log "STAGE 1 — skipped (RUN_COARSE=0)"
fi

# ---- 2. Find latest coarse weights and wire them into the global config -----
log "Locating latest stage-1 checkpoint ($COARSE_TAG) under $RESULTS_DIR"
COARSE_CKPT="$(find_latest_ckpt "$RESULTS_DIR" "$COARSE_TAG")"
log "Stage-1 checkpoint: $COARSE_CKPT"
set_pretrained_path "$GLOBAL_CONFIG" "$COARSE_CKPT"

# ---- 3. Fine (stage 2) ------------------------------------------------------
if [[ "$RUN_FINE" == "1" ]]; then
  log "STAGE 2 — global residual training (depth>0, pretrained coarse path frozen) on $NPROC_FINE GPU(s)"
  torchrun --nnodes=1 --nproc_per_node="$NPROC_FINE" train.py --config "$CONFIG"
else
  log "STAGE 2 — skipped (RUN_FINE=0)"
fi

# ---- 4. Evaluation ----------------------------------------------------------
if [[ "$RUN_EVAL" == "1" ]]; then
  log "Locating latest stage-2 checkpoint ($FINE_TAG)"
  FINE_CKPT="$(find_latest_ckpt "$RESULTS_DIR" "$FINE_TAG")"
  log "Stage-2 checkpoint: $FINE_CKPT"

  log "Generating $EVAL_TOTAL_SAMPLES volumes (Euler, steps=$EVAL_SAMPLING_STEPS, NFE=$EVAL_SAMPLING_STEPS) -> $EVAL_OUTPUT_DIR"
  "$PYTHON" sample.py \
    --config "$CONFIG" \
    --ckpt "$FINE_CKPT" \
    --num-samples "$EVAL_NUM_SAMPLES" \
    --total-samples "$EVAL_TOTAL_SAMPLES" \
    --num-sampling-steps "$EVAL_SAMPLING_STEPS" \
    --output-dir "$EVAL_OUTPUT_DIR"

  FAKE_DIR="${EVAL_OUTPUT_DIR}/xs"     # final t=1 samples
  if [[ -n "$FID_PRETRAIN_PATH" && -n "$DATA_ROOT_REAL" ]]; then
    ACT_DIR="${EVAL_OUTPUT_DIR}/activations"
    mkdir -p "$ACT_DIR"
    SPLIT_DIR="$(dirname "$TRAIN_LIST")"   # real mode resolves train.txt/val.txt from here
    log "3D FID (real=$DATA_ROOT_REAL, fake=$FAKE_DIR)"
    "$PYTHON" evaluations/fid.py \
      --dataset "$EVAL_DATASET" --img_size "$IMG_SIZE" \
      --data_root_real "$DATA_ROOT_REAL" --data_root_fake "$FAKE_DIR" \
      --split_dir "$SPLIT_DIR" \
      --pretrain_path "$FID_PRETRAIN_PATH" --path_to_activations "$ACT_DIR" \
      --num_samples "$EVAL_TOTAL_SAMPLES"
    log "3D MMD (real=$DATA_ROOT_REAL, fake=$FAKE_DIR)"
    "$PYTHON" evaluations/mmd.py \
      --dataset "$EVAL_DATASET" --img_size "$IMG_SIZE" \
      --data_root_real "$DATA_ROOT_REAL" --data_root_fake "$FAKE_DIR" \
      --split_dir "$SPLIT_DIR" \
      --pretrain_path "$FID_PRETRAIN_PATH" --path_to_activations "$ACT_DIR" \
      --num_samples "$EVAL_TOTAL_SAMPLES"
  else
    log "FID/MMD skipped — set FID_PRETRAIN_PATH (3D ResNet-50) and DATA_ROOT_REAL to enable."
    log "Generated samples are in: $FAKE_DIR"
  fi
else
  log "EVALUATION — skipped (RUN_EVAL=0)"
fi

log "Pipeline complete."
