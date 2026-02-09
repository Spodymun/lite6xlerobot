#!/usr/bin/env bash
set -euo pipefail

cd ~/src/lite6xlerobot

# -------------------------
# Gemeinsame Einstellungen
# -------------------------
export CUDA_VISIBLE_DEVICES=""
export HF_HUB_OFFLINE=1
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export LEROBOT_TASK_JITTER_MODE=uniform
export LEROBOT_TASK_JITTER_SEED=0

CONFIG="configs/act_env_smoke.yaml"
DATASET="records/throws/fs_2026-02-09_ALL_THROWS_COMBINED"

# Save alle 10k Steps
SAVE_FREQ=10000
LOG_FREQ=50

# -------------------------
# Helper: neuesten Checkpoint finden
# -------------------------
latest_checkpoint () {
  local outdir="$1"
  if [[ ! -d "$outdir/checkpoints" ]]; then
    echo ""
    return
  fi
  ls -d "$outdir"/checkpoints/* 2>/dev/null | sort -V | tail -n 1
}

# -------------------------
# Helper: letzte Output-Dir finden
# -------------------------
latest_output_dir () {
  ls -d outputs/train/*/*_gym_manipulator_act 2>/dev/null | sort -V | tail -n 1
}

run_phase () {
  local phase_name="$1"
  local steps="$2"
  local jitter_mm="$3"     # "" => unset
  local pretrained="$4"   # "" => none

  echo "=============================="
  echo "${phase_name}"
  echo "steps=${steps} save_freq=${SAVE_FREQ} log_freq=${LOG_FREQ} jitter=${jitter_mm:-OFF}"
  echo "=============================="

  if [[ -n "$jitter_mm" ]]; then
    export LEROBOT_TASK_JITTER_MM="$jitter_mm"
  else
    unset LEROBOT_TASK_JITTER_MM || true
  fi

  if [[ -n "$pretrained" ]]; then
    python3 tools/train_with_jitter.py \
      --config_path "$CONFIG" \
      --dataset.root "$DATASET" \
      --steps "$steps" \
      --log_freq "$LOG_FREQ" \
      --save_checkpoint true \
      --save_freq "$SAVE_FREQ" \
      --policy.pretrained_path "$pretrained"
  else
    python3 tools/train_with_jitter.py \
      --config_path "$CONFIG" \
      --dataset.root "$DATASET" \
      --steps "$steps" \
      --log_freq "$LOG_FREQ" \
      --save_checkpoint true \
      --save_freq "$SAVE_FREQ"
  fi
}

# -------------------------
# PHASE 1
# -------------------------
run_phase "PHASE 1: Interpolation (±3mm)" 200000 3 ""

OUT1="$(latest_output_dir)"
echo "[INFO] Phase 1 output dir: $OUT1"
CKPT1="$(latest_checkpoint "$OUT1")"
echo "[INFO] Phase 1 latest checkpoint: $CKPT1"
if [[ -z "$CKPT1" ]]; then
  echo "[ERROR] Kein Checkpoint nach Phase 1 gefunden."
  exit 1
fi

# -------------------------
# PHASE 2
# -------------------------
run_phase "PHASE 2: Precision (±1mm)" 50000 1 "$CKPT1/pretrained_model"

OUT2="$(latest_output_dir)"
echo "[INFO] Phase 2 output dir: $OUT2"
CKPT2="$(latest_checkpoint "$OUT2")"
echo "[INFO] Phase 2 latest checkpoint: $CKPT2"
if [[ -z "$CKPT2" ]]; then
  echo "[ERROR] Kein Checkpoint nach Phase 2 gefunden."
  exit 1
fi

# -------------------------
# PHASE 3
# -------------------------
run_phase "PHASE 3: Final (no jitter)" 30000 "" "$CKPT2/pretrained_model"

OUT3="$(latest_output_dir)"
echo "[INFO] Phase 3 output dir: $OUT3"
CKPT3="$(latest_checkpoint "$OUT3")"
echo "[INFO] Phase 3 latest checkpoint: $CKPT3"

echo "=============================="
echo "ALL PHASES FINISHED SUCCESSFULLY"
echo "Final output dir: $OUT3"
echo "Final checkpoint: $CKPT3"
echo "=============================="
