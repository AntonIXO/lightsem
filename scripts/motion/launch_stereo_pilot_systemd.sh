#!/usr/bin/env bash
# Start the stereo pilot behind the shared GPU lock and retain the unit after
# completion so a curator or a later Codex turn can inspect its final state.
set -euo pipefail

cd /root/lightsem
ARTIFACT_ROOT="${STEREO_PILOT_ROOT:-/mnt/lightsem-data/experiments/lightsem/audio_stereo_spatial_20261007_pilot_v1}"
MANIFEST="${STEREO_PILOT_MANIFEST:-$ARTIFACT_ROOT/STEREO_FEATURE_MANIFEST.json}"
LOG="$ARTIFACT_ROOT/logs/TRAINING_LAUNCH.log"
UNIT="${STEREO_PILOT_UNIT:-lightsem-stereo-pilot}"

mkdir -p "$ARTIFACT_ROOT/logs"
exec systemd-run --user --unit="$UNIT" --remain-after-exit \
  --working-directory=/root/lightsem \
  --setenv=TORCH_ENV=/mnt/lightsem-data/venv-torch-fixed \
  --property="StandardOutput=append:$LOG" \
  --property="StandardError=append:$LOG" \
  bash run_torch.sh scripts/motion/run_stereo_pilot.py \
    --stage train --manifest "$MANIFEST" --out "$ARTIFACT_ROOT" \
    --device cuda --steps 3000 --batch-size 16 --split train \
    --wait-for-gpu --max-wait-hours 24
