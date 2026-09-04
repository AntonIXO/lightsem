#!/usr/bin/env bash
# LightSem torch runner — DEDICATED venv on /mnt (replaces run_gpu.sh, which
# borrowed /root/optiHealth-EiV/.venv and coupled two unrelated projects).
#
#   bash run_torch.sh train_light.py --epochs 30
#
# Never invoke the venv python binary directly from a tool shell: it trips an
# "embedded null byte" error in this environment. Always go through this
# wrapper (PATH-based dispatch).
set -e
cd /root/lightsem
TORCH_ENV="${TORCH_ENV:-/mnt/lightsem-data/venv-torch}"
export PATH="$TORCH_ENV/bin:$PATH"
export PYTHONPATH="/root/lightsem:$PYTHONPATH"
export UV_CACHE_DIR=/mnt/lightsem-data/uv-cache
exec python "$@"
