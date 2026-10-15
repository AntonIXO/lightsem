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
# The NTFS-backed venv has a non-resolving Python symlink in this mount.  The
# fixed launcher uses the same Torch site-packages with the host uv Python,
# leaving the host NVIDIA driver/CUDA, ffmpeg, and EGL/OpenGL libraries in use.
TORCH_ENV="${TORCH_ENV:-/mnt/lightsem-data/venv-torch-fixed}"
# A missing data mount must not silently select the system Python.
if [[ ! -x "$TORCH_ENV/bin/python" ]]; then
    printf 'LightSem Torch environment is unavailable: %s\nRestore the data mount or set TORCH_ENV to a dedicated LightSem environment.\n' "$TORCH_ENV" >&2
    exit 1
fi
export PATH="$TORCH_ENV/bin:$PATH"
export PYTHONPATH="/root/lightsem:$PYTHONPATH"
export UV_CACHE_DIR=/mnt/lightsem-data/uv-cache
exec python "$@"
