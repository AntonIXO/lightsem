#!/usr/bin/env bash
# LightSem GPU runner.
#
# WHY IT POINTS AT ANOTHER DIRECTORY: this is purely a disk-space measure on
# this one machine. A working torch 2.11.0+cu130 with sm_120 support for the
# RTX 5070 already exists in another project's venv (1.2 GB torch + 2.7 GB
# nvidia wheels). That venv is python 3.11 and lightsem/.venv is 3.12, so the
# compiled wheels are ABI-incompatible and cannot be symlinked; running from
# that interpreter with PYTHONPATH set here reuses all 3.9 GB instead of
# installing a second copy. Nothing about LightSem's design depends on it --
# swap TORCH_ENV for any python 3.11 env with torch and this still works.
#
# CPU-only / rendering work keeps using ./run.sh (python 3.12 venv).
# Use THIS script for anything that touches torch.
#
# usage: bash scripts/legacy_stages/run_gpu.sh scripts/training/train_macro.py --epochs 30
set -e
cd /root/lightsem
TORCH_ENV="${TORCH_ENV:-/root/optiHealth-EiV/.venv}"
export PATH="$TORCH_ENV/bin:$PATH"
export PYTHONPATH="/root/lightsem:$PYTHONPATH"
exec python "$@"
