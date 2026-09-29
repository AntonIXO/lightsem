#!/usr/bin/env bash
# H-B: three arms on identical descriptors from the SAME causal checkpoint.
set -e
cd /root/lightsem
CK=/mnt/lightsem-data/ckpt/light_trunk_v3causal.pt
bash run_torch.sh scripts/legacy_stages/run_stage1_eval.py --ckpt $CK --no-smoothing --tag v3hb_raw --n 40
bash run_torch.sh scripts/training/calibrate_smoothing.py --ckpt $CK --tag v3causal
bash run_torch.sh scripts/legacy_stages/run_stage1_eval.py --ckpt /mnt/lightsem-data/ckpt/light_trunk_v3causal_smooth.pt --tag v3hb_ema --n 40
bash run_torch.sh scripts/legacy_stages/run_stage1_eval.py --ckpt $CK --no-smoothing --zero-phase 12 --tag v3hb_zp12 --n 40
echo HB_DONE
