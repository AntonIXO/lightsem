#!/usr/bin/env bash
# H-A evaluation once training has written its JSON: raw offline eval on the
# offline bus, then the H-B smoothing fitted on the offline model, then the
# smoothed eval, so both raw and smoothed offline numbers sit next to their
# L=0 counterparts (v3causal raw / v3hb_ema).
set -e
cd /root/lightsem
until grep -q "causality leak" logs_train_v4offline.txt; do sleep 10; done
CK=/mnt/lightsem-data/ckpt/light_trunk_v4offline.pt
AD=/mnt/lightsem-data/cache/audio_offline
bash run_torch.sh calibrate_events.py --ckpt $CK --tag v4offline --audio-dir $AD
bash run_torch.sh run_stage1_eval.py --ckpt $CK --audio-dir $AD --tag v4offline --n 40
bash run_torch.sh calibrate_smoothing.py --ckpt $CK --tag v4offline --audio-dir $AD
bash run_torch.sh run_stage1_eval.py --ckpt /mnt/lightsem-data/ckpt/light_trunk_v4offline_smooth.pt --audio-dir $AD --tag v4offline_smooth --n 40
echo HA_EVAL_DONE
