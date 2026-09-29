#!/usr/bin/env bash
# Resumed Stage 8 chain (the first died of memory during arm 1's evaluation
# while the extraction still held 12 GB). Arm 1 is evaluated from its saved
# checkpoint; the rest is the original chain from the second arm on.
set -e
cd /root/lightsem
T=/mnt/lightsem-data/cache
DROP=hip_yaw_speed,footfall_rate
echo "STAGE8 chain2 start $(date +%H:%M)"
if [ -f /mnt/lightsem-data/ckpt/light_trunk_jd_none.pt ]; then
  bash run_torch.sh scripts/training/train_light.py --v2 --tag jd_none --tensor $T/train_tensor_jd_none.npz --drop-a $DROP --eval-only /mnt/lightsem-data/ckpt/light_trunk_jd_none.pt
else
  bash run_torch.sh scripts/training/train_light.py --v2 --steps 12000 --tag jd_none --tensor $T/train_tensor_jd_none.npz --drop-a $DROP
fi
bash run_torch.sh scripts/training/train_light.py --v2 --steps 12000 --tag jd_struct --tensor $T/train_tensor_jd_real.npz --drop-a $DROP
bash run_torch.sh scripts/training/train_light.py --v2 --steps 12000 --tag jd_shuf --tensor $T/train_tensor_jd_shuffled.npz --drop-a $DROP
for pair in "jd_none:none" "jd_struct:real" "jd_shuf:shuffled"; do
  tag=${pair%%:*}; st=${pair##*:}
  bash run_torch.sh scripts/training/calibrate_events.py --corpus jd --tensor $T/train_tensor_jd_$st.npz --ckpt /mnt/lightsem-data/ckpt/light_trunk_$tag.pt --tag $tag
  bash run_torch.sh scripts/training/calibrate_smoothing.py --corpus jd --tensor $T/train_tensor_jd_$st.npz --ckpt /mnt/lightsem-data/ckpt/light_trunk_$tag.pt --tag $tag
  bash run_torch.sh scripts/legacy_stages/run_stage1_eval.py --corpus jd --tensor $T/train_tensor_jd_none.npz --split test_dancer --ckpt /mnt/lightsem-data/ckpt/light_trunk_${tag}_smooth.pt --structure $st --tag ${tag}_smooth --n 40
  bash run_torch.sh scripts/legacy_stages/run_stage1_eval.py --corpus jd --tensor $T/train_tensor_jd_none.npz --split test_dancer --ckpt /mnt/lightsem-data/ckpt/light_trunk_$tag.pt --structure $st --tag ${tag}_raw --n 40
done
bash run_torch.sh scripts/legacy_stages/run_stage1_eval.py --corpus jd --tensor $T/train_tensor_jd_none.npz --split test_dancer --ckpt /mnt/lightsem-data/ckpt/light_trunk_v3causal_smooth.pt --tag jd_aistonly --n 40
bash run_torch.sh scripts/training/train_operator_head.py --with-hazard --root /mnt/lightsem-data/cache/justdance_visual --run-tag _jdvis
echo STAGE8_DONE
