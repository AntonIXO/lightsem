#!/usr/bin/env bash
# Stage 8 chain, fired when the pose extraction finishes: caches, the audit,
# the three tensors, the H-C arms (none / real / shuffled structure), the
# evaluations on held-out songs, and the AIST-only model scored on the same
# held-out songs as the 8.5 baseline. Every step is idempotent.
set -e
cd /root/lightsem
until grep -q "^done .* files in" logs_jd_pose.txt; do sleep 60; done
echo "STAGE8: extraction finished $(date +%H:%M)"
bash run.sh build_justdance_motion.py
bash run.sh build_justdance_visual.py --workers 4
bash run.sh run_justdance_audit.py --tag jd_full
for st in none real shuffled; do
  bash run.sh -m lightsem.model.jd_data --structure $st --out /mnt/lightsem-data/cache/train_tensor_jd_$st.npz
done
T=/mnt/lightsem-data/cache
DROP=hip_yaw_speed,footfall_rate
bash run_torch.sh train_light.py --v2 --steps 12000 --tag jd_none --tensor $T/train_tensor_jd_none.npz --drop-a $DROP
bash run_torch.sh train_light.py --v2 --steps 12000 --tag jd_struct --tensor $T/train_tensor_jd_real.npz --drop-a $DROP
bash run_torch.sh train_light.py --v2 --steps 12000 --tag jd_shuf --tensor $T/train_tensor_jd_shuffled.npz --drop-a $DROP
# per arm: event thresholds and the H-B smoothing constants, fitted on the val songs
# with the arm's own structure channels; then the smoothed checkpoint is scored
for pair in "jd_none:none" "jd_struct:real" "jd_shuf:shuffled"; do
  tag=${pair%%:*}; st=${pair##*:}
  bash run_torch.sh calibrate_events.py --corpus jd --tensor $T/train_tensor_jd_$st.npz --ckpt /mnt/lightsem-data/ckpt/light_trunk_$tag.pt --tag $tag
  bash run_torch.sh calibrate_smoothing.py --corpus jd --tensor $T/train_tensor_jd_$st.npz --ckpt /mnt/lightsem-data/ckpt/light_trunk_$tag.pt --tag $tag
  bash run_torch.sh run_stage1_eval.py --corpus jd --tensor $T/train_tensor_jd_none.npz --split test_dancer --ckpt /mnt/lightsem-data/ckpt/light_trunk_${tag}_smooth.pt --structure $st --tag ${tag}_smooth --n 40
  bash run_torch.sh run_stage1_eval.py --corpus jd --tensor $T/train_tensor_jd_none.npz --split test_dancer --ckpt /mnt/lightsem-data/ckpt/light_trunk_$tag.pt --structure $st --tag ${tag}_raw --n 40
done
bash run_torch.sh run_stage1_eval.py --corpus jd --tensor $T/train_tensor_jd_none.npz --split test_dancer --ckpt /mnt/lightsem-data/ckpt/light_trunk_v3causal_smooth.pt --tag jd_aistonly --n 40
# H-F: the background as a designed show, operator-head test on the visual cache
bash run_torch.sh train_operator_head.py --with-hazard --root /mnt/lightsem-data/cache/justdance_visual --run-tag _jdvis
echo STAGE8_DONE
