#!/usr/bin/env bash
# Resume from the structure arm's calibration (the calibrators now carry the
# structure channels the checkpoint expects). Arms are trained; jd_none is done.
set -e
cd /root/lightsem
T=/mnt/lightsem-data/cache
echo "STAGE8 chain3 start $(date +%H:%M)"
for pair in "jd_struct:real" "jd_shuf:shuffled"; do
  tag=${pair%%:*}; st=${pair##*:}
  bash run_torch.sh calibrate_events.py --corpus jd --structure $st --tensor $T/train_tensor_jd_$st.npz --ckpt /mnt/lightsem-data/ckpt/light_trunk_$tag.pt --tag $tag
  bash run_torch.sh calibrate_smoothing.py --corpus jd --structure $st --tensor $T/train_tensor_jd_$st.npz --ckpt /mnt/lightsem-data/ckpt/light_trunk_$tag.pt --tag $tag
  bash run_torch.sh run_stage1_eval.py --corpus jd --tensor $T/train_tensor_jd_none.npz --split test_dancer --ckpt /mnt/lightsem-data/ckpt/light_trunk_${tag}_smooth.pt --structure $st --tag ${tag}_smooth --n 40
  bash run_torch.sh run_stage1_eval.py --corpus jd --tensor $T/train_tensor_jd_none.npz --split test_dancer --ckpt /mnt/lightsem-data/ckpt/light_trunk_$tag.pt --structure $st --tag ${tag}_raw --n 40
done
bash run_torch.sh run_stage1_eval.py --corpus jd --tensor $T/train_tensor_jd_none.npz --split test_dancer --ckpt /mnt/lightsem-data/ckpt/light_trunk_v3causal_smooth.pt --tag jd_aistonly --n 40
bash run_torch.sh train_operator_head.py --with-hazard --root /mnt/lightsem-data/cache/justdance_visual --run-tag _jdvis
echo STAGE8_DONE
