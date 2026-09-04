#!/usr/bin/env bash
# After the stem chain: causal stem features (CPU), then the neural operator-head arms on the GPU.
cd /root/lightsem
while kill -0 $(cat logs/stems_chain.pid) 2>/dev/null; do sleep 15; done
echo "stems chain done $(date)"
bash run.sh build_stem_feats.py --mode causal --workers 6 > logs/stem_feats_causal.txt 2>&1
bash run.sh build_jdvis_stems.py > logs/build_jdvis_stems.txt 2>&1
# H-H neural headline (exp-016 recipe on Beat Saber scene events), audio-only and with hazard, in parallel
bash run_torch.sh train_operator_head.py --root /mnt/lightsem-data/cache/beatsaver_visual --event-key ev_scene --run-tag _bsaber > logs/head_bsaber_audio.txt 2>&1 &
bash run_torch.sh train_operator_head.py --root /mnt/lightsem-data/cache/beatsaver_visual --event-key ev_scene --with-hazard --run-tag _bsaber > logs/head_bsaber_haz.txt 2>&1 &
wait
# H-G neural arm: JD visual with stems appended, vs exp-016 (operator_head_haz_jdvis, lift 3.17x)
bash run_torch.sh train_operator_head.py --root /mnt/lightsem-data/cache/justdance_visual_stems --with-hazard --run-tag _jdvis_stems > logs/head_jdvis_stems_haz.txt 2>&1 &
bash run_torch.sh train_operator_head.py --root /mnt/lightsem-data/cache/justdance_visual_stems --run-tag _jdvis_stems > logs/head_jdvis_stems_audio.txt 2>&1 &
wait
echo "GPU_CHAIN_DONE $(date)"
