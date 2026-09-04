#!/usr/bin/env bash
# v2 decoder (sampled look) training, then the fixed G9.3 evals for v1 and v2 and the G9.4 planner eval on v2.
cd /root/lightsem
bash run_torch.sh train_group_decoder.py --flow --steps 6000 --corpora rpmc,jd --run-tag v2 > logs/decoder_v2.txt 2>&1
bash run_torch.sh run_stage9_decoder_eval.py --n 60 --tag v1 --ckpt /mnt/lightsem-data/ckpt/group_decoder_v1.pt > logs/decoder_eval_v1.txt 2>&1 &
bash run_torch.sh run_stage9_decoder_eval.py --n 60 --tag v2 --ckpt /mnt/lightsem-data/ckpt/group_decoder_v2.pt > logs/decoder_eval_v2.txt 2>&1 &
wait
bash run_torch.sh run_stage9_planner_eval.py --n 60 --ckpt /mnt/lightsem-data/ckpt/group_decoder_v2.pt > logs/planner_eval.txt 2>&1
echo "V2_CHAIN_DONE $(date)"
