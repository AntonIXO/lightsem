#!/usr/bin/env bash
# After the first HI chain (planner trainings): the fixed decoder eval, then the planner eval.
cd /root/lightsem
mkdir -p runs/logs
while kill -0 $(cat runs/logs/hi_chain.pid) 2>/dev/null; do sleep 15; done
bash run_torch.sh scripts/legacy_stages/run_stage9_decoder_eval.py --n 60 --tag v1 > runs/logs/decoder_eval_v1.txt 2>&1
bash run_torch.sh scripts/legacy_stages/run_stage9_planner_eval.py --n 60 > runs/logs/planner_eval.txt 2>&1
echo "HI_CHAIN2_DONE $(date)"
