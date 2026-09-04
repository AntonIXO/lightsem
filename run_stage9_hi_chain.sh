#!/usr/bin/env bash
# After the decoder trains: its G9.3 eval, then the planner (offline + causal) and its G9.4 eval.
cd /root/lightsem
while kill -0 $(cat logs/decoder_v1.pid) 2>/dev/null; do sleep 15; done
echo "decoder trained $(date)"
bash run_torch.sh run_stage9_decoder_eval.py --n 60 --tag v1 > logs/decoder_eval_v1.txt 2>&1
bash run_torch.sh train_planner.py --steps 4000 --run-tag v1 > logs/planner_v1.txt 2>&1 &
bash run_torch.sh train_planner.py --steps 4000 --run-tag v1 --causal > logs/planner_v1_causal.txt 2>&1 &
wait
bash run_torch.sh run_stage9_planner_eval.py --n 60 > logs/planner_eval.txt 2>&1
echo "HI_CHAIN_DONE $(date)"
