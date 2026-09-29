#!/usr/bin/env bash
# H-O size sweep, three at a time on the 5070 (each ~2-5 GiB), all --flow (the v2 look head), causal.
cd /root/lightsem
mkdir -p runs/logs
run() { bash run_torch.sh scripts/training/train_group_decoder.py --flow --steps 6000 --corpora rpmc,jd "$@" > runs/logs/sweep_$1_$2.txt 2>&1; }
( bash run_torch.sh scripts/training/train_group_decoder.py --flow --self-hazard --steps 6000 --run-tag s_d192b3_self --d-model 192 --n-blocks 3 > runs/logs/sweep_d192b3_self.txt 2>&1 ) &
( bash run_torch.sh scripts/training/train_group_decoder.py --flow --steps 6000 --run-tag s_d256b6_audio --d-model 256 --n-blocks 6 > runs/logs/sweep_d256b6_audio.txt 2>&1 ) &
( bash run_torch.sh scripts/training/train_group_decoder.py --flow --self-hazard --steps 6000 --run-tag s_d256b6_self --d-model 256 --n-blocks 6 > runs/logs/sweep_d256b6_self.txt 2>&1 ) &
wait
( bash run_torch.sh scripts/training/train_group_decoder.py --flow --self-hazard --steps 6000 --run-tag s_d384b6_self --d-model 384 --n-blocks 6 > runs/logs/sweep_d384b6_self.txt 2>&1 ) &
( bash run_torch.sh scripts/training/train_group_decoder.py --flow --steps 6000 --run-tag s_d384b6_audio --d-model 384 --n-blocks 6 > runs/logs/sweep_d384b6_audio.txt 2>&1 ) &
wait
echo "SWEEP_DONE $(date)"
