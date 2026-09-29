#!/usr/bin/env bash
# H-O size sweep, one training at a time (three in parallel OOM'd the 12 GiB card).
cd /root/lightsem
mkdir -p runs/logs
while pgrep -f "run-tag s_d256b6_self" > /dev/null; do sleep 15; done
bash run_torch.sh scripts/training/train_group_decoder.py --flow --self-hazard --steps 6000 --run-tag s_d192b3_self --d-model 192 --n-blocks 3 > runs/logs/sweep_d192b3_self.txt 2>&1
bash run_torch.sh scripts/training/train_group_decoder.py --flow --steps 6000 --run-tag s_d256b6_audio --d-model 256 --n-blocks 6 > runs/logs/sweep_d256b6_audio.txt 2>&1
bash run_torch.sh scripts/training/train_group_decoder.py --flow --self-hazard --steps 6000 --run-tag s_d384b6_self --d-model 384 --n-blocks 6 > runs/logs/sweep_d384b6_self.txt 2>&1
bash run_torch.sh scripts/training/train_group_decoder.py --flow --steps 6000 --run-tag s_d384b6_audio --d-model 384 --n-blocks 6 > runs/logs/sweep_d384b6_audio.txt 2>&1
echo "SWEEP_DONE $(date)"
