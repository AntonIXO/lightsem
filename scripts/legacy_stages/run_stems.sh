#!/usr/bin/env bash
# H-G step 1: three GPU shards, each does its whole-track share then its causal (L=0) share.
cd /root/lightsem
mkdir -p runs/logs
for s in 0 1 2; do
  ( bash run_torch.sh scripts/data/extract_stems.py --mode whole --shard $s --nshards 3 > runs/logs/stems_whole_$s.txt 2>&1
    bash run_torch.sh scripts/data/extract_stems.py --mode causal --shard $s --nshards 3 --limit 90 --batch 8 > runs/logs/stems_causal_$s.txt 2>&1 ) &
  echo $! > runs/logs/stems_$s.pid
done
wait
echo ALL_STEMS_DONE
