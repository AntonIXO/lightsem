#!/usr/bin/env bash
# H-G step 1: three GPU shards, each does its whole-track share then its causal (L=0) share.
cd /root/lightsem
for s in 0 1 2; do
  ( bash run_torch.sh extract_stems.py --mode whole --shard $s --nshards 3 > logs/stems_whole_$s.txt 2>&1
    bash run_torch.sh extract_stems.py --mode causal --shard $s --nshards 3 --limit 90 --batch 8 > logs/stems_causal_$s.txt 2>&1 ) &
  echo $! > logs/stems_$s.pid
done
wait
echo ALL_STEMS_DONE
