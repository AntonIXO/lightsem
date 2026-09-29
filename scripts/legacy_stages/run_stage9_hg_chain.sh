#!/usr/bin/env bash
# H-G: redo the OOM'd causal shard, its features, then the full stem audit.
cd /root/lightsem
mkdir -p runs/logs
bash run_torch.sh scripts/data/extract_stems.py --mode causal --shard 0 --nshards 3 --limit 90 --batch 6 > runs/logs/stems_causal_0b.txt 2>&1
bash run.sh scripts/data/build_stem_feats.py --mode causal --workers 6 > runs/logs/stem_feats_causal2.txt 2>&1
bash run.sh scripts/legacy_stages/run_stage9_stems.py > runs/logs/hg_stems.txt 2>&1
echo "HG_CHAIN_DONE $(date)"
