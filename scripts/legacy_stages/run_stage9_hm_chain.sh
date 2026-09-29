#!/usr/bin/env bash
cd /root/lightsem
mkdir -p runs/logs
while kill -0 $(cat runs/logs/gpu_chain.pid) 2>/dev/null; do sleep 15; done
bash run_torch.sh scripts/legacy_stages/run_stage9_embedding.py extract > runs/logs/hm_extract.txt 2>&1
bash run.sh scripts/legacy_stages/run_stage9_embedding.py audit > runs/logs/hm_audit.txt 2>&1
echo "HM_CHAIN_DONE $(date)"
