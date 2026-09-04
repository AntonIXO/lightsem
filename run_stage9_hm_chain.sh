#!/usr/bin/env bash
cd /root/lightsem
while kill -0 $(cat logs/gpu_chain.pid) 2>/dev/null; do sleep 15; done
bash run_torch.sh run_stage9_embedding.py extract > logs/hm_extract.txt 2>&1
bash run.sh run_stage9_embedding.py audit > logs/hm_audit.txt 2>&1
echo "HM_CHAIN_DONE $(date)"
