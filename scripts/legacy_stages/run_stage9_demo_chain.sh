#!/usr/bin/env bash
cd /root/lightsem
mkdir -p runs/logs
while kill -0 $(cat runs/logs/critic_v1.pid) 2>/dev/null; do sleep 15; done
bash run_torch.sh scripts/rendering/render_stage9_demo.py /root/shpongle.flac > runs/logs/demo_shpongle.txt 2>&1
echo "DEMO_DONE $(date)"
