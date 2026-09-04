#!/usr/bin/env bash
cd /root/lightsem
while kill -0 $(cat logs/critic_v1.pid) 2>/dev/null; do sleep 15; done
bash run_torch.sh render_stage9_demo.py /root/shpongle.flac > logs/demo_shpongle.txt 2>&1
echo "DEMO_DONE $(date)"
