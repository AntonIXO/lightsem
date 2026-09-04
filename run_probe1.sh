#!/bin/bash
cd /root/lightsem
export PATH="/root/lightsem/.venv/bin:$PATH"
exec python probe1_pose_vs_audio.py 2>&1 | tee probe1_log.txt
