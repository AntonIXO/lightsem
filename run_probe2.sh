#!/bin/bash
cd /root/lightsem
export PATH="/root/lightsem/.venv/bin:$PATH"
exec python probe2_artifact_controls.py 2>&1 | tee probe2_log.txt
