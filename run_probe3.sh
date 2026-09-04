#!/bin/bash
cd /root/lightsem
export PATH="/root/lightsem/.venv/bin:$PATH"
exec python probe3_segment_first.py 2>&1 | tee probe3_log.txt
