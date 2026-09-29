#!/bin/bash
cd /root/lightsem
export PATH="/root/lightsem/.venv/bin:$PATH"
exec python scripts/diagnostics/probe3b_segment_first.py 2>&1 | grep -v "Warning\|_methods\|fromnumeric\|  ret =\|  arrmean\|  return _" | tee probe3b_log.txt
