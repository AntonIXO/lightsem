#!/usr/bin/env bash
# Canonical wrapper. NEVER call .venv/bin/python directly (embedded null byte).
cd /root/lightsem
export PATH="/root/lightsem/.venv/bin:$PATH"
export PYTHONPATH="/root/lightsem:$PYTHONPATH"
exec python "$@"
