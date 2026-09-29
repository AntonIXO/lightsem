#!/bin/bash
cd /root/lightsem
export PATH="/root/lightsem/.venv/bin:$PATH"
exec python scripts/rendering/make_figure.py
