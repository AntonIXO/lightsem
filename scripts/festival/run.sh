#!/usr/bin/env bash
# Blender owns its interpreter; audio/ML Python dependencies are unnecessary.
set -euo pipefail
cd "$(dirname "$0")/../.."
festival_blender="${LIGHTSEM_BLENDER:-/opt/blender-lightsem/blender}"
if [[ ! -x "$festival_blender" ]]; then
  festival_blender="$(command -v blender || true)"
fi
if [[ -z "$festival_blender" || ! -x "$festival_blender" ]]; then
  echo 'Set LIGHTSEM_BLENDER to a Blender 4.5 LTS executable.' >&2
  exit 1
fi
exec "$festival_blender" --background --threads 8 --python-exit-code 1 \
  --python scripts/festival/render_festival.py -- "$@"
