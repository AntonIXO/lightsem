#!/usr/bin/env bash
# Audit what optiHealth already has installed, so LightSem can share instead of
# duplicating ~3 GB of torch + nvidia wheels.
echo "=== python versions ==="
export PATH="/root/optiHealth-EiV/.venv/bin:$PATH"
python -V
echo -n "lightsem: "; PATH="/root/lightsem/.venv/bin:$PATH" python -V

echo
echo "=== optiHealth torch ==="
python - <<'PY'
try:
    import torch
    print("torch", torch.__version__)
    print("built against cuda", torch.version.cuda)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device:", torch.cuda.get_device_name(0))
        print("capability:", torch.cuda.get_device_capability(0))
except Exception as e:
    print("torch MISSING:", e)
PY

echo
echo "=== disk footprint of optiHealth venv ==="
SP=$(PATH="/root/optiHealth-EiV/.venv/bin:$PATH" python -c "import site;print(site.getsitepackages()[0])")
echo "site-packages: $SP"
du -sh "$SP" 2>/dev/null
du -sh "$SP"/torch "$SP"/nvidia 2>/dev/null

echo
echo "=== system CUDA via pacman ==="
pacman -Q cuda cudnn nvidia-utils 2>/dev/null || echo "(not installed via pacman)"
