#!/usr/bin/env bash
# Can optiHealth's py3.11 interpreter run LightSem code? Checks every dep the
# training path needs, so we can reuse its 3.9 GB torch+nvidia stack instead of
# installing a second copy.
export PATH="/root/optiHealth-EiV/.venv/bin:$PATH"
export PYTHONPATH="/root/lightsem:$PYTHONPATH"
cd /root/lightsem
python - <<'PY'
import importlib
need = ["torch", "numpy", "h5py", "requests", "librosa", "scipy",
        "sklearn", "soundfile", "moderngl", "imageio", "av", "pandas"]
missing = []
for m in need:
    try:
        mod = importlib.import_module(m)
        print(f"  {m:12s} {getattr(mod,'__version__','?')}")
    except Exception as e:
        print(f"  {m:12s} MISSING ({type(e).__name__})")
        missing.append(m)

print()
try:
    from lightsem.data.rpmc_remote import TOTAL
    print(f"lightsem package importable  (remote view {TOTAL/1e9:.1f} GB)")
except Exception as e:
    print("lightsem import FAILED:", e)

import torch
print(f"\ntorch {torch.__version__} cuda={torch.cuda.is_available()}")
if torch.cuda.is_available():
    x = torch.randn(4096, 4096, device="cuda")
    import time
    torch.cuda.synchronize(); t = time.time()
    for _ in range(20):
        x = x @ x.T / 4096.0
    torch.cuda.synchronize()
    dt = time.time() - t
    tf = 20 * 2 * 4096**3 / dt / 1e12
    print(f"matmul smoke: {tf:.1f} TFLOP/s fp32  ({dt:.2f}s)")
    print("free/total GB:", [round(v/1e9, 1) for v in torch.cuda.mem_get_info()])

print("\nMISSING:", missing if missing else "none")
PY
