#!/usr/bin/env bash
cd /root/lightsem
export PATH="/root/lightsem/.venv/bin:$PATH"
exec python - <<'PY'
import sys, importlib
print("python", sys.version)
for m in ["librosa","numpy","scipy","sklearn","moderngl","imageio","av","PIL","pandas","pyarrow","soundfile"]:
    try:
        mod=importlib.import_module(m)
        print(f"  {m:12s} {getattr(mod,'__version__','?')}")
    except Exception as e:
        print(f"  {m:12s} MISSING {e}")
import moderngl
ctx = moderngl.create_context(standalone=True, backend="egl")
print("GL:", ctx.info["GL_RENDERER"], ctx.info["GL_VERSION"])
PY
