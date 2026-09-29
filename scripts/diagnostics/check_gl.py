import moderngl
import numpy as np

for backend in ("egl", "standalone-default"):
    try:
        ctx = (moderngl.create_context(standalone=True, backend="egl")
               if backend == "egl" else moderngl.create_context(standalone=True))
        print(f"{backend:20s} OK  {ctx.info['GL_RENDERER']}  GL {ctx.info['GL_VERSION']}")
        fbo = ctx.simple_framebuffer((64, 64))
        fbo.use(); ctx.clear(0.2, 0.4, 0.8)
        px = np.frombuffer(fbo.read(components=3), dtype=np.uint8)
        print("                     pixel sample:", px[:3])
        ctx.release()
        break
    except Exception as e:
        print(f"{backend:20s} FAIL {type(e).__name__}: {str(e)[:120]}")
