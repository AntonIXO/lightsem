"""Headless volumetric stage renderer (moderngl + standalone EGL).

Consumes a Stage C DMX stream and renders frames to an offscreen FBO on the
GPU, then muxes with the source audio via ffmpeg.

Verified working on this host: standalone EGL gives
``NVIDIA GeForce RTX 5070/PCIe/SSE2, GL 3.3.0``. No X/Wayland display needed.
"""

from __future__ import annotations

import math
import subprocess
from dataclasses import dataclass
from pathlib import Path

import moderngl
import numpy as np

from ..dmx.fixtures import Fixture, Rig
from .shaders import FRAGMENT_SHADER, VERTEX_SHADER

MAX_LIGHTS = 16


def _decode_fixture(f: Fixture, universe: np.ndarray) -> dict:
    """Read a fixture's live parameters back OUT of the DMX universe.

    This is deliberate: the renderer consumes the DMX stream, not the
    in-memory scene objects. If the universe encoding is wrong, the render
    shows it — which makes the picture a genuine test of Stage C.
    """
    a = f.address - 1

    def ch(off: int) -> float:
        i = a + off
        return float(universe[i]) / 255.0 if 0 <= i < 512 else 0.0

    if f.kind == "moving_head":
        pan = (float(universe[a]) * 256 + float(universe[a + 1])) / 65535.0
        tilt = (float(universe[a + 2]) * 256 + float(universe[a + 3])) / 65535.0
        return {"pan": pan, "tilt": tilt, "dimmer": ch(4), "strobe": ch(5),
                "rgb": (ch(6), ch(7), ch(8)), "zoom": ch(9), "kind": 0}
    if f.kind == "wash":
        return {"pan": 0.5, "tilt": 0.5, "dimmer": ch(0), "strobe": ch(1),
                "rgb": (ch(2), ch(3), ch(4)), "zoom": ch(5), "kind": 1}
    if f.kind == "blinder":
        return {"pan": 0.5, "tilt": 0.5, "dimmer": ch(0), "strobe": ch(1),
                "rgb": (1.0, 0.95, 0.85), "zoom": 1.0, "kind": 2}
    if f.kind == "strobe":
        return {"pan": 0.5, "tilt": 0.5, "dimmer": ch(0), "strobe": ch(1),
                "rgb": (1.0, 1.0, 1.0), "zoom": 0.8, "kind": 2}
    if f.kind == "projector":
        return {"pan": ch(8), "tilt": ch(9), "dimmer": ch(0), "strobe": ch(1),
                "rgb": (ch(2), ch(3), ch(4)), "zoom": ch(5), "kind": 3}
    return {"pan": 0.5, "tilt": 0.5, "dimmer": 0.0, "strobe": 0.0,
            "rgb": (0, 0, 0), "zoom": 0.5, "kind": 0}


def _beam_direction(f: Fixture, pan: float, tilt: float) -> tuple[float, float, float]:
    """Convert normalised pan/tilt into a world-space beam axis."""
    if f.kind in ("wash", "blinder", "strobe"):
        d = np.array(f.direction, dtype=np.float64)
        return tuple(d / (np.linalg.norm(d) + 1e-9))
    pan_rad = math.radians((pan - 0.5) * f.pan_range_deg)
    tilt_rad = math.radians((tilt - 0.5) * f.tilt_range_deg)
    # Rest pose points down (-y); tilt lifts it toward +z, pan spins about y.
    x = math.sin(pan_rad) * math.cos(tilt_rad)
    z = math.sin(tilt_rad)
    y = -math.cos(pan_rad) * math.cos(tilt_rad)
    v = np.array([x, y, z], dtype=np.float64)
    return tuple(v / (np.linalg.norm(v) + 1e-9))


@dataclass
class RenderConfig:
    width: int = 1920
    height: int = 1080
    fps: float = 30.0
    haze: float = 0.115
    cam_pos: tuple[float, float, float] = (0.0, 2.6, 11.0)
    cam_target: tuple[float, float, float] = (0.0, 3.2, -3.0)
    cam_orbit: float = 0.07          # slow orbit amplitude (radians)
    intensity_gain: float = 2.4
    exposure_profile: str = "legacy"
    fixed_exposure_scale: float = 0.12
    fixed_role_gains: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    gpu_device: int = 0
    require_gpu: bool = True

    def __post_init__(self):
        if self.exposure_profile not in ("legacy", "fixed"):
            raise ValueError("Unknown exposure profile")
        if not math.isfinite(self.fixed_exposure_scale) or self.fixed_exposure_scale <= 0:
            raise ValueError("Fixed exposure scale must be finite and positive")
        if len(self.fixed_role_gains) != 4 or any(not math.isfinite(v) or v < 0 for v in self.fixed_role_gains):
            raise ValueError("Fixed role gains require four finite nonnegative values: beam, wash, flash, projector")


class StageRenderer:
    """Offscreen volumetric renderer for the club rig."""

    def __init__(self, rig: Rig, cfg: RenderConfig | None = None):
        self.rig = rig
        self.cfg = cfg or RenderConfig()
        self.ctx = moderngl.create_context(
            standalone=True, backend="egl", device_index=self.cfg.gpu_device
        )
        renderer = self.ctx.info["GL_RENDERER"]
        software_renderers = ("llvmpipe", "softpipe", "swiftshader", "software rasterizer")
        if self.cfg.require_gpu and any(name in renderer.lower() for name in software_renderers):
            raise RuntimeError(f"GPU rendering required, got software renderer: {renderer}")
        self.prog = self.ctx.program(
            vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER
        )
        quad = np.array([-1, -1, 3, -1, -1, 3], dtype="f4")
        self.vbo = self.ctx.buffer(quad.tobytes())
        self.vao = self.ctx.simple_vertex_array(self.prog, self.vbo, "in_pos")
        self.fbo = self.ctx.simple_framebuffer(
            (self.cfg.width, self.cfg.height), components=4
        )
        self.fbo.use()

    @property
    def renderer_name(self) -> str:
        return f"{self.ctx.info['GL_RENDERER']} / {self.ctx.info['GL_VERSION']}"

    def _set_uniform(self, name: str, value) -> bool:
        """Set a uniform, returning whether it existed.

        moderngl exposes a uniform ARRAY under its base name (``u_lpos``),
        NOT under ``u_lpos[0]``. An earlier version guarded every write with
        ``if name in self.prog`` and addressed elements as ``u_lpos[0]``,
        so every per-light uniform was silently skipped and all frames
        rendered identically. Never guard-and-skip a uniform write without
        surfacing the miss.
        """
        member = self.prog.get(name, None)
        if member is None:
            return False
        member.value = value
        return True

    def _set_light_array(self, base: str, index: int, value) -> bool:
        """Write element ``index`` of a uniform array."""
        member = self.prog.get(base, None)
        if member is None:
            return False
        try:
            member.__setitem__(index, value)          # moderngl array write
            return True
        except (TypeError, AttributeError, KeyError):
            pass
        # Fallback: some drivers do expose the indexed name.
        return self._set_uniform(f"{base}[{index}]", value)

    def render_frame(self, universe: np.ndarray, t: float) -> np.ndarray:
        cfg = self.cfg
        self._set_uniform("u_res", (float(cfg.width), float(cfg.height)))
        self._set_uniform("u_time", float(t))
        ang = cfg.cam_orbit * math.sin(t * 0.11)
        cx, cy, cz = cfg.cam_pos
        self._set_uniform("u_cam_pos", (
            cx * math.cos(ang) + cz * math.sin(ang), cy,
            -cx * math.sin(ang) + cz * math.cos(ang)))
        self._set_uniform("u_cam_target", cfg.cam_target)
        self._set_uniform("u_haze", cfg.haze)
        if not self._set_uniform("u_fixed_exposure", int(cfg.exposure_profile == "fixed")):
            raise RuntimeError("Renderer shader is missing the exposure-profile uniform")
        if not self._set_uniform("u_exposure_scale", cfg.fixed_exposure_scale):
            raise RuntimeError("Renderer shader is missing the fixed-exposure scale")

        fixtures = self.rig.fixtures[:MAX_LIGHTS]
        n = len(fixtures)
        self._set_uniform("u_n_lights", n)

        # Build the whole uniform arrays and write them in one call each.
        # Element-wise assignment via "u_lpos[i]" does NOT work: moderngl
        # exposes an array under its BASE name only.
        pos = np.zeros((MAX_LIGHTS, 3), dtype="f4")
        dirs = np.zeros((MAX_LIGHTS, 3), dtype="f4")
        cols = np.zeros((MAX_LIGHTS, 3), dtype="f4")
        coss = np.zeros(MAX_LIGHTS, dtype="f4")
        ints = np.zeros(MAX_LIGHTS, dtype="f4")
        kinds = np.zeros(MAX_LIGHTS, dtype="i4")

        for i, f in enumerate(fixtures):
            p = _decode_fixture(f, universe)
            d = _beam_direction(f, p["pan"], p["tilt"])
            half_deg = (
                f.beam_min_deg + p["zoom"] * (f.beam_max_deg - f.beam_min_deg)
            ) * 0.5
            r, g, b = p["rgb"]
            pos[i] = f.position
            dirs[i] = d
            cols[i] = (r, g, b)
            coss[i] = math.cos(math.radians(half_deg))
            ints[i] = p["dimmer"] * cfg.intensity_gain
            if cfg.exposure_profile == "fixed":
                ints[i] *= cfg.fixed_role_gains[p["kind"]]
            kinds[i] = p["kind"]

        for name, arr in (
            ("u_lpos", pos), ("u_ldir", dirs), ("u_lcol", cols),
            ("u_lcos", coss), ("u_lint", ints), ("u_lkind", kinds),
        ):
            member = self.prog.get(name, None)
            if member is not None:
                member.write(arr.tobytes())

        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        self.vao.render(moderngl.TRIANGLES)
        buf = self.fbo.read(components=3, alignment=1)
        img = np.frombuffer(buf, dtype=np.uint8).reshape(
            self.cfg.height, self.cfg.width, 3
        )
        return np.flipud(img)   # GL origin is bottom-left

    def render_video(
        self,
        frames: np.ndarray,
        stream_fps: float,
        out_path: str | Path,
        audio_path: str | Path | None = None,
        duration_s: float | None = None,
        progress_every: int = 60,
    ) -> Path:
        """Render the DMX stream to an MP4, muxing audio when supplied."""
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cfg = self.cfg

        total_s = duration_s if duration_s is not None else frames.shape[0] / stream_fps
        n_out = int(round(total_s * cfg.fps))

        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{cfg.width}x{cfg.height}", "-r", str(cfg.fps),
            "-i", "pipe:0",
        ]
        if audio_path is not None:
            cmd += ["-i", str(audio_path)]
        cmd += [
            "-c:v", "av1_nvenc", "-gpu", str(cfg.gpu_device),
            "-preset", "p7", "-tune", "uhq",
            "-rc", "vbr", "-cq", "18", "-b:v", "0",
            "-multipass", "fullres", "-rc-lookahead", "32",
            "-spatial-aq", "1", "-temporal-aq", "1",
            "-pix_fmt", "yuv420p",
            "-colorspace", "bt709", "-color_primaries", "bt709",
            "-color_trc", "bt709", "-movflags", "+faststart",
        ]
        if audio_path is not None:
            cmd += ["-c:a", "aac", "-b:a", "192k", "-shortest"]
        cmd += [str(out_path)]

        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        assert proc.stdin is not None
        try:
            for i in range(n_out):
                t = i / cfg.fps
                src = min(int(round(t * stream_fps)), frames.shape[0] - 1)
                img = self.render_frame(frames[src], t)
                proc.stdin.write(img.tobytes())
                if progress_every and i % progress_every == 0:
                    print(f"  frame {i}/{n_out}  t={t:.2f}s", flush=True)
        finally:
            proc.stdin.close()
            err = proc.stderr.read().decode() if proc.stderr else ""
            rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"ffmpeg failed (rc={rc}): {err[:2000]}")
        return out_path
