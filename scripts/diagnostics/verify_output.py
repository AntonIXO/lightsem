"""Verify the produced MP4 actually contains ANIMATED, audio-synced light.

A file that plays is not proof of a working show: a frozen frame or a black
video would also "play". This decodes the real MP4 and checks:
  * frames genuinely differ over time (not static)
  * brightness tracks the structure (drop sections brighter than intro)
  * the audio stream is present and non-silent
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import av
import numpy as np


def check(path: Path, drop_windows, intro_window) -> int:
    print(f"=== {path.name} ===")
    container = av.open(str(path))
    vs = next(s for s in container.streams if s.type == "video")
    aud = [s for s in container.streams if s.type == "audio"]
    print(f"  video {vs.codec_context.width}x{vs.codec_context.height} "
          f"avg_rate={float(vs.average_rate):.2f}")
    print(f"  audio streams: {len(aud)}")

    lum, times = [], []
    for frame in container.decode(video=0):
        img = frame.to_ndarray(format="rgb24")
        lum.append(float(img.mean()))
        times.append(float(frame.pts * vs.time_base))
    container.close()

    lum_a = np.array(lum)
    times_a = np.array(times)
    print(f"  decoded {len(lum)} frames, t=[{times_a.min():.2f},{times_a.max():.2f}]")
    print(f"  luminance mean={lum_a.mean():.2f} std={lum_a.std():.2f} "
          f"min={lum_a.min():.2f} max={lum_a.max():.2f}")

    # 1. Not static.
    diffs = np.abs(np.diff(lum_a))
    print(f"  frame-to-frame |dLum| mean={diffs.mean():.3f} max={diffs.max():.3f}")
    assert lum_a.std() > 1.0, "video is static - luminance never changes"
    assert diffs.max() > 0.5, "no frame-to-frame change - frozen render"
    print("  -> animated OK")

    # 2. Structure tracking.
    def win_mean(a, b):
        m = (times_a >= a) & (times_a < b)
        return float(lum_a[m].mean()) if m.any() else float("nan")

    intro = win_mean(*intro_window)
    drops = [win_mean(*w) for w in drop_windows]
    print(f"  intro window luminance : {intro:.2f}")
    for w, d in zip(drop_windows, drops):
        print(f"  drop  {w} luminance : {d:.2f}")
    if np.isfinite(intro) and any(np.isfinite(d) for d in drops):
        best = np.nanmax(drops)
        assert best > intro, "drops are not brighter than intro in the VIDEO"
        print(f"  -> drop ({best:.2f}) brighter than intro ({intro:.2f}) OK")

    # 3. Audio non-silent.
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=codec_name,sample_rate,channels,duration",
         "-print_format", "json", str(path)],
        capture_output=True, text=True)
    info = json.loads(r.stdout)
    print(f"  audio: {info['streams'][0] if info.get('streams') else 'NONE'}")
    assert info.get("streams"), "no audio stream in the MP4"

    # volumedetect reports on the INFO log level; at "-v error" the lines are
    # suppressed entirely, which is why an earlier version failed to find
    # mean_volume even on perfectly good audio.
    vol = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-v", "info",
         "-i", str(path), "-af", "volumedetect", "-f", "null", "-"],
        capture_output=True, text=True)
    blob = vol.stderr + vol.stdout
    for line in blob.splitlines():
        if "mean_volume" in line or "max_volume" in line:
            print(f"  {line.strip()}")
    assert "max_volume" in blob, "could not measure audio level"
    mv = [l for l in blob.splitlines() if "max_volume" in l][0]
    max_db = float(mv.split(":")[1].strip().split()[0])
    assert max_db > -50.0, f"audio is effectively silent ({max_db} dB)"
    print(f"  -> audio present and non-silent ({max_db} dB peak) OK")
    return 0


def main() -> int:
    base = Path("/root/lightsem/experiments/legacy/misc/outputs")
    check(base / "lightsem_demo.mp4",
          drop_windows=[(17.0, 24.0), (37.0, 45.0)],
          intro_window=(1.0, 8.0))
    print()
    check(base / "aist_mBR0_show.mp4",
          drop_windows=[(10.5, 12.5), (20.0, 24.0)],
          intro_window=(0.5, 3.0))
    print("\nVIDEO VERIFICATION: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
