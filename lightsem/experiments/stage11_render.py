"""Stage 11 rendering and prospective blind-comparison materials.

The scene vector is a proxy learned from dance video, not authored DMX truth.
``frames_scene`` maps all arms through one fixed assign-look/GroupEngine
recipe. ``render_frames`` and ``render_descriptors`` are independent of any
checkpoint format. Importing this module does not create an EGL context.

``prepare_pairs(root, protocol, candidates, n_songs=20)`` accepts already
excerpt-aligned 40 Hz DMX arrays (or .npy paths). Each candidate must contain
``canonical_id``, ``audio_path``, ``offset_s`` (absolute source-audio offset),
``duration_s``, ``arms`` (name -> frames), and two ``comparisons`` (arm-name
pairs); comparisons may instead be fixed in ``protocol['render']``. It writes
only to a newly created ``root/results/renders`` directory, never overwrites
existing materials/ratings, and produces 40 primary + four repeat + four sham
pairs with the defaults. Arm provenance is retained in pairs.json; the rating
CLI displays only anonymous A/B filenames.

Frozen scientific candidates also provide ``arm_provenance`` keyed by the
same displayed arm names, containing the training seed, generation seed and
checkpoint SHA256. Those records follow A/B swapping and repeat/sham reuse.

Brightness matching means *screen-luma matching at this camera and exposure*.
It is not equal physical illuminance or equality of colour perception.
Only the brighter stream is attenuated. A final encoded-video audit must
pass the 3% symmetric mean-luma tolerance before a pair is published.
"""

from __future__ import annotations

import colorsys
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
from typing import Iterable

import numpy as np

from lightsem.conductor.rig import canonical_rig
from lightsem.dmx.decode import decode_stream


DMX_FPS = 40.0
SCENE_FPS = 10.0
LUMA_TOLERANCE = 0.03
LUMA_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)
FIXED_RENDER_CONFIG = {
    "width": 854, "height": 480, "fps": 30.0, "haze": 0.115,
    "cam_pos": (0.0, 2.6, 11.0), "cam_target": (0.0, 3.2, -3.0),
    "cam_orbit": 0.0, "intensity_gain": 2.4,
}
FIXED_SCENE_MAPPING = {
    "version": "stage11_scene_mapping_v1", "scene_fps": SCENE_FPS,
    "dmx_fps": DMX_FPS, "tau_level": 0.3, "tau_colour": 0.8,
    "effect": "pulse", "rate_beats": 1.0, "spread": 0.0,
    "density": 0.5, "initial_saturation": 1.0,
    "wave_amp": 0.6, "wave_bars": 2.0, "wave_spread": 1.0,
    "hue_spread": 0.12, "hue_drift": 0.01,
}


def _frames(value) -> np.ndarray:
    if isinstance(value, (str, Path)):
        value = np.load(value, allow_pickle=False)
    value = np.asarray(value)
    if value.ndim != 2 or value.shape[1] != 512 or len(value) == 0:
        raise ValueError("frames must be a nonempty (T, 512) DMX array")
    if value.dtype != np.uint8:
        raise ValueError("DMX frames must have uint8 dtype")
    return value


def _positive_duration(duration_s: float) -> float:
    value = float(duration_s)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("duration_s must be finite and positive")
    return value


def _write_new_json(path: Path, value) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _hash_frames(frames: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(frames).tobytes()).hexdigest()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_excerpt_start(
    canonical_id: str, duration_s: float, *, salt: str,
    clip_s: float = 30.0, margin_s: float = 5.0,
) -> float:
    """Choose an excerpt without model scores, aligned to the 40 Hz grid."""
    duration_s, clip_s = _positive_duration(duration_s), _positive_duration(clip_s)
    if not np.isfinite(margin_s) or margin_s < 0:
        raise ValueError("margin_s must be nonnegative")
    first = int(math.ceil(margin_s * DMX_FPS))
    last = int(math.floor((duration_s - clip_s - margin_s) * DMX_FPS))
    if last < first:
        raise ValueError("recording is too short for the requested excerpt and margins")
    key = hashlib.sha256(f"{salt}\0{canonical_id}".encode()).digest()
    return (first + int.from_bytes(key, "big") % (last - first + 1)) / DMX_FPS


def frames_scene(scene_native, beats, bar_phase, duration_s: float, *, rig=None):
    """Map native scene channels to one deterministic 40 Hz ``DmxStream``.

    Native means hue cos/sin, saturation, *recording-normalized* value and
    p90, and concentration. Undo only the checkpoint's dataset target scaler
    before calling; do not restore an absolute video exposure. The supplied
    bar phase must already cover the excerpt on its 40 Hz grid.
    """
    from lightsem.model.group_decoder import EFFECTS, RATES, ROLES, GroupEngine, assign_look

    scene = np.asarray(scene_native, dtype=np.float64)
    duration_s = _positive_duration(duration_s)
    if scene.ndim != 2 or scene.shape[1] != 6 or len(scene) == 0 or not np.isfinite(scene).all():
        raise ValueError("scene_native must be finite with nonempty shape (T, 6)")
    if abs(len(scene) / SCENE_FPS - duration_s) > 1 / SCENE_FPS + 1e-9:
        raise ValueError("scene duration does not match duration_s at 10 Hz")
    n_out = int(round(duration_s * DMX_FPS))
    if n_out < 1:
        raise ValueError("duration_s must cover at least one DMX frame")
    beats = np.asarray(beats, dtype=np.float64)
    phase = np.asarray(bar_phase, dtype=np.float64)
    if beats.ndim != 1 or not np.isfinite(beats).all() or np.any(np.diff(beats) <= 0):
        raise ValueError("beats must be finite and strictly increasing")
    if phase.ndim != 1 or len(phase) != n_out or not np.isfinite(phase).all():
        raise ValueError("bar_phase must cover exactly the 40 Hz output timeline")
    if np.any((phase < 0) | (phase >= 1)):
        raise ValueError("bar_phase must lie in [0, 1)")
    n, roles = len(scene), len(ROLES)
    params = {
        "inten": np.zeros((n, roles)), "hue": np.zeros((n, roles, 2)),
        "sat": np.ones((n, roles)), "effect": np.zeros((n, roles, len(EFFECTS))),
        "rate": np.zeros((n, roles, len(RATES))), "spread": np.zeros((n, roles)),
        "density": np.full((n, roles), 0.5),
    }
    params["hue"][..., 0] = 1
    params["effect"][..., EFFECTS.index("pulse")] = 1
    params["rate"][..., RATES.index(1.0)] = 1
    params = assign_look(scene, params, fps=SCENE_FPS, tau_level=0.3, tau_colour=0.8)
    engine = GroupEngine(
        rig or canonical_rig(), fps=DMX_FPS, wave_amp=0.6, wave_bars=2.0,
        wave_spread=1.0, hue_spread=0.12, hue_drift=0.01,
    )
    result = engine.render(params, np.arange(n) / SCENE_FPS, beats, phase, duration_s)
    result.cues = [{"time_s": 0.0, "type": "engine", **FIXED_SCENE_MAPPING}]
    return result


def decode_metric_v2(rig, frames) -> dict[str, np.ndarray]:
    """Decode radiometric proxies consistently with the current screen renderer.

    Historical decoding is unchanged. Blinders have fixed warm RGB and a
    separate dimmer; their value is one, not the dimmer again. The current
    renderer likewise displays strobes as fixed white and ignores their DMX
    white-value channel, so v2 records that behavior explicitly.
    """
    out = decode_stream(rig, _frames(frames))
    for kind, rgb in (("blinder", (1.0, 0.95, 0.85)), ("strobe", (1.0, 1.0, 1.0))):
        mask = out["kind"] == kind
        hue, sat, val = colorsys.rgb_to_hsv(*rgb)
        out["hue"][:, mask] = hue
        out["sat"][:, mask] = sat
        out["val"][:, mask] = val
    return out


def emitted_v2(decoded: dict[str, np.ndarray]) -> np.ndarray:
    """Dimmer × fixed/RGB value proxy, with exact zero for dark RGB output."""
    value = np.asarray(decoded["val"]).copy()
    if "kind" in decoded:
        fixed = np.isin(decoded["kind"], ["blinder", "strobe"])
        value[:, fixed] = 1.0
    return np.asarray(decoded["dimmer"]) * value


def scale_dimmer(frames, gain: float, *, rig=None) -> np.ndarray:
    """Scale only dimmer bytes; preserve pan, RGB, gobo and all other bytes."""
    frames = _frames(frames)
    if not np.isfinite(gain) or not 0 <= gain <= 1:
        raise ValueError("screen matching only permits attenuation gains in [0, 1]")
    out = frames.copy()
    for fixture in (rig or canonical_rig()).fixtures:
        offset = 4 if fixture.kind == "moving_head" else 0
        column = fixture.address - 1 + offset
        out[:, column] = np.floor(out[:, column].astype(np.float64) * gain + 0.5).astype(np.uint8)
    return out


def symmetric_luma_gap(a: float, b: float) -> float:
    if not np.isfinite([a, b]).all() or min(a, b) < 0:
        raise ValueError("luma means must be finite and nonnegative")
    return 0.0 if a + b == 0 else 2 * abs(a - b) / (a + b)


def image_luma_stats(images: Iterable[np.ndarray]) -> dict:
    """Whole-frame RGB -> 64×36 area-filtered BT.709 luma, normalized to 0..1."""
    from PIL import Image

    lumas, full_clip, full_dark = [], [], []
    for image in images:
        image = np.asarray(image)
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError("renderer images must be uint8 RGB arrays")
        full_clip.append(float(np.mean(image >= 254)))
        full_dark.append(float(np.mean(image <= 1)))
        small = np.asarray(Image.fromarray(image).resize((64, 36), Image.Resampling.BOX))
        lumas.append((small.astype(np.float64) @ LUMA_WEIGHTS) / 255)
    if not lumas:
        raise ValueError("at least one rendered image is required")
    data = np.stack(lumas)
    quantiles = np.quantile(data, [0.01, 0.05, 0.5, 0.95, 0.99])
    return {
        "mean": float(data.mean()), "frame_mean_std": float(data.mean((1, 2)).std()),
        "quantiles": dict(zip(("p01", "p05", "p50", "p95", "p99"), map(float, quantiles))),
        "rgb_clip_fraction": float(np.mean(full_clip)), "rgb_near_black_fraction": float(np.mean(full_dark)),
        "n_frames": len(lumas), "roi": "whole frame", "luma_size": [64, 36],
        "luma_coefficients": LUMA_WEIGHTS.tolist(),
    }


def _probe_indices(n_dmx: int, duration_s: float, n_probe: int = 24) -> tuple[np.ndarray, np.ndarray]:
    # Choose video frames, then reproduce render_video's nearest-DMX lookup.
    n_video = max(1, int(round(duration_s * FIXED_RENDER_CONFIG["fps"])))
    indices = np.unique(np.rint(np.linspace(0, n_video - 1, min(n_probe, n_video))).astype(int))
    times = indices / FIXED_RENDER_CONFIG["fps"]
    return np.minimum(np.rint(times * DMX_FPS).astype(int), n_dmx - 1), times


def probe_luma(frames, renderer, duration_s: float, *, n_probe: int = 24) -> dict:
    frames = _frames(frames)
    indices, times = _probe_indices(len(frames), duration_s, n_probe)
    result = image_luma_stats(renderer.render_frame(frames[i], float(t)) for i, t in zip(indices, times))
    result["video_times_s"] = times.tolist()
    result["clipping_resolution"] = [renderer.cfg.width, renderer.cfg.height]
    return result


def match_luma(
    frames_a, frames_b, renderer, duration_s: float, *,
    tolerance: float = LUMA_TOLERANCE, max_probes: int = 12, rig=None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Bounded gain bisection on sparse screen probes; final video audit is separate."""
    a, b = _frames(frames_a), _frames(frames_b)
    if a.shape != b.shape:
        raise ValueError("paired streams must share the same timeline")
    if not 0 < tolerance <= LUMA_TOLERANCE or max_probes < 2:
        raise ValueError("matching needs tolerance in (0, .03] and at least two probes")
    before = [probe_luma(a, renderer, duration_s), probe_luma(b, renderer, duration_s)]
    means = [s["mean"] for s in before]
    bright, dark = int(np.argmax(means)), int(np.argmin(means))
    original = [a, b]
    matched, after, gains = list(original), list(before), [1.0, 1.0]
    history = []
    if symmetric_luma_gap(*means) > tolerance:
        target = means[dark]
        best = (symmetric_luma_gap(means[bright], target), 1.0, before[bright], original[bright])
        low, high = 0.0, 1.0
        for iteration in range(max_probes):
            gain = 0.0 if iteration == 0 else (low + high) / 2
            candidate = scale_dimmer(original[bright], gain, rig=rig)
            stats = probe_luma(candidate, renderer, duration_s)
            gap = symmetric_luma_gap(stats["mean"], target)
            history.append({"gain": gain, "mean": stats["mean"], "gap": gap})
            if gap < best[0]:
                best = (gap, gain, stats, candidate)
            if gap <= tolerance / 2:
                break
            if stats["mean"] > target:
                high = gain
            else:
                low = gain
        _, gains[bright], after[bright], matched[bright] = best
    gap = symmetric_luma_gap(after[0]["mean"], after[1]["mean"])
    return matched[0], matched[1], {
        "method": "dim-brighter-only; bounded bisection on fixed-camera screen probes",
        "gains": gains, "before": before, "after": after, "probe_history": history,
        "symmetric_gap": gap, "tolerance": tolerance, "probe_pass": gap <= tolerance,
        "physical_brightness_matched": False,
    }


def _make_renderer(rig=None):
    from lightsem.sim.renderer import RenderConfig, StageRenderer
    return StageRenderer(rig or canonical_rig(), RenderConfig(**FIXED_RENDER_CONFIG))


def _check_renderer(renderer) -> None:
    actual = asdict(renderer.cfg)
    for name, expected in FIXED_RENDER_CONFIG.items():
        if actual[name] != expected:
            raise ValueError(f"renderer must use frozen {name}={expected!r}")


def render_frames(
    frames, out_path, *, audio_path=None, duration_s=None, renderer=None, rig=None,
) -> Path:
    """Render already aligned 40 Hz DMX; audio must already be cut to the excerpt."""
    frames = _frames(frames)
    duration_s = _positive_duration(duration_s if duration_s is not None else len(frames) / DMX_FPS)
    if abs(len(frames) / DMX_FPS - duration_s) > 1 / DMX_FPS + 1e-9:
        raise ValueError("DMX frames must already cover the requested duration")
    out_path = Path(out_path)
    if out_path.exists():
        raise FileExistsError(out_path)
    renderer = renderer or _make_renderer(rig)
    _check_renderer(renderer)
    return renderer.render_video(
        frames, DMX_FPS, out_path, audio_path=audio_path,
        duration_s=duration_s, progress_every=0,
    )


def render_scene_stream(
    scene_native, beats, bar_phase, duration_s, out_path, *, audio_path=None, renderer=None, rig=None,
) -> Path:
    stream = frames_scene(scene_native, beats, bar_phase, duration_s, rig=rig)
    return render_frames(stream.frames, out_path, audio_path=audio_path,
                         duration_s=duration_s, renderer=renderer, rig=rig)


def render_descriptors(desc, bus, out_path, *, audio_path=None, renderer=None, conductor=None) -> Path:
    """Render externally constructed checkpoint-aware descriptors with a fixed conductor."""
    from lightsem.conductor.hybrid import ConductorHybrid
    stream = (conductor or ConductorHybrid()).render(desc, bus)
    return render_frames(stream.frames, out_path, audio_path=audio_path,
                         duration_s=stream.duration_s, renderer=renderer)


def extract_audio_excerpt(audio_path, out_path, offset_s: float, duration_s: float) -> Path:
    audio_path, out_path = Path(audio_path), Path(out_path)
    duration_s = _positive_duration(duration_s)
    if not audio_path.is_file():
        raise FileNotFoundError(audio_path)
    if not np.isfinite(offset_s) or offset_s < 0:
        raise ValueError("audio offset_s must be finite and nonnegative")
    if out_path.exists():
        raise FileExistsError(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-v", "error", "-nostdin", "-n", "-ss", str(offset_s),
        "-i", str(audio_path), "-t", str(duration_s), "-vn", "-ac", "1", "-ar", "44100", str(out_path),
    ], check=True)
    length = subprocess.run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(out_path),
    ], check=True, capture_output=True, text=True)
    if abs(float(length.stdout.strip()) - duration_s) > 0.05:
        raise ValueError("extracted audio is shorter/different than the requested excerpt")
    return out_path


def measure_video_luma(path) -> dict:
    """Audit every encoded frame, using one fixed 64×36 RGB observation grid."""
    result = subprocess.run([
        "ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-an",
        "-vf", "scale=64:36:flags=area", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ], check=True, capture_output=True)
    data = np.frombuffer(result.stdout, dtype=np.uint8)
    frame_size = 64 * 36 * 3
    if len(data) == 0 or len(data) % frame_size:
        raise ValueError("encoded video has no complete decoded RGB frames")
    stats = image_luma_stats(data.reshape(-1, 36, 64, 3))
    stats["clipping_resolution"] = [64, 36]
    stats["audit"] = "all encoded video frames"
    return stats


def render_matched_pair(frames_a, frames_b, audio_path, duration_s, directory, *, renderer=None, rig=None) -> dict:
    """Match through bounded feedback from the final encoded-video observer.

    Encode both original streams, then attenuate only the brighter original.
    Every gain trial uses the same all-frame observer as final acceptance;
    raw rendering probes cannot substitute for encoded screen-luma evidence.
    All originals, corrections and the final audit are retained.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    renderer = renderer or _make_renderer(rig)
    _check_renderer(renderer)
    originals = [_frames(frames_a), _frames(frames_b)]
    if originals[0].shape != originals[1].shape:
        raise ValueError("paired streams must share the same timeline")
    rendered, audited, current, gains, history = [], [], list(originals), [1.0, 1.0], []
    max_corrections = 12
    for side, label in enumerate(("left", "right")):
        path = render_frames(originals[side], directory / f"{label}__original.mp4", audio_path=audio_path,
                             duration_s=duration_s, renderer=renderer, rig=rig)
        rendered.append(path)
        audited.append(measure_video_luma(path))
    before = list(audited)
    gap = symmetric_luma_gap(audited[0]["mean"], audited[1]["mean"])
    bright = int(np.argmax([value["mean"] for value in audited]))
    dark = 1 - bright
    if gap > LUMA_TOLERANCE:
        target = audited[dark]["mean"]
        low, high = 0.0, 1.0
        low_mean, high_mean = None, audited[bright]["mean"]
        best = (gap, 1.0, originals[bright], rendered[bright], audited[bright])
        for attempt in range(max_corrections):
            if attempt == 0:
                gain = 0.0  # Include the actual encoded ambient/codec floor.
            elif attempt == 1 and high_mean > low_mean:
                gain = float(np.clip((target - low_mean) / (high_mean - low_mean), 0.001, 0.999))
            else:
                gain = (low + high) / 2
            trial = scale_dimmer(originals[bright], gain, rig=rig)
            path = render_frames(trial, directory / f"correction__attempt{attempt:02d}.mp4",
                                 audio_path=audio_path, duration_s=duration_s, renderer=renderer, rig=rig)
            observed = measure_video_luma(path)
            distance = symmetric_luma_gap(observed["mean"], target)
            history.append({"attempt": attempt, "side": bright, "gain": gain,
                            "path": str(path), "video_sha256": _hash_file(path),
                            "encoded_mean": observed["mean"], "target_encoded_mean": target,
                            "symmetric_gap": distance})
            if distance < best[0]:
                best = (distance, gain, trial, path, observed)
            if distance <= LUMA_TOLERANCE / 2:
                break
            if attempt == 0 and observed["mean"] > target:
                # No dimming gain can reach the target: never brighten the other arm.
                break
            if observed["mean"] > target:
                high, high_mean = gain, observed["mean"]
            else:
                low, low_mean = gain, observed["mean"]
        gap, gains[bright], current[bright], rendered[bright], audited[bright] = best
    result = {
        "paths": [str(path) for path in rendered],
        "calibration": {
            "method": "dim-brighter-only bounded gain search against every encoded video frame",
            "gains": gains, "before": before, "after": audited,
            "encoded_history": history, "maximum_corrections": max_corrections,
            "target_dark_side": dark, "tolerance": LUMA_TOLERANCE,
            "pre_encoding_probe_used_for_acceptance": False, "physical_brightness_matched": False,
        },
        "encoded_video": audited, "symmetric_gap": gap,
        "passed": gap <= LUMA_TOLERANCE, "frames_sha256": [_hash_frames(value) for value in current],
        "renderer": FIXED_RENDER_CONFIG, "physical_brightness_matched": False,
        "distribution_diagnostics": {
            "absolute_quantile_gaps": {
                name: abs(audited[0]["quantiles"][name] - audited[1]["quantiles"][name])
                for name in ("p01", "p05", "p50", "p95", "p99")
            },
            "encoded_rgb_clip_fractions": [value["rgb_clip_fraction"] for value in audited],
            "matching_scope": "mean encoded screen luma; distributions and colour perception remain distinct",
        },
    }
    _write_new_json(directory / "match.json", result)
    if not result["passed"]:
        raise RuntimeError("encoded-video screen-luma gap exceeds 3%; pair was not published")
    return result


def _validated_candidates(protocol, candidates, n_songs):
    if not isinstance(n_songs, int) or n_songs < 2 or len(candidates) < n_songs:
        raise ValueError("provide at least n_songs>=2 candidate songs")
    selected, identities = [], set()
    fallback = protocol.get("render", {}).get("comparisons")
    for item in candidates[:n_songs]:
        canonical = str(item["canonical_id"])
        if not canonical or canonical in identities:
            raise ValueError("candidate canonical_id values must be nonempty and unique")
        identities.add(canonical)
        audio_path = Path(item["audio_path"])
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        duration = _positive_duration(item["duration_s"])
        offset = float(item["offset_s"])
        if not np.isfinite(offset) or offset < 0:
            raise ValueError("candidate offset_s must be finite and nonnegative")
        comparisons = item.get("comparisons", fallback)
        if not comparisons or len(comparisons) != 2:
            raise ValueError("each song needs exactly two predeclared comparisons")
        arms = item["arms"]
        provenance = item.get("arm_provenance", {})
        for comparison in comparisons:
            if len(comparison) != 2 or comparison[0] == comparison[1]:
                raise ValueError("a primary comparison must name two different arms")
            for arm in comparison:
                if arm not in arms:
                    raise ValueError(f"comparison refers to missing arm {arm!r}")
                frames = _frames(arms[arm])
                if abs(len(frames) / DMX_FPS - duration) > 1 / DMX_FPS + 1e-9:
                    raise ValueError("candidate DMX is not aligned to the excerpt duration")
                if arm not in provenance:
                    if protocol.get("frozen") is True:
                        raise ValueError("frozen scientific candidates require per-arm checkpoint provenance")
                    continue
                info = provenance[arm]
                for name in ("training_seed", "generation_seed"):
                    if not isinstance(info.get(name), int) or isinstance(info[name], bool) or info[name] < 0:
                        raise ValueError(f"arm provenance requires a nonnegative integer {name}")
                sha = info.get("checkpoint_sha256", "")
                if not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
                    raise ValueError("arm provenance requires a lowercase checkpoint SHA256")
                if "dmx_file_sha256" in info:
                    if not isinstance(arms[arm], (str, Path)) or _hash_file(Path(arms[arm])) != info["dmx_file_sha256"]:
                        raise ValueError("source DMX candidate differs from its per-arm provenance")
        selected.append({**item, "canonical_id": canonical, "audio_path": audio_path,
                         "duration_s": duration, "offset_s": offset, "comparisons": comparisons})
    return selected


def prepare_pairs(root, protocol, candidates: list[dict], n_songs: int = 20, *, renderer=None) -> dict:
    """Publish anonymous videos only after every prospective matching gate passes.

    Selection/order is provided by the frozen candidate list; no model metric
    is consulted. Presentation uses a separate deterministic seed. Repeated
    and sham materials reuse exact encoded bytes and count as controls only.
    A failed attempt leaves diagnostics in its new output directory and does
    not create pairs.json or ratings.jsonl, avoiding an incomplete rating set.
    """
    selected = _validated_candidates(protocol, candidates, n_songs)
    directory = Path(root).resolve() / "results" / "renders"
    directory.parent.mkdir(parents=True, exist_ok=True)
    directory.mkdir(exist_ok=False)
    renderer = renderer or _make_renderer()
    _check_renderer(renderer)
    seed = int(protocol.get("render", {}).get("presentation_seed", 1100))
    rng = np.random.default_rng(seed)
    primary = []
    for song_index, candidate in enumerate(selected):
        wav = extract_audio_excerpt(
            candidate["audio_path"], directory / "audio" / f"song{song_index:03d}.wav",
            candidate["offset_s"], candidate["duration_s"],
        )
        for comparison_index, arms in enumerate(candidate["comparisons"]):
            number = len(primary)
            frame_pair = [_frames(candidate["arms"][arm]) for arm in arms]
            match = render_matched_pair(
                *frame_pair, wav, candidate["duration_s"], directory / "materials" / f"primary{number:03d}",
                renderer=renderer,
            )
            primary.append({
                "uid": f"primary{number:03d}", "kind": "primary", "control": False,
                "canonical_id": candidate["canonical_id"], "music_id": candidate.get("music_id", candidate["canonical_id"]),
                "song_key": candidate["canonical_id"], "offset_s": candidate["offset_s"],
                "clip_s": candidate["duration_s"], "comparison_index": comparison_index,
                "arms": list(arms), "source_audio": str(candidate["audio_path"]),
                "arm_provenance": [candidate.get("arm_provenance", {}).get(arm) for arm in arms],
                "source_frames_sha256": [_hash_frames(f) for f in frame_pair],
                "match": match, "source_primary_uid": None,
            })
    # Four evenly spaced repeats and four disjoint-ish sham sources. This
    # deterministic schedule is independent of scores and brightness gains.
    repeated_indices = np.linspace(0, len(primary) - 1, 4).round().astype(int)
    sham_indices = (repeated_indices + max(1, len(primary) // 8)) % len(primary)
    records = list(primary)
    for index, source in enumerate(repeated_indices):
        p = primary[int(source)]
        records.append({**p, "uid": f"repeat{index:03d}", "kind": "repeat", "control": True,
                        "source_primary_uid": p["uid"]})
    for index, source in enumerate(sham_indices):
        p = primary[int(source)]
        side = index % 2
        sham_match = {**p["match"], "paths": [p["match"]["paths"][side]] * 2,
                      "encoded_video": [p["match"]["encoded_video"][side]] * 2,
                      "frames_sha256": [p["match"]["frames_sha256"][side]] * 2,
                      "symmetric_gap": 0.0}
        records.append({**p, "uid": f"sham{index:03d}", "kind": "sham", "control": True,
                        "arms": [p["arms"][side]] * 2, "match": sham_match,
                        "arm_provenance": [p["arm_provenance"][side]] * 2,
                        "source_frames_sha256": [p["source_frames_sha256"][side]] * 2,
                        "source_primary_uid": p["uid"]})
    order = rng.permutation(len(records))
    ids = {records[int(index)]["uid"]: f"pair{position:03d}" for position, index in enumerate(order)}
    pairs = []
    for index in order:
        record = records[int(index)]
        pid = ids[record["uid"]]
        swap = rng.permutation(2)
        arm_identity, brightness, videos, hashes, arm_provenance = {}, {}, {}, {}, {}
        for tag, side in zip(("A", "B"), swap):
            source = Path(record["match"]["paths"][int(side)])
            target = directory / f"{pid}__{tag}.mp4"
            # copyfile preserves exact repeated/sham bytes without coupling
            # future writes to the diagnostic files via hard links.
            with source.open("rb") as src, target.open("xb") as dst:
                shutil.copyfileobj(src, dst)
            arm_identity[tag] = record["arms"][int(side)]
            arm_provenance[tag] = record["arm_provenance"][int(side)]
            brightness[tag] = record["match"]["encoded_video"][int(side)]
            videos[tag], hashes[tag] = target.name, _hash_file(target)
        pairs.append({
            "id": pid, "canonical_id": record["canonical_id"], "song_key": record["song_key"],
            "music_id": record["music_id"], "offset_s": record["offset_s"], "clip_s": record["clip_s"],
            "kind": record["kind"], "control": record["control"], "comparison_index": record["comparison_index"],
            "arm_identity": arm_identity, "shown_source_indices": [int(v) for v in swap],
            "arm_provenance": arm_provenance,
            "videos": videos, "video_sha256": hashes, "brightness": brightness,
            "symmetric_luma_gap": record["match"]["symmetric_gap"],
            "source_primary_pair": ids.get(record["source_primary_uid"]),
            "source_audio": record["source_audio"], "source_frames_sha256": record["source_frames_sha256"],
            "match_audit": str(Path(record["match"]["paths"][0]).parent.relative_to(directory) / "match.json"),
            "foil_overbright": None,
        })
    protocol_canonical_sha = hashlib.sha256(json.dumps(protocol, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    protocol_path = Path(root).resolve() / "protocol.json"
    protocol_sha = _hash_file(protocol_path) if protocol_path.exists() else protocol_canonical_sha
    result = {
        "schema": "stage11_blind_pairs_v1", "protocol_sha256": protocol_sha,
        "protocol_canonical_sha256": protocol_canonical_sha,
        "n_songs": n_songs, "n_primary": 2 * n_songs, "n_repeat": 4, "n_sham": 4,
        "presentation_seed": seed, "renderer": FIXED_RENDER_CONFIG, "scene_mapping": FIXED_SCENE_MAPPING,
        "luma_tolerance": LUMA_TOLERANCE, "physical_brightness_matched": False,
        "preference_status": "pending human ratings", "metadata_contains_unblinding_key": True,
        "pairs": pairs,
    }
    _write_new_json(directory / "pairs.json", result)
    (directory / "ratings.jsonl").open("x").close()
    return result
