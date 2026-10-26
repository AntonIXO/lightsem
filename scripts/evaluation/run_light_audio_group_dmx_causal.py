#!/usr/bin/env python3
"""Carry cached LightLayers audio-group masks through fixed mapper DMX.

The LightLayers 94-D screens already establish palette-state sensitivity.  This
runner takes those saved palette/cue outputs, applies the same P2 two-group RGB
delivery to the sealed selected-C body+mapper DMX, decodes the resulting DMX,
and reports angular motion, RGB, brightness, inactivity, saturation and exact
duplicate-stream diagnostics.  It does not regenerate audio, body or DMX
geometry, and it never renders video.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def circular_deg(delta: np.ndarray) -> np.ndarray:
    return np.abs((delta + 180.0) % 360.0 - 180.0)


def duplicate_fraction(dmx: np.ndarray, patch: list[dict]) -> tuple[float, list[list[str]]]:
    streams: dict[bytes, list[str]] = {}
    for fixture in patch:
        u = int(fixture["universe"])
        a = int(fixture["address"]) - 1
        w = int(fixture["n_channels"])
        key = np.ascontiguousarray(dmx[:, u, a:a + w]).tobytes()
        streams.setdefault(key, []).append(fixture["name"])
    groups = [names for names in streams.values() if len(names) > 1]
    duplicated = sum(len(names) for names in groups)
    return duplicated / max(len(patch), 1), groups


def _metrics(base_dmx: np.ndarray, candidate_dmx: np.ndarray, patch: list[dict]) -> dict:
    from lightsem.conductor.body_transfer import decode

    base = decode(base_dmx, patch)
    cand = decode(candidate_dmx, patch)
    angle_delta = circular_deg(cand["angles_deg"] - base["angles_deg"])
    angular = np.sqrt(np.sum(angle_delta ** 2, axis=-1))
    moving = [i for i, fixture in enumerate(patch) if fixture["kind"] in ("beam", "laser")]
    rgb_delta = cand["rgb"].astype(np.float64) - base["rgb"].astype(np.float64)
    dim_delta = cand["dimmer"] - base["dimmer"]
    # body_transfer.decode returns normalized RGB/dimmer arrays but no HSV
    # value field; reproduce dmx.decode.emitted_light with max(R,G,B).
    base_luma = (base["dimmer"] * np.maximum(np.max(base["rgb"], axis=-1), 1e-6)).mean(axis=1)
    cand_luma = (cand["dimmer"] * np.maximum(np.max(cand["rgb"], axis=-1), 1e-6)).mean(axis=1)
    luma_delta = cand_luma - base_luma
    duplicate_rate, duplicate_groups = duplicate_fraction(candidate_dmx, patch)
    return {
        "finite_dmx": bool(np.isfinite(candidate_dmx).all()),
        "exact_dmx_equal": bool(np.array_equal(base_dmx, candidate_dmx)),
        "angular_rms_deg_all": float(np.sqrt(np.mean(angular ** 2))),
        "angular_rms_deg_moving": float(np.sqrt(np.mean(angular[:, moving] ** 2))) if moving else 0.0,
        "angular_max_deg": float(np.max(angular)) if angular.size else 0.0,
        "rgb_rms_8bit": float(np.sqrt(np.mean(rgb_delta ** 2))),
        "brightness_rms": float(np.sqrt(np.mean(luma_delta ** 2))),
        "dimmer_rms": float(np.sqrt(np.mean(dim_delta ** 2))),
        "brightness_mean": float(np.mean(cand_luma)),
        "dimmer_mean": float(np.mean(cand["dimmer"])),
        "brightness_inactive_fraction": float(np.mean(cand["dimmer"] < 0.02)),
        "brightness_saturated_low_fraction": float(np.mean(cand["dimmer"] <= (1.0 / 255.0))),
        "brightness_saturated_high_fraction": float(np.mean(cand["dimmer"] >= (254.0 / 255.0))),
        "rgb_zero_channel_fraction": float(np.mean(cand["rgb"] == 0)),
        "rgb_full_channel_fraction": float(np.mean(cand["rgb"] == 255)),
        "rgb_inactive_fixture_fraction": float(np.mean(np.all(cand["rgb"] == 0, axis=-1))),
        "exact_duplicate_fixture_rate": float(duplicate_rate),
        "exact_duplicate_groups": duplicate_groups,
    }


def _palette_rgb(prototypes: np.ndarray) -> np.ndarray:
    """Reproduce the fixed P2 two-group palette choices without canonical-rig assumptions."""
    from lightsem.experiments.visual_next.delivery import global_palette_rgb
    from lightsem.experiments.visual_next.palettes import prototype_representatives

    global_rgb = global_palette_rgb(prototypes)
    representatives = np.asarray(
        [[m["rgb"] for m in prototype_representatives(p)] for p in prototypes],
        dtype=np.float64,
    )
    return np.concatenate([global_rgb[:, None, :], representatives], axis=1)


def _apply_two_group_palette(base_dmx: np.ndarray, palette_ids10: np.ndarray,
                             cues10: np.ndarray, patch: list[dict],
                             prototypes: np.ndarray) -> tuple[np.ndarray, dict]:
    """Apply the P2 two-group RGB policy directly to the arbitrary-N DMX patch.

    The sealed native mapper uses 63 fixtures, while ``color_delivery`` is bound
    to the 14-fixture canonical review rig. This implementation preserves every
    custom patch channel except RGB, assigns palette mode 1 to stage-left and
    mode 2 to stage-right, and uses the same 300-ms causal smoothstep fades.
    """
    from lightsem.model.light_layers.transitions import fade_rgb_targets

    dmx = np.asarray(base_dmx, dtype=np.uint8)
    if dmx.ndim != 2 or dmx.shape[1] != 512:
        raise ValueError(f"Expected (T,512) DMX frames, got {dmx.shape}")
    ids = np.asarray(palette_ids10)
    cues = np.asarray(cues10, dtype=bool)
    if ids.ndim != 1 or cues.shape != ids.shape or len(ids) * 4 != len(dmx):
        raise ValueError("Palette state timeline must exactly cover the 40-Hz prefix")
    if np.any((ids < 0) | (ids >= len(prototypes))):
        raise ValueError("Palette state outside frozen vocabulary")
    times = np.arange(len(dmx), dtype=np.float64) / 40.0
    ids40 = np.repeat(ids.astype(np.int64), 4)
    cue_times = np.flatnonzero(cues) / 10.0
    active = np.zeros(len(dmx), dtype=bool) if not len(cue_times) else times >= cue_times[0]
    choices = _palette_rgb(prototypes)
    out = dmx.copy()
    rgb_fixture_count = 0
    for fixture in patch:
        kind = fixture["kind"]
        if kind not in ("beam", "laser", "bulb"):
            continue
        address = int(fixture["address"]) - 1
        # Moving-head/laser: four 16-bit angle bytes, then dimmer, then RGB.
        # Bulb: dimmer, then RGB.
        rgb_offset = 5 if kind in ("beam", "laser") else 1
        slots = np.arange(address + rgb_offset, address + rgb_offset + 3)
        original = out[:, slots].astype(np.float64) / 255.0
        mode = 1 if float(fixture.get("position", [0.0])[0]) < 0.0 else 2
        target = choices[ids40, mode]
        track = original.copy()
        track[active] = target[active]
        faded = fade_rgb_targets(track, times, duration_s=0.3, active=active)
        out[:, slots] = np.floor(np.clip(faded, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        rgb_fixture_count += 1
    if not np.isfinite(out).all():
        raise ValueError("Nonfinite RGB overlay")
    return out, {
        "layout": "two_groups",
        "mode_left": 1,
        "mode_right": 2,
        "fade_s": 0.3,
        "color_cue_count": int(cues.sum()),
        "active_color_seconds": float(active.sum() / 40.0),
        "rgb_fixture_count": rgb_fixture_count,
        "scope": "RGB-only overlay on sealed arbitrary-N mapper patch; body, angles and dimmers held fixed",
    }


def run_song(args, song: str) -> dict:
    from lightsem.conductor.body_transfer import decode

    group_root = Path(args.group_root).resolve()
    baseline_root = Path(args.baseline_root).resolve()
    transforms = json.loads(Path(args.transforms).resolve().read_text())
    body_dir = baseline_root / song
    lights_path = body_dir / "lights.npz"
    patch_path = body_dir / "patch.json"
    generation_path = body_dir / "GENERATION.json"
    if not lights_path.exists() or not patch_path.exists() or not generation_path.exists():
        raise FileNotFoundError(f"Missing sealed body+mapper DMX for {song}")
    with np.load(lights_path, allow_pickle=False) as z:
        base_dmx_full = np.asarray(z["universes"], dtype=np.uint8)
        body = np.asarray(z["body"], dtype=np.float64)
        base_time = np.asarray(z["time"], dtype=np.float64)
    patch = json.loads(patch_path.read_text())
    generation = json.loads(generation_path.read_text())
    raw_safety = generation.get("raw_safety_gate", {})
    if raw_safety.get("status") != "pass":
        raise ValueError(f"Selected body lacks a passing sealed raw-safety receipt: {song}")
    if base_dmx_full.ndim != 3 or base_dmx_full.shape[1:] != (1, 512):
        raise ValueError(f"Unexpected DMX shape for {song}: {base_dmx_full.shape}")
    if not np.isfinite(body).all() or not np.isfinite(base_time).all():
        raise ValueError(f"Nonfinite fixed body/time for {song}")

    arm_reports = sorted(([group_root / "REPORT.json"] if (group_root / "REPORT.json").exists() else []) + list(group_root.glob("*/REPORT.json")))
    if not arm_reports:
        raise FileNotFoundError(f"No group reports in {group_root}")
    arms = {}
    for report_path in arm_reports:
        report = json.loads(report_path.read_text())
        if report.get("song") != song:
            continue
        for npz_path in sorted(report_path.parent.glob("*.npz")):
            arm = npz_path.stem
            with np.load(npz_path, allow_pickle=False) as z:
                if "generated_palette_ids" not in z or "color_cues" not in z:
                    continue
                ids10 = np.asarray(z["generated_palette_ids"])
                cues10 = np.asarray(z["color_cues"], dtype=bool)
            arms[arm] = (ids10, cues10, report_path)
    if "full" not in arms:
        raise ValueError(f"Full arm missing for {song}")

    out = Path(args.out).resolve() / song
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    # The saved body/mapper DMX is 40 Hz.  Cached LightLayers outputs are 10 Hz;
    # compare the common prefix and preserve exact provenance rather than pad.
    for arm, (ids10, cues10, report_path) in arms.items():
        if ids10.ndim != 1 or cues10.shape != ids10.shape:
            raise ValueError(f"Malformed LightLayers arm {song}/{arm}")
        ids40 = np.repeat(ids10.astype(np.int64), 4)
        cues40 = np.repeat(cues10, 4)
        length = min(len(base_dmx_full), len(ids40))
        if length < 4:
            raise ValueError(f"Too few aligned frames for {song}/{arm}")
        # Keep a whole 10-Hz state prefix to avoid partial state comparisons.
        length -= length % 4
        base_dmx = base_dmx_full[:length, 0, :]
        ids10_aligned = ids10[:length // 4]
        cues10_aligned = cues10[:length // 4]
        times = np.arange(length, dtype=np.float64) / 40.0
        if arm == "full":
            candidate_dmx = base_dmx.copy()
            delivery = {
                "layout": "two_groups",
                "history": "sealed selected-C P2 full arm",
                "fade_s": 0.3,
                "color_cue_count": int(cues10_aligned.sum()),
                "active_color_seconds": float(cues10_aligned.sum() / 10.0),
                "scope": "full arm is the fixed selected-C body+mapper DMX reference",
            }
        else:
            candidate_dmx, delivery = _apply_two_group_palette(
                base_dmx, ids10_aligned, cues10_aligned, patch,
                np.asarray(transforms["palette_prototypes"], dtype=np.float64),
            )
        candidate_dmx_3d = candidate_dmx[:, None, :]
        metrics = _metrics(base_dmx[:, None, :], candidate_dmx_3d, patch)
        np.savez_compressed(
            out / f"{arm}.npz",
            time=times,
            baseline_dmx=base_dmx[:, None, :],
            candidate_dmx=candidate_dmx_3d,
            palette_ids10=ids10_aligned,
            color_cues10=cues10_aligned,
        )
        rows.append({
            "arm": arm,
            "source_palette_report": str(report_path),
            "source_palette_report_sha256": sha(report_path),
            "frames": int(length),
            "seconds": float(length / 40.0),
            "palette_frames": int(len(ids10_aligned)),
            "delivery": delivery,
            **metrics,
        })

    base_row = next(row for row in rows if row["arm"] == "full")
    with np.load(out / "full.npz", allow_pickle=False) as z:
        full_candidate = np.asarray(z["candidate_dmx"], dtype=np.uint8)
    for row in rows:
        with np.load(out / f"{row['arm']}.npz", allow_pickle=False) as z:
            candidate = np.asarray(z["candidate_dmx"], dtype=np.uint8)
        effect = _metrics(full_candidate, candidate, patch)
        row["effect_vs_full"] = {
            "angular_rms_deg_moving": effect["angular_rms_deg_moving"],
            "rgb_rms_8bit": effect["rgb_rms_8bit"],
            "brightness_rms": effect["brightness_rms"],
            "exact_dmx_equal": effect["exact_dmx_equal"],
        }
    for row in rows:
        row["delta_vs_full_rgb_rms_8bit"] = float(row["rgb_rms_8bit"] - base_row["rgb_rms_8bit"])
        row["delta_vs_full_brightness_rms"] = float(row["brightness_rms"] - base_row["brightness_rms"])
    return {
        "song": song,
        "lights_path": str(lights_path),
        "lights_sha256": sha(lights_path),
        "patch_path": str(patch_path),
        "patch_sha256": sha(patch_path),
        "generation_path": str(generation_path),
        "generation_sha256": sha(generation_path),
        "raw_safety_gate": raw_safety,
        "body_sha256": hashlib.sha256(body.tobytes()).hexdigest(),
        "frames_available": int(len(base_dmx_full)),
        "rows": rows,
        "fixed_body_finite": True,
        "mapper": "sealed mapper-01 selected-C DMX; RGB overlay via P2 two_groups",
        "scope": "cached LightLayers audio masks -> fixed body+mapper DMX; no geometry/audio regeneration/video",
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--group-root", required=True, type=Path, help="directory containing song group REPORT.json and arm npz files")
    p.add_argument("--baseline-root", required=True, type=Path, help="root containing song/lights.npz and song/patch.json")
    p.add_argument("--transforms", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--songs", default="shpongle,cure_for_me")
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    songs = [s.strip() for s in args.songs.split(",") if s.strip()]
    results = [run_song(args, song) for song in songs]
    report = {
        "schema_version": 1,
        "status": "diagnostic_only",
        "qualification_status": "unqualified_cached_palette_to_dmx",
        "group_root": str(args.group_root.resolve()),
        "baseline_root": str(args.baseline_root.resolve()),
        "transforms": str(args.transforms.resolve()),
        "transforms_sha256": sha(args.transforms.resolve()),
        "songs": results,
        "safety": {
            "raw_body_recomputed": False,
            "fixed_body_finite": all(result["fixed_body_finite"] for result in results),
            "geometry_intervention": "none; sealed selected-C body and mapper DMX held fixed",
            "audio_intervention": "LightLayers 94-D group masks from cached two-song screen",
        },
        "missing_evidence": [
            "authenticated raw waveform own-song recomputation",
            "learned output-stage audio path beyond cached palette-to-RGB overlay",
            "audio-induced angular DMX motion (expected null under fixed body)",
        ],
    }
    (args.out / "PORTFOLIO_RECEIPT.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
