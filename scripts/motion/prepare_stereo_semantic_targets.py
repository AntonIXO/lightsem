#!/usr/bin/env python3
"""Build source-aware stereo targets from MUSDB TRAIN/DEV stems.

The mixture descriptors are the model input.  The four MUSDB stems are used
only to construct authenticated offline targets: source event/activity,
per-source stereo image, and pairwise source relations.  MUSDB TEST is
rejected explicitly.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from lightsem.audio.stereo_v1 import (
    FEATURE_NAMES, SPATIAL_NAMES, StereoAudioV1, extract_stereo_v1,
    resample_stream, sha256,
)
from lightsem.experiments.audio_stereo.semantic import (
    SOURCE_EVENT_NAMES, SOURCE_RELATION_NAMES, SOURCE_SPATIAL_NAMES, STEM_NAMES,
)


# Indices are bound to the StereoAudioV1 schema, never inferred by position in
# this script's output.
FEATURE_INDEX = {name: i for i, name in enumerate(FEATURE_NAMES)}
SPATIAL_INDEX = {name: i for i, name in enumerate(SPATIAL_NAMES)}
EVENT_INDEX = {name: i for i, name in enumerate((
    "onset_left", "onset_right", "onset_mid", "onset_side",
    "flux_left", "flux_right", "transient_ratio", "event_strength"))}


def write_json(path: Path, value: dict) -> None:
    if path.exists():
        raise FileExistsError(f"Preserving existing semantic artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    path.with_name(path.name + ".sha256.json").write_text(json.dumps({"sha256": sha256(path)}) + "\n")


def _load_audio(path: Path) -> tuple[np.ndarray, int]:
    wave, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    return wave.T, int(sample_rate)


def _align(value: np.ndarray, source_time: np.ndarray, target_time: np.ndarray) -> np.ndarray:
    return resample_stream(value, source_time, target_time)


def _stem_targets(descriptors: dict[str, StereoAudioV1], mixture: StereoAudioV1) -> tuple[np.ndarray, np.ndarray]:
    event_rows, spatial_rows = [], []
    for stem in STEM_NAMES:
        descriptor = descriptors[stem]
        energy = descriptor.features_40[:, [FEATURE_INDEX["left_log_rms"], FEATURE_INDEX["right_log_rms"]]].mean(1)
        onset_left = descriptor.events_40[:, EVENT_INDEX["onset_left"]]
        onset_right = descriptor.events_40[:, EVENT_INDEX["onset_right"]]
        active = (energy > -4.0).astype(np.float32)
        event = np.stack((energy, active, onset_left, onset_right), axis=1)
        event_rows.append(_align(event, descriptor.time_40, mixture.time_40))
        spatial_image = descriptor.spatial_10[:, [
            SPATIAL_INDEX["ild_db"], SPATIAL_INDEX["width_fraction"],
            SPATIAL_INDEX["pan_balance"], SPATIAL_INDEX["coherence"],
        ]]
        energy_10 = _align(
            descriptor.features_40[:, [FEATURE_INDEX["mid_log_rms"]]],
            descriptor.time_40, mixture.time_10,
        )
        spatial_rows.append(np.concatenate((
            _align(spatial_image, descriptor.time_10, mixture.time_10), energy_10,
        ), axis=1))
    event_target = np.concatenate(event_rows, axis=1).astype(np.float32)
    spatial_target = np.concatenate(spatial_rows, axis=1).astype(np.float32)
    return event_target, spatial_target


def _relation_targets(descriptors: dict[str, StereoAudioV1], mixture: StereoAudioV1) -> np.ndarray:
    # Pairwise relations are derived from source event-strength envelopes. The
    # lag convention matches StereoAudioV1: positive lag means the left member
    # of the pair leads the right member.
    events = {}
    for stem in STEM_NAMES:
        descriptor = descriptors[stem]
        source = descriptor.events_40[:, EVENT_INDEX["event_strength"]][:, None]
        events[stem] = _align(source, descriptor.time_40, mixture.time_40)[:, 0]
    rows = []
    max_lag = 20  # 0.5 s at 40 Hz.
    half_window = 80  # 2 s on each side at the 40 Hz event clock.
    centers = np.rint(mixture.time_2 * 40.0).astype(int)
    for center in centers:
        frame_row = []
        for i, left_name in enumerate(STEM_NAMES):
            for right_name in STEM_NAMES[i + 1:]:
                left_full, right_full = events[left_name], events[right_name]
                lo = max(0, center - half_window); hi = min(len(left_full), center + half_window + 1)
                left = left_full[lo:hi].astype(np.float64); right = right_full[lo:hi].astype(np.float64)
                left = left - left.mean(); right = right - right.mean()
                correlations, lags = [], []
                for lag in range(-max_lag, max_lag + 1):
                    if lag < 0:
                        a, b = left[:lag], right[-lag:]
                    elif lag > 0:
                        a, b = left[lag:], right[:-lag]
                    else:
                        a, b = left, right
                    denom = np.linalg.norm(a) * np.linalg.norm(b)
                    correlations.append(float(np.dot(a, b) / denom) if denom > 1e-8 else 0.)
                    lags.append(lag)
                best = int(np.argmax(correlations))
                confidence = max(0., correlations[best] - float(np.median(correlations)))
                frame_row.extend((float(correlations[best]), lags[best] / 40., confidence))
        rows.append(frame_row)
    return np.asarray(rows, dtype=np.float32)


def build(manifest_path: Path, out: Path, *, limit: int | None = None) -> dict:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("test_access", True):
        raise ValueError("Semantic target preparation requires test_access=false")
    entries = [row for row in manifest.get("entries", []) if row.get("split") in ("train", "dev", "val")]
    if limit is not None:
        entries = entries[:limit]
    if not entries:
        raise ValueError("No TRAIN/DEV entries available")
    rows = []
    for row in entries:
        mixture_path = Path(row["audio_path"])
        if mixture_path.name != "mixture.wav":
            raise ValueError(f"Expected mixture.wav, got {mixture_path}")
        mixture_wave, mixture_sr = _load_audio(mixture_path)
        mixture = StereoAudioV1.load(row["feature_path"])
        descriptors = {}
        source_paths = {}
        for stem in STEM_NAMES:
            source_path = mixture_path.parent / f"{stem}.wav"
            if not source_path.exists():
                raise FileNotFoundError(source_path)
            wave, sr = _load_audio(source_path)
            if sr != mixture_sr:
                raise ValueError(f"sample-rate mismatch in {row['id']}: {sr} vs {mixture_sr}")
            descriptors[stem] = extract_stereo_v1(wave, sr)
            source_paths[stem] = {"path": str(source_path), "sha256": sha256(source_path)}
        event, spatial = _stem_targets(descriptors, mixture)
        relation = _relation_targets(descriptors, mixture)
        destination = out / "targets" / f"{row['id'].replace(':', '_')}.npz"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(destination)
        with destination.open("xb") as handle:
            np.savez_compressed(handle, time_40=mixture.time_40, source_event_40=event,
                                time_10=mixture.time_10, source_spatial_10=spatial,
                                time_2=mixture.time_2, source_relation_2=relation)
        rows.append({"id": row["id"], "split": row.get("split"), "mixture_path": str(mixture_path),
                     "mixture_sha256": sha256(mixture_path), "mixture_feature_path": row["feature_path"],
                     "mixture_feature_sha256": sha256(row["feature_path"]), "stems": source_paths,
                     "target_path": str(destination), "target_sha256": sha256(destination),
                     "frames_40": len(event), "frames_10": len(spatial), "frames_2": len(relation)})
    result = {"schema_version": 1, "status": "complete", "test_access": False,
              "source_manifest": str(manifest_path), "source_manifest_sha256": sha256(manifest_path),
              "thresholds": {"active_log_energy_gt": -4.0}, "stem_names": list(STEM_NAMES),
              "source_event_names": list(SOURCE_EVENT_NAMES), "source_spatial_names": list(SOURCE_SPATIAL_NAMES),
              "source_relation_names": list(SOURCE_RELATION_NAMES), "entries": rows,
              "spatial_semantics": "source stereo image inferred from mastered mixture; not physical 3D labels"}
    write_json(out / "STEREO_SEMANTIC_TARGET_MANIFEST.json", result)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    result = build(args.manifest, args.out, limit=args.limit)
    print(json.dumps({"status": result["status"], "rows": len(result["entries"]), "test_access": result["test_access"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
