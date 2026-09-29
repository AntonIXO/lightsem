#!/usr/bin/env python3
"""Derive a 2x2 review from authenticated variant06 outputs, without inference."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from lightsem.conductor.rig import canonical_rig
from lightsem.model.light_layers.overlay import apply_overlay
from lightsem.model.light_layers.transitions import fade_rgb_targets

RGB_OFFSETS = {"moving_head": (6, 7, 8), "wash": (2, 3, 4), "projector": (2, 3, 4)}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def replay_choices(choices, action_indices, frames, *, history):
    """Replay the saved all-previous-state GPU table; final column is trained null."""
    choices, indices = np.asarray(choices), np.asarray(action_indices)
    if (choices.ndim != 2 or choices.shape[1] < 2 or choices.dtype.kind not in "iu"
            or indices.ndim != 1 or indices.dtype.kind not in "iu"
            or len(indices) != len(choices) or not len(indices)
            or indices[0] != 0 or np.any(np.diff(indices.astype(np.int64)) <= 0)
            or indices[-1] >= frames or np.any(choices >= choices.shape[1] - 1)
            or np.any(choices < 0)):
        raise ValueError("invalid saved palette action table")
    null_id = choices.shape[1] - 1
    current = null_id
    generated = np.empty(frames, np.int64)
    previous = np.empty(frames, np.int64)
    for j, index in enumerate(indices):
        previous_id = current if history else null_id
        current = int(choices[j, previous_id])
        end = int(indices[j + 1]) if j + 1 < len(indices) else frames
        generated[index:end] = current
        previous[index] = previous_id
        previous[index + 1:end] = current if history else null_id
    return generated, previous


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", required=True, type=Path)
    args = ap.parse_args()
    plan = json.loads(args.plan.read_text())
    out = Path(plan["output_root"])
    for entry in plan["inputs"].values():
        if sha(entry["path"]) != entry["sha256"]:
            raise RuntimeError("Input changed: " + entry["path"])
    for path, expected in plan["source_files"].items():
        if sha(path) != expected:
            raise RuntimeError("Bound source changed: " + path)
    with np.load(plan["inputs"]["overlay"]["path"], allow_pickle=False) as z:
        original = z["overlay"].copy()
        times, active, palette_rgb = z["time"], z["color_active40"], z["palette_rgb"]
        allowed, accent_times = z["allowed_slot_mask"], z["accent_times_s"]
        saved_ids40 = z["palette_ids40"]
    with np.load(plan["inputs"]["model_outputs"]["path"], allow_pickle=False) as z:
        own_ids, previous = replay_choices(z["action_argmax_choices"], z["action_indices"],
                                           len(z["generated_palette_ids"]), history=True)
        null_ids, _ = replay_choices(z["action_argmax_choices"], z["action_indices"],
                                     len(own_ids), history=False)
        assert np.array_equal(own_ids, z["generated_palette_ids"])
        assert np.array_equal(previous, z["previous_ids"])
        assert np.array_equal(previous[z["action_indices"]], z["action_previous_ids"])
        assert np.array_equal(np.unique(np.r_[0, np.flatnonzero(z["color_cues"])]), z["action_indices"])
        assert np.array_equal(np.repeat(own_ids, 4), saved_ids40)
        cue_times = np.flatnonzero(z["color_cues"]) / 10
    assert original.dtype == np.uint8 and original.shape == (11092, 512)
    assert np.array_equal(times, np.arange(len(original)) / 40)
    assert len(accent_times) == 30 and len(cue_times) == 262
    assert int((~allowed).sum()) == 469
    rig = canonical_rig()
    rgb_mask = np.zeros(512, bool)
    own_tracks, null_tracks = {}, {}
    for f in rig.fixtures:
        if f.kind not in RGB_OFFSETS:
            continue
        slots = f.address - 1 + np.array(RGB_OFFSETS[f.kind])
        rgb_mask[slots] = True
        own_tracks[f.name] = original[:, slots].astype(float) / 255
        expected = np.floor(palette_rgb[np.repeat(own_ids, 4)] * 255 + .5).astype(np.uint8)
        assert np.array_equal(original[np.ix_(active, slots)], expected[active])
        null_tracks[f.name] = own_tracks[f.name].copy()
        # The same channel quantization is applied in both arms before fades.
        null_tracks[f.name][active] = np.floor(
            palette_rgb[np.repeat(null_ids, 4)][active] * 255 + .5) / 255
    assert np.array_equal(apply_overlay(original, rig, times, rgb=own_tracks), original)
    receipt = {"scope": "Engineering 2x2 human review, no new hypothesis test or training",
               "encoder_forwards": 0, "palette_head_forwards": 0,
               "saved_GPU_own_state_replay_exact": True,
               "original_model_outputs_unchanged": True,
               "cue_times_identical": True, "accent_times_identical": True,
               "color_cue_count": len(cue_times), "accent_count": len(accent_times),
               "history_differing_frames10": int(np.count_nonzero(own_ids != null_ids)),
               "history_differing_seconds": float(np.count_nonzero(own_ids != null_ids) / 10),
               "variants": []}
    changes = np.flatnonzero(np.diff((own_ids != null_ids).astype(int), prepend=0, append=0))
    receipt["history_difference_intervals_s"] = [[float(a/10), float(b/10)]
                                                   for a, b in zip(changes[::2], changes[1::2])]
    for variant in plan["variants"]:
        tracks = own_tracks if variant["history"] else null_tracks
        if variant["transition_s"]:
            tracks = {name: fade_rgb_targets(track, times, duration_s=variant["transition_s"], active=active)
                      for name, track in tracks.items()}
        stream = apply_overlay(original, rig, times, rgb=tracks)
        assert np.array_equal(stream[:, ~rgb_mask], original[:, ~rgb_mask])
        assert np.array_equal(stream[:, ~allowed], original[:, ~allowed])
        if variant["id"] == "01":
            assert np.array_equal(stream, original)
        if shutil.disk_usage(out).free < 2**30 + 8 * 2**20:
            raise RuntimeError("Disk floor")
        path = out / "streams" / (variant["id"] + ".npz")
        path.parent.mkdir(exist_ok=True)
        with path.open("xb") as f:
            np.savez_compressed(f, candidate=stream, time=times)
        with np.load(path, allow_pickle=False) as z:
            assert np.array_equal(z["candidate"], stream)
        ids = own_ids if variant["history"] else null_ids
        ids_active = np.repeat(ids, 4)[active]
        steps = np.abs(np.diff(stream[:, rgb_mask].astype(int), axis=0))
        receipt["variants"].append({**variant, "streams": str(path), "streams_sha256": sha(path),
            "dmx_sha256": hashlib.sha256(stream.tobytes()).hexdigest(),
            "protected_slots_unchanged": int((~rgb_mask).sum()),
            "changed_bytes_vs_selected_06": int(np.count_nonzero(stream != original)),
            "subsequent_target_palette_changes": int(np.count_nonzero(np.diff(ids_active))),
            "max_RGB_channel_step_byte": int(steps.max()),
            "step_metric_scope": "channel discontinuity only; not perceived color distance/quality"})
    for entry in plan["inputs"].values():
        assert sha(entry["path"]) == entry["sha256"]
    receipt["status"] = "prepared_and_verified"
    save(out / "PREPARATION.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
