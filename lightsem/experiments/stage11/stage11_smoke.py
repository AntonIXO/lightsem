"""Training-only, explicitly non-scientific Stage 11 integration smoke.

Reads completed versioned features for two *old training* canonical songs.
The isolated engineering root never accesses validation/test target arrays and
never contributes time or scores to the scientific campaign. The three model
arms execute sequentially while holding the campaign accelerator lock.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import math
import time

import numpy as np

from .stage11_common import (
    ROOT, artifact_manifest, authenticated_json, digest, gpu_lock, json_digest,
    read_authenticated, read_json, verify_artifacts,
)


ARMS = ("D_old", "S_pointwise", "S_temporal")


def _verify_state_artifacts(state):
    if state.get("status") != "ok" or not state.get("artifacts"):
        raise RuntimeError("Only completed feature/audio states are eligible for the smoke")
    for path, expected in state["artifacts"].items():
        if digest(path) != expected:
            raise RuntimeError("Prepared source artifact changed: " + path)


def _select_training_recordings(campaign_root):
    data = campaign_root / "data"
    scan = read_json(data / "recordings_scan.json")
    canonical = read_json(data / "canonical_map.json")
    selected, identities = [], set()
    # Metadata determines eligibility before any audio/visual/motion arrays load.
    for row in sorted(scan, key=lambda item: item["recording_id"]):
        rid = row["recording_id"]
        identity = canonical["records"][rid]
        feature_path = data / "feature_state" / f"{rid}.json"
        if (not row.get("is_old") or not row.get("eligible_core")
                or identity["partition"] != "train" or not feature_path.exists()
                or identity["canonical_id"] in identities):
            continue
        feature = read_json(feature_path)
        audio_path = data / "audio_state" / f"{rid}.json"
        if feature.get("status") != "ok" or not audio_path.exists():
            continue
        audio = read_json(audio_path)
        if audio.get("status") != "ok":
            continue
        if (feature["recording_id"] != rid or not feature.get("coaches")
                or any(coach["canonical_id"] != identity["canonical_id"] or coach["partition"] != "train"
                       for coach in feature["coaches"])):
            raise RuntimeError("Feature state does not agree with the old-training identity map")
        _verify_state_artifacts(feature)
        _verify_state_artifacts(audio)
        selected.append({"scan": row, "identity": identity, "feature": feature, "audio": audio,
                         "feature_state_path": str(feature_path), "audio_state_path": str(audio_path)})
        identities.add(identity["canonical_id"])
        if len(selected) == 2:
            break
    if len(selected) != 2:
        raise RuntimeError("Two completed old-training canonical recordings are not available yet")
    return selected


def prepare_smoke_fixture(campaign_root=ROOT):
    from .__main__ import initialize_protocol
    from .stage11_metrics import fit_scales
    from .stage11_training import A_NAMES, B_NAMES, C_NAMES, _audio_features
    from lightsem.motion.io import load_bus, load_descriptors, slice_bus

    campaign_root = Path(campaign_root)
    root = campaign_root / "engineering_integration"
    fixture_path = root / "fixture.json"
    if fixture_path.exists():
        existing = read_authenticated(fixture_path)
        verify_artifacts(existing["artifacts"])
        if not existing.get("engineering_only") or existing.get("target_partitions_read") != ["train"]:
            raise RuntimeError("Existing smoke fixture is not explicitly training-only")
        return root, existing
    selected = _select_training_recordings(campaign_root)
    protocol = initialize_protocol(root)
    protocol.update({"engineering_only": True, "exclude_from_campaign_gpu_budget": True,
                     "engineering_steps": 100, "source_campaign_root": str(campaign_root.resolve())})
    authenticated_json(root / "protocol.json", protocol)
    motion, scene, motion_y, motion_x, hazards, scene_x, scene_y = [], [], [], [], [], [], []
    provenance, source_files = [], []
    for selected_row in selected:
        row, identity = selected_row["scan"], selected_row["identity"]
        feature, audio = selected_row["feature"], selected_row["audio"]
        rid = row["recording_id"]
        bus = load_bus(Path(audio["audio_path"]))
        if bus.fps != 40:
            raise RuntimeError("Smoke descriptor audio requires 40 Hz source features")
        used_audio = False
        for coach in feature["coaches"]:
            descriptor = load_descriptors(Path(coach["motion_path"]))
            if descriptor.fps != 40:
                raise RuntimeError("Smoke descriptor targets must be 40 Hz")
            offset, duration = float(coach["start_s"]), float(coach["duration_s"])
            sliced = slice_bus(bus, duration_s=duration, offset_s=offset)
            n = min(len(descriptor.times), len(sliced.times))
            x = _audio_features(sliced)[:n]
            y = np.stack([descriptor.channels[name][:n] for name in A_NAMES + C_NAMES], axis=1).astype(np.float32)
            b = np.zeros((n, len(B_NAMES)), dtype=np.float32)
            for column, name in enumerate(B_NAMES):
                indices = np.rint(np.asarray(descriptor.events.get(name, [])) * sliced.fps).astype(int)
                indices = indices[(indices >= 0) & (indices < n)]
                b[indices, column] = 1
            if x.shape != (n, 92) or y.shape != (n, 21) or n < 160:
                raise RuntimeError("Smoke descriptor tensor contract mismatch")
            if not all(np.isfinite(values).all() for values in (x, y, b)):
                raise RuntimeError("Smoke descriptor input is nonfinite")
            path = root / "tensors" / "motion" / f"{coach['coach_id']}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, x=x, y=y, b=b)
            motion.append({"key": coach["coach_id"], "canonical_id": identity["canonical_id"],
                           "recording_id": rid, "partition": "train", "is_old": True,
                           "path": str(path), "sha256": digest(path), "n_frames": n,
                           "offset_s": offset, "duration_s": n / 40})
            motion_y.append(y[np.linspace(0, n - 1, min(n, 2048), dtype=int)])
            hazards.append(b)
            if not used_audio:
                motion_x.append(x)
                used_audio = True
        with np.load(feature["visual_path"], allow_pickle=False) as visual:
            meta = json.loads(str(visual["meta"]))
            if (meta["partition"] != "train" or meta["canonical_id"] != identity["canonical_id"]
                    or meta["normalization_scope"] != "complete_active_recording_before_crop"):
                raise RuntimeError("Smoke scene target provenance is not training-only/full-recording normalized")
            x, y = visual["x"].astype(np.float32), visual["y"].astype(np.float32)
        if len(x) < 600 or x.shape != (len(y), 52) or y.shape[1] != 6:
            raise RuntimeError("Smoke scene tensor contract mismatch")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise RuntimeError("Smoke scene input is nonfinite")
        path = root / "tensors" / "scene" / f"{rid}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, x=x, y=y)
        scene.append({"key": rid, "canonical_id": identity["canonical_id"], "recording_id": rid,
                      "partition": "train", "is_old": True, "path": str(path), "sha256": digest(path),
                      "n_frames": len(x), "offset_s": row["active_start_s"], "duration_s": len(x) / 10})
        scene_x.append(x)
        scene_y.append(y)
        provenance.append({"recording_id": rid, "canonical_id": identity["canonical_id"],
                           "is_old": True, "partition": "train", "source_start_s": row["active_start_s"],
                           "motion_clips": len(feature["coaches"]), "scene_frames": len(x),
                           "feature_state_sha256": digest(selected_row["feature_state_path"]),
                           "audio_state_sha256": digest(selected_row["audio_state_path"])})
        source_files.extend([selected_row["feature_state_path"], selected_row["audio_state_path"]])
        source_files.extend(feature["artifacts"])
        source_files.extend(audio["artifacts"])
    mx, my, mb = np.concatenate(motion_x), np.concatenate(motion_y), np.concatenate(hazards)
    sx, sy = np.concatenate(scene_x), np.concatenate([values[::10] for values in scene_y])
    probability = np.clip(mb.mean(axis=0, dtype=np.float64), 1e-4, 1 - 1e-4)
    norms = {
        "motion": {"x_mean": mx.mean(axis=0, dtype=np.float64).tolist(),
                   "x_std": np.maximum(mx.astype(float).std(axis=0), 1e-6).tolist(),
                   "y_center": np.median(my, axis=0).tolist(),
                   "y_scale": np.maximum(1.5 * (np.quantile(my, 0.75, axis=0) - np.quantile(my, 0.25, axis=0)), 1e-3).tolist(),
                   "pos_weight": ((1 - probability) / probability).tolist(), "a_names": list(A_NAMES),
                   "b_names": list(B_NAMES), "c_names": list(C_NAMES),
                   "fit_population": "engineering_two_old_training_recordings_only"},
        "scene": {"x_mean": sx.mean(axis=0, dtype=np.float64).tolist(),
                  "x_std": np.maximum(sx.astype(float).std(axis=0), 1e-6).tolist(),
                  "y_center": sy.mean(axis=0).tolist(), "y_scale": np.maximum(sy.std(axis=0), 1e-3).tolist(),
                  "fit_population": "engineering_two_old_training_recordings_only"},
        "scene_metrics": fit_scales(scene_y, fps=10),
    }
    authenticated_json(root / "normalizers.json", norms)
    data = {"schema_version": 1, "engineering_only": True,
            "manifest_sha256": json_digest(provenance), "normalizers_path": str(root / "normalizers.json"),
            "normalizers_sha256": digest(root / "normalizers.json"), "motion": motion, "scene": scene}
    authenticated_json(root / "training_data.json", data)
    files = [root / "protocol.json", root / "normalizers.json", root / "training_data.json"]
    files += [Path(entry["path"]) for entry in motion + scene]
    result = {"schema_version": 1, "engineering_only": True, "exclude_from_campaign_gpu_budget": True,
              "target_partitions_read": ["train"], "validation_is_training_reconstruction": True,
              "canonical_count": 2, "selected_recordings": provenance,
              "input_provenance": artifact_manifest(sorted(set(source_files))),
              "identity_metadata_sha256": {"recordings_scan": digest(campaign_root / "data" / "recordings_scan.json"),
                                           "canonical_map": digest(campaign_root / "data" / "canonical_map.json")},
              "artifacts": artifact_manifest(files), "created_at": time.time()}
    authenticated_json(fixture_path, result)
    return root, result


def _verify_engineering_run(root, arm, steps):
    import torch
    from .stage11_training import _model, _selection_score, evaluate_motion, evaluate_scene

    directory = root / "engineering_runs" / arm / "seed_0"
    report = read_authenticated(directory / "report.json")
    verify_artifacts(report["artifacts"])
    validation_path = directory / f"validation_step_{steps}.json"
    validation = read_authenticated(validation_path)
    checkpoint = directory / "best.pt"
    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    latest = torch.load(directory / "latest.pt", map_location="cpu", weights_only=False)
    if (report["status"] != "completed" or report["steps"] != steps or report["best_step"] != steps
            or blob["step"] != steps or latest["step"] != steps or latest.get("pending_validation") is not None
            or report["identity"] != blob["identity"] or validation["identity"] != blob["identity"]):
        raise RuntimeError("Engineering report, checkpoint and validation identities disagree")
    if report["nonfinite"] or report["oom"]:
        raise RuntimeError("Engineering trainer recorded numerical or memory failure")
    if not all(torch.isfinite(value).all() for value in blob["model"].values() if value.is_floating_point()):
        raise RuntimeError("Engineering checkpoint has nonfinite parameters")
    for name in ("optimizer", "scheduler", "rng", "donor_rng", "flow_rng", "normalizers", "metric_scales"):
        if name not in blob:
            raise RuntimeError("Engineering checkpoint lacks recovery state: " + name)
    logs = [json.loads(line) for line in (directory / "training.jsonl").read_text().splitlines()]
    if not logs or not all(math.isfinite(row["loss"]) and math.isfinite(row["grad_norm"]) for row in logs):
        raise RuntimeError("Engineering gradient/loss diagnostics are absent or nonfinite")
    data = read_authenticated(root / "training_data.json")
    entries = data[blob["family"]][:2]
    if any(entry["partition"] != "train" for entry in entries):
        raise RuntimeError("Engineering evaluation attempted a non-training example")
    model = _model(blob["family"], arm)
    model.load_state_dict(blob["model"])
    model = model.to("cuda").eval()
    capture_dir = directory / "verified_training_captures"
    try:
        if blob["family"] == "motion":
            rows = evaluate_motion(model, entries, blob["normalizers"], "cuda", capture_dir=capture_dir)
        else:
            rows = evaluate_scene(model, entries, blob["normalizers"], blob["metric_scales"], "cuda",
                                  generation_seeds=(101,), controls=False, capture_dir=capture_dir, circular_shifts=0)
        reproduced = _selection_score(rows, blob["family"])
        if not math.isclose(reproduced, validation["selection_score"], rel_tol=1e-7, abs_tol=1e-9):
            raise RuntimeError("Reloaded engineering checkpoint does not reproduce its saved validation score")
        captures = artifact_manifest([row["capture_path"] for row in rows])
        verify_artifacts(captures)
        for row in rows:
            with np.load(row["capture_path"], allow_pickle=False) as z:
                if z["truth"].shape != z["prediction"].shape or not np.isfinite(z["prediction"]).all():
                    raise RuntimeError("Invalid engineering validation capture")
        result = {"schema_version": 1, "engineering_only": True, "target_partition": "train",
                  "arm": arm, "steps": steps, "gradient_checks_finite": True,
                  "checkpoint_reload_reproduced_validation": True, "reproduced_selection_score": reproduced,
                  "report_sha256": digest(directory / "report.json"), "checkpoint_sha256": digest(checkpoint),
                  "validation_sha256": digest(validation_path), "rows": rows, "artifacts": captures,
                  "training_seconds": report["training_seconds"],
                  "peak_allocated_gib": report["peak_allocated_gib"], "peak_reserved_gib": report["peak_reserved_gib"]}
        authenticated_json(directory / "integration_verification.json", result)
        return result
    finally:
        del model, blob, latest
        torch.cuda.empty_cache()


def run_smoke(campaign_root=ROOT, *, resume=False):
    from .stage11_training import train_run

    campaign_root = Path(campaign_root)
    root, fixture = prepare_smoke_fixture(campaign_root)
    verify_artifacts(fixture["input_provenance"])
    steps = 100
    started = time.monotonic()
    verified = {}
    with gpu_lock(campaign_root):
        print(json.dumps({"event": "engineering_gpu_acquired", "root": str(root), "arms": ARMS, "steps_per_arm": steps}), flush=True)
        for arm in ARMS:
            directory = root / "engineering_runs" / arm / "seed_0"
            if not (directory / "report.json").exists():
                train_run(root, arm, 0, resume=resume and (directory / "latest.pt").exists(), engineering_steps=steps)
            verified[arm] = _verify_engineering_run(root, arm, steps)
    result = {"schema_version": 1, "status": "completed", "engineering_only": True,
              "exclude_from_campaign_gpu_budget": True, "target_partitions_read": ["train"],
              "scientific_outcomes_evaluated": False, "steps_per_arm": steps,
              "fixture_sha256": digest(root / "fixture.json"), "arms": verified,
              "training_seconds_total": sum(value["training_seconds"] for value in verified.values()),
              "elapsed_s": time.monotonic() - started,
              "artifacts": artifact_manifest([root / "engineering_runs" / arm / "seed_0" / "integration_verification.json" for arm in ARMS])}
    authenticated_json(root / "smoke_report.json", result)
    print(json.dumps({"event": "engineering_gpu_released", "report": str(root / "smoke_report.json"),
                      "training_seconds_total": result["training_seconds_total"]}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.prepare_only:
        root, fixture = prepare_smoke_fixture(args.root)
        print(json.dumps({"root": str(root), "selected_recordings": fixture["selected_recordings"]}, indent=2))
    else:
        run_smoke(args.root, resume=args.resume)


if __name__ == "__main__":
    main()
