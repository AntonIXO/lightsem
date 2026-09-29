"""Authenticated evaluation and canonical-song aggregation for Stage 11.

The CLI/supervisor owns the accelerator lock. ``evaluate_run`` never opens a
locked test recording without an authenticated, checkpoint-specific unlock;
``aggregate`` is CPU-only and never creates that unlock. Generation draws and
recordings are averaged before the paired training-seed/song bootstrap.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import gc
import math
import time

import numpy as np

from .stage11_common import (
    artifact_manifest, authenticated_json, digest, json_digest, read_authenticated,
    verify_artifacts, verify_sources,
)
from .stage11_metrics import (
    evaluate_conditioning_gate, evaluate_motion_gate, evaluate_temporal_gate,
    paired_hierarchical_bootstrap,
)


CORE_SEEDS = (0, 1, 2, 3, 4)
GENERATION_SEEDS = (101, 102, 103, 104)
SCENE_ARMS = ("S_pointwise", "S_temporal", "S_swapped")
EXPECTED_SONGS = 40


def _finite(value):
    return value is not None and isinstance(value, (int, float, np.number)) and math.isfinite(value)


def _verify_inputs(root):
    # Lazy import keeps CPU aggregation importable in the NumPy-only environment.
    from .stage11_training import verify_training_inputs
    verify_training_inputs(root)


def _context(root):
    protocol = read_authenticated(root / "protocol.json")
    data = read_authenticated(root / "training_data.json")
    normalizers = read_authenticated(root / "normalizers.json")
    source = verify_sources(root)
    if list(protocol.get("training_seeds", [])) != list(CORE_SEEDS):
        raise RuntimeError("Stage 11 primary protocol requires training seeds 0 through 4")
    if "generation_seeds" in protocol and list(protocol["generation_seeds"]) != list(GENERATION_SEEDS):
        raise RuntimeError("Stage 11 primary evaluation requires generation seeds 101 through 104")
    if digest(root / "normalizers.json") != data["normalizers_sha256"]:
        raise RuntimeError("Training-data normalizer binding changed")
    return protocol, data, normalizers, {
        "protocol_sha256": digest(root / "protocol.json"),
        "data_sha256": digest(root / "training_data.json"),
        "normalizers_sha256": digest(root / "normalizers.json"),
        "source_sha256": source["source_sha256"],
    }


def _roles(protocol):
    motion = {arm: cfg for arm, cfg in protocol["arms"].items() if cfg["family"] == "motion"}
    old = [arm for arm, cfg in motion.items() if cfg.get("population") == "old"]
    full = [arm for arm, cfg in motion.items() if cfg.get("population") != "old"]
    if len(old) != 1 or len(full) != 1 or any(arm not in protocol["arms"] for arm in SCENE_ARMS):
        raise RuntimeError("Protocol must contain one old/full descriptor pair and all three scene arms")
    return old[0], full[0]


def _entries(data, family, split):
    entries = [entry for entry in data[family] if entry["partition"] == split]
    keys = [entry["key"] for entry in entries]
    songs = {entry["canonical_id"] for entry in entries}
    if len(keys) != len(set(keys)):
        raise RuntimeError(f"Duplicate {family}/{split} input keys")
    if len(songs) != EXPECTED_SONGS:
        raise RuntimeError(f"Expected {EXPECTED_SONGS} {family}/{split} canonical songs, found {len(songs)}")
    return entries


def _run_directory(root, split, arm, seed, steps=8):
    path = root / "results" / split / arm / f"seed_{seed}"
    return path if steps == 8 else path / f"sensitivity_steps_{steps}"


def _test_unlock(root, protocol, binding):
    """Authenticate selection before a caller obtains any held-out entries."""
    path = root / "test_unlock.json"
    if not path.exists():
        raise RuntimeError("Locked test has no authenticated test_unlock.json")
    unlock = read_authenticated(path)
    validation_path = root / "results" / "aggregate_val.json"
    validation = read_authenticated(validation_path)
    if (unlock.get("schema_version") != 1 or unlock.get("protocol_sha256") != binding["protocol_sha256"]
            or unlock.get("validation_aggregate_sha256") != digest(validation_path)
            or validation.get("status") != "complete"
            or validation.get("identity") != binding):
        raise RuntimeError("Test unlock does not bind a complete current validation aggregate")
    eligible = unlock.get("eligible_families", [])
    if not eligible or not set(eligible).issubset(validation.get("eligible_families", [])):
        raise RuntimeError("Test unlock includes a family that did not pass validation")
    for name in (("H11_D",) if eligible == ["motion"] else
                 ("H11_T", "H11_A") if eligible == ["scene"] else ("H11_D", "H11_T", "H11_A")):
        if validation.get("gates", {}).get(name, {}).get("passed") is not True:
            raise RuntimeError("Test unlock eligibility disagrees with validation gates")
    if not validation.get("input_reports"):
        raise RuntimeError("Validation aggregate has no authenticated input-report provenance")
    for report_path, expected_hash in validation["input_reports"].items():
        if digest(report_path) != expected_hash:
            raise RuntimeError("Validation input report changed after selection: " + report_path)
    old, full = _roles(protocol)
    expected_arms = ({old, full} if "motion" in eligible else set()) | (set(SCENE_ARMS) if "scene" in eligible else set())
    if set(unlock.get("arms", {})) != expected_arms:
        raise RuntimeError("Test unlock must freeze every paired arm of each eligible family")
    for arm, selected in unlock["arms"].items():
        if selected.get("config_sha256") != json_digest(protocol["arms"][arm]):
            raise RuntimeError(f"Test unlock config mismatch: {arm}")
        seeds = selected.get("training_seeds", [])
        if seeds != validation.get("arm_seeds", {}).get(arm):
            raise RuntimeError(f"Test unlock training seed mismatch: {arm}")
        if set(selected.get("checkpoints", {})) != {str(seed) for seed in seeds}:
            raise RuntimeError(f"Test unlock checkpoint matrix incomplete: {arm}")
        for seed in seeds:
            checkpoint = root / "runs" / arm / f"seed_{seed}" / "best.pt"
            if digest(checkpoint) != selected["checkpoints"][str(seed)]:
                raise RuntimeError(f"Selected test checkpoint changed: {arm}/seed_{seed}")
    return unlock


def _training_report(root, arm, seed, protocol, binding):
    directory = root / "runs" / arm / f"seed_{seed}"
    report = read_authenticated(directory / "report.json")
    identity = report.get("identity", {})
    expected = {**binding, "arm": arm, "seed": seed, "config": protocol["arms"][arm]}
    if report.get("status") != "completed" or any(identity.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"Training report identity/status mismatch: {arm}/seed_{seed}")
    if report.get("steps") != protocol["arms"][arm]["steps"]:
        raise RuntimeError(f"Training run did not complete its full horizon: {arm}/seed_{seed}")
    if report.get("nonfinite") is not False or report.get("oom") is not False:
        raise RuntimeError(f"Training report records a numerical/execution failure: {arm}/seed_{seed}")
    verify_artifacts(report["artifacts"])
    return report


def _validate_rows(rows, entries, family, *, donors_required=False, shifts_required=False):
    by_key = {entry["key"]: entry for entry in entries}
    expected = ({(key, seed) for key in by_key for seed in GENERATION_SEEDS} if family == "scene"
                else {(key, None) for key in by_key})
    seen = set()
    for row in rows:
        key = (row.get("key"), row.get("generation_seed") if family == "scene" else None)
        if key in seen or key not in expected:
            raise RuntimeError(f"Duplicate or unexpected evaluation row: {key}")
        seen.add(key)
        entry = by_key[key[0]]
        if row.get("canonical_id") != entry["canonical_id"] or row.get("n_frames") != entry["n_frames"]:
            raise RuntimeError(f"Evaluation row identity/length mismatch: {key}")
        if family == "scene":
            metrics = row.get("metrics", {})
            if not all(_finite(metrics.get(name)) for name in ("temporal_error", "marginal_error")):
                raise RuntimeError(f"Missing/nonfinite scene primary metric: {key}")
            novelty = metrics.get("novelty", {})
            if novelty.get("eligible"):
                if (not _finite(novelty.get("ap")) or novelty.get("event_count", 0) < 3
                        or novelty.get("positive_frames", 0) < 1 or novelty.get("negative_frames", 0) < 1):
                    raise RuntimeError(f"Invalid claimed novelty eligibility: {key}")
                if shifts_required and (len(novelty.get("circular_shift_aps", [])) != 200
                                         or not _finite(novelty.get("circular_shift_mean_ap"))):
                    raise RuntimeError(f"Incomplete circular-shift controls: {key}")
            if donors_required:
                donors = row.get("donors", [])
                if len(donors) != 6 or len({donor["key"] for donor in donors}) != 6:
                    raise RuntimeError(f"Expected six distinct donor recordings: {key}")
                donor_songs = set()
                for donor in donors:
                    donor_entry = by_key.get(donor["key"])
                    if donor_entry is None or donor_entry["canonical_id"] == entry["canonical_id"]:
                        raise RuntimeError(f"Donor is missing or belongs to recipient song: {key}")
                    donor_songs.add(donor_entry["canonical_id"])
                    if not 41 <= donor["n_frames"] <= min(entry["n_frames"], donor_entry["n_frames"]):
                        raise RuntimeError(f"Invalid donor paired support: {key}")
                    matched = donor["matched_metrics"]["novelty"]
                    control = donor["donor_metrics"]["novelty"]
                    if any(matched.get(name) != control.get(name) for name in ("event_count", "positive_frames", "valid_frames")):
                        raise RuntimeError(f"Donor/matched references use different support: {key}")
                if len(donor_songs) != 6:
                    raise RuntimeError(f"Expected six distinct canonical donor songs: {key}")
        elif "motion" not in row or "events" not in row:
            raise RuntimeError(f"Incomplete motion evaluation row: {key}")
    if seen != expected:
        missing = sorted(expected - seen, key=str)
        raise RuntimeError(f"Incomplete evaluation row matrix; missing {len(missing)} rows: {missing[:12]}")


def _verify_captures(report, entries, family):
    artifacts = report.get("artifacts", {})
    verify_artifacts(artifacts)
    expected_paths = set()
    for row in report["rows"]:
        path = Path(row.get("capture_path", ""))
        resolved = str(path.resolve())
        if resolved in expected_paths or resolved not in artifacts:
            raise RuntimeError(f"Missing/duplicated evaluation capture: {row.get('key')}")
        expected_paths.add(resolved)
        if row.get("capture_sha256") != artifacts[resolved]["sha256"]:
            raise RuntimeError(f"Capture digest binding mismatch: {path}")
        with np.load(path, allow_pickle=False) as capture:
            channels = 6 if family == "scene" else 4
            for name in ("truth", "prediction"):
                values = capture[name]
                if values.shape != (row["n_frames"], channels) or not np.isfinite(values).all():
                    raise RuntimeError(f"Invalid {name} capture shape/values: {path}")
            if family == "motion":
                for name in ("hazard_truth", "hazard_prediction"):
                    if capture[name].shape != (row["n_frames"], 3) or not np.isfinite(capture[name]).all():
                        raise RuntimeError(f"Invalid hazard capture: {path}")
    if set(artifacts) != expected_paths:
        raise RuntimeError("Evaluation artifact manifest must contain exactly its output captures")


def evaluate_run(root, arm, seed, split="val", controls=True, steps=8):
    """Evaluate a completed frozen checkpoint; the caller owns the GPU lock."""
    root = Path(root)
    if split not in ("val", "test") or steps not in (4, 8, 16):
        raise ValueError("Evaluation supports val/test and 4/8/16 integration steps")
    protocol, data, normalizers, binding = _context(root)
    cfg = protocol["arms"][arm]
    family = cfg["family"]
    if seed not in list(CORE_SEEDS) + protocol.get("extension_seeds", []):
        raise RuntimeError("Evaluation seed is outside the frozen protocol")
    if family == "motion" and (seed not in CORE_SEEDS or steps != 8):
        raise RuntimeError("Descriptor evaluation has five seeds and no Euler sensitivity variants")
    if split == "test":
        unlock = _test_unlock(root, protocol, binding)
        if arm not in unlock["arms"] or seed not in unlock["arms"][arm]["training_seeds"]:
            raise RuntimeError(f"Requested test arm/seed was not selected: {arm}/{seed}")
    # No test array is loaded before the selection check above.
    _verify_inputs(root)
    entries = _entries(data, family, split)
    training = _training_report(root, arm, seed, protocol, binding)
    run = root / "runs" / arm / f"seed_{seed}"
    directory = _run_directory(root, split, arm, seed, steps)
    donor_controls = bool(controls and family == "scene" and arm == "S_temporal")
    identity = {**binding, "training_report_sha256": digest(run / "report.json"),
                "checkpoint_sha256": digest(run / "best.pt"), "arm": arm, "seed": seed,
                "split": split, "family": family, "config": cfg,
                "generation_seeds": list(GENERATION_SEEDS) if family == "scene" else [],
                "integration_steps": steps, "controls": bool(controls), "donor_controls": donor_controls}
    path = directory / "report.json"
    if path.exists():
        existing = read_authenticated(path)
        if existing.get("identity") != identity or existing.get("status") != "completed":
            raise RuntimeError(f"Existing evaluation has a different identity: {path}")
        _validate_rows(existing["rows"], entries, family, donors_required=donor_controls,
                       shifts_required=bool(controls and family == "scene"))
        _verify_captures(existing, entries, family)
        return existing
    from .stage11_training import evaluate_motion, evaluate_scene, load_run_model
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("GPU evaluation requested but CUDA is unavailable")
    model, blob = load_run_model(root, arm, seed)
    if (blob["identity"] != training["identity"] or blob["step"] != training["best_step"]
            or blob["family"] != family or json_digest(blob["normalizers"]) != json_digest(normalizers[family])):
        raise RuntimeError("Selected checkpoint differs from its authenticated training report/calibration")
    if json_digest(blob["metric_scales"]) != json_digest(normalizers["scene_metrics"]):
        raise RuntimeError("Selected checkpoint uses different scene-metric calibration")
    started = time.monotonic()
    model = model.to("cuda").eval()
    captures = directory / "captures"
    try:
        if family == "motion":
            rows = evaluate_motion(model, entries, blob["normalizers"], "cuda", capture_dir=captures)
        else:
            rows = evaluate_scene(model, entries, blob["normalizers"], blob["metric_scales"], "cuda",
                                  generation_seeds=GENERATION_SEEDS, steps=steps,
                                  controls=donor_controls, capture_dir=captures,
                                  circular_shifts=200 if controls else 0)
        _validate_rows(rows, entries, family, donors_required=donor_controls,
                       shifts_required=bool(controls and family == "scene"))
        report = {"schema_version": 1, "status": "completed", "identity": identity,
                  "rows": rows, "expected_canonical_ids": sorted({e["canonical_id"] for e in entries}),
                  "canonical_count": EXPECTED_SONGS, "row_count": len(rows),
                  "elapsed_s": time.monotonic() - started,
                  "artifacts": artifact_manifest([row["capture_path"] for row in rows])}
        _verify_captures(report, entries, family)
        authenticated_json(path, report)
        return report
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


def _false_gates(reason):
    return {name: {"passed": False, "reason": reason, "checks": {"complete_evidence": False}}
            for name in ("H11_D", "H11_T", "H11_A")}


def _matrix_contract(root, split, protocol, binding):
    old, full = _roles(protocol)
    if split == "test":
        unlock = _test_unlock(root, protocol, binding)
        return {arm: selected["training_seeds"] for arm, selected in unlock["arms"].items()}, []
    arms = {arm: list(CORE_SEEDS) for arm in (old, full, *SCENE_ARMS)}
    extension = []
    for seed in protocol.get("extension_seeds", []):
        started = any((root / "runs" / arm / f"seed_{seed}").exists()
                      or _run_directory(root, split, arm, seed).exists() for arm in SCENE_ARMS)
        if started:
            for arm in SCENE_ARMS:
                arms[arm].append(seed)
            extension.append(seed)
    return arms, extension


def _read_evaluation(root, arm, seed, split, protocol, data, binding):
    path = _run_directory(root, split, arm, seed) / "report.json"
    report = read_authenticated(path)
    family = protocol["arms"][arm]["family"]
    identity = report.get("identity", {})
    expected = {**binding, "arm": arm, "seed": seed, "split": split, "family": family,
                "config": protocol["arms"][arm], "integration_steps": 8,
                "generation_seeds": list(GENERATION_SEEDS) if family == "scene" else []}
    if report.get("status") != "completed" or any(identity.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"Evaluation report identity/status mismatch: {arm}/{seed}/{split}")
    training = _training_report(root, arm, seed, protocol, binding)
    run = root / "runs" / arm / f"seed_{seed}"
    if (identity.get("training_report_sha256") != digest(run / "report.json")
            or identity.get("checkpoint_sha256") != digest(run / "best.pt")):
        raise RuntimeError(f"Evaluation belongs to a different training artifact: {arm}/{seed}")
    if family == "scene" and (identity.get("controls") is not True
                              or arm == "S_temporal" and identity.get("donor_controls") is not True):
        raise RuntimeError(f"Primary scene evaluation lacks its frozen controls: {arm}/{seed}")
    entries = _entries(data, family, split)
    _validate_rows(report["rows"], entries, family, donors_required=arm == "S_temporal", shifts_required=family == "scene")
    _verify_captures(report, entries, family)
    if report.get("row_count") != len(report["rows"]) or report.get("canonical_count") != EXPECTED_SONGS:
        raise RuntimeError(f"Evaluation report counts disagree: {arm}/{seed}")
    return report


def _eligible_novelty(metrics, name="ap"):
    novelty = metrics.get("novelty", {})
    value = novelty.get(name)
    return value if novelty.get("eligible") and _finite(value) else None


def _donor_value(row, side):
    donors = row.get("donors", [])
    if len(donors) != 6:
        return None
    values = [_eligible_novelty(donor[f"{side}_metrics"]) for donor in donors]
    return float(np.mean(values)) if all(_finite(value) for value in values) else None


def _song_values(rows, entries, getter):
    """Average draws/coaches per recording, then recordings per canonical song."""
    by_key = {entry["key"]: entry for entry in entries}
    grouped = defaultdict(lambda: defaultdict(list))
    attempts = defaultdict(int)
    invalid = []
    valid_rows = 0
    for row in rows:
        entry = by_key[row["key"]]
        recording = (entry["canonical_id"], entry["recording_id"])
        attempts[recording] += 1
        value = getter(row)
        if _finite(value):
            grouped[recording[0]][recording[1]].append(float(value))
            valid_rows += 1
        else:
            invalid.append({"key": row["key"], "generation_seed": row.get("generation_seed")})
    values = {song: float(np.mean([np.mean(draws) for draws in recordings.values()]))
              for song, recordings in grouped.items()}
    expected_songs = {entry["canonical_id"] for entry in entries}
    return values, {
        "attempted_rows": len(rows), "valid_rows": valid_rows, "excluded_rows": invalid,
        "attempted_recordings": len(attempts), "valid_recordings": sum(len(v) for v in grouped.values()),
        "attempted_songs": len(expected_songs), "valid_songs": len(values),
        "excluded_songs": sorted(expected_songs - set(values)),
    }


def _paired(baseline, candidate, seeds, songs, *, higher_is_better):
    if not songs:
        return None
    first = np.asarray([[baseline[seed][song] for song in songs] for seed in seeds])
    second = np.asarray([[candidate[seed][song] for song in songs] for seed in seeds])
    result = paired_hierarchical_bootstrap(first, second, higher_is_better=higher_is_better)
    result.update({"canonical_ids": songs, "training_seeds": seeds,
                   "baseline_by_seed_song": first.tolist(), "candidate_by_seed_song": second.tolist()})
    return result


def _intersection(mappings, seeds, enrolled):
    songs = set(enrolled)
    for mapping in mappings:
        for seed in seeds:
            songs &= set(mapping[seed])
    return sorted(songs)


def _compute_aggregate(protocol, data, reports, arm_seeds, split):
    """Pure aggregation kernel; its caller authenticates every input report."""
    old, full = _roles(protocol)
    values, denominators, summaries = {}, {}, {}
    scene_getters = {
        "temporal_error": lambda row: row["metrics"].get("temporal_error"),
        "marginal_error": lambda row: row["metrics"].get("marginal_error"),
        "variation_ratio": lambda row: row["metrics"].get("variation_ratio"),
        "hf_energy_ratio_2to5": lambda row: row["metrics"].get("hf_energy_ratio_2to5"),
        "reference_variation": lambda row: row["metrics"].get("reference_variation"),
        "generated_variation": lambda row: row["metrics"].get("generated_variation"),
        "reference_hf_energy_2to5": lambda row: row["metrics"].get("reference_hf_energy_2to5"),
        "generated_hf_energy_2to5": lambda row: row["metrics"].get("generated_hf_energy_2to5"),
        "novelty_ap": lambda row: _eligible_novelty(row["metrics"]),
        "novelty_prevalence": lambda row: row["metrics"].get("novelty", {}).get("prevalence"),
        "novelty_event_count": lambda row: row["metrics"].get("novelty", {}).get("event_count"),
        "beat_only": lambda row: _eligible_novelty(row["metrics"], "beat_only_ap"),
        "circular_shift": lambda row: _eligible_novelty(row["metrics"], "circular_shift_mean_ap"),
        "donor_matched": lambda row: _donor_value(row, "matched"),
        "donor_audio": lambda row: _donor_value(row, "donor"),
    }
    for arm, seeds in arm_seeds.items():
        family = protocol["arms"][arm]["family"]
        entries = _entries(data, family, split)
        getters = dict(scene_getters) if family == "scene" else {"primary_r2": lambda row: row["motion"].get("primary_r2")}
        if family == "motion":
            for name in ("metric_level", "vert_period_s", "vert_bounce_amp", "com_speed"):
                getters[f"r2_{name}"] = lambda row, name=name: row["motion"].get("r2_by_channel", {}).get(name)
            for name in ("footfall", "jump", "hit"):
                getters[f"event_ap_{name}"] = lambda row, name=name: row["events"].get(name, {}).get("ap")
        elif arm != "S_temporal":
            getters.pop("donor_matched")
            getters.pop("donor_audio")
        values[arm], denominators[arm] = {key: {} for key in getters}, {}
        for seed in seeds:
            report = reports[(arm, seed)]
            _validate_rows(report["rows"], entries, family, donors_required=arm == "S_temporal", shifts_required=family == "scene")
            denominators[arm][str(seed)] = {}
            for name, getter in getters.items():
                metric, counts = _song_values(report["rows"], entries, getter)
                values[arm][name][seed] = metric
                denominators[arm][str(seed)][name] = counts
        summaries[arm] = {}
        for name, per_seed in values[arm].items():
            means = {str(seed): float(np.mean(list(mapping.values()))) if mapping else None
                     for seed, mapping in per_seed.items()}
            finite_means = [mean for mean in means.values() if mean is not None]
            summaries[arm][name] = {"mean": float(np.mean(finite_means)) if finite_means else None,
                                   "per_seed_mean": means,
                                   "per_seed_song_count": {str(seed): len(mapping) for seed, mapping in per_seed.items()}}
    gates = _false_gates("family_not_selected_for_this_split")
    comparisons = {}
    eligibility = {}
    if old in arm_seeds and full in arm_seeds:
        seeds = arm_seeds[old]
        enrolled = sorted({entry["canonical_id"] for entry in _entries(data, "motion", split)})
        songs = _intersection([values[old]["primary_r2"], values[full]["primary_r2"]], seeds, enrolled)
        result = _paired(values[old]["primary_r2"], values[full]["primary_r2"], seeds, songs, higher_is_better=True)
        comparisons["H11_D"] = result
        gates["H11_D"] = evaluate_motion_gate(result or {})
        gates["H11_D"]["checks"]["eligible_at_least_80_percent"] = len(songs) / EXPECTED_SONGS >= 0.8
        gates["H11_D"]["passed"] = all(gates["H11_D"]["checks"].values())
        eligibility["H11_D"] = {"enrolled": EXPECTED_SONGS, "eligible": len(songs), "fraction": len(songs) / EXPECTED_SONGS,
                                  "canonical_ids": songs, "excluded_canonical_ids": sorted(set(enrolled) - set(songs))}
    if all(arm in arm_seeds for arm in SCENE_ARMS):
        seeds = arm_seeds["S_temporal"]
        if any(arm_seeds[arm] != seeds for arm in SCENE_ARMS):
            raise RuntimeError("Scene arms do not contain the same complete training-seed matrix")
        enrolled = sorted({entry["canonical_id"] for entry in _entries(data, "scene", split)})
        baseline, temporal = values["S_pointwise"], values["S_temporal"]
        temporal_names = ("temporal_error", "marginal_error", "variation_ratio", "hf_energy_ratio_2to5")
        mappings = [baseline[name] for name in ("temporal_error", "marginal_error")]
        mappings += [temporal[name] for name in temporal_names]
        songs = _intersection(mappings, seeds, enrolled)
        result = _paired(baseline["temporal_error"], temporal["temporal_error"], seeds, songs, higher_is_better=False)
        comparisons["H11_T"] = result

        def average(mapping):
            return float(np.mean([mapping[seed][song] for seed in seeds for song in songs])) if songs else None

        base_metrics = {name: average(baseline[name]) for name in ("temporal_error", "marginal_error")}
        candidate_metrics = {name: average(temporal[name]) for name in temporal_names}
        gates["H11_T"] = evaluate_temporal_gate(base_metrics, candidate_metrics, result or {},
                                                eligible_fraction=len(songs) / EXPECTED_SONGS)
        comparisons["H11_T_aggregate_metrics"] = {"baseline": base_metrics, "candidate": candidate_metrics}
        eligibility["H11_T"] = {"enrolled": EXPECTED_SONGS, "eligible": len(songs), "fraction": len(songs) / EXPECTED_SONGS,
                                  "canonical_ids": songs, "excluded_canonical_ids": sorted(set(enrolled) - set(songs))}
        controls = {
            "beat_only": (temporal["beat_only"], temporal["novelty_ap"]),
            "circular_shift": (temporal["circular_shift"], temporal["novelty_ap"]),
            "shuffled_training": (values["S_swapped"]["novelty_ap"], temporal["novelty_ap"]),
            "donor_audio": (temporal["donor_audio"], temporal["donor_matched"]),
        }
        songs = _intersection([mapping for pair in controls.values() for mapping in pair], seeds, enrolled)
        results = {name: _paired(first, second, seeds, songs, higher_is_better=True)
                   for name, (first, second) in controls.items()}
        comparisons["H11_A"] = results
        gates["H11_A"] = evaluate_conditioning_gate({key: value or {} for key, value in results.items()},
                                                    eligible_fraction=len(songs) / EXPECTED_SONGS)
        eligibility["H11_A"] = {"enrolled": EXPECTED_SONGS, "eligible": len(songs), "fraction": len(songs) / EXPECTED_SONGS,
                                  "canonical_ids": songs, "excluded_canonical_ids": sorted(set(enrolled) - set(songs))}
        comparisons["conditioning_control_policy"] = {
            "circular_shift_statistic": "mean_of_200_predeclared_shifts",
            "donor_statistic": "mean_of_six_distinct_canonical_donors_with_paired_support",
            "strongest_control_by_smallest_paired_improvement": (min(results, key=lambda key: results[key]["delta"]) if songs else None),
            "all_controls_must_pass": True,
            "raw_null_means": {key: result["baseline_mean"] if result else None for key, result in results.items()},
            "paired_matched_means": {key: result["candidate_mean"] if result else None for key, result in results.items()},
        }
    families = []
    if gates["H11_D"]["passed"]:
        families.append("motion")
    if gates["H11_T"]["passed"] and gates["H11_A"]["passed"]:
        families.append("scene")
    return {"gates": gates, "comparisons": comparisons, "eligibility": eligibility,
            "metric_denominators": denominators, "arm_summaries": summaries, "eligible_families": families,
            "aggregation": "generation draws/coaches within recording; recordings within canonical song; paired crossed training-seed/song bootstrap",
            "temporal_variation_definition": "mean_absolute_lag_difference_normalized_by_frozen_training_lag_IQR"}


def aggregate(root, split="val"):
    """Authenticate the complete arm/seed matrix and write prospective verdicts.

    Missing core runs or a started but incomplete extension triplet yield a
    named incomplete report with every gate closed. Integrity failures raise;
    they are not treated as inconvenient missing observations.
    """
    root = Path(root)
    if split not in ("val", "test"):
        raise ValueError("Aggregation supports val or test")
    protocol, data, _, binding = _context(root)
    arm_seeds, extensions = _matrix_contract(root, split, protocol, binding)
    expected = [{"arm": arm, "seed": seed} for arm, seeds in arm_seeds.items() for seed in seeds]
    missing = [{**item, "reason": "evaluation_report_missing"} for item in expected
               if not (_run_directory(root, split, item["arm"], item["seed"]) / "report.json").exists()]
    result = {"schema_version": 1, "split": split, "identity": binding,
              "status": "incomplete" if missing else "complete", "expected_runs": expected,
              "missing_runs": missing, "arm_seeds": arm_seeds, "included_extension_seeds": extensions,
              "gates": _false_gates("incomplete_expected_run_matrix"), "eligible_families": [],
              "input_reports": {}, "computed_at": time.time()}
    # Authenticate every present report even if another run is still missing.
    reports = {}
    for item in expected:
        path = _run_directory(root, split, item["arm"], item["seed"]) / "report.json"
        if path.exists():
            reports[(item["arm"], item["seed"])] = _read_evaluation(root, item["arm"], item["seed"], split,
                                                                   protocol, data, binding)
            result["input_reports"][str(path)] = digest(path)
    if not missing:
        _verify_inputs(root)
        result.update(_compute_aggregate(protocol, data, reports, arm_seeds, split))
    path = root / "results" / f"aggregate_{split}.json"
    if path.exists():
        existing = read_authenticated(path)
        # Repeating an unchanged aggregate must not invalidate an existing
        # checkpoint-specific unlock merely by changing a wall-clock field.
        if ({key: value for key, value in existing.items() if key != "computed_at"}
                == {key: value for key, value in result.items() if key != "computed_at"}):
            return existing
    authenticated_json(path, result)
    return result
