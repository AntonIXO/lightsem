"""Frozen center-clip observer-proxy evaluation. No lighting-truth claim.

CLI stages are opt-in and hash-bound. Importing this module reads no data.
Students see one normalized (2560,94) window, never a whole-track model context.
The 94-feature bus itself was constructed offline from the full source record.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from . import metrics
from .training import (AUDIO_SCALARS, RHYTHM_COLUMNS, OBSERVER_SCOPE_BINDINGS,
                       ContractError, SourceStore, TrainConfig, validate_run_scope,
                       _normalize, campaign_space, deterministic_donors, generated_history, json_hash,
                       load_checkpoint, make_targets, sample_linear, sha256)

SCHEMA = 1
PROTOCOL_ID = "observer-proxy-20260922-v1"
SCOPE = ("selected_historicalDEV_center8_frozen_authored_video_OBSERVER_proxy; "
         "unknown_cuts_INCLUDED; not_LIGHTtruth_finebeataccuracy_or_generalization")
RECIPES = ("S_none_P0", "S_V6_P0", "S_VR_P0", "I_VR_P0", "S_VR_P0_Ashuf")
SEEDS = (0, 1, 2)
CENTER = (28., 36.)
RATES = {"accent": 40, "color": 10}
BASELINE_NAMES = ("prevalence", "audio_onset", "audio_novelty", "logistic")
CONTRASTS = {"V6-none": ("S_V6_P0", "S_none_P0"),
             "VR-V6": ("S_VR_P0", "S_V6_P0"),
             "S-I": ("S_VR_P0", "I_VR_P0"),
             "real-Ashuf": ("S_VR_P0", "S_VR_P0_Ashuf")}
SHIFT_SCHEMA = {
    "count": 20, "minimum_wrapped_displacement_strict_s": .4,
    "segment": "each_contiguous_eligible_requested_grid_segment_in_center8",
    "allowed_integer_offsets": "positive_j/native_rate < actual_segment_duration; min(offset,duration-offset) > .4",
    "selection": "allowed[floor(linspace(0,len(allowed)-1,20))]",
    "transform": "fixed_postNMS_candidate_event_time modulo_actual_segment_duration; preserve_confidences_counts",
    "too_short": "exclude_only_from_shift_paired_null; real_rescored_on_identical_shift_support",
    "post_shift_NMS": False,
    "wrap_limitation": "report_wrapped_candidates_and_close_pairs; no_independent_replicate_claim",
    "minimum_fraction_of_primary_eligible_duration": .8,
    "qualified_groups_minimum": 30,
    "event_bearing_groups_per_head_minimum": 10,
    "distinct_offsets_required": True,
}


def evaluation_definition():
    """Data-free schema to save/register BEFORE any baseline fit or scoring."""
    return {"schema_version": SCHEMA, "protocol_id": PROTOCOL_ID, "scope": SCOPE,
        "model_input": "one_exact64s_normalized94_window; fullsource_offline_feature_provenance",
        "prediction_precision": "FP32_autocast_disabled_TF32_disabled",
        "fit_exposure": {
            "audio_normalizer": "FULL_SOURCE_94feature_buses_of_final_usable_TRAIN_records_only; not_DEV; not_64s_only",
            "visual_transforms": "selected_TRAIN64s_windows_only: q95_thresholds_posweights_palette_vocab_and_Rich/V6_global_auxnorms; palette_sampling_as_frozen_transforms",
            "supervised_labels": "center8_of_selected_TRAIN64s_only_for_neural_and_logistic; no_DEVfit",
            "legacy_V6_local": "cached_z[y]_value_columns3/4_already_full_active_record_mean_std_normalized_before_crop_then_TRAIN_global_auxnorm",
            "feature_generation": "bus_globalbeatgrid_BeatThis_OFFLINE_FULL_SOURCE_for_all_inputs_controls; NOT64s_rawaudio_or_causal_claim"},
        "generated_history": {"initial_state": "null_at_clip_start", "cue_input": "own_color_probability_only",
            "cue_threshold": .5, "cue_nms_s": .5, "mark_choice": "own_argmax",
            "clip_start_display_initialization_is_event": False, "teacher_history": False},
        "checkpoint_reporting": "separate_complete15_cell_indices_for_selected_primary_and_final_companion",
        "selection_scope": "selected_on_same_historicalDEV_commonmainloss; conditional_development_estimates_not_unbiased_or_fresh_generalization",
        "score_interval_relative_s": list(CENTER), "reference_clock": "actual_source_PTS",
        "prediction_clock": "absolute_requested_audio_grid",
        "reference": {"train_quantile": .95, "pre_post_s": .3, "nms_s": .5,
                      "accent": "one_native10Hz_point_not_four_repeated_training_bins"},
        "candidates": {"nms_s": .2, "quota": None,
            "local_max": "interior_plateau_strictly_above_both_neighbors_earliest_plateau_time",
            "constant_and_segment_endpoints": "no_candidates",
            "nms_order": "global_after_valid_segment_localmax; descending_confidence_then_earlier_time",
            "scope": "full64s_valid_support_peak_extraction_then_identical_center8_clip"},
        "visual_tolerance_s": .2, "shift_schema": SHIFT_SCHEMA,
        "baselines": {"prevalence": "TRAIN_center_eligible_label_mean_constant",
            "audio_onset": "standardized_bus_onset_ema_fixed_not_best_channel",
            "audio_novelty": "Euclidean_.3s_pre_post_TRAIN_standardized64mel_means_at_native_rate",
            "logistic": {"C": 1., "penalty": "l2", "class_weight": "balanced",
                "max_iter": 1000, "random_state": 0, "solver": "lbfgs",
                "input": "same94_at40Hz_accent_mean4_pool10Hz_color",
                "fit_support": "TRAIN_center8_same_observed_labels_and_validity",
                "loss_difference": "sklearn_balanced_L2_not_neural_clamped_posweight_BCE"},
            "training_Ashuf_donor": "nearest_abs_log_tempo_ratio_fullsource_median_beat_period; samepartition_distinctcanonical; hash_rid_donor_tie; many_to_one",
            "donor_scope": "correct_record_association_of_nonretained_columns_only; mel_onset_flux_may_carry_rhythm; NOT_all_rhythm_held",
            "donor": "EXACT_training_tempo_nearest_map; same_partition_distinct_canonical; >=640frames; observer_cropindex0; many_to_one_not_bijection",
            "donor_transform": "raw_columns0:86_replaced; raw86:94_retained; then_common_TRAIN_affine_normalization",
            "Ashuf_primary_input": "same_frozen_training_tempo_donor_route_as_TRAIN_and_DEV_selection",
            "preserved_rhythm_columns": list(RHYTHM_COLUMNS)},
        "deployment": {"model": .5, "donor": .5, "logistic": .5,
            "audio_onset": "TRAIN_center_eligible_q95",
            "audio_novelty": "TRAIN_center_eligible_q95", "prevalence": "TRAIN_prevalence"},
        "cohort": {"original": 40, "enrolled": 31, "original_fraction": 31/40,
            "original80percentgate": "FAILED_NOT_WAIVED", "qualified_min": 30,
            "qualified_center_coverage_min": .8, "event_bearing_min_per_head": 10,
            "capture_exclusions": "pre_fit_hashed_source_only_receipts; immutable31_enrolled; all15_cells_for_exact_usable_subset; no_missing_record_imputation"},
        "statistics": {"contrasts": CONTRASTS, "heads": list(RATES), "seeds": list(SEEDS),
            "canonical_song_bootstraps": 10000, "bootstrap_seed": 0,
            "multiplicity": "Holm8_p_values_only_CI_unadjusted",
            "primary": "VR-V6_color_AP_gain_at_least_.01",
            "observer_effect_gate": {"registered_hypothesis": "h-layers-20260922-observer-proxy",
                "color_AP_gain_minimum": .01, "paired95_CI_lower_strictly_positive": True,
                "Holm8_p_strictly_below": .05, "accent_AP_gain_minimum": -.01,
                "prerequisites": "support_null_complete_controls_and15_cells; otherwise_undefined_not_zero",
                "numerical_gain_alone_is_success": False}},
        "palette": "separate_exploratory_phase_fixed_common_cues_seeds101_102_103_104_inverseCDF"}


def implementation_hashes():
    root = Path(__file__).resolve().parents[3]
    paths = [Path(__file__), Path(__file__).with_name("metrics.py"),
             Path(__file__).with_name("palette_metrics.py"),
             Path(__file__).with_name("training.py"), Path(__file__).with_name("targets.py"),
             Path(__file__).with_name("model.py"), root/"lightsem/model/light_trunk.py",
             root/"scripts"/"light_layers"/"run_light_layers_evaluate.py"]
    return {str(p): sha256(p) for p in paths}


def checked_ref(reference):
    if not isinstance(reference, dict) or not {"path", "sha256"} <= reference.keys():
        raise ContractError("Every artifact requires path and sha256")
    path = Path(reference["path"])
    if not path.is_file() or sha256(path) != reference["sha256"]:
        raise ContractError(f"Missing or changed source/artifact: {path}")
    return path


def reference(path):
    return {"path": str(Path(path).resolve()), "sha256": sha256(path)}


def validate_transforms(transforms):
    if transforms.get("fit_partition") != "train":
        raise ContractError("Frozen transforms must be TRAIN fitted")
    for key, expected in (("threshold_quantile", .95), ("novelty_window_s", .3), ("event_nms_s", .5)):
        if transforms.get(key) != expected:
            raise ContractError(f"Reference target definition drift: {key}")
    for key, value in (("norm_sha256", {"audio": transforms.get("audio"), "aux": transforms.get("aux")}),
                       ("vocab_sha256", transforms.get("palette_prototypes"))):
        if transforms.get(key) != json_hash(value):
            raise ContractError(f"Frozen transform hash mismatch: {key}")
    norm = transforms.get("audio") or {}
    mean, std = np.asarray(norm.get("mean")), np.asarray(norm.get("std"))
    if (mean.shape != (94,) or std.shape != (94,) or not np.isfinite(mean).all()
            or not np.isfinite(std).all() or np.any(std <= 0)):
        raise ContractError("Frozen TRAIN audio normalizer must have94 finite coordinates")
    p = np.asarray(transforms["palette_prototypes"])
    if (p.shape != (transforms.get("palette_size"), 72) or not np.isfinite(p).all()
            or np.any(p < 0) or not np.allclose(p.sum(1), 1, atol=1e-5)):
        raise ContractError("Frozen palette target shape/distribution mismatch")


def _vector(value, n, name, *, boolean=False):
    x = np.asarray(value)
    if x.shape != (n,) or (boolean and x.dtype != bool) or not np.isfinite(x).all():
        raise ContractError(f"{name} must be finite shape ({n},)" + (" boolean" if boolean else ""))
    return x


def validate_clocks(requested, source):
    requested = _vector(requested, 640, "requested_times10_s").astype(float)
    source = _vector(source, 640, "source_times10_s").astype(float)
    if (not np.allclose(np.diff(requested), .1, atol=3e-5, rtol=0)
            or not np.allclose(requested*10, np.rint(requested*10), atol=3e-4, rtol=0)
            or np.any(np.diff(source) <= 0)):
        raise ContractError("Clip clocks must preserve absolute aligned10Hz requested and increasing source time")
    return requested, source


def validate_audio_window(audio40):
    values = np.asarray(audio40, np.float32)
    if values.shape != (2560, 94) or not np.isfinite(values).all():
        raise ContractError("Student requires exactly one finite normalized (2560,94)64s window; no clipping/padding/expansion")
    return values


def audio_window(store, record, transforms):
    """Slice raw offline fullsource bus, then apply the immutable TRAIN transform."""
    validate_clocks(record.times, record.times if record.source_times is None else record.source_times)
    audio = store.audio(record.recording_id)
    q = record.times[0] + np.arange(2560)/40
    raw, supported = sample_linear(np.arange(len(audio))/40, audio, q, max_gap_s=.026)
    if not supported.all():
        raise ContractError("Source does not cover exact64s clip; no boundary clipping or padding")
    return validate_audio_window(_normalize(raw, transforms["audio"]))


def donor_map(rows, tempos_bpm):
    """Exact training donor algorithm, with PREDECLARED fullsource tempo evidence.

    Observer scope guarantees64s windows; actual640-frame windows are checked
    before prediction. This metadata-only function reads no source features.
    """
    rows = sorted(rows, key=lambda r: r["recording_id"])
    ids = [r["recording_id"] for r in rows]
    if len(set(ids)) != len(ids) or set(tempos_bpm) != set(ids):
        raise ContractError("Duplicate donor inventory or missing exact fullsource tempo evidence")
    if any(not np.isfinite(v) or v <= 0 for v in tempos_bpm.values()):
        raise ContractError("Frozen donor tempos must be finite and positive")
    descriptors = [SimpleNamespace(recording_id=r["recording_id"], canonical_id=r["canonical_id"],
        partition=r["partition"], times=range(640)) for r in rows]
    # Calls training's exact implementation: same partition spelling, >=640,
    # nearest abs(log tempo ratio), compact-JSON hash tie, many-to-one allowed.
    return deterministic_donors(descriptors, tempos_bpm)


def donor_audit(rows, mapping):
    result = {}
    for partition in sorted({row["partition"] for row in rows}):
        recipients = [row["recording_id"] for row in rows if row["partition"] == partition]
        counts = {donor: sum(mapping[rid] == donor for rid in recipients)
                  for donor in sorted({mapping[rid] for rid in recipients})}
        result[partition] = {"recipients": len(recipients), "unique_donors": len(counts),
            "reuse_counts": counts, "max_reuse": max(counts.values(), default=0),
            "bijective_permutation_claim": False}
    return result


def verify_donor_tempos(store, recording_ids, tempos_bpm):
    """Verify frozen tempo metadata against the same FULL SOURCE TRAIN/DEV bus."""
    for rid in recording_ids:
        period = store.audio(rid)[:, 64+AUDIO_SCALARS.index("beat_period_s")]
        if not len(period) or not np.isfinite(period).all() or np.any(period <= 0):
            raise ContractError("Donor matching requires positive fullsource beat periods")
        actual = 60/float(np.median(period))
        if rid not in tempos_bpm or actual != tempos_bpm[rid]:
            raise ContractError("Frozen donor fullsource tempo evidence changed")


def nonrhythm_donor(recipient40, donor40):
    x, donor = validate_audio_window(recipient40), validate_audio_window(donor40)
    result = donor.copy()
    result[:, RHYTHM_COLUMNS] = x[:, RHYTHM_COLUMNS]
    return result


def predict_window(model, normalized_audio40, *, requested_times10_s, source_times10_s):
    """One model forward, one own-history rollout from null at clip start.

    No gold label/palette/cue/history argument exists. Full64s raw logits and
    sigmoid scores plus generated states are frozen; only center8 is scored.
    """
    import torch
    x = validate_audio_window(normalized_audio40)
    requested, source = validate_clocks(requested_times10_s, source_times10_s)
    if model.cfg.in_dim != 94 or model.cfg.context_radius10 > 280:
        raise ContractError("Model context/features incompatible with frozen64s window")
    previous_training = model.training
    model.eval()
    device = next(model.parameters()).device
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
            output = model(torch.from_numpy(x[None]).to(device))
            if (tuple(output["accent_logit"].shape) != (1, 2560)
                    or tuple(output["chromatic_change_logit"].shape) != (1, 640)
                    or tuple(output["context10"].shape[:2]) != (1, 640)):
                raise ContractError("Model output target shape mismatch")
            if any(not bool(torch.isfinite(output[key]).all()) for key in
                   ("accent_logit", "chromatic_change_logit", "context10")):
                raise ContractError("Nonfinite model head/context before generated rollout")
            rollout = generated_history(model, output["context10"], output["chromatic_change_logit"],
                                        initial_ids=None, state_ablation=False, cue_threshold=.5, initialize=True)
            if not bool(torch.isfinite(model.palette_logits(output["context10"], rollout["previous_ids"])).all()):
                raise ContractError("Nonfinite own-history palette logits; token argmax is not valid")
            result = {"accent_logit": output["accent_logit"][0].float().cpu().numpy(),
                "color_logit": output["chromatic_change_logit"][0].float().cpu().numpy(),
                "accent": torch.sigmoid(output["accent_logit"].float())[0].cpu().numpy(),
                "color": torch.sigmoid(output["chromatic_change_logit"].float())[0].cpu().numpy(),
                "generated_palette_ids": rollout["generated_palette_ids"][0].cpu().numpy(),
                "previous_ids": rollout["previous_ids"][0].cpu().numpy(),
                "generated_color_cues": rollout["color_cues"][0].cpu().numpy(),
                "requested_times10_s": requested, "source_times10_s": source,
                "requested_times40_s": requested[0]+np.arange(2560)/40}
            if any(not np.isfinite(v).all() for v in result.values()):
                raise ContractError("Nonfinite frozen model output")
            if result["previous_ids"][0] != model.cfg.palette_size:
                raise ContractError("Generated state did not start from null")
            return result
    finally:
        model.train(previous_training)


def predict_conditions(model, normalized_windows, recording_id, frozen_donor_map, *, recipe,
                       requested_times10_s, source_times10_s):
    """Apply the frozen map to EVERY donor path; Ashuf primary stays intervened."""
    if recipe not in RECIPES:
        raise ContractError("Undeclared observer recipe")
    donor_id = frozen_donor_map.get(recording_id)
    if donor_id is None or donor_id == recording_id or donor_id not in normalized_windows:
        raise ContractError("Missing distinct mapped donor input")
    recipient = validate_audio_window(normalized_windows[recording_id])
    donor = nonrhythm_donor(recipient, normalized_windows[donor_id])
    clocks = dict(requested_times10_s=requested_times10_s, source_times10_s=source_times10_s)
    primary = donor if recipe == "S_VR_P0_Ashuf" else recipient
    result = predict_window(model, primary, **clocks)
    # For Ashuf, primary and donor are the SAME condition, not two draws.
    control = result if recipe == "S_VR_P0_Ashuf" else predict_window(model, donor, **clocks)
    return {"model": result, "donor": control}


def load_bound_model(checkpoint_reference, transforms, *, transforms_sha256, expected_bindings, device="cpu"):
    validate_transforms(transforms)
    required = {"manifest_sha256", "source_sha256", "split_sha256", "config_sha256", *OBSERVER_SCOPE_BINDINGS}
    if not required <= expected_bindings.keys():
        raise ContractError("Missing exact checkpoint provenance bindings")
    bindings = {**expected_bindings, "norm_sha256": transforms["norm_sha256"],
                "vocab_sha256": transforms["vocab_sha256"], "transforms_sha256": transforms_sha256}
    return load_checkpoint(checked_ref(checkpoint_reference), device=device, expected_bindings=bindings)


@dataclass
class FrozenLabels:
    recording_id: str
    canonical_id: str
    partition: str
    requested_times10_s: np.ndarray
    source_times10_s: np.ndarray
    accent10: np.ndarray
    color: np.ndarray
    accent_mask10: np.ndarray
    color_mask: np.ndarray

    def validate(self):
        validate_clocks(self.requested_times10_s, self.source_times10_s)
        for head, key, mask in (("accent", "accent10", "accent_mask10"), ("color", "color", "color_mask")):
            y = _vector(getattr(self, key), 640, head+" labels")
            m = _vector(getattr(self, mask), 640, head+" validity", boolean=True)
            if not np.isin(y, [0, 1]).all() or np.any(y[~m]):
                raise ContractError("Reference labels must be binary and within frozen observed support")
        return self

    def content_hash(self):
        self.validate()
        digest = hashlib.sha256(json_hash([self.recording_id, self.canonical_id, self.partition]).encode())
        for name in ("requested_times10_s", "source_times10_s", "accent10", "color", "accent_mask10", "color_mask"):
            # Canonical array representation independent of native endian/dtype.
            digest.update(name.encode()); digest.update(np.asarray(getattr(self, name), dtype="<f8").tobytes())
        return digest.hexdigest()


def novelty_window_support(source_times, observed):
    """Validate .3s pre/post sample-hold support, with no gap bridging."""
    times = _vector(source_times, 640, "source_times")
    valid = _vector(observed, 640, "observed", boolean=True)
    if np.any(np.diff(times) <= 0):
        raise ContractError("Novelty support requires increasing actual-source times")
    result = np.zeros(640, bool)
    for i, center in enumerate(times):
        left, right = center-.3, center+.3
        if left < times[0]-1e-8 or right > times[-1]+1e-8:
            continue
        active = (times[:-1] < right-1e-9) & (times[1:] > left+1e-9)
        result[i] = (active.any() and valid[:-1][active].all()
                     and np.all(np.diff(times)[active] <= .2+1e-8))
    return result


def freeze_labels(record, transforms):
    if len(record.times) != 640:
        raise ContractError("Observer evaluation needs exactly64s captured targets")
    target = make_targets(record, transforms)
    source = record.times if record.source_times is None else record.source_times
    for head, valid, mask_key in (("accent", record.value_valid, "accent_mask10"),
                                  ("color", record.color_valid, "color_mask")):
        observed = np.asarray(valid, bool).copy()
        if record.frame_valid is not None:
            observed &= record.frame_valid
        support = novelty_window_support(source, observed)
        mask = _vector(target[mask_key], 640, head+" validity", boolean=True)
        if np.any(mask & ~support):
            raise ContractError("Frozen target eligible window crosses observed invalidity/source gap")
    return FrozenLabels(record.recording_id, record.canonical_id, record.partition,
        record.times.copy(), (record.times if record.source_times is None else record.source_times).copy(),
        target["accent10"].copy(), target["color"].copy(), target["accent_mask10"].copy(),
        target["color_mask"].copy()).validate()


def _runs(mask):
    boundaries = np.flatnonzero(np.diff(np.r_[False, np.asarray(mask, bool), False]))
    return list(zip(boundaries[::2], boundaries[1::2]))


def eligible_intervals(labels, head, *, center=True):
    """Actual-source sample holds, already .3s-eroded by frozen target builder."""
    labels.validate()
    source = labels.source_times10_s
    mask = labels.accent_mask10 if head == "accent" else labels.color_mask
    start, stop = labels.requested_times10_s[0] + np.asarray(CENTER if center else (0., 64.))
    intervals = []
    for i in np.flatnonzero(mask):
        end = source[i+1] if i+1 < len(source) else source[i]+.1
        if end-source[i] > .2+1e-8:
            raise ContractError("Frozen eligible label bridges an unsupported source gap")
        a, b = max(start, source[i]), min(stop, end)
        if a >= b:
            continue
        if intervals and abs(intervals[-1][1]-a) < 1e-9:
            intervals[-1][1] = b
        else:
            intervals.append([a, b])
    return np.asarray(intervals, float).reshape(-1, 2)


def _regions(times, intervals):
    region = np.full(len(times), -1, int)
    for i, (a, b) in enumerate(intervals):
        region[(times >= a) & (times < b)] = i
    return region


def _region_runs(regions):
    starts = np.r_[0, np.flatnonzero(regions[1:] != regions[:-1])+1]
    stops = np.r_[starts[1:], len(regions)]
    return [(int(a), int(b)) for a, b in zip(starts, stops) if regions[a] >= 0]


def candidate_peaks(times, scores, intervals):
    """Interior strict plateau maxima, earliest tie, then score-ranked .2s NMS.

    No quota or data-dependent threshold. Constant curves and isolated eligible
    points do not invent peaks. Neighbors across invalid gaps are never used.
    """
    times, scores = np.asarray(times, float), np.asarray(scores, float)
    if (times.ndim != 1 or scores.shape != times.shape or not np.isfinite(times).all()
            or not np.isfinite(scores).all() or np.any(np.diff(times) <= 0)):
        raise ContractError("Candidate times/scores must be finite aligned increasing vectors")
    regions = _regions(times, intervals)
    candidates = []
    for start, stop in _region_runs(regions):
        i = start
        while i < stop:
            end = i+1
            while end < stop and scores[end] == scores[i]:
                end += 1
            if i > start and end < stop and scores[i] > scores[i-1] and scores[i] > scores[end]:
                candidates.append(i)
            i = end
    keep = []
    for i in sorted(candidates, key=lambda i: (-scores[i], times[i])):
        if all(abs(times[i]-times[j]) >= .2-1e-12 for j in keep):
            keep.append(i)
    keep = np.asarray(sorted(keep, key=lambda i: times[i]), int)
    return times[keep], scores[keep]


def shift_plan(times, intervals, rate):
    """Data-free deterministic offsets within each actual valid-source segment.

    Candidate predictions are on the requested grid; reference support endpoints
    are actual source PTS. Offset increments are 1/rate, but modulo uses the
    actual interval duration, so wrapping never moves a candidate into a gap.
    """
    times = np.asarray(times, float)
    if rate not in (10, 40) or len(times) < 2 or not np.allclose(np.diff(times), 1/rate, atol=1e-8):
        raise ContractError("Shifts require the frozen native uniform prediction rate")
    rows, included, excluded = [], [], []
    for a, b in np.asarray(intervals).reshape(-1, 2):
        duration = float(b-a)
        allowed = [j for j in range(1, int(np.ceil(duration*rate)))
                   if min(j/rate, duration-j/rate) > .4+1e-12]
        offsets = ([allowed[i] for i in np.floor(np.linspace(0, len(allowed)-1, 20)).astype(int)]
                   if len(allowed) >= 20 else None)
        (included if offsets is not None else excluded).append([float(a), float(b)])
        rows.append({"start_s": float(a), "stop_s": float(b), "duration_s": duration,
            "requested_grid_samples": int(np.sum((times >= a) & (times < b))),
            "offsets_frames": offsets, "offsets_s": None if offsets is None else [j/rate for j in offsets],
            "undefined_reason": None if offsets is not None else "fewer_than20_admissible_unique_offsets"})
    total = sum(r["duration_s"] for r in rows)
    duration = sum(b-a for a, b in included)
    fraction = duration/total if total else None
    return {"schema": SHIFT_SCHEMA, "rate": rate, "segments": rows,
            "eligible_intervals_s": included, "excluded_intervals_s": excluded,
            "primary_eligible_duration_s": total, "shift_eligible_duration_s": duration,
            "retained_fraction_of_primary": fraction, "defined": bool(included),
            "qualified": fraction is not None and fraction >= .8-1e-10}


def shifted_candidates(candidate_times, confidences, plan):
    """Shift fixed NMS events; never redetect or suppress after the wrap.

    Returns paired real candidates and20 null draws on IDENTICAL eligible
    segments. Counts/confidences are preserved. Wrap-created close pairs are an
    explicit null limitation, not silently repaired by another NMS.
    """
    candidates, confidence = np.asarray(candidate_times, float), np.asarray(confidences, float)
    if candidates.shape != confidence.shape or candidates.ndim != 1:
        raise ContractError("Shift candidate shape mismatch")
    intervals = np.asarray(plan["eligible_intervals_s"], float).reshape(-1, 2)
    keep = _regions(candidates, intervals) >= 0
    real, scores = candidates[keep], confidence[keep]
    if not plan["defined"]:
        return {"real_times_s": real, "confidences": scores, "draws": None}
    draws = []
    for draw in range(20):
        shifted = real.copy(); wrapped = 0; close_pairs = 0; jitter = []
        for row in plan["segments"]:
            if row["offsets_s"] is None:
                continue
            a, b, duration = row["start_s"], row["stop_s"], row["duration_s"]
            take = (real >= a) & (real < b)
            local = real[take]-a+row["offsets_s"][draw]
            wrapped += int(np.sum(local >= duration))
            shifted[take] = a+np.mod(local, duration)
            displacement = shifted[take]-real[take]
            jitter.extend(np.abs(displacement*plan["rate"]-np.rint(displacement*plan["rate"]))/plan["rate"])
            sorted_times = np.sort(shifted[take])
            close_pairs += sum(int(np.sum(sorted_times[i+1:]-t < .2-1e-12))
                               for i, t in enumerate(sorted_times))
        if np.any(_regions(shifted, intervals) < 0):
            raise ContractError("Circular shift escaped its valid segment")
        draws.append({"times_s": shifted, "confidences": scores.copy(),
                      "wrapped_candidate_count": wrapped, "close_pair_count_lt_0p2s": close_pairs,
                      "wrap_grid_jitter_max_abs_s": float(max(jitter, default=0.)),
                      "off_requested_grid_candidate_count": int(np.sum(np.asarray(jitter) > 1e-10))})
    return {"real_times_s": real, "confidences": scores, "draws": draws}


def audio_novelty(features, rate):
    """Audio-only .3s pre/post Euclidean standardized64mel-mean distance."""
    x = np.asarray(features, float)
    if x.ndim != 2 or x.shape[1] != 94 or rate not in (10, 40) or not np.isfinite(x).all():
        raise ContractError("Audio novelty requires finite standardized94 at10/40Hz")
    width = int(.3*rate)
    out = np.zeros(len(x)); supported = np.zeros(len(x), bool)
    sums = np.vstack([np.zeros(64), np.cumsum(x[:, :64], axis=0)])
    for i in range(width, len(x)-width+1):
        out[i] = np.linalg.norm((sums[i+width]-sums[i] - sums[i]+sums[i-width])/width)
        supported[i] = True
    return out, supported


def _features(audio40, head):
    x = validate_audio_window(audio40)
    return x if head == "accent" else x.reshape(640, 4, 94).mean(1)


@dataclass
class BaselineExample:
    labels: FrozenLabels
    normalized_audio40: np.ndarray


def prepare_baselines(examples, transforms):
    """Fit four declared controls using TRAIN center8 labels ONLY, never DEV."""
    import sklearn
    from sklearn.linear_model import LogisticRegression
    from sklearn.exceptions import ConvergenceWarning
    import warnings
    validate_transforms(transforms)
    examples = sorted(examples, key=lambda e: e.labels.recording_id)
    if not examples or any(e.labels.partition != "train" for e in examples):
        raise ContractError("Baseline fitting accepts nonempty TRAIN only")
    ids = [e.labels.recording_id for e in examples]
    if len(ids) != len(set(ids)) or set(ids) != set(transforms["fit_recording_ids"]):
        raise ContractError("Baseline TRAIN inventory differs from frozen transform fit inventory")
    result = {"schema_version": SCHEMA, "definition_sha256": json_hash(evaluation_definition()),
        "fit_partition": "train", "fit_recording_ids": ids,
        "software": {"numpy": np.__version__, "sklearn": sklearn.__version__},
        "fit_exposure": evaluation_definition()["fit_exposure"],
        "norm_sha256": transforms["norm_sha256"], "vocab_sha256": transforms["vocab_sha256"],
        "target_hashes": {e.labels.recording_id: e.labels.content_hash() for e in examples}, "heads": {}}
    for head, rate in RATES.items():
        xs, ys, onsets, novelties = [], [], [], []
        for example in examples:
            labels = example.labels.validate()
            x = _features(example.normalized_audio40, head)
            y = labels.accent10 if head == "accent" else labels.color
            valid = labels.accent_mask10 if head == "accent" else labels.color_mask
            if head == "accent":
                y, valid = np.repeat(y, 4), np.repeat(valid, 4)
            center = (np.arange(len(y)) >= 28*rate) & (np.arange(len(y)) < 36*rate)
            novelty, supported = audio_novelty(x, rate)
            mask = center & valid
            if np.any(mask & ~supported):
                raise ContractError("Novelty baseline lacks common supervised support")
            xs.append(x[mask]); ys.append(y[mask])
            onsets.append(x[mask, 64+AUDIO_SCALARS.index("onset_ema")]); novelties.append(novelty[mask])
        x, y = np.concatenate(xs), np.concatenate(ys)
        if len(np.unique(y)) != 2:
            raise ContractError(f"Missing {head} TRAIN classes for logistic control; no constant fallback")
        # lbfgs with finite C and default l1_ratio/penalty is L2. Avoid the
        # deprecated explicit penalty keyword in sklearn>=1.8 (same objective).
        model = LogisticRegression(C=1., class_weight="balanced", max_iter=1000,
                                   random_state=0, solver="lbfgs")
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model.fit(x, y)
        prevalence = float(y.mean())
        result["heads"][head] = {"prevalence": prevalence, "eligible_training_bins": len(y),
            "positive_training_bins": int(y.sum()), "coef": model.coef_[0].tolist(),
            "intercept": float(model.intercept_[0]), "classes": model.classes_.tolist(),
            "iterations": int(model.n_iter_[0]), "deployment_thresholds": {"model": .5, "donor": .5,
                "logistic": .5, "prevalence": prevalence,
                "audio_onset": float(np.quantile(np.concatenate(onsets), .95)),
                "audio_novelty": float(np.quantile(np.concatenate(novelties), .95))}}
    return result


def baseline_curves(normalized_audio40, fitted, head):
    if fitted.get("definition_sha256") != json_hash(evaluation_definition()) or fitted.get("fit_partition") != "train":
        raise ContractError("Baseline definition/provenance mismatch")
    x = _features(normalized_audio40, head)
    parameters = fitted["heads"][head]
    coef = np.asarray(parameters["coef"])
    if coef.shape != (94,) or parameters["classes"] != [0., 1.]:
        raise ContractError("Frozen logistic coefficient/class shape mismatch")
    logits = x @ coef + parameters["intercept"]
    return {"prevalence": np.full(len(x), parameters["prevalence"]),
            "audio_onset": x[:, 64+AUDIO_SCALARS.index("onset_ema")],
            "audio_novelty": audio_novelty(x, RATES[head])[0],
            "logistic": 1/(1+np.exp(-np.clip(logits, -700, 700)))}


def score_record(labels, curves, thresholds, head, *, include_shifts=True):
    """Same full64s candidate extraction, central clip, and support for all arms."""
    labels.validate()
    rate = RATES[head]
    start = labels.requested_times10_s[0]+28
    times = labels.requested_times10_s[0]+np.arange(64*rate)/rate
    intervals = eligible_intervals(labels, head)
    full_intervals = eligible_intervals(labels, head, center=False)
    y = labels.accent10 if head == "accent" else labels.color
    source = labels.source_times10_s
    refs = source[(y > 0) & (_regions(source, intervals) >= 0)]
    plan = shift_plan(times, intervals, rate)
    supplied = {k: _vector(v, 64*rate, k) for k, v in curves.items()}
    if set(supplied)-set(thresholds):
        raise ContractError("Missing frozen deployment threshold")
    scores = {}; candidates = {}
    def measure(reference_times, candidate_times, confidence, eligible, threshold):
        return asdict(metrics.visual_event_ap(reference_times-start, candidate_times-start, confidence,
            duration_s=8., deployment_threshold=threshold, channel=head,
            eligible_intervals_s=np.asarray(eligible, float).reshape(-1, 2)-start))
    for method, values in supplied.items():
        candidate, confidence = candidate_peaks(times, values, full_intervals)
        keep = _regions(candidate, intervals) >= 0
        candidate, confidence = candidate[keep], confidence[keep]
        candidates[method] = (candidate, confidence)
        scores[method] = measure(refs, candidate, confidence, intervals, thresholds[method])
    null = None
    if include_shifts:
        if "model" not in supplied:
            raise ContractError("Shift control requires explicit model scores")
        shifted = shifted_candidates(*candidates["model"], plan)
        restricted = np.asarray(plan["eligible_intervals_s"], float).reshape(-1, 2)
        null_refs = refs[_regions(refs, restricted) >= 0]
        null = {"real": None, "draws": None}
        if shifted["draws"] is not None:
            null["real"] = measure(null_refs, shifted["real_times_s"], shifted["confidences"],
                                   restricted, thresholds["model"])
            null["draws"] = [{"score": measure(null_refs, row["times_s"], row["confidences"],
                restricted, thresholds["model"]), "wrapped_candidate_count": row["wrapped_candidate_count"],
                "close_pair_count_lt_0p2s": row["close_pair_count_lt_0p2s"],
                "wrap_grid_jitter_max_abs_s": row["wrap_grid_jitter_max_abs_s"],
                "off_requested_grid_candidate_count": row["off_requested_grid_candidate_count"]}
                for row in shifted["draws"]]
    return {"recording_id": labels.recording_id, "canonical_id": labels.canonical_id,
        "head": head, "source_reference_times_s": refs, "scored_interval_s": [start, start+8],
        "eligible_intervals_s": intervals, "target_sha256": labels.content_hash(),
        "shift_plan": plan, "paired_shift_null": null,
        "missing_controls": [] if include_shifts else ["within_valid_segment_20_shifts"], "scores": scores}


def observer_effect_gate(contrasts, *, support_qualified, null_qualified,
                         controls_complete, cell_inventory_complete=True):
    """Conjunction from preregistered hypothesis gate ``observer-effect``.

    A numerical engineering gain alone is NOT hypothesis success. Failed or
    unavailable prerequisites make the overall qualified result undefined.
    The paired95% CI is unadjusted; the p-value must be the Holm8-adjusted one.
    """
    color = contrasts.get("VR-V6_color")
    accent = contrasts.get("VR-V6_accent")
    def available(row, key):
        return row is not None and key in row and row[key] is not None
    components = {
        "color_AP_gain_ge_0p01": (bool(color["difference"] >= .01)
                                 if available(color, "difference") else None),
        "color_paired95_CI_lower_gt_0": (bool(color["interval"][0] > 0)
            if available(color, "interval") and color.get("confidence") == .95 else None),
        "color_Holm8_p_lt_0p05": (bool(color["holm8_p_value"] < .05)
                                if available(color, "holm8_p_value") else None),
        "accent_AP_gain_ge_minus_0p01": (bool(accent["difference"] >= -.01)
                                        if available(accent, "difference") else None),
    }
    prerequisites = {"support_qualified": bool(support_qualified), "null_qualified": bool(null_qualified),
        "controls_complete": bool(controls_complete), "cell_inventory_complete": bool(cell_inventory_complete),
        "registered_contrast_statistics_available": all(value is not None for value in components.values())}
    ready = all(prerequisites.values())
    return {"gate_id": "observer-effect", "hypothesis_id": "h-layers-20260922-observer-proxy",
        "components": components, "prerequisites": prerequisites,
        "qualified_pass": all(components.values()) if ready else None,
        "unmet_prerequisites": [name for name, passed in prerequisites.items() if not passed],
        "numerical_gain_alone_is_hypothesis_success": False,
        "scope": SCOPE, "CI_multiplicity_adjusted": False}


def summarize_campaign(records, *, enrolled_songs, exclusions, usable_inventory=None,
                       capture_exclusions=()):
    """Macro only the usable inventory; capture failures are NOT metric records.

    ``capture_exclusions`` are DEV receipts already hash/source/scope-validated
    by validate_run_scope in the CLI. ``usable_inventory`` is the final manifest
    (canonical_id, recording_id) inventory, not inferred from successful scores.
    The original31 denominator remains explicit; excluded records never become
    zero AP, no-event observations, or fake empty-support predictions.
    """
    enrolled = sorted(enrolled_songs)
    if (len(enrolled) != 31 or len(set(enrolled)) != 31 or not isinstance(exclusions, dict)
            or len(exclusions) != 9 or set(exclusions) & set(enrolled)
            or any(not isinstance(v, str) or not v.strip() for v in exclusions.values())):
        raise ContractError("Must retain31 enrolled /40 original and all9 original exclusions")
    capture_exclusions = list(capture_exclusions)
    excluded_inventory = set()
    for row in capture_exclusions:
        if (row.get("partition") not in ("dev", "val") or row.get("pre_fit") is not True
                or row.get("not_based_on_target_or_model_scores") is not True
                or row.get("basis") not in ("capture_integrity", "source_availability", "source_identity")
                or not isinstance(row.get("reason"), str) or not row["reason"].strip()
                or not row.get("receipt", {}).get("path") or not row.get("receipt", {}).get("sha256")
                or not row.get("recording_id") or row.get("canonical_id") not in enrolled):
            raise ContractError("DEV capture exclusions require explicit pre-fit source-only receipt provenance")
        excluded_inventory.add((row["canonical_id"], row["recording_id"]))
    if len(excluded_inventory) != len(capture_exclusions):
        raise ContractError("Duplicate capture exclusions")
    index = {}
    for row in records:
        key = row["recipe"], row["seed"], row["head"], row["canonical_id"], row["recording_id"]
        if key in index or key[0] not in RECIPES or key[1] not in SEEDS or key[2] not in RATES or key[3] not in enrolled:
            raise ContractError("Duplicate/undeclared score inventory")
        index[key] = row
    observed_base = {(key[3], key[4]) for key in index if key[:3] == (RECIPES[0], 0, "accent")}
    base_inventory = observed_base if usable_inventory is None else set(map(tuple, usable_inventory))
    if usable_inventory is not None and len(base_inventory) != len(usable_inventory):
        raise ContractError("Duplicate declared usable recording inventory")
    songs = sorted({song for song, _ in base_inventory})
    excluded_songs = {song for song, _ in excluded_inventory}
    if (not base_inventory or observed_base != base_inventory
            or {rid for _, rid in base_inventory} & {rid for _, rid in excluded_inventory}
            or set(songs) | excluded_songs != set(enrolled)):
        raise ContractError("Missing usable score inventory or unreceipted enrolled capture exclusion")
    unavailable = sorted(set(enrolled)-set(songs))
    for recipe in RECIPES:
        for seed in SEEDS:
            for head in RATES:
                actual = {(key[3], key[4]) for key in index if key[:3] == (recipe, seed, head)}
                if actual != base_inventory:
                    raise ContractError("Missing model/seed/head/record score cells; no silent exclusion")
    qualified = {head: set(songs) for head in RATES}
    events = {head: set() for head in RATES}
    null_qualified = {head: set(songs) for head in RATES}
    null_events = {head: set() for head in RATES}
    missing = set()
    methods = ("model", "donor", *BASELINE_NAMES)
    for head in RATES:
        for song, rid in sorted(base_inventory):
            rows = [index[(r, seed, head, song, rid)] for r in RECIPES for seed in SEEDS]
            first = rows[0]; first_score = first["scores"]["model"]
            for row in rows:
                if (row["target_sha256"] != first["target_sha256"]
                        or not np.array_equal(row["eligible_intervals_s"], first["eligible_intervals_s"])
                        or row["shift_plan"] != first["shift_plan"]):
                    raise ContractError("Comparator target/support/shift-plan drift")
                missing.update(row["missing_controls"])
                missing.update(set(methods)-row["scores"].keys())
                for score in row["scores"].values():
                    if any(score[key] != first_score[key] for key in
                           ("reference_count", "eligible_duration_s", "duration_s", "coverage")):
                        raise ContractError("Comparator reference/validity differs")
            if first_score["coverage"] < .8-1e-10:
                qualified[head].discard(song)
            if first_score["reference_count"]:
                events[head].add(song)
            if not first["shift_plan"]["qualified"]:
                null_qualified[head].discard(song)
            null = first.get("paired_shift_null")
            if null is None or null["real"] is None:
                null_qualified[head].discard(song)
            elif null["real"]["reference_count"]:
                null_events[head].add(song)
    common = set.intersection(*qualified.values())
    support = {}
    for head in RATES:
        positive = events[head] & common
        support[head] = {"qualified_groups": len(qualified[head]), "qualified_ids": sorted(qualified[head]),
            "common_qualified_groups": len(common), "common_qualified_ids": sorted(common),
            "event_bearing_groups": len(positive), "event_bearing_ids": sorted(positive),
            "all_enrolled_event_bearing_ids": sorted(events[head]),
            "nonevent_ids": sorted(set(songs)-events[head]),
            "unqualified_ids": sorted(set(enrolled)-qualified[head]),
            "unqualified_usable_ids": sorted(set(songs)-qualified[head]),
            "capture_unavailable_ids": unavailable,
            "gate": len(common) >= 30 and len(positive) >= 10}
    null_common = common & set.intersection(*null_qualified.values())
    null_support = {head: {"qualified_groups": len(null_common), "qualified_ids": sorted(null_common),
        "event_bearing_groups": len(null_events[head] & null_common),
        "gate": len(null_common) >= 30 and len(null_events[head] & null_common) >= 10} for head in RATES}
    gates = all(x["gate"] for x in support.values())
    null_gates = all(x["gate"] for x in null_support.values())
    observations, macros = {}, {}
    for head in RATES:
        for recipe in RECIPES:
            rows = [v for k, v in index.items() if k[0] == recipe and k[2] == head]
            for method in methods:
                if any(method not in r["scores"] for r in rows):
                    continue
                for metric in ("average_precision", "false_cues_per_minute", "nonevent_false_cues_per_minute", "coverage"):
                    cells = []
                    for r in rows:
                        score = r["scores"][method]
                        if metric == "nonevent_false_cues_per_minute":
                            value = score["false_cues_per_minute"] if score["reference_count"] == 0 else None
                        else:
                            value = score[metric]
                        if metric == "average_precision" and r["canonical_id"] not in common:
                            value = None
                        cells.append(metrics.MetricObservation(r["canonical_id"], r["recording_id"],
                            "observer", r["seed"], 0, value))
                    if method == "model" and metric == "average_precision":
                        observations[(recipe, head)] = cells
                    macros[recipe+":"+head+":"+method+":"+metric] = asdict(metrics.macro_song_mean(cells,
                        expected_songs=songs, expected_seeds=SEEDS, expected_draws=(0,)))
    contrast_results = {}; pvalues = {}
    if gates:
        for name, (candidate, comparator) in CONTRASTS.items():
            for head in RATES:
                key = name+"_"+head
                result = metrics.paired_song_bootstrap(observations[(candidate, head)], observations[(comparator, head)],
                    expected_songs=songs, expected_seeds=SEEDS, expected_draws=(0,), random_seed=0)
                contrast_results[key] = asdict(result)
                pvalues[key] = result.p_value_two_sided
        corrected = metrics.holm_adjust(pvalues)
        for key in contrast_results:
            contrast_results[key]["holm8_p_value"] = corrected[key]
            contrast_results[key]["CI_multiplicity_adjusted"] = False
    null_macro = {}
    for head in RATES:
        for recipe in RECIPES:
            real, shifted = [], []
            for key, row in index.items():
                if key[0] != recipe or key[2] != head:
                    continue
                null = row.get("paired_shift_null")
                available = null is not None and null["real"] is not None and row["canonical_id"] in null_common
                for draw in range(20):
                    args = (row["canonical_id"], row["recording_id"], "observer", row["seed"], draw)
                    real.append(metrics.MetricObservation(*args, null["real"]["average_precision"] if available else None))
                    shifted.append(metrics.MetricObservation(*args, null["draws"][draw]["score"]["average_precision"] if available else None))
            kwargs = dict(expected_songs=songs, expected_seeds=SEEDS, expected_draws=tuple(range(20)))
            null_macro[recipe+":"+head] = {"real": asdict(metrics.macro_song_mean(real, **kwargs)),
                "mean20_shifts": asdict(metrics.macro_song_mean(shifted, **kwargs)),
                "same_support": True, "shifts_are_independent_replicates": False,
                "inference_gate": null_support[head]["gate"],
                "comparison_scope": "paired_AP_descriptive_null_not_in_Holm8"}
    primary = contrast_results.get("VR-V6_color")
    interpretable = gates and null_gates and not missing
    effect_gate = observer_effect_gate(contrast_results, support_qualified=gates,
        null_qualified=null_gates, controls_complete=not missing, cell_inventory_complete=True)
    return {"schema_version": SCHEMA, "scope": SCOPE,
        "status": "blocked_inference" if not gates else
                  "scored_observer_proxy_only" if interpretable else "scored_interpretation_blocked",
        "interpretation_gate_pass": interpretable,
        "original80percentgate": "FAILED_NOT_WAIVED", "original_groups": 40, "enrolled_groups": 31,
        "enrolled_song_ids": enrolled, "usable_groups": len(songs), "usable_song_ids": songs,
        "capture_unavailable_ids": unavailable, "DEV_capture_exclusions": capture_exclusions,
        "usable_fraction_of_enrolled31": len(songs)/31, "usable_fraction_of_original40": len(songs)/40,
        "capture_failure_imputed_as_metric_record": False,
        "macro_inventory_scope": "usable_recordings_only; excluded_capture_is_not_a_nonevent_or_zeroAP",
        "original_coverage": 31/40, "original_exclusions": exclusions, "replacement_records": False,
        "support_gates": support, "paired_null_support_gates": null_support, "missing_controls": sorted(missing),
        "macro_scores": macros, "paired_shift_null_macro": null_macro,
        "contrasts": contrast_results, "primary_VR_minus_V6_color_AP_minimum_gain": .01,
        "primary_gain_pass": None if primary is None or not interpretable else primary["difference"] >= .01,
        "primary_gain_pass_scope": "numerical_engineering_gain_component_only_NOT_hypothesis_success",
        "observer_effect_gate": effect_gate,
        "observer_effect_qualified_pass": effect_gate["qualified_pass"],
        "records": records, "CI_note": "95percent percentile CI unadjusted; Holm corrects8P only",
        "checkpoint_selection_scope": "selected_on_same_historicalDEV_commonmainloss; conditional_development_estimates_not_unbiased_or_fresh_generalization",
        "generalization_or_LIGHT_quality_claim": False}


def palette_rollouts(model, context10, fixed_cue_indices, *, draw_seeds=(101, 102, 103, 104)):
    """Optional secondary instrument: inverse-CDF own/null paired common noise.

    Cues must be frozen from a common AUDIO-only inventory outside this call.
    First cue is clip-start initialization, not an event. No reference tokens.
    Returns generated64s IDs for the separate palette_metrics instrument.
    """
    import torch
    if tuple(draw_seeds) != (101, 102, 103, 104):
        raise ContractError("Palette draws fixed at seeds101:104")
    cues = np.asarray(fixed_cue_indices)
    if (cues.ndim != 1 or not len(cues) or not np.issubdtype(cues.dtype, np.integer)
            or cues[0] != 0 or np.any(np.diff(cues) <= 0) or cues[-1] >= 640
            or tuple(context10.shape[:2]) != (1, 640)):
        raise ContractError("Palette cues/context must cover one64s clip and start with initialization0")
    k = model.cfg.palette_size
    with torch.no_grad():
        context = context10[:, cues][0]
        expanded = context[:, None].expand(-1, k+1, -1)
        states = torch.arange(k+1, device=context.device).expand(len(cues), -1)
        probabilities = torch.softmax(model.palette_logits(expanded, states).float(), -1).cpu().numpy()
    result = {}
    for seed in draw_seeds:
        noise = np.random.default_rng(seed).random(len(cues))
        pair = {"uniform_noise": noise}
        for ablation in (False, True):
            state = k; generated = np.empty(640, int); previous = np.empty(640, int)
            for j, cue in enumerate(cues):
                end = cues[j+1] if j+1 < len(cues) else 640
                effective = k if ablation else state
                previous[cue:end] = effective
                cdf = np.cumsum(probabilities[j, effective]); cdf[-1] = 1.
                state = int(np.searchsorted(cdf, noise[j], side="right"))
                generated[cue:end] = state
            pair["null_state" if ablation else "own_state"] = {"generated_palette_ids": generated,
                                                                  "cue_previous_ids": previous[cues]}
        result[seed] = pair
    return result


def _serial(value):
    if is_dataclass(value):
        return _serial(asdict(value))
    if isinstance(value, dict):
        return {str(k): _serial(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_serial(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json_exclusive(path, value, campaign_root):
    path = Path(path)
    data = (json.dumps(_serial(value), sort_keys=True, indent=2, allow_nan=False)+"\n").encode()
    campaign_space(campaign_root, len(data))
    if not path.resolve().is_relative_to(Path(campaign_root).resolve()):
        raise ContractError("Evaluation output must remain inside bounded campaign root")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)


def write_npz_exclusive(path, arrays, campaign_root):
    path = Path(path)
    campaign_space(campaign_root, sum(np.asarray(x).nbytes for x in arrays.values())+2**20)
    if not path.resolve().is_relative_to(Path(campaign_root).resolve()):
        raise ContractError("Prediction output must remain inside bounded campaign root")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def _registration(path, manifest_path, transforms_path, *, phase):
    """No scientific data access until parent freezes exact phase bindings."""
    registration = json.loads(Path(path).read_text())
    expected = {"definition_sha256": json_hash(evaluation_definition()),
        "implementation_sha256": json_hash(implementation_hashes()),
        "manifest_sha256": sha256(manifest_path), "transforms_sha256": sha256(transforms_path)}
    if (registration.get("protocol_id") != PROTOCOL_ID or not registration.get("registered_before_fit")
            or not registration.get("vault_experiment_id") or registration.get("scope") != SCOPE
            or phase not in registration.get("authorized_phases", [])):
        raise ContractError("Missing parent preregistered observer calibration/source/scope/phase authorization")
    for key, value in expected.items():
        if registration.get("bindings", {}).get(key) != value:
            raise ContractError(f"Evaluation preregistration binding mismatch: {key}")
    return registration


def _inputs(args, phase):
    registration = _registration(args.registration, args.manifest, args.transforms, phase=phase)
    manifest = json.loads(Path(args.manifest).read_text())
    scope = validate_run_scope(manifest, TrainConfig(), observer_proxy=True)
    if registration.get("scope_sha256") != scope["scope_sha256"]:
        raise ContractError("Evaluation observer scope hash mismatch")
    transforms = json.loads(Path(args.transforms).read_text())
    validate_transforms(transforms)
    if transforms.get("study_scope") != scope:
        raise ContractError("Frozen TRAIN transforms belong to a different observer scope")
    train_rows = [row for row in manifest["records"] if row["partition"] == "train"]
    if (set(transforms.get("fit_recording_ids", [])) != {row["recording_id"] for row in train_rows}
            or set(transforms.get("fit_canonical_ids", [])) != {row["canonical_id"] for row in train_rows}):
        raise ContractError("Frozen transforms do not use exact final identity-qualified TRAIN inventory")
    store = SourceStore(manifest)
    mapping = donor_map(manifest["records"], registration.get("donor_tempos_bpm", {}))
    if (registration.get("donor_map") != mapping
            or registration.get("donor_map_sha256") != json_hash(mapping)):
        raise ContractError("Donor map+hash must be frozen before data use")
    return registration, manifest, transforms, store


def validate_training_receipt(checkpoint, registration):
    """Bind Ashuf evaluation to the actual TRAIN/DEV-selection donor run map."""
    run = json.loads(checked_ref(checkpoint.get("training_run_receipt")).read_text())
    config = run.get("config", {})
    if (run.get("status") != "complete" or run.get("smoke", True)
            or config.get("recipe") != checkpoint["recipe"] or config.get("seed") != checkpoint["seed"]
            or json_hash(config) != checkpoint["bindings"].get("config_sha256")
            or any(run.get("bindings", {}).get(k) != v for k, v in checkpoint["bindings"].items())):
        raise ContractError("Training run receipt differs from checkpoint provenance/config")
    weight_key = "best_weights_sha256" if checkpoint["role"] == "selected" else "final_weights_sha256"
    if run.get(weight_key) != checkpoint["sha256"]:
        raise ContractError("Selected/final weight hash differs from completed training run")
    if checkpoint["recipe"] == "S_VR_P0_Ashuf" and run.get("donors") != registration["donor_map"]:
        raise ContractError("Ashuf evaluation donor map differs from actual TRAIN/DEV-selection run receipt")
    return run


def validate_prediction_receipt(receipt, registration, baseline_reference):
    """Bind arm/seed/role labels to the registered checkpoint, not a copied tag."""
    if (receipt.get("status") != "complete" or receipt.get("bindings") != registration["bindings"]
            or receipt.get("baselines") != baseline_reference or receipt.get("scope") != SCOPE
            or receipt.get("donor_map") != registration["donor_map"]
            or receipt.get("donor_map_sha256") != json_hash(registration["donor_map"])
            or receipt.get("donor_tempos_bpm") != registration["donor_tempos_bpm"]
            or receipt.get("primary_audio_route") != ("training_tempo_donor" if receipt.get("recipe") == "S_VR_P0_Ashuf" else "recipient")):
        raise ContractError("Prediction receipt binding mismatch")
    checkpoint = receipt.get("checkpoint")
    if checkpoint not in registration.get("checkpoints", {}).values():
        raise ContractError("Prediction checkpoint missing from current frozen model inventory")
    for receipt_key, checkpoint_key in (("recipe", "recipe"), ("seed", "seed"), ("checkpoint_role", "role")):
        if receipt.get(receipt_key) != checkpoint.get(checkpoint_key):
            raise ContractError("Prediction recipe/seed/checkpoint role relabeling")
    metadata = receipt.get("metadata", {})
    config = metadata.get("config", {})
    if (config.get("recipe") != checkpoint["recipe"] or config.get("seed") != checkpoint["seed"]
            or json_hash(config) != checkpoint["bindings"].get("config_sha256")
            or any(metadata.get(key) != value for key, value in checkpoint["bindings"].items())):
        raise ContractError("Prediction metadata does not match checkpoint config/provenance")
    step = metadata.get("step")
    if (type(step) is not int or step <= 0 or step > 6000 or step % 300
            or checkpoint["role"] == "final" and (step != 6000 or metadata.get("smoke", False))
            or checkpoint["role"] == "selected" and not np.isfinite(metadata.get("selection_loss", float("nan")))):
        raise ContractError("Prediction selected/final role does not match checkpoint metadata")
    return checkpoint


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="phase", required=True)
    schema = sub.add_parser("schema", help="Print data-free exact definitions for preregistration")
    for phase in ("prepare-baselines", "predict", "score"):
        p = sub.add_parser(phase)
        for name in ("manifest", "transforms", "registration", "output"):
            p.add_argument("--"+name, required=True)
        if phase == "predict":
            p.add_argument("--checkpoint", required=True, help="Hash+metadata binding key in registration checkpoints")
            p.add_argument("--baselines", required=True)
            p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
            p.add_argument("--gpu-authorized", action="store_true")
        if phase == "score":
            p.add_argument("--prediction-index", required=True,
                           help="JSON list of hash references to complete predict-stage receipts")
            p.add_argument("--baselines", required=True)
    args = parser.parse_args(argv)
    if args.phase == "schema":
        print(json.dumps({"definition": evaluation_definition(), "definition_sha256": json_hash(evaluation_definition()),
            "implementation_files": implementation_hashes(),
            "implementation_sha256": json_hash(implementation_hashes())}, sort_keys=True, indent=2))
        return
    if Path(args.output).exists():
        raise ContractError("Output exists; no overwrite/resume")
    registration, manifest, transforms, store = _inputs(args, args.phase)
    root = manifest["campaign_root"]
    campaign_space(root, 2**20)
    if args.phase == "prepare-baselines":
        examples = []
        for row in manifest["records"]:
            if row["partition"] != "train":
                continue
            verify_donor_tempos(store, [row["recording_id"]], registration["donor_tempos_bpm"])
            record = store.visual(row["recording_id"])
            examples.append(BaselineExample(freeze_labels(record, transforms), audio_window(store, record, transforms)))
        fitted = prepare_baselines(examples, transforms)
        fitted["bindings"] = registration["bindings"]
        fitted["study_scope"] = transforms["study_scope"]
        fitted["donor_map"] = registration["donor_map"]
        fitted["donor_map_sha256"] = json_hash(registration["donor_map"])
        fitted["donor_tempos_bpm"] = registration["donor_tempos_bpm"]
        fitted["donor_audit"] = donor_audit(manifest["records"], registration["donor_map"])
        write_json_exclusive(args.output, fitted, root)
        return
    baseline_reference = registration.get("baselines")
    if baseline_reference is None or checked_ref(baseline_reference).resolve() != Path(args.baselines).resolve():
        raise ContractError("Frozen TRAIN baselines must be hash-bound in phase registration")
    fitted = json.loads(Path(args.baselines).read_text())
    if (fitted["norm_sha256"] != transforms["norm_sha256"] or fitted["vocab_sha256"] != transforms["vocab_sha256"]
            or fitted["bindings"] != registration["bindings"] or fitted["donor_map"] != registration["donor_map"]
            or fitted["donor_tempos_bpm"] != registration["donor_tempos_bpm"]):
        raise ContractError("Baseline transform/source/donor bindings changed")
    dev_rows = [r for r in manifest["records"] if r["partition"] in ("dev", "val")]
    # validate_run_scope already proved exact131-ID disjoint usable+excluded
    # union and hashed each pre-fit source-only exclusion receipt.
    dev_exclusions = [r for r in transforms["study_scope"]["capture_exclusions"]
                      if r["partition"] in ("dev", "val")]
    enrolled_dev = sorted({r["canonical_id"] for r in dev_rows+dev_exclusions})
    usable_dev = {r["canonical_id"] for r in dev_rows}
    if len(enrolled_dev) != 31 or len(usable_dev) < 30:
        raise ContractError("Keep immutable31 enrolled DEV and >=30 usable with explicit pre-fit capture receipts")
    if args.phase == "predict":
        if args.device == "cuda" and not args.gpu_authorized:
            raise ContractError("CUDA launch belongs to parent; explicit --gpu-authorized required")
        checkpoint = registration.get("checkpoints", {}).get(args.checkpoint)
        if (checkpoint is None or checkpoint.get("recipe") not in RECIPES or checkpoint.get("seed") not in SEEDS
                or checkpoint.get("role") not in ("selected", "final")):
            raise ContractError("Checkpoint not bound to frozen recipe/seed/selected-or-final inventory")
        from .training import implementation_hashes as training_implementation_hashes
        scope = transforms["study_scope"]
        required_bindings = {"manifest_sha256": sha256(args.manifest), "split_sha256": store.split_sha256,
            "source_sha256": json_hash(training_implementation_hashes()),
            "config_sha256": json_hash(asdict(TrainConfig(recipe=checkpoint["recipe"], seed=checkpoint["seed"]))),
            **{key: scope[key] for key in OBSERVER_SCOPE_BINDINGS}}
        if any(checkpoint["bindings"].get(key) != value for key, value in required_bindings.items()):
            raise ContractError("Checkpoint source/config/observer scope does not match current frozen implementation")
        validate_training_receipt(checkpoint, registration)
        model, metadata = load_bound_model(checkpoint, transforms, transforms_sha256=sha256(args.transforms),
            expected_bindings=checkpoint["bindings"], device=args.device)
        if (metadata.get("config", {}).get("recipe") != checkpoint["recipe"]
                or metadata.get("config", {}).get("seed") != checkpoint["seed"]):
            raise ContractError("Checkpoint recipe/seed label differs from trained metadata")
        if (checkpoint["role"] == "final" and metadata.get("step") != 6000
                or checkpoint["role"] == "selected" and "selection_loss" not in metadata):
            raise ContractError("Checkpoint role does not match saved selection/final metadata")
        output = Path(args.output)
        if not output.resolve().is_relative_to(Path(root).resolve()):
            raise ContractError("Prediction output must remain inside bounded campaign root")
        rows = []; estimates = len(dev_rows)*300_000+2**20
        campaign_space(root, estimates)
        output.mkdir(parents=True, exist_ok=False)
        for row in dev_rows:
            rid = row["recording_id"]
            record = store.visual(rid)
            labels = freeze_labels(record, transforms)
            bound_target = registration.get("target_hashes", {}).get(rid)
            if labels.content_hash() != bound_target:
                raise ContractError("Frozen target checksum missing/changed before prediction")
            requested, source = validate_clocks(record.times, record.times if record.source_times is None else record.source_times)
            donor_id = registration["donor_map"][rid]
            verify_donor_tempos(store, [rid, donor_id], registration["donor_tempos_bpm"])
            donor = store.visual(donor_id)
            x = audio_window(store, record, transforms)
            donor_x = audio_window(store, donor, transforms)
            clocks = dict(requested_times10_s=requested, source_times10_s=source)
            conditions = predict_conditions(model, {rid: x, donor_id: donor_x}, rid,
                registration["donor_map"], recipe=checkpoint["recipe"], **clocks)
            values = {method+"__"+key: value for method, result in conditions.items()
                      for key, value in result.items()}
            for head in RATES:
                values.update({name+"__"+head: curve for name, curve in baseline_curves(x, fitted, head).items()})
            values.update(target__color_mask10=labels.color_mask, target__accent_mask10=labels.accent_mask10,
                          observer__color_valid=record.color_valid, observer__value_valid=record.value_valid)
            if record.frame_valid is not None:
                values["observer__frame_valid"] = record.frame_valid
            if record.mask_activity_known is not None:
                values["observer__mask_activity_known"] = record.mask_activity_known
            path = output/(hashlib.sha256(rid.encode()).hexdigest()[:24]+".npz")
            write_npz_exclusive(path, values, root)
            rows.append({"recording_id": rid, "canonical_id": row["canonical_id"], "source_row": row,
                         "donor_source_row": store.rows[donor_id], "target_sha256": bound_target,
                         "artifact": reference(path)})
        receipt = {"schema_version": SCHEMA, "scope": SCOPE, "status": "complete", "recipe": checkpoint["recipe"],
            "seed": checkpoint["seed"], "checkpoint_role": checkpoint["role"],
            "study_scope": transforms["study_scope"], "fit_exposure": evaluation_definition()["fit_exposure"],
            "checkpoint": checkpoint, "metadata": metadata,
            "bindings": registration["bindings"], "baselines": baseline_reference, "records": rows,
            "primary_audio_route": "training_tempo_donor" if checkpoint["recipe"] == "S_VR_P0_Ashuf" else "recipient",
            "donor_map": registration["donor_map"], "donor_map_sha256": json_hash(registration["donor_map"]),
            "donor_tempos_bpm": registration["donor_tempos_bpm"],
            "donor_audit": donor_audit(manifest["records"], registration["donor_map"])}
        write_json_exclusive(output/"complete.json", receipt, root)
        return
    index_ref = registration.get("prediction_index")
    if index_ref is None or checked_ref(index_ref).resolve() != Path(args.prediction_index).resolve():
        raise ContractError("Score input index must be hash-bound after raw prediction freeze")
    prediction_refs = json.loads(Path(args.prediction_index).read_text())
    records = []; checkpoint_roles = set()
    labels = {row["recording_id"]: freeze_labels(store.visual(row["recording_id"]), transforms) for row in dev_rows}
    for prediction_ref in prediction_refs:
        receipt = json.loads(checked_ref(prediction_ref).read_text())
        checkpoint_roles.add(receipt.get("checkpoint_role"))
        if receipt.get("checkpoint_role") not in ("selected", "final") or len(checkpoint_roles) != 1:
            raise ContractError("Score selected and final checkpoints in separate complete15-cell indices")
        checkpoint = validate_prediction_receipt(receipt, registration, baseline_reference)
        checked_ref(checkpoint)
        validate_training_receipt(checkpoint, registration)
        if {r["recording_id"] for r in receipt["records"]} != set(labels):
            raise ContractError("Prediction inventory differs from exact declared usable DEV subset")
        for row in receipt["records"]:
            rid = row["recording_id"]; label = labels[rid]
            donor_id = registration["donor_map"][rid]
            for source_row in (store.rows[rid], store.rows[donor_id]):
                for key in ("audio", "rhythm", "visual", "visual_complete"):
                    if key in source_row:
                        store.checked(source_row[key])
            if (row["source_row"] != store.rows[rid] or row["donor_source_row"] != store.rows[donor_id]
                    or row["target_sha256"] != label.content_hash()
                    or registration.get("target_hashes", {}).get(rid) != label.content_hash()):
                raise ContractError("Source/target changed between predict and score")
            with np.load(checked_ref(row["artifact"]), allow_pickle=False) as z:
                if (not np.array_equal(z["target__color_mask10"], label.color_mask)
                        or not np.array_equal(z["target__accent_mask10"], label.accent_mask10)):
                    raise ContractError("Frozen prediction target validity changed")
                for method in ("model", "donor"):
                    if not np.array_equal(z[method+"__requested_times10_s"], label.requested_times10_s) or not np.array_equal(
                            z[method+"__source_times10_s"], label.source_times10_s):
                        raise ContractError("Prediction requested/source clocks changed")
                for head in RATES:
                    curves = {name: z[name+"__"+head] for name in ("model", "donor", *BASELINE_NAMES)}
                    scored = score_record(label, curves, fitted["heads"][head]["deployment_thresholds"], head)
                    records.append({**scored, "recipe": receipt["recipe"], "seed": receipt["seed"]})
    summary = summarize_campaign(records, enrolled_songs=enrolled_dev,
        usable_inventory=[(r["canonical_id"], r["recording_id"]) for r in dev_rows],
        capture_exclusions=dev_exclusions, exclusions=registration.get("original_exclusions"))
    summary["bindings"] = registration["bindings"]
    summary["study_scope"] = transforms["study_scope"]
    summary["fit_exposure"] = evaluation_definition()["fit_exposure"]
    summary["TRAIN_fit_groups"] = len({row["canonical_id"] for row in manifest["records"] if row["partition"] == "train"})
    summary["TRAIN_fit_recordings"] = len(transforms["fit_recording_ids"])
    summary["identity_exclusion_ledger"] = manifest.get("identity_exclusions")
    summary["original_capture_inventory_recordings"] = transforms["study_scope"]["original_inventory_recordings"]
    summary["identity_quarantined_recording_ids"] = transforms["study_scope"]["identity_quarantined_recording_ids"]
    summary["capture_exclusions"] = transforms["study_scope"]["capture_exclusions"]
    summary["donor_map"] = registration["donor_map"]
    summary["donor_map_sha256"] = json_hash(registration["donor_map"])
    summary["donor_tempos_bpm"] = registration["donor_tempos_bpm"]
    summary["donor_audit"] = donor_audit(manifest["records"], registration["donor_map"])
    summary["checkpoint_role"] = next(iter(checkpoint_roles))
    summary["companion_checkpoint_role_still_required"] = "final" if summary["checkpoint_role"] == "selected" else "selected"
    write_json_exclusive(args.output, summary, root)


if __name__ == "__main__":
    main()
