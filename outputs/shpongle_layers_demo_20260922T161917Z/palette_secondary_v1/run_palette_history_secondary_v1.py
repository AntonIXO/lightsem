"""New opt-in palette observer secondary; frozen event/model/metric modules unchanged.
No data or inference on import. Real phases require parent authorization and hashes.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from lightsem.experiments.light_layers import evaluation as ev, palette_metrics as pm, metrics
from lightsem.experiments.light_layers import training as tr

HERE = Path(__file__).resolve().parent
PROTOCOL_PATH = HERE / "PALETTE_SECONDARY_PROTOCOL_V1.json"
PROTOCOL_ID = "palette-history-observer-secondary-20260922-v1"
SEEDS = (0, 1, 2)
DRAWS = (101, 102, 103, 104)
ROLES = ("selected", "final")
CONDITIONS = ("own_state", "null_state", "persistence", "tempo_donor_own")
CAP = 64 * 2**20
FLOOR = 2**30
RESERVE = 32 * 2**20
DERIVED_ROOT = Path("/mnt/lightsem-data/experiments/lightsem/palette-history-secondary-20260922-v1")


def require(condition, message):
    if not condition:
        raise tr.ContractError(message)


def serial(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if hasattr(value, "__dataclass_fields__"): return serial(asdict(value))
    if isinstance(value, dict): return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [serial(v) for v in value]
    return value


def space(root, reserve=0):
    root = Path(root)
    parent = root if root.exists() else root.parent
    require(parent.is_dir(), "Authorized output parent does not exist")
    used = sum(p.stat().st_size for p in root.rglob("*") if p.is_file()) if root.exists() else 0
    free = shutil.disk_usage(parent).free
    require(used + reserve <= CAP and free - reserve >= FLOOR,
            f"Storage gate failed: used={used}, reserve={reserve}, free={free}")
    return dict(derived_bytes=used, free_bytes=free, reserve_bytes=reserve)


def write_json(path, value, root):
    data = (json.dumps(serial(value), indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    write_bytes(path, data, root)


def write_bytes(path, data, root):
    path, root = Path(path), Path(root)
    require(path.resolve().is_relative_to(root.resolve()), "Output escapes bounded root")
    require(not path.exists(), "Output exists; no overwrite/resume")
    space(root, len(data))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream: stream.write(data)


def write_npz(path, values, root):
    output = io.BytesIO()
    np.savez_compressed(output, **values)
    write_bytes(path, output.getvalue(), root)


def cue_indices(mask):
    mask = np.asarray(mask)
    require(mask.shape == (640,) and mask.dtype == np.bool_, "Frozen cue mask must be bool640")
    return np.unique(np.r_[0, np.flatnonzero(mask)]).astype(np.int64)


def validate_cues(cues):
    cues = np.asarray(cues)
    require(cues.ndim == 1 and len(cues) > 0 and np.issubdtype(cues.dtype, np.integer)
            and cues[0] == 0 and np.all(np.diff(cues) > 0) and cues[-1] < 640,
            "Cues must be increasing integer640 indices including0")
    return cues.astype(np.int64)


def common_observer(record):
    require(len(record.times) == 640, "Exact64s reference required")
    source = record.times if record.source_times is None else record.source_times
    requested, source = ev.validate_clocks(record.times, source)
    require(np.allclose(requested - requested[0], np.arange(640)/10, atol=1e-8, rtol=0),
            "Requested grid is not regular10Hz")
    valid = np.asarray(record.color_valid, bool).copy()
    if record.frame_valid is not None: valid &= record.frame_valid
    valid &= ~np.asarray(record.cuts, bool)
    gaps = np.flatnonzero(np.diff(source) > .2 + 1e-8)
    valid[gaps] = False
    valid[gaps+1] = False
    hs = np.asarray(record.hs, np.float64).copy()
    require(hs.shape == (640, 72) and np.isfinite(hs).all() and np.all(hs >= 0), "Invalid HS")
    require(np.allclose(hs[valid].sum(1), 1., atol=1e-6, rtol=0), "Observed HS not normalized")
    hs[~valid] = 0
    return hs, valid


def canonical_track(track, prototypes):
    """Unsupported cue intervals have a canonical, metrically inert token0.

    This also removes arbitrary placeholder influence on invalid-change counts.
    Actual generated IDs remain preserved separately; no state is fed back here.
    """
    cues = np.rint(np.asarray(track.cue_times_s)*10).astype(int)
    ids = np.asarray(track.token_ids).copy()
    for j, start in enumerate(cues):
        end = cues[j+1] if j+1 < len(cues) else 640
        if not np.any(track.valid[start:end]): ids[j] = 0
    return replace(track, token_ids=ids)


def reference_track(record, cues, prototypes):
    cues = validate_cues(cues)
    hs, valid = common_observer(record)
    ids = np.zeros(len(cues), np.int64)
    for j, start in enumerate(cues):
        end = cues[j+1] if j+1 < len(cues) else 640
        observed = hs[start:end][valid[start:end]]
        if len(observed):
            ids[j] = np.argmin(((np.asarray(prototypes)-observed.mean(0))**2).sum(1))
    return pm.PaletteTrack(record.canonical_id, record.recording_id, np.arange(640)/10,
        hs, valid, cues/10, ids, pm.palette_prototypes_hash(prototypes))


def generated_track(reference, ids, prototypes):
    ids = np.asarray(ids)
    require(ids.shape == (640,) and np.issubdtype(ids.dtype, np.integer)
            and np.all((ids >= 0) & (ids < len(prototypes))), "Bad generated palette IDs")
    cues = np.rint(reference.cue_times_s*10).astype(int)
    held = ids[cues][np.searchsorted(cues, np.arange(640), side="right")-1]
    require(np.array_equal(held, ids), "Generated state changed outside fixed cues")
    hs = np.asarray(prototypes)[ids].copy()
    hs[~reference.valid] = 0
    return canonical_track(replace(reference, hs_histograms=hs, token_ids=ids[cues]), prototypes)


def active_tokens(track):
    cues = np.rint(track.cue_times_s*10).astype(int)
    return track.token_ids[np.searchsorted(cues, np.arange(640), side="right")-1][track.valid]


def diversity(track):
    values = active_tokens(track)
    if not len(values): return dict(occupancy=None, effective_diversity=None)
    _, counts = np.unique(values, return_counts=True)
    p = counts/counts.sum()
    return dict(occupancy=int(len(counts)), effective_diversity=float(np.exp(-np.sum(p*np.log(p)))))


def safe_rollouts(model, context, cues):
    cues = validate_cues(cues)
    require(not model.training and next(model.parameters()).device.type == "cpu", "CPU eval model required")
    require(tuple(context.shape[:2]) == (1, 640) and context.dtype == torch.float32
            and bool(torch.isfinite(context).all()), "Nonfinite or non-FP32 context")
    k = model.cfg.palette_size
    require(k == 32, "Frozen palette32 required")
    with torch.no_grad(), torch.autocast(device_type="cpu", enabled=False):
        c = context[:, cues][0]
        logits = model.palette_logits(c[:, None].expand(-1, k+1, -1),
            torch.arange(k+1).expand(len(cues), -1))
        require(bool(torch.isfinite(logits).all()), "Nonfinite palette logits")
        probabilities = torch.softmax(logits.float(), -1)
        require(bool(torch.isfinite(probabilities).all()) and bool((probabilities >= 0).all())
                and bool(torch.allclose(probabilities.sum(-1),torch.ones_like(probabilities.sum(-1)),atol=1e-6,rtol=0)),
                "Palette probabilities must be finite nonnegative normalized rows")
        result = ev.palette_rollouts(model, context, cues, draw_seeds=DRAWS)
    for draw in DRAWS:
        for condition in ("own_state", "null_state"):
            ids = result[draw][condition]["generated_palette_ids"]
            require(np.all((ids >= 0) & (ids < k)), "InverseCDF produced invalid ID")
    return result


def paired_conditions(model, recipient_context, donor_context, cues):
    own = safe_rollouts(model, recipient_context, cues)
    donor = safe_rollouts(model, donor_context, cues)
    result = {}
    for draw in DRAWS:
        require(np.array_equal(own[draw]["uniform_noise"], donor[draw]["uniform_noise"]), "Unpaired noise")
        initial = int(own[draw]["own_state"]["generated_palette_ids"][0])
        require(initial == int(own[draw]["null_state"]["generated_palette_ids"][0]), "Initial draws differ")
        result[draw] = {
            "own_state": own[draw]["own_state"]["generated_palette_ids"],
            "null_state": own[draw]["null_state"]["generated_palette_ids"],
            "persistence": np.full(640, initial, np.int64),
            "tempo_donor_own": donor[draw]["own_state"]["generated_palette_ids"],
            "uniform_noise": own[draw]["uniform_noise"],
        }
    return result


def emitted_score(labels, ids, cues, color, palette_valid):
    """Different-token deployed events, not raw-head candidate AP."""
    cues = validate_cues(cues)
    labels.validate()
    intervals = ev.eligible_intervals(labels, "color")
    starts = cues[cues > 0]
    keep = (ids[starts] != ids[starts-1]) & palette_valid[starts] & palette_valid[starts-1]
    starts = starts[keep]
    times = labels.requested_times10_s[starts]
    keep = ev._regions(times, intervals) >= 0
    times, confidence = times[keep], np.asarray(color)[starts][keep]
    refs = labels.source_times10_s[(labels.color > 0) & (ev._regions(labels.source_times10_s, intervals) >= 0)]
    begin = labels.requested_times10_s[0] + 28.
    score = metrics.visual_event_ap(refs-begin, times-begin, confidence,
        duration_s=8., deployment_threshold=.5, channel="color", eligible_intervals_s=intervals-begin)
    return {key: getattr(score, key) for key in ("average_precision", "reference_count", "candidate_count",
        "emitted_count", "matched_count", "false_cues_per_minute", "coverage", "eligible_duration_s")}


def unsupported_intervals(track):
    cues = np.rint(track.cue_times_s*10).astype(int)
    return sum(not np.any(track.valid[start:(cues[j+1] if j+1 < len(cues) else 640)])
               for j,start in enumerate(cues))


def compact_score(score, track):
    def summary(s):
        return {k: v for k, v in asdict(s).items() if k not in ("component_samples", "marginal_hs")}
    return dict(temporal_raw=score.temporal_raw, temporal_scaled=score.temporal_scaled,
        temporal_mean_scaled=score.temporal_mean_scaled, marginal_raw=score.marginal_raw,
        marginal_scaled=score.marginal_scaled, undefined_components=score.undefined_components,
        generated=summary(score.generated), reference=summary(score.reference),
        unsupported_interval_count=unsupported_intervals(track))


def score_draw(reference, conditions, prototypes, calibration, labels, cues, color):
    reference = canonical_track(reference, prototypes)
    refdiv = diversity(reference)
    result = {}
    for name in CONDITIONS:
        ids = np.asarray(conditions[name])
        track = generated_track(reference, ids, prototypes)
        score = compact_score(pm.compare_palette_tracks(track, reference, prototypes, calibration), reference)
        div = diversity(track)
        score["generated_diversity"] = div
        score["reference_diversity"] = refdiv
        score["occupancy_ratio"] = div["occupancy"]/refdiv["occupancy"] if refdiv["occupancy"] else None
        score["diversity_ratio"] = div["effective_diversity"]/refdiv["effective_diversity"] if refdiv["effective_diversity"] else None
        score["timing"] = emitted_score(labels, ids, cues, color, reference.valid)
        result[name] = score
    return result


def check_inventory(rows, dev_rows):
    expected = {(role, seed, r["recording_id"], draw) for role in ROLES for seed in SEEDS
                for r in dev_rows for draw in DRAWS}
    actual = [(r["role"], r["seed"], r["recording_id"], r["draw"]) for r in rows]
    require(len(actual) == len(set(actual)) and set(actual) == expected, "Incomplete/duplicated role seed record draw inventory")
    canonical = {r["recording_id"]: r["canonical_id"] for r in dev_rows}
    for r in rows:
        require(r["canonical_id"] == canonical[r["recording_id"]] and set(r["conditions"]) == set(CONDITIONS),
                "Changed canonical/condition inventory")
        for score in r["conditions"].values():
            require(set(score.get("temporal_raw",{})) == set(pm.COMPONENTS)
                    and set(score.get("temporal_scaled",{})) == set(pm.COMPONENTS), "Missing temporal component")
            raw, scaled = score["temporal_raw"],score["temporal_scaled"]
            require(all((raw[k] is None) == (scaled[k] is None) for k in pm.COMPONENTS), "Component undefined mismatch")
            require(all(v is None or (np.isfinite(v) and v >= 0) for v in (*raw.values(),*scaled.values())),
                    "Nonfinite/negative temporal component")
            undefined = {k for k in pm.COMPONENTS if scaled[k] is None}
            require(set(score.get("undefined_components",[])) == undefined, "Undefined component inventory mismatch")
            total = score["temporal_mean_scaled"]
            require((total is None) == bool(undefined), "Aggregate silently omitted undefined component")
            if not undefined:
                require(np.isfinite(total) and np.isclose(total,np.mean(list(scaled.values())),rtol=1e-10,atol=1e-12),
                        "Aggregate differs from all four equal-weight components")


def summarize(rows, dev_rows):
    check_inventory(rows, dev_rows)
    songs = sorted({r["canonical_id"] for r in dev_rows})
    kwargs = dict(expected_songs=songs, expected_seeds=SEEDS, expected_draws=DRAWS)
    reports = {}
    for role in ROLES:
        selected = [r for r in rows if r["role"] == role]
        # Qualification is common across ALL seeds/draws/conditions, never arm-dependent.
        support = {}
        timing_support = {}
        timing_events = {}
        for dev in dev_rows:
            rid = dev["recording_id"]
            cells = [r for r in selected if r["recording_id"] == rid]
            support[rid] = all(c["generated"]["observed_valid_time_fraction"] >= .8
                and c["temporal_mean_scaled"] is not None and c["marginal_scaled"] is not None
                for r in cells for c in r["conditions"].values())
            timing_support[rid] = support[rid] and all(c["timing"]["coverage"] >= .8
                for r in cells for c in r["conditions"].values())
            timing_events[rid] = timing_support[rid] and all(c["timing"]["average_precision"] is not None
                for r in cells for c in r["conditions"].values())
        def observations(condition, key, mask):
            return [metrics.MetricObservation(r["canonical_id"], r["recording_id"], "recording", r["seed"], r["draw"],
                (r["conditions"][condition]["timing"][key] if key in ("average_precision", "false_cues_per_minute")
                 else r["conditions"][condition][key]) if mask[r["recording_id"]] else None) for r in selected]
        def aggregate(condition, key, mask=support):
            return metrics.macro_song_mean(observations(condition, key, mask), **kwargs)
        condition_means = {name: {key: aggregate(name, key).macro_mean for key in
            ("temporal_mean_scaled", "marginal_scaled", "occupancy_ratio", "diversity_ratio")} for name in CONDITIONS}
        contrasts = {}
        for name, condition in (("null_minus_own_temporal", "null_state"), ("tempo_donor_minus_own_temporal", "tempo_donor_own")):
            a, b = observations(condition, "temporal_mean_scaled", support), observations("own_state", "temporal_mean_scaled", support)
            enough = len(aggregate(condition, "temporal_mean_scaled").eligible_songs) >= 2
            contrasts[name] = ({"descriptive95": serial(metrics.paired_song_bootstrap(a, b, **kwargs, confidence=.95)),
                "family97p5": serial(metrics.paired_song_bootstrap(a, b, **kwargs, confidence=.975))} if enough else None)
        holm = metrics.holm_adjust({k: v["descriptive95"]["p_value_two_sided"] for k, v in contrasts.items()}) if all(contrasts.values()) else None
        timing_means = {name: aggregate(name, "average_precision", timing_events).macro_mean for name in CONDITIONS}
        timing_ci = None
        if len(aggregate("own_state", "average_precision", timing_events).eligible_songs) >= 2:
            timing_ci = serial(metrics.paired_song_bootstrap(observations("own_state", "average_precision", timing_events),
                observations("null_state", "average_precision", timing_events), **kwargs))
        ngroups = len({d["canonical_id"] for d in dev_rows if support[d["recording_id"]]})
        ntiming = len({d["canonical_id"] for d in dev_rows if timing_support[d["recording_id"]]})
        nevents = len({d["canonical_id"] for d in dev_rows if timing_events[d["recording_id"]]})
        own, null = condition_means["own_state"], condition_means["null_state"]
        delta = null["temporal_mean_scaled"]-own["temporal_mean_scaled"] if null["temporal_mean_scaled"] is not None else None
        relative = delta/null["temporal_mean_scaled"] if null["temporal_mean_scaled"] else None
        prerequisites = ngroups >= 30 and ntiming >= 30 and nevents >= 10 and all(contrasts.values()) and relative is not None
        gates = dict(common_palette_support=ngroups >= 30, emitted_timing_support=ntiming >= 30 and nevents >= 10,
            absolute_gain=None if delta is None else delta >= .02, relative_gain=None if relative is None else relative >= .15,
            family_positive=None if not all(contrasts.values()) else all(v["family97p5"]["interval"][0] > 0 for v in contrasts.values()),
            holm2_significant=None if holm is None else all(p < .05 for p in holm.values()),
            occupancy=None if own["occupancy_ratio"] is None else own["occupancy_ratio"] >= .7,
            diversity=None if own["diversity_ratio"] is None else own["diversity_ratio"] >= .7,
            marginal=None if own["marginal_scaled"] is None else own["marginal_scaled"] <= 1.05*null["marginal_scaled"],
            emitted_timing=None if timing_means["own_state"] is None else timing_means["own_state"]-timing_means["null_state"] >= -.01)
        reports[role] = dict(condition_means=condition_means, contrasts=contrasts, holm2=holm,
            absolute_gain=delta, relative_reduction=relative, palette_qualified_groups=ngroups,
            emitted_timing_groups=ntiming, emitted_timing_event_groups=nevents, timing_means=timing_means,
            timing_difference95=timing_ci, gates=gates, secondary_pass=all(gates.values()) if prerequisites else None,
            support_by_record=support, timing_support_by_record=timing_support,
            false_cues_per_minute={name: aggregate(name,"false_cues_per_minute",timing_support).macro_mean for name in CONDITIONS})
    return dict(reports=reports, original_source_gate="FAILED31/40=.775<.80;unchanged",
        original_event_null_gate="FAILED29/30;unchanged;not_palette_prerequisite", original_palette_qualified=None,
        original_shuffled_palette_control="MISSING;tempo-donor is not its substitute", promotion=False)


def validate_native_vault_receipt(native, registration):
    start = native.get("start", {})
    require(start.get("id") == registration["vault_experiment_id"]
            and start.get("hypothesis_id") == "h-layers-20260922-palette"
            and start.get("external_run") == str(DERIVED_ROOT), "Native vault experiment identity mismatch")
    try:
        config = json.loads(start["config_note"])
    except (KeyError,TypeError,ValueError) as error:
        raise tr.ContractError("Native vault config_note missing/invalid") from error
    require(config.get("protocol") == registration["protocol"], "Native vault protocol binding mismatch")


def validate_cpu_receipt(cpu, registration):
    require(cpu.get("status") == "passed" and cpu.get("synthetic_only") is True
            and cpu.get("data_access") is False and cpu.get("cuda_used") is False
            and cpu.get("tests_failed") == 0 and cpu.get("tests_passed",0) > 0,
            "CPU receipt is not a passed synthetic-only run")
    require(cpu.get("driver_sha256") == registration["driver"]["sha256"]
            and cpu.get("protocol_sha256") == registration["protocol"]["sha256"]
            and cpu.get("tests_sha256") == registration["tests"]["sha256"]
            and cpu.get("clarification_sha256") == registration["clarification"]["sha256"],
            "CPU receipt source/protocol/tests mismatch")


def validate_saved_draw(z, draw, cues):
    noise = z[f"{draw}__uniform_noise"]
    require(noise.dtype == np.float64 and np.array_equal(noise,np.random.default_rng(draw).random(len(cues))),
            "Uniform noise dtype/content changed")
    conditions = {}
    for name in CONDITIONS:
        ids = z[f"{draw}__{name}"]
        require(ids.dtype == np.uint8 and ids.shape == (640,) and np.all(ids < 32),
                "Saved IDs require uint8[640] in palette32 before any conversion")
        conditions[name] = ids.astype(np.int64)
    initial = conditions["own_state"][0]
    require(initial == conditions["null_state"][0] and np.all(conditions["persistence"] == initial),
            "Persistence or paired initial draw changed")
    return conditions


def load_authorized(path, phase):
    registration = json.loads(Path(path).read_text())
    require(registration.get("protocol_id") == PROTOCOL_ID and registration.get("authorized") is True
            and registration.get("registered_before_palette_results") is True
            and phase in registration.get("authorized_phases", []) and bool(registration.get("vault_experiment_id")),
            "Parent preregistration/phase authorization missing")
    vault_path = ev.checked_ref(registration.get("vault_receipt"))
    vault = json.loads(vault_path.read_text())
    require(vault.get("experiment_id") == registration["vault_experiment_id"] and vault.get("status") == "registered"
            and vault.get("protocol_sha256") == registration["protocol"]["sha256"], "Vault authorization receipt does not bind this experiment/protocol")
    native_vault = json.loads(ev.checked_ref(vault["native_vault_receipt"]).read_text())
    validate_native_vault_receipt(native_vault,registration)
    require(ev.checked_ref(registration["protocol"]).resolve() == PROTOCOL_PATH, "Wrong protocol location")
    clarification = json.loads(ev.checked_ref(registration["clarification"]).read_text())
    require(clarification.get("protocol") == registration["protocol"] and clarification.get("no_gate_change") is True,
            "Clarification does not bind the frozen protocol")
    require(vault.get("clarification_sha256") == registration["clarification"]["sha256"],
            "Vault receipt does not bind clarification")
    require(ev.checked_ref(registration["driver"]).resolve() == Path(__file__).resolve(), "Driver source mismatch")
    tests_path = ev.checked_ref(registration["tests"])
    cpu = json.loads(ev.checked_ref(registration["cpu_test_receipt"]).read_text())
    validate_cpu_receipt(cpu,registration)
    ev.checked_ref(cpu["test_log"])
    ev.checked_ref(cpu["junit"])
    for source, sha in registration["frozen_sources"].items():
        require(tr.sha256(source) == sha, "Frozen source changed: " + source)
    protocol = json.loads(PROTOCOL_PATH.read_text())
    require(protocol["protocol_id"] == PROTOCOL_ID and protocol["resources"]["derived_root"] == str(DERIVED_ROOT), "Protocol/root mismatch")
    eventreg_path = ev.checked_ref(registration["event_registration"])
    manifest_path = ev.checked_ref(registration["manifest"])
    transforms_path = ev.checked_ref(registration["transforms"])
    # Reuse frozen observer provenance validation; it rejects TEST before source opens.
    args = argparse.Namespace(registration=str(eventreg_path), manifest=str(manifest_path), transforms=str(transforms_path))
    eventreg, manifest, transforms, store = ev._inputs(args, "predict")
    baselines_path = ev.checked_ref(registration["baselines"])
    require(registration["baselines"] == eventreg["baselines"], "Baseline binding mismatch")
    baselines = json.loads(baselines_path.read_text())
    train = [r for r in manifest["records"] if r["partition"] == "train"]
    dev = [r for r in manifest["records"] if r["partition"] in ("dev", "val")]
    require(len(train) == 91 and len(dev) == 31 and len({r["canonical_id"] for r in dev}) == 31,
            "Frozen91/31 inventory changed")
    receipts = []
    for ref in registration["event_predictions"]:
        receipt = json.loads(ev.checked_ref(ref).read_text())
        checkpoint = ev.validate_prediction_receipt(receipt, eventreg, registration["baselines"])
        require(checkpoint["recipe"] == "S_VR_P0", "Wrong recipe")
        require({r["recording_id"] for r in receipt["records"]} == {r["recording_id"] for r in dev}, "Prediction records changed")
        receipts.append(receipt)
    require(len(receipts) == 6 and {(r["checkpoint_role"],r["seed"]) for r in receipts} == {(role,seed) for role in ROLES for seed in SEEDS},
            "Exact six prediction/checkpoint cells required")
    return registration, eventreg, manifest, transforms, store, baselines, receipts, train, dev


def validate_secondary_entries(prediction, receipts, eventreg):
    expected = {(r["checkpoint_role"], r["seed"], row["recording_id"]): (r, row)
                for r in receipts for row in r["records"]}
    entries = prediction.get("records", [])
    keys = [(e["role"], e["seed"], e["recording_id"]) for e in entries]
    require(len(keys) == len(set(keys)) and set(keys) == set(expected), "Secondary prediction cell inventory changed")
    metadata = {}
    for e in entries:
        key = (e["role"],e["seed"],e["recording_id"])
        receipt, old = expected[key]
        require(e["canonical_id"] == old["canonical_id"] and e["original_prediction"] == old["artifact"]
                and e["target_sha256"] == old["target_sha256"]
                and e["donor_id"] == eventreg["donor_map"][e["recording_id"]],
                "Secondary cell role/seed/source/target/donor provenance mismatch")
        metadata[key] = dict(role=e["role"],seed=e["seed"],recording_id=e["recording_id"],
            original_prediction_sha256=old["artifact"]["sha256"], checkpoint_sha256=receipt["checkpoint"]["sha256"],
            donor_id=e["donor_id"],cues_sha256=e["cues_sha256"])
    return metadata


def validate_calibration_inventory(calibration, train):
    expected = sorted((r["canonical_id"],r["recording_id"]) for r in train)
    require(sorted(tuple(k) for k in calibration.training_record_ids) == expected
            and sorted(calibration.training_song_ids) == sorted({r["canonical_id"] for r in train}),
            "Calibration exact canonical/record TRAIN inventory changed")


def check_receipt(path, registration_path, status="complete"):
    value = json.loads(Path(path).read_text())
    require(value["registration_sha256"] == tr.sha256(registration_path) and value["status"] == status,
            "Secondary artifact registration/status mismatch")
    return value


def deadline(start):
    require(time.monotonic()-start <= 3600, "Phase exceeded3600s CPU wall budget")


def execute(args):
    started = time.monotonic()
    reg, eventreg, manifest, transforms, store, baselines, receipts, train, dev = load_authorized(args.registration,args.phase)
    root = DERIVED_ROOT
    output = Path(args.output)
    require(output.resolve().is_relative_to(root.resolve()) and not output.exists(), "Exclusive bounded output required")
    space(root, RESERVE)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    common = dict(protocol_id=PROTOCOL_ID, registration_sha256=tr.sha256(args.registration),
                  status="complete", scope=json.loads(PROTOCOL_PATH.read_text())["study_scope"])
    prototypes = np.asarray(transforms["palette_prototypes"], np.float64)
    if args.phase == "calibrate":
        tracks = []
        cue_hashes = {}
        for row in sorted(train,key=lambda r:r["recording_id"]):
            deadline(started)
            record = store.visual(row["recording_id"])
            x = ev.audio_window(store,record,transforms)
            probability = ev.baseline_curves(x,baselines,"color")["logistic"]
            cues = np.unique(np.r_[0,tr._cue_indices(probability[None],threshold=.5)[0]]).astype(np.int64)
            cue_hashes[row["recording_id"]] = tr.json_hash(cues.tolist())
            tracks.append(reference_track(record,cues,prototypes))
        try:
            calibration = pm.fit_palette_scales(tracks,prototypes,partition="TRAIN")
        except ValueError as error:
            write_json(output,{**common,"status":"blocked","reason":str(error),"cue_hashes":cue_hashes},root)
            return
        deadline(started)
        write_json(output,{**common,"calibration":serial(calibration),"cue_hashes":cue_hashes,
                           "elapsed_s":time.monotonic()-started},root)
        return
    calibration_receipt = check_receipt(args.calibration,args.registration)
    calibration = pm.PaletteCalibration(**calibration_receipt["calibration"])
    pm._validate_calibration(calibration,pm.palette_prototypes_hash(prototypes))
    validate_calibration_inventory(calibration,train)
    if args.phase == "predict":
        output.mkdir(parents=True,exist_ok=False)
        entries = []
        for receipt in sorted(receipts,key=lambda r:(ROLES.index(r["checkpoint_role"]),r["seed"])):
            checkpoint = receipt["checkpoint"]
            ev.validate_training_receipt(checkpoint,eventreg)
            model,_ = ev.load_bound_model(checkpoint,transforms,transforms_sha256=reg["transforms"]["sha256"],
                                          expected_bindings=checkpoint["bindings"],device="cpu")
            for row in receipt["records"]:
                deadline(started)
                rid = row["recording_id"]
                oldpath = ev.checked_ref(row["artifact"])
                with np.load(oldpath,allow_pickle=False) as z:
                    cues = cue_indices(z["model__generated_color_cues"])
                    saved_logits = z["model__color_logit"].copy()
                    saved_donor_logits = z["donor__color_logit"].copy()
                    old_times = z["model__requested_times10_s"].copy()
                record = store.visual(rid)
                require(np.array_equal(record.times,old_times),"Source/requested clock mismatch")
                label = ev.freeze_labels(record,transforms)
                require(label.content_hash() == row["target_sha256"] == eventreg["target_hashes"][rid], "Frozen target changed")
                donor_id = eventreg["donor_map"][rid]
                ev.verify_donor_tempos(store,[rid,donor_id],eventreg["donor_tempos_bpm"])
                x = ev.audio_window(store,record,transforms)
                dx = ev.audio_window(store,store.visual(donor_id),transforms)
                dx = ev.nonrhythm_donor(x,dx)
                with torch.no_grad(),torch.autocast(device_type="cpu",enabled=False):
                    o = model(torch.from_numpy(x[None]))
                    d = model(torch.from_numpy(dx[None]))
                logits = o["chromatic_change_logit"][0].float().numpy()
                require(np.isfinite(logits).all() and np.allclose(logits,saved_logits,rtol=1e-5,atol=1e-5),"FP32 color replay differs from saved predictions")
                donor_logits = d["chromatic_change_logit"][0].float().numpy()
                require(np.isfinite(donor_logits).all() and np.allclose(donor_logits,saved_donor_logits,rtol=1e-5,atol=1e-5),
                        "FP32 donor color replay differs from saved predictions")
                recomputed = np.unique(np.r_[0,tr._cue_indices(torch.sigmoid(o["chromatic_change_logit"].float()).numpy(),.5)[0]])
                require(np.array_equal(recomputed,cues),"Regenerated cue inventory differs")
                draws = paired_conditions(model,o["context10"],d["context10"],cues)
                meta = dict(role=receipt["checkpoint_role"],seed=receipt["seed"],recording_id=rid,
                    original_prediction_sha256=row["artifact"]["sha256"],checkpoint_sha256=checkpoint["sha256"],
                    donor_id=donor_id,cues_sha256=tr.json_hash(cues.tolist()))
                values = {"cues":cues.astype(np.uint16), "metadata_json":np.array(json.dumps(meta,sort_keys=True))}
                for draw in DRAWS:
                    for name in CONDITIONS: values[f"{draw}__{name}"] = draws[draw][name].astype(np.uint8)
                    values[f"{draw}__uniform_noise"] = draws[draw]["uniform_noise"]
                filename = f"{receipt['checkpoint_role']}_s{receipt['seed']}_{hashlib.sha256(rid.encode()).hexdigest()[:20]}.npz"
                path = output/filename
                write_npz(path,values,root)
                entries.append(dict(role=receipt["checkpoint_role"],seed=receipt["seed"],recording_id=rid,
                    canonical_id=row["canonical_id"],artifact=ev.reference(path),original_prediction=row["artifact"],
                    cues_sha256=tr.json_hash(cues.tolist()),target_sha256=row["target_sha256"],donor_id=donor_id))
            del model
        deadline(started)
        write_json(output/"complete.json",{**common,"calibration":ev.reference(args.calibration),"records":entries,
                   "elapsed_s":time.monotonic()-started},root)
        return
    prediction = check_receipt(args.predictions,args.registration)
    require(prediction["calibration"] == ev.reference(args.calibration),"Prediction/calibration mismatch")
    expected_metadata = validate_secondary_entries(prediction,receipts,eventreg)
    rows, floors = [], []
    for entry in prediction["records"]:
        deadline(started)
        record = store.visual(entry["recording_id"])
        label = ev.freeze_labels(record,transforms)
        require(label.content_hash() == entry["target_sha256"],"Scoring target changed")
        with np.load(ev.checked_ref(entry["original_prediction"]),allow_pickle=False) as z:
            cues = cue_indices(z["model__generated_color_cues"])
            color = z["model__color"].copy()
        require(tr.json_hash(cues.tolist()) == entry["cues_sha256"],"Cues changed")
        reference = reference_track(record,cues,prototypes)
        floor_self = compact_score(pm.compare_palette_tracks(reference,reference,prototypes,calibration),reference)
        cueids = reference.token_ids[np.searchsorted(cues,np.arange(640),side="right")-1]
        quantized = generated_track(reference,cueids,prototypes)
        floor_quant = compact_score(pm.compare_palette_tracks(quantized,reference,prototypes,calibration),reference)
        floors.append(dict(role=entry["role"],seed=entry["seed"],recording_id=entry["recording_id"],self_floor=floor_self,quantization_floor=floor_quant))
        with np.load(ev.checked_ref(entry["artifact"]),allow_pickle=False) as z:
            key = (entry["role"],entry["seed"],entry["recording_id"])
            require(json.loads(str(z["metadata_json"])) == expected_metadata[key], "Secondary artifact cell relabeling")
            require(z["cues"].dtype == np.uint16 and np.array_equal(z["cues"],cues),"Outputcue dtype/inventory changed")
            for draw in DRAWS:
                conditions = validate_saved_draw(z,draw,cues)
                scored = score_draw(reference,conditions,prototypes,calibration,label,cues,color)
                rows.append(dict(role=entry["role"],seed=entry["seed"],draw=draw,recording_id=entry["recording_id"],
                    canonical_id=entry["canonical_id"],conditions=scored))
    summary = summarize(rows,dev)
    deadline(started)
    write_json(output,{**common,**summary,"rows":rows,"floors":floors,"calibration":ev.reference(args.calibration),
        "predictions":ev.reference(args.predictions),"elapsed_s":time.monotonic()-started},root)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="phase",required=True)
    sub.add_parser("schema")
    for phase in ("calibrate","predict","score"):
        p = sub.add_parser(phase)
        p.add_argument("--registration",required=True)
        p.add_argument("--output",required=True)
        if phase != "calibrate": p.add_argument("--calibration",required=True)
        if phase == "score": p.add_argument("--predictions",required=True)
        if phase == "predict": p.add_argument("--device",choices=("cpu",),default="cpu")
    args = parser.parse_args(argv)
    if args.phase == "schema":
        print(json.dumps(dict(protocol=ev.reference(PROTOCOL_PATH),driver=ev.reference(__file__),
                              definition=json.loads(PROTOCOL_PATH.read_text())),indent=2))
        return
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):
        execute(args)


if __name__ == "__main__": main()
