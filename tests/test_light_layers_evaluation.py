"""Synthetic CPU-only observer orchestration contract tests. No datasets/GPU."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lightsem.model.light_layers import evaluation as ev
from lightsem.model.light_layers.training import ContractError, RHYTHM_COLUMNS, json_hash
from lightsem.model.light_layers.model import LightLayersConfig, SharedLightLayers


@pytest.fixture(autouse=True)
def cpu_single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def transforms(ids=("r0", "r1")):
    audio = {"mean": [0.]*94, "std": [1.]*94}
    prototypes = np.zeros((2, 72)); prototypes[0, 0] = 1; prototypes[1, 1] = 1
    result = {"fit_partition": "train", "fit_recording_ids": list(ids), "audio": audio, "aux": {},
        "palette_prototypes": prototypes.tolist(), "palette_size": 2,
        "color_threshold": .5, "accent_threshold": .5, "threshold_quantile": .95,
        "novelty_window_s": .3, "event_nms_s": .5}
    result["norm_sha256"] = json_hash({"audio": audio, "aux": {}})
    result["vocab_sha256"] = json_hash(result["palette_prototypes"])
    return result


def labels(rid="r0", partition="train", offset=17., pts_delta=.013):
    requested = offset+np.arange(640)/10
    source = requested+pts_delta
    y = np.zeros(640); y[[290, 310, 330, 350]] = 1
    valid = np.ones(640, bool); valid[:4] = False; valid[-4:] = False
    return ev.FrozenLabels(rid, "song-"+rid, partition, requested, source,
                           y.copy(), y.copy(), valid.copy(), valid.copy()).validate()


def features(seed=0):
    return np.random.default_rng(seed).normal(size=(2560, 94)).astype(np.float32)


class SpyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.))
        self.cfg = SimpleNamespace(in_dim=94, context_radius10=20, palette_size=3)
        self.seen = []
    def forward(self, x):
        self.seen.append(x.detach().clone())
        color = torch.full((1, 640), -20., device=x.device)
        color[0, [70, 150, 300]] = 20.
        return {"accent_logit": x[:, :, 0], "chromatic_change_logit": color,
                "context10": torch.zeros(1, 640, 2, device=x.device)}
    def palette_logits(self, context, ids):
        chosen = torch.where(ids == 3, 0, (ids+1)%3)
        return torch.nn.functional.one_hot(chosen, 3).float()*20


def test_exact_one64s_forward_and_generated_history_from_null():
    m = SpyModel(); x = features(); l = labels()
    out = ev.predict_window(m, x, requested_times10_s=l.requested_times10_s, source_times10_s=l.source_times10_s)
    assert len(m.seen) == 1 and tuple(m.seen[0].shape) == (1, 2560, 94)
    np.testing.assert_array_equal(m.seen[0][0].numpy(), x)
    assert out["previous_ids"][0] == 3
    assert out["generated_palette_ids"][0] == 0
    assert out["generated_palette_ids"][70] == 1
    assert out["generated_palette_ids"][150] == 2
    assert out["generated_palette_ids"][300] == 0
    assert out["accent"].shape == (2560,) and out["color"].shape == (640,)
    np.testing.assert_array_equal(out["source_times10_s"], l.source_times10_s)
    assert out["requested_times40_s"][0] == 17
    assert m.training  # inference restores caller mode


@pytest.mark.parametrize("shape", [(2556, 94), (2564, 94), (12000, 94), (2560, 93)])
def test_context_clipping_padding_expansion_and_extra_input_forbidden(shape):
    l = labels(); m = SpyModel()
    with pytest.raises(ContractError, match="exactly one"):
        ev.predict_window(m, np.zeros(shape), requested_times10_s=l.requested_times10_s,
                          source_times10_s=l.source_times10_s)
    assert not m.seen


def test_no_teacher_history_api():
    l = labels()
    with pytest.raises(TypeError):
        ev.predict_window(SpyModel(), features(), requested_times10_s=l.requested_times10_s,
                          source_times10_s=l.source_times10_s, gold_palette=np.zeros(640))


def test_true_source_boundary_fails_no_padding():
    store = SimpleNamespace(audio=lambda rid: np.zeros((2560, 94)))
    l = labels(offset=1.)
    rec = SimpleNamespace(times=l.requested_times10_s, source_times=l.source_times10_s, recording_id="r0")
    with pytest.raises(ContractError, match="no boundary clipping"):
        ev.audio_window(store, rec, transforms())


def test_source_window_slices_absolute_clock_and_only_then_normalizes():
    whole = np.broadcast_to(np.arange(4000)[:, None], (4000, 94)).astype(np.float32)
    store = SimpleNamespace(audio=lambda rid: whole)
    l = labels(offset=17.)
    rec = SimpleNamespace(times=l.requested_times10_s, source_times=l.source_times10_s, recording_id="r0")
    t = transforms(); t["audio"]["mean"] = [10.]*94; t["audio"]["std"] = [2.]*94
    got = ev.audio_window(store, rec, t)
    assert got.shape == (2560, 94)
    assert got[0, 0] == (680-10)/2 and got[-1, 0] == (3239-10)/2


def test_source_and_model_hash_checks_fail_closed(tmp_path):
    path = tmp_path/"data"; path.write_bytes(b"a")
    ref = ev.reference(path)
    assert ev.checked_ref(ref) == path
    path.write_bytes(b"b")
    with pytest.raises(ContractError, match="changed"):
        ev.checked_ref(ref)
    with pytest.raises(ContractError, match="provenance"):
        ev.load_bound_model(ref, transforms(), transforms_sha256="x", expected_bindings={})


def test_safe_checkpoint_reconstruction_and_bindings(tmp_path):
    cfg = LightLayersConfig(d_model=8, n_blocks=1, n_heads=2, ff_mult=1, conv_kernel=3,
                           attn_window=4, dropout=0., palette_size=2)
    model = SharedLightLayers(cfg); t = transforms()
    bindings = {key: "synthetic" for key in ({"manifest_sha256", "source_sha256", "split_sha256", "config_sha256"}
                                          | set(ev.OBSERVER_SCOPE_BINDINGS))}
    metadata = {**bindings, "norm_sha256": t["norm_sha256"], "vocab_sha256": t["vocab_sha256"],
        "transforms_sha256": "tf", "reference_model_config": asdict(cfg), "model_config": asdict(cfg),
        "architecture": "shared"}
    path = tmp_path/"m.pt"; torch.save({"metadata": metadata, "model_state_dict": model.state_dict()}, path)
    loaded, got = ev.load_bound_model(ev.reference(path), t, transforms_sha256="tf", expected_bindings=bindings)
    assert not loaded.training and got == metadata
    with pytest.raises(ContractError, match="binding mismatch"):
        ev.load_bound_model(ev.reference(path), t, transforms_sha256="changed", expected_bindings=bindings)


def test_transform_hash_and_target_shape_fail_closed():
    t = transforms(); ev.validate_transforms(t)
    t["audio"]["mean"][0] = 1
    with pytest.raises(ContractError, match="hash"):
        ev.validate_transforms(t)
    l = labels(); l.accent10 = np.repeat(l.accent10, 4)
    with pytest.raises(ContractError, match="shape"):
        l.validate()
    l = labels(); l.color_mask = np.ones((640, 1), bool)
    with pytest.raises(ContractError, match="shape"):
        l.validate()


def test_target_hash_binds_both_clocks_and_values():
    a, b = labels(), labels()
    assert a.content_hash() == b.content_hash()
    b.source_times10_s += .001
    assert a.content_hash() != b.content_hash()
    b = labels(); b.color[300] = 1
    assert a.content_hash() != b.content_hash()


def test_constant_and_endpoints_never_invent_predictions():
    times = np.arange(20)/10
    for curve in (np.ones(20), np.zeros(20), np.arange(20), -np.arange(20)):
        t, s = ev.candidate_peaks(times, curve, [[0, 2]])
        assert len(t) == len(s) == 0


def test_plateau_earliest_tie_and_score_nms_without_quota():
    times = np.arange(40)/40; scores = np.zeros(40)
    scores[2:4] = 1; scores[7] = 1; scores[15] = 2; scores[25] = 1; scores[35] = 1
    t, s = ev.candidate_peaks(times, scores, [[0, 1]])
    np.testing.assert_array_equal(t, times[[2, 15, 25, 35]])
    np.testing.assert_array_equal(s, [1, 2, 1, 1])


def test_mask_precedes_nms_and_invalid_gaps_never_supply_neighbors():
    times = np.arange(40)/40; scores = np.zeros(40)
    scores[10] = 100.; scores[14] = 1.
    t, _ = ev.candidate_peaks(times, scores, [[0, .2], [.3, 1]])
    assert times[14] in t and times[10] not in t
    scores[:] = 0.; scores[12] = 2  # first supported sample is not an interior max
    t, _ = ev.candidate_peaks(times, scores, [[.3, 1]])
    assert not len(t)


def test_full64_context_max_at_center_boundary_survives_clipping():
    l = labels(pts_delta=0.); l.color[:] = 0.; l.color[280] = 1
    curve = np.zeros(640); curve[[279, 280, 360]] = [1, 2, 10]
    out = ev.score_record(l, {"model": curve}, {"model": .5}, "color")
    score = out["scores"]["model"]
    assert score["candidate_count"] == score["reference_count"] == 1
    assert score["average_precision"] == 1
    assert score["duration_s"] == score["eligible_duration_s"] == 8


def test_actual_clock_references_one_accent_event_not_four():
    l = labels(); l.accent10[:] = 0.; l.accent10[300] = 1
    curve = np.zeros(2560); curve[1200] = 1
    out = ev.score_record(l, {"model": curve}, {"model": .5}, "accent")
    assert out["scores"]["model"]["reference_count"] == 1
    assert out["source_reference_times_s"][0] == pytest.approx(47.013)
    assert out["scores"]["model"]["average_precision"] == 1


def test_nonevent_ap_undefined_fp_rate_retains_full_and_eligible_duration():
    l = labels(pts_delta=0.); l.color[:] = 0; l.color_mask[300:320] = False
    curve = np.zeros(640); curve[290] = 1; curve[330] = 1
    out = ev.score_record(l, {"model": curve}, {"model": .5}, "color")
    score = out["scores"]["model"]
    assert score["average_precision"] is None
    assert score["duration_s"] == 8 and score["eligible_duration_s"] == 6
    assert score["false_cues_per_minute"] == 20


def test_shift_schema_is_fixed_before_any_scores():
    times = np.arange(320)/40; intervals = np.array([[0., 2.], [3., 8.]])
    plan = ev.shift_plan(times, intervals, 40)
    assert plan == ev.shift_plan(times, intervals, 40)
    assert plan["defined"] and plan["qualified"]
    for row in plan["segments"]:
        assert len(row["offsets_frames"]) == len(set(row["offsets_frames"])) == 20
        assert min(min(x, row["duration_s"]-x) for x in row["offsets_s"]) > .4


def test_shift_gaps_counts_and_confidences_preserved_without_postnms():
    times = np.arange(80)/10; intervals = [[0., 3.], [4., 8.]]
    plan = ev.shift_plan(times, intervals, 10)
    candidate = np.array([.05, 2.95, 4.1, 7.9]); confidence = np.array([.8, .4, .2, .9])
    shifted = ev.shifted_candidates(candidate, confidence, plan)
    assert len(shifted["draws"]) == 20
    for row in shifted["draws"]:
        assert len(row["times_s"]) == 4
        np.testing.assert_array_equal(row["confidences"], confidence)
        assert not np.any((row["times_s"] >= 3) & (row["times_s"] < 4))
        assert row["close_pair_count_lt_0p2s"] >= 1


def test_short_shift_segments_only_removed_from_paired_null_not_primary():
    l = labels(pts_delta=0.)
    l.color_mask[290:292] = False; l.color[290:292] = 0
    curve = np.zeros(640); curve[[285, 310, 330, 350]] = 1
    out = ev.score_record(l, {"model": curve}, {"model": .5}, "color")
    plan = out["shift_plan"]
    assert plan["excluded_intervals_s"] == [[45., 46.]]
    assert out["scores"]["model"]["eligible_duration_s"] == pytest.approx(7.8)
    null = out["paired_shift_null"]
    assert null["real"]["eligible_duration_s"] == pytest.approx(6.8)
    assert all(row["score"]["eligible_duration_s"] == null["real"]["eligible_duration_s"] for row in null["draws"])
    assert all(row["score"]["candidate_count"] == null["real"]["candidate_count"] for row in null["draws"])


def test_all_short_shifts_undefined_not_fake_zero():
    plan = ev.shift_plan(np.arange(80)/10, [[0, 1], [2, 3]], 10)
    assert not plan["defined"] and not plan["qualified"]
    shifted = ev.shifted_candidates([.5, 2.5], [1, 2], plan)
    assert shifted["draws"] is None


def test_donor_distinct_canonical_partition_and_all_named_rhythm_preserved():
    rows = [{"recording_id": str(i), "canonical_id": str(i//2), "partition": part}
            for part in ("train", "dev") for i in (range(4) if part == "train" else range(4, 8))]
    tempos = {r["recording_id"]: 100.+int(r["recording_id"])*5 for r in rows}
    donors = ev.donor_map(rows, tempos); lookup = {r["recording_id"]: r for r in rows}
    for rid, donor in donors.items():
        assert lookup[rid]["canonical_id"] != lookup[donor]["canonical_id"]
        assert lookup[rid]["partition"] == lookup[donor]["partition"]
    x, y = features(1), features(2); output = ev.nonrhythm_donor(x, y)
    np.testing.assert_array_equal(output[:, RHYTHM_COLUMNS], x[:, RHYTHM_COLUMNS])
    other = sorted(set(range(94))-set(RHYTHM_COLUMNS))
    np.testing.assert_array_equal(output[:, other], y[:, other])
    assert 64+ev.AUDIO_SCALARS.index("pulse_clarity") in RHYTHM_COLUMNS
    with pytest.raises(ContractError, match="different-canonical"):
        ev.donor_map(rows[:2], {r["recording_id"]: tempos[r["recording_id"]] for r in rows[:2]})
    assert donors["0"] == "2"  # exact training tempo-nearest donor, not pure hash
    audit = ev.donor_audit(rows, donors)
    assert audit["train"]["recipients"] == 4
    assert not audit["train"]["bijective_permutation_claim"]


def test_audio_novelty_uses_only_standardized64mel_and_fixed_prepost():
    x = np.zeros((640, 94)); x[300:, :64] = 1
    novelty, support = ev.audio_novelty(x, 10)
    assert novelty[300] == pytest.approx(8.)
    x[:, 64:] = np.random.default_rng(0).normal(size=(640, 30))*100
    got, _ = ev.audio_novelty(x, 10)
    np.testing.assert_array_equal(got, novelty)
    assert not support[:3].any() and not support[-2:].any()


def test_baselines_train_only_and_center_only_and_same94_rates():
    t = transforms(); a, b = labels(), labels("r1")
    x, y = features(1), features(2)
    examples = [ev.BaselineExample(a, x), ev.BaselineExample(b, y)]
    fitted = ev.prepare_baselines(examples, t)
    for head, n in (("accent", 640), ("color", 160)):
        assert fitted["heads"][head]["eligible_training_bins"] == n
        curves = ev.baseline_curves(x, fitted, head)
        assert set(curves) == set(ev.BASELINE_NAMES)
        assert all(len(c) == (2560 if head == "accent" else 640) for c in curves.values())
        assert np.ptp(curves["prevalence"]) == 0
    # Change noncentral observed labels only. Coefficients cannot use them.
    a.accent10[100] = a.color[100] = 1
    b.accent10[500] = b.color[500] = 1
    changed = ev.prepare_baselines(examples, t)
    assert changed["heads"] == fitted["heads"]
    b.partition = "dev"
    with pytest.raises(ContractError, match="TRAIN only"):
        ev.prepare_baselines(examples, t)


def test_missing_logistic_class_blocks_not_constant_fallback():
    a = labels(); a.accent10[:] = 0; a.color[:] = 0
    with pytest.raises(ContractError, match="no constant fallback"):
        ev.prepare_baselines([ev.BaselineExample(a, features())], transforms(("r0",)))


def test_palette_common_cues_noise_and_no_teacher_history():
    model = SpyModel(); context = torch.zeros(1, 640, 2)
    got = ev.palette_rollouts(model, context, [0, 100, 300])
    assert set(got) == {101, 102, 103, 104}
    for seed, row in got.items():
        np.testing.assert_array_equal(row["uniform_noise"], np.random.default_rng(seed).random(3))
        assert row["own_state"]["generated_palette_ids"][100] == 1
        assert row["null_state"]["generated_palette_ids"][100] == 0
        assert row["own_state"]["cue_previous_ids"][0] == 3
        assert np.all(row["null_state"]["cue_previous_ids"] == 3)
    with pytest.raises(ContractError, match="seeds"):
        ev.palette_rollouts(model, context, [0, 100], draw_seeds=(1,))


def campaign_rows(positive=12, low_coverage=0, null_qualified=True):
    out = []
    for group in range(31):
        l = labels(str(group), partition="dev", pts_delta=0.)
        if group >= positive:
            l.accent10[:] = l.color[:] = 0
        if group < low_coverage:
            l.color_mask[290:330] = False; l.color[290:330] = 0
        for head in ev.RATES:
            n = ev.RATES[head]*64
            curves = {name: np.zeros(n) for name in ("model", "donor", *ev.BASELINE_NAMES)}
            base = ev.score_record(l, curves, {name: .5 for name in curves}, head)
            if not null_qualified:
                base["shift_plan"]["qualified"] = False
            for recipe in ev.RECIPES:
                for seed in ev.SEEDS:
                    out.append({**base, "recipe": recipe, "seed": seed})
    return out


def summarize(rows):
    return ev.summarize_campaign(rows, enrolled_songs=["song-"+str(i) for i in range(31)],
        exclusions={"excluded-"+str(i): "frozen_source_presence_failed" for i in range(9)})


def test_cohort_failure_blocked_not_zero_and_original_gate_always_visible():
    result = summarize(campaign_rows(positive=9))
    assert result["status"] == "blocked_inference" and result["primary_gain_pass"] is None
    assert result["contrasts"] == {}
    assert result["original80percentgate"] == "FAILED_NOT_WAIVED"
    assert result["original_coverage"] == .775
    assert result["support_gates"]["color"]["event_bearing_groups"] == 9
    result = summarize(campaign_rows(low_coverage=2))
    assert result["support_gates"]["accent"]["common_qualified_groups"] == 29
    assert result["status"] == "blocked_inference"


def test_complete15_cell_cohort_bootstrap_holm8_and_null_gate():
    result = summarize(campaign_rows(null_qualified=False))
    assert len(result["contrasts"]) == 8
    assert all(row["replicates"] == 10000 and not row["CI_multiplicity_adjusted"]
               and "holm8_p_value" in row for row in result["contrasts"].values())
    assert result["status"] == "scored_interpretation_blocked"
    assert result["primary_gain_pass"] is None
    assert not result["generalization_or_LIGHT_quality_claim"]


def test_missing_seed_not_treated_as_zero_or_extra_frame_sample():
    rows = campaign_rows()
    with pytest.raises(ContractError, match="Missing model"):
        summarize(rows[:-1])
    with pytest.raises(ContractError, match="Duplicate"):
        summarize(rows+rows[:1])


def test_no_overwrite_and_disk_floor(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(ev, "campaign_space", lambda root, reserve: calls.append((root, reserve)))
    path = tmp_path/"x.json"
    ev.write_json_exclusive(path, {"x": np.array([1])}, tmp_path)
    assert json.loads(path.read_text()) == {"x": [1]}
    with pytest.raises(FileExistsError):
        ev.write_json_exclusive(path, {}, tmp_path)
    assert calls and all(reserve > 0 for _, reserve in calls)
    monkeypatch.setattr(ev, "campaign_space", lambda *a: (_ for _ in ()).throw(ContractError("Storage gate")))
    with pytest.raises(ContractError, match="Storage gate"):
        ev.write_npz_exclusive(tmp_path/"new.npz", {"a": np.zeros(4)}, tmp_path)
    assert not (tmp_path/"new.npz").exists()


def test_schema_and_registration_fail_closed_before_any_source_access(tmp_path):
    schema = ev.evaluation_definition()
    assert schema["cohort"]["qualified_center_coverage_min"] == .8
    assert schema["candidates"]["quota"] is None
    assert schema["shift_schema"]["post_shift_NMS"] is False
    manifest = tmp_path/"manifest.json"; transform = tmp_path/"transform.json"; registration = tmp_path/"reg.json"
    manifest.write_text("{}"); transform.write_text("{}"); registration.write_text("{}")
    with pytest.raises(ContractError, match="preregistered"):
        ev._registration(registration, manifest, transform, phase="score")


def test_novelty_support_erodes_observed_gaps_and_actual_timestamps():
    times = 12.+np.arange(640)/10
    valid = np.ones(640, bool); valid[300] = False
    support = ev.novelty_window_support(times, valid)
    assert not support[298:304].any()
    assert support[297] and support[304]
    times[320:] += .3
    support = ev.novelty_window_support(times, valid)
    assert not support[319] and not support[320]


def test_frozen_target_cannot_claim_uneroded_invalid_support(monkeypatch):
    l = labels(); valid = np.ones(640, bool); valid[300] = False
    record = SimpleNamespace(times=l.requested_times10_s, source_times=l.source_times10_s,
        value_valid=valid, color_valid=valid, frame_valid=valid)
    target = {"accent_mask10": np.ones(640, bool), "color_mask": np.ones(640, bool)}
    monkeypatch.setattr(ev, "make_targets", lambda *args: target)
    with pytest.raises(ContractError, match="invalidity"):
        ev.freeze_labels(record, transforms())


def test_donor_tempo_evidence_is_exact_fullsource_and_requires_all_ids():
    rows = [{"recording_id": "a", "canonical_id": "a", "partition": "train"},
            {"recording_id": "b", "canonical_id": "b", "partition": "train"}]
    with pytest.raises(ContractError, match="tempo evidence"):
        ev.donor_map(rows, {"a": 100.})
    raw = np.zeros((4000, 94)); raw[:, 64+ev.AUDIO_SCALARS.index("beat_period_s")] = .5
    store = SimpleNamespace(audio=lambda rid: raw)
    ev.verify_donor_tempos(store, ["a"], {"a": 120.})
    with pytest.raises(ContractError, match="evidence changed"):
        ev.verify_donor_tempos(store, ["a"], {"a": 121.})


def test_actual_source_wrap_jitter_is_logged_never_repaired():
    plan = ev.shift_plan(np.arange(320)/40, [[.013, 4.026]], 40)
    result = ev.shifted_candidates([.05, 3.95], [.8, .9], plan)
    assert any(row["wrap_grid_jitter_max_abs_s"] > .001 for row in result["draws"])
    assert any(row["off_requested_grid_candidate_count"] for row in result["draws"])
    assert all(len(row["times_s"]) == 2 for row in result["draws"])


def test_donor_swap_commutes_exactly_with_common_affine_train_normalization():
    x, donor = features(9), features(10)
    norm = {"mean": np.arange(94).tolist(), "std": (np.arange(94)+.01).tolist()}
    raw_then_norm = ev._normalize(ev.nonrhythm_donor(x, donor), norm)
    norm_then_swap = ev.nonrhythm_donor(ev._normalize(x, norm), ev._normalize(donor, norm))
    np.testing.assert_array_equal(raw_then_norm, norm_then_swap)


def test_palette_nan_cannot_silently_become_argmax_token_zero():
    class BrokenPalette(SpyModel):
        def palette_logits(self, context, ids):
            return super().palette_logits(context, ids)*float("nan")
    l = labels()
    with pytest.raises(ContractError, match="palette logits"):
        ev.predict_window(BrokenPalette(), features(), requested_times10_s=l.requested_times10_s,
                          source_times10_s=l.source_times10_s)


def test_all_distinct_fit_exposure_scopes_are_explicit():
    definition = ev.evaluation_definition()["fit_exposure"]
    assert "FULL_SOURCE" in definition["audio_normalizer"]
    assert "TRAIN64s" in definition["visual_transforms"]
    assert "center8" in definition["supervised_labels"]
    assert "full_active_record" in definition["legacy_V6_local"]
    assert "NOT64s_rawaudio" in definition["feature_generation"]


def test_global_nms_after_segment_localmax_handles_short_invalid_gaps():
    times = np.arange(40)/40; curve = np.zeros(40)
    curve[8] = 1.; curve[13] = 2.
    intervals = [[0, .25], [.275, 1.]]
    result, _ = ev.candidate_peaks(times, curve, intervals)
    np.testing.assert_array_equal(result, times[[13]])
    curve[13] = 1.
    result, _ = ev.candidate_peaks(times, curve, intervals)
    np.testing.assert_array_equal(result, times[[8]])


def test_prediction_receipt_cannot_relabel_registered_checkpoint_cells():
    config = {"recipe": "S_VR_P0", "seed": 0}
    bindings = {"config_sha256": json_hash(config), "norm_sha256": "norm"}
    cp = {"path": "synthetic.pt", "sha256": "sha", "recipe": config["recipe"], "seed": 0,
          "role": "selected", "bindings": bindings}
    registration = {"bindings": {}, "donor_map": {"a": "b"}, "donor_tempos_bpm": {"a": 120.},
                    "checkpoints": {"cp": cp}}
    baseline = {"path": "synthetic.json", "sha256": "base"}
    receipt = {"status": "complete", "bindings": {}, "baselines": baseline, "scope": ev.SCOPE,
        "recipe": config["recipe"], "seed": 0, "checkpoint_role": "selected", "checkpoint": cp,
        "metadata": {**bindings, "config": config, "step": 300, "selection_loss": 1.},
        "donor_map": registration["donor_map"], "donor_map_sha256": json_hash(registration["donor_map"]),
        "donor_tempos_bpm": registration["donor_tempos_bpm"], "primary_audio_route": "recipient"}
    assert ev.validate_prediction_receipt(receipt, registration, baseline) == cp
    for key, value in (("recipe", "S_none_P0"), ("seed", 1), ("checkpoint_role", "final")):
        altered = deepcopy(receipt); altered[key] = value
        with pytest.raises(ContractError, match="relabeling"):
            ev.validate_prediction_receipt(altered, registration, baseline)
    altered = deepcopy(receipt); altered["metadata"]["step"] = 301
    with pytest.raises(ContractError, match="role"):
        ev.validate_prediction_receipt(altered, registration, baseline)


def test_ashuf_primary_uses_frozen_map_while_real_primary_does_not():
    l = labels(); windows = {"a": features(1), "b": features(2), "c": features(3)}
    mapping = {"a": "c", "b": "a", "c": "b"}
    clocks = dict(requested_times10_s=l.requested_times10_s, source_times10_s=l.source_times10_s)
    model = SpyModel()
    ashuf = ev.predict_conditions(model, windows, "a", mapping, recipe="S_VR_P0_Ashuf", **clocks)
    assert len(model.seen) == 1
    expected = ev.nonrhythm_donor(windows["a"], windows["c"])
    np.testing.assert_array_equal(model.seen[0][0].numpy(), expected)
    assert ashuf["model"] is ashuf["donor"]  # samecondition, noextra independentdraw
    model = SpyModel()
    ev.predict_conditions(model, windows, "a", mapping, recipe="S_VR_P0", **clocks)
    assert len(model.seen) == 2
    np.testing.assert_array_equal(model.seen[0][0].numpy(), windows["a"])
    np.testing.assert_array_equal(model.seen[1][0].numpy(), expected)


def test_ashuf_donor_map_must_equal_completed_training_run_receipt(tmp_path):
    config = {"recipe": "S_VR_P0_Ashuf", "seed": 0}
    bindings = {"config_sha256": json_hash(config)}
    run = {"status": "complete", "smoke": False, "config": config, "bindings": bindings,
           "donors": {"a": "b"}, "best_weights_sha256": "weights"}
    path = tmp_path/"complete.json"; path.write_text(json.dumps(run))
    checkpoint = {"recipe": "S_VR_P0_Ashuf", "seed": 0, "role": "selected", "sha256": "weights",
                  "bindings": bindings, "training_run_receipt": ev.reference(path)}
    assert ev.validate_training_receipt(checkpoint, {"donor_map": {"a": "b"}}) == run
    with pytest.raises(ContractError, match="actual TRAIN/DEV"):
        ev.validate_training_receipt(checkpoint, {"donor_map": {"a": "c"}})


def effect_contrasts(color=.02, low=.005, p=.03, accent=0.):
    return {"VR-V6_color": {"difference": color, "interval": [low, .04], "confidence": .95,
                            "holm8_p_value": p},
            "VR-V6_accent": {"difference": accent}}


def effect_gate(contrasts, **overrides):
    args = {"support_qualified": True, "null_qualified": True, "controls_complete": True,
            "cell_inventory_complete": True}
    args.update(overrides)
    return ev.observer_effect_gate(contrasts, **args)


def test_observer_effect_positive_gain_with_ci_crossing_zero_is_not_success():
    gate = effect_gate(effect_contrasts(low=-.001))
    assert gate["components"]["color_AP_gain_ge_0p01"]
    assert not gate["components"]["color_paired95_CI_lower_gt_0"]
    assert gate["qualified_pass"] is False


def test_observer_effect_significant_color_gain_cannot_hide_accent_decline():
    gate = effect_gate(effect_contrasts(accent=-.011))
    assert gate["components"]["color_Holm8_p_lt_0p05"]
    assert not gate["components"]["accent_AP_gain_ge_minus_0p01"]
    assert gate["qualified_pass"] is False


@pytest.mark.parametrize("prerequisite", ["support_qualified", "null_qualified", "controls_complete", "cell_inventory_complete"])
def test_observer_effect_missing_prerequisites_is_undefined_not_false_or_zero(prerequisite):
    gate = effect_gate(effect_contrasts(), **{prerequisite: False})
    assert all(gate["components"].values())
    assert gate["qualified_pass"] is None
    assert prerequisite in gate["unmet_prerequisites"]


def test_observer_effect_all_registered_thresholds_and_missing_statistics():
    assert effect_gate(effect_contrasts(color=.01, accent=-.01))["qualified_pass"] is True
    assert effect_gate(effect_contrasts(low=0.))["qualified_pass"] is False
    assert effect_gate(effect_contrasts(p=.05))["qualified_pass"] is False
    assert effect_gate({})["qualified_pass"] is None


def capture_exclusion(rid="30"):
    return {"recording_id": rid, "canonical_id": "song-"+rid, "partition": "dev",
            "pre_fit": True, "not_based_on_target_or_model_scores": True,
            "basis": "capture_integrity", "reason": "synthetic pre-fit capture receipt failure",
            "receipt": {"path": "/synthetic/pre_fit_receipt.json", "sha256": "a"*64}}


def summarize_usable30(rows, excluded=None):
    return ev.summarize_campaign(rows, enrolled_songs=["song-"+str(i) for i in range(31)],
        exclusions={"excluded-"+str(i): "original source gate failed" for i in range(9)},
        usable_inventory=[("song-"+str(i), str(i)) for i in range(30)],
        capture_exclusions=[capture_exclusion()] if excluded is None else excluded)


def test_predeclared_usable30_preserves_enrolled31_without_fake_metric_record():
    rows = [row for row in campaign_rows() if row["recording_id"] != "30"]
    result = summarize_usable30(rows)
    assert result["usable_groups"] == 30 and result["enrolled_groups"] == 31
    assert result["original_groups"] == 40 and result["original_coverage"] == .775
    assert result["capture_unavailable_ids"] == ["song-30"]
    assert result["support_gates"]["color"]["common_qualified_groups"] == 30
    assert "song-30" not in result["support_gates"]["color"]["nonevent_ids"]
    assert not result["capture_failure_imputed_as_metric_record"]
    assert all(row["recording_id"] != "30" for row in result["records"])
    assert result["observer_effect_qualified_pass"] is False  # syntheticzero gain, notsuccess
    for macro in result["macro_scores"].values():
        assert "song-30" not in macro["values"]


def test_usable30_still_requires_every_declared_recipe_seed_head_cell():
    rows = [row for row in campaign_rows() if row["recording_id"] != "30"]
    with pytest.raises(ContractError, match="Missing model"):
        summarize_usable30(rows[:-1])
    # Remove a usable record from ALL arms: the manifest inventory still catches it.
    with pytest.raises(ContractError, match="Missing usable"):
        summarize_usable30([row for row in rows if row["recording_id"] != "29"])


def test_no_unreceipted_or_posthoc_usable30_capture_exclusion():
    rows = [row for row in campaign_rows() if row["recording_id"] != "30"]
    with pytest.raises(ContractError, match="unreceipted"):
        summarize_usable30(rows, excluded=[])
    excluded = capture_exclusion(); excluded["pre_fit"] = False
    with pytest.raises(ContractError, match="pre-fit"):
        summarize_usable30(rows, excluded=[excluded])
