#!/usr/bin/env python3
"""Four fixed, existing-checkpoint Shpongle deliveries. Never train or tune.

NEW versioned adapter. Original demo, model, mapping and renderer stay immutable.
No project import or model load occurs until a bound phase passes its guards.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import colorsys
import hashlib
import io
import json
import os
from pathlib import Path
import random
import shutil
import time

RECIPES = ("S_none_P0", "S_V6_P0", "I_VR_P0", "S_VR_P0_Ashuf")
WIDTHS = {r: 76 if r == "I_VR_P0" else 128 for r in RECIPES}
AUX = dict(zip(RECIPES, (0, 6, 96, 96)))
GIB = 2**30
CAP = 1020054732
RESERVATION = 32 * 2**20
PAYLOAD_BOUND = 32069076
METADATA_BOUND = 2**20
NPZ_OVERHEAD_BOUND = 128 * 1024
TOTAL_BOUND = PAYLOAD_BOUND + METADATA_BOUND + NPZ_OVERHEAD_BOUND


class Blocked(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise Blocked(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def checked(ref):
    path = Path(ref["path"])
    require(path.is_file() and digest(path) == ref["sha256"], "Changed/missing artifact: " + str(path))
    return path


def reference(path):
    path = Path(path)
    return {"path": str(path), "sha256": digest(path), "bytes": path.stat().st_size}


def read_json(path):
    return json.loads(Path(path).read_text())


def source_checks(sources):
    require(bool(sources), "Empty source binding")
    for path, expected in sources.items():
        checked({"path": path, "sha256": expected})


def size_tree(root):
    root = Path(root)
    if not root.exists():
        return 0
    require(not root.is_symlink(), "Symlink output root")
    total = 0
    for path in root.rglob("*"):
        require(not path.is_symlink(), "Symlink in accounted output tree")
        if path.is_file():
            total += path.stat().st_size
    return total


def space_values(campaign_used, derived_used, free, pending=0):
    """Reserve ALL four outputs, including when only the first arm is requested."""
    require(0 <= derived_used <= campaign_used, "Invalid aggregate accounting")
    require(pending >= 0 and derived_used + pending <= RESERVATION, "Derived32MiB cap")
    remaining_reservation = RESERVATION - derived_used
    require(campaign_used + remaining_reservation <= CAP, "Original campaign0.95GiB cap")
    require(free - remaining_reservation >= GIB, "Data1GiB free floor")
    return {"campaign_bytes": campaign_used, "derived_bytes": derived_used,
            "remaining_reserved_bytes": remaining_reservation, "free_bytes": free,
            "reserved_final_campaign_bytes": campaign_used + remaining_reservation}


def guard(plan, pending=0):
    campaign = Path(plan["campaign_root"]).resolve()
    derived = Path(plan["derived_root"]).resolve()
    require(campaign.is_dir() and derived != campaign and derived.is_relative_to(campaign),
            "Derived outputs must be inside original campaign")
    return space_values(size_tree(campaign), size_tree(derived), shutil.disk_usage(campaign).free, pending)


def exclusive_bytes(path, data, plan):
    """Publish once; interrupted partials remain evidence and block overwrite."""
    path = Path(path)
    partial = path.with_name(path.name + ".partial")
    require(not path.exists() and not partial.exists(), "Refuse overwrite/resume: " + str(path))
    require(path.resolve().is_relative_to(Path(plan["derived_root"]).resolve()), "Unaccounted write")
    guard(plan, len(data))
    with partial.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(partial, path)
    partial.unlink()
    guard(plan)


def save_json(path, value, plan):
    data = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    root = Path(plan["derived_root"])
    json_bytes = sum(p.stat().st_size for p in root.rglob("*")
                     if p.is_file() and p.suffix in (".json", ".log", ".py", ".txt"))
    require(json_bytes + len(data) <= METADATA_BOUND, "Bounded metadata/log budget")
    exclusive_bytes(path, data, plan)


def save_npz(path, arrays, plan):
    import numpy as np
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    exclusive_bytes(path, buffer.getvalue(), plan)


def validate_plan(plan):
    require(plan.get("schema_version") == 1, "Unknown plan")
    require(tuple(a["recipe"] for a in plan["arms"]) == RECIPES, "Four fixed recipes/order required")
    require(plan.get("scientific_cap_bytes") == CAP and plan.get("derived_reservation_bytes") == RESERVATION,
            "No storage-cap override")
    require(plan.get("minimum_free_bytes") == GIB and TOTAL_BOUND <= RESERVATION, "Resource policy changed")
    require(plan.get("audio_route") == "recipient_audio_delivery", "No alternate/donor demo audio")
    require(plan.get("anchor_inference_allowed") is False, "Anchor cannot be inferred again")
    require(plan.get("mapping_id") == "sep22_fixed_overlay_v1", "No mapping retuning")
    require(len({a["subdir"] for a in plan["arms"]}) == 4, "Unique variant directories required")
    for arm in plan["arms"]:
        ck = arm["checkpoint"]
        require((ck["recipe"], ck["seed"], ck["role"]) == (arm["recipe"], 0, "selected"), "Wrong checkpoint role/seed")
        require(Path(arm["subdir"]).name == arm["subdir"] and arm["subdir"] not in ("", ".", ".."), "Unsafe subdir")
        require(arm["audio_route"] == "recipient_audio_delivery", "Per-arm audio route mismatch")
        require(arm["non_primary_donor_route"] is (arm["recipe"] == "S_VR_P0_Ashuf"), "Ashuf route must be disclosed")


def authorize(plan_path, binding_path, phase):
    plan, binding = read_json(plan_path), read_json(binding_path)
    validate_plan(plan)
    require(binding.get("plan_sha256") == digest(plan_path), "Plan binding changed")
    require(binding.get("status") == "authorized_delivery" and phase in binding.get("phases", []), "Phase not authorized")
    require(binding.get("source_files") == plan["source_files"], "Exact source binding required")
    require(binding.get("test_sources") == plan["test_sources"], "Exact test binding required")
    source_checks(plan["source_files"])
    source_checks(plan["test_sources"])
    require(str(Path(__file__).resolve()) in plan["source_files"], "Driver omitted from binding")
    if phase == "infer":
        receipt = read_json(checked(binding.get("native_test_receipt", {})))
        require(receipt.get("status") == "passed" and receipt.get("source_files") == plan["source_files"]
                and receipt.get("test_sources") == plan["test_sources"]
                and receipt.get("real_model_forward_calls") == 0
                and receipt.get("sealed_TEST_reads") == 0, "Native synthetic tests not bound/passed")
    guard(plan)
    return plan, binding


def load_helpers():
    from lightsem.experiments.light_layers.training import load_checkpoint, generated_history, _normalize
    from lightsem.experiments.light_layers.evaluation import candidate_peaks, validate_transforms
    from lightsem.experiments.light_layers.overlay import (
        AccentConfig, AccentEvent, apply_overlay, apply_accent_events, allowed_slot_mask, sample_accent_events)
    from lightsem.conductor.rig import canonical_rig
    return locals()


def validate_metadata(metadata, arm):
    recipe = arm["recipe"]
    cfg, ref = metadata["model_config"], metadata["reference_model_config"]
    require(metadata["config"]["recipe"] == recipe and metadata["config"]["seed"] == 0, "Loaded recipe/seed mismatch")
    architecture = "independent" if recipe == "I_VR_P0" else "shared"
    require(metadata["architecture"] == architecture, "Loaded architecture mismatch")
    for config, width in ((cfg, WIDTHS[recipe]), (ref, 128)):
        require(config["d_model"] == width and config["in_dim"] == 94 and config["palette_size"] == 32
                and config["visual_aux_dim"] == AUX[recipe] and config["pose_aux_dim"] == 0
                and config["max_frames40"] == 12000, "Wrong architecture dimensions/double narrowing")
    for key, expected in arm["checkpoint"]["bindings"].items():
        require(metadata.get(key) == expected, "Loaded checkpoint binding mismatch: " + key)


def load_bound_model(arm, loader, device):
    """Canonical loader uses reference128, never factory(realized76)."""
    model, metadata = loader(arm["checkpoint"]["path"], device=device,
                             expected_bindings=arm["checkpoint"]["bindings"])
    validate_metadata(metadata, arm)
    require(asdict(model.cfg) == metadata["model_config"], "Loaded model/metadata config mismatch")
    return model, metadata


def validate_clock(raw, normalized, baseline, times, bus_times):
    import numpy as np
    require(raw.shape == normalized.shape == (11092, 94), "Audio shape")
    require(raw.dtype == normalized.dtype == np.float32, "Audio dtype")
    require(baseline.shape == (11092, 512) and baseline.dtype == np.uint8, "Baseline DMX shape/type")
    require(times.dtype == np.float64 and np.array_equal(times, np.arange(11092) / 40), "Canonical clock")
    require(bus_times.dtype == np.float32 and np.array_equal(bus_times, times.astype(np.float32)), "Quantized bus clock")
    require(np.isfinite(raw).all() and np.isfinite(normalized).all(), "Nonfinite audio")


def load_inputs(plan, helpers):
    import numpy as np
    refs = plan["inputs"]
    for ref in refs.values():
        checked(ref)
    registration = read_json(refs["evaluation_binding"]["path"])
    transforms = read_json(refs["transforms"]["path"])
    helpers["validate_transforms"](transforms)
    for arm in plan["arms"]:
        ck = arm["checkpoint"]
        checked(ck)
        training = read_json(checked(ck["training_run_receipt"]))
        require(registration["checkpoints"][arm["checkpoint_key"]] == ck, "Registered checkpoint differs")
        require(training["status"] == "complete" and training["best_weights_sha256"] == ck["sha256"], "Not completed selected checkpoint")
        require(training["bindings"] == ck["bindings"] and training["frozen_transforms"] == transforms,
                "Checkpoint common transforms/bindings differ")
        require(refs["transforms"]["sha256"] == ck["bindings"]["transforms_sha256"], "Transform file mismatch")
        for key in ("norm_sha256", "vocab_sha256"):
            require(transforms[key] == ck["bindings"][key], "Transform identity mismatch")
    with np.load(refs["audio94"]["path"], allow_pickle=False) as z:
        raw, normalized = z["audio94"], z["normalized_audio94"]
        times, bus_times = z["time40_s"], z["source_bus_time40_s"]
    with np.load(refs["baseline"]["path"], allow_pickle=False) as z:
        baseline, baseline_times = z["companion"], z["time"]
    validate_clock(raw, normalized, baseline, times, bus_times)
    require(np.array_equal(times, baseline_times), "Accepted companion clock differs")
    require(np.array_equal(normalized, helpers["_normalize"](raw, transforms["audio"])), "Cached normalization differs")
    provenance = read_json(refs["original_provenance"]["path"])
    require(provenance["artifacts"]["audio94.npz"] == refs["audio94"], "Audio provenance differs")
    require(provenance["inputs"]["baseline"] == refs["baseline"], "Baseline provenance differs")
    require(provenance["inputs"]["audio"] == refs["source_audio"], "Soundtrack provenance differs")
    return transforms, raw, normalized, baseline, times


def palette_rgb(prototypes):
    """Exact Sep22 prototype visualization; no new palette policy."""
    import numpy as np
    p = np.asarray(prototypes, np.float64).reshape(-1, 12, 6)
    hue = np.arange(12, dtype=np.float64) / 12
    saturation = np.arange(6, dtype=np.float64) / 5
    z = np.sum(p * saturation[None, None, :] * np.exp(2j * np.pi * hue)[None, :, None], axis=(1, 2))
    h = np.where(np.abs(z) == 0, 0., np.mod(np.angle(z) / (2 * np.pi), 1.))
    s = np.abs(z)
    require(np.all(s <= 1 + 1e-12), "Invalid prototype saturation")
    hsv = np.stack([h, np.minimum(s, 1), np.ones(len(s))], 1)
    rgb = np.array([colorsys.hsv_to_rgb(*v) for v in hsv], np.float64)
    return rgb, hsv, z


def independent_slots(rig):
    import numpy as np
    rgb, dim, angles = [], [], []
    for fixture in rig.fixtures:
        a = fixture.address - 1
        if fixture.kind == "moving_head":
            rgb.extend([a + 6, a + 7, a + 8]); dim.append(a + 4); angles.extend(a + j for j in range(4))
        elif fixture.kind == "wash":
            rgb.extend([a + 2, a + 3, a + 4]); dim.append(a)
        elif fixture.kind == "projector":
            rgb.extend([a + 2, a + 3, a + 4])
    allowed = np.zeros(512, bool)
    allowed[rgb + dim] = True
    return allowed, np.asarray(rgb, int), np.asarray(dim, int), np.asarray(angles, int)


def map_outputs(scores, baseline, times, prototypes, helpers):
    import numpy as np
    rig = helpers["canonical_rig"]()
    config = helpers["AccentConfig"](max_strength=.35, min_width_s=.25, max_width_s=.25, refractory_s=.2)
    candidate_times, candidate_scores = helpers["candidate_peaks"](
        times, scores["accent_score"], np.array([[0., len(times) / 40]]))
    take = candidate_scores >= .5
    event_times, confidences = candidate_times[take], candidate_scores[take]
    events = [helpers["AccentEvent"](float(t), .35, .25) for t in event_times]
    pulse = helpers["sample_accent_events"](times, events, config=config)
    rgb, hsv, z = palette_rgb(prototypes)
    cues = scores["color_cues"]
    cue_times = scores["time10_s"][cues]
    token40 = np.repeat(scores["generated_palette_ids"], 4)
    active = np.zeros(len(times), bool) if not len(cue_times) else times >= cue_times[0]
    tracks = {}
    offsets = {"moving_head": (6, 7, 8), "wash": (2, 3, 4), "projector": (2, 3, 4)}
    for f in rig.fixtures:
        if f.kind in offsets:
            slots = f.address - 1 + np.array(offsets[f.kind])
            track = baseline[:, slots].astype(float) / 255
            track[active] = rgb[token40[active]]
            tracks[f.name] = track
    colored = helpers["apply_overlay"](baseline, rig, times, rgb=tracks)
    accented = {f.name: events for f in rig.fixtures if f.kind in ("moving_head", "wash")}
    overlay = helpers["apply_accent_events"](colored, rig, times, accented, accent_config=config)
    result = dict(overlay=overlay, time=times, allowed_slot_mask=helpers["allowed_slot_mask"](rig),
                  color_active40=active, palette_ids40=token40, accent_pulse40=pulse.astype(np.float32),
                  accent_times_s=event_times, accent_confidences=confidences, color_cue_times_s=cue_times,
                  palette_rgb=rgb, palette_hsv=hsv, palette_complex_z=z)
    verify_mapping(scores, result, baseline, rig)
    return result


def array_digest(value):
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def verify_mapping(scores, overlay, baseline, rig):
    """Independent byte reconstruction, no neural calls or cue-policy changes."""
    import numpy as np
    o = overlay
    allowed, rgbslots, dimslots, angles = independent_slots(rig)
    require(int(allowed.sum()) == 43, "Canonical rig43/469 contract")
    require(np.array_equal(o["allowed_slot_mask"], allowed), "Saved allowed mask differs")
    require(o["overlay"].shape == baseline.shape and o["overlay"].dtype == np.uint8, "Output shape/type")
    require(np.array_equal(o["overlay"][:, ~allowed], baseline[:, ~allowed]), "Protected bytes changed")
    require(np.array_equal(o["overlay"][:, angles], baseline[:, angles]), "Angles changed")
    require(np.array_equal(o["time"], scores["time40_s"])
            and np.array_equal(o["time"], np.arange(len(baseline)) / 40)
            and np.array_equal(scores["time10_s"], np.arange(len(baseline) // 4) / 10), "Saved clock differs")
    require(np.array_equal(o["palette_ids40"], np.repeat(scores["generated_palette_ids"], 4)), "Token clock differs")
    require(np.array_equal(o["color_cue_times_s"], scores["time10_s"][scores["color_cues"]]), "Cue clock differs")
    cue_times = o["color_cue_times_s"]
    expected_active = np.zeros(len(o["time"]), bool) if len(cue_times) == 0 else o["time"] >= cue_times[0]
    require(np.array_equal(o["color_active40"], expected_active), "Invented cue/initialization")
    active = expected_active
    color = np.floor(o["palette_rgb"][o["palette_ids40"]] * 255 + .5).astype(np.uint8)
    for i in range(0, len(rgbslots), 3):
        slots = rgbslots[i:i+3]
        require(np.array_equal(o["overlay"][np.ix_(active, slots)], color[active]), "RGB differs")
        require(np.array_equal(o["overlay"][np.ix_(~active, slots)], baseline[np.ix_(~active, slots)]), "RGB before cue changed")
    pulse = np.zeros(len(o["time"]))
    for t in o["accent_times_s"]:
        distance = np.abs(o["time"] - t) / .125
        pulse = np.maximum(pulse, np.where(distance < 1, .35 * (.5 + .5 * np.cos(np.pi * distance)), 0.))
    expected = np.floor(np.minimum(baseline[:, dimslots] * (1 + pulse[:, None]), 255) + .5).astype(np.uint8)
    require(np.array_equal(expected, o["overlay"][:, dimslots]), "Accent dimmer differs")
    require(np.allclose(pulse, o["accent_pulse40"], rtol=0, atol=2e-8), "Pulse differs")
    require(float(pulse.max(initial=0)) <= .35 + 1e-12, "Accent strength cap")
    ids, previous = scores["generated_palette_ids"], scores["previous_ids"]
    require(ids.shape == previous.shape == scores["color_cues"].shape
            and ids.dtype.kind in "iu" and previous.dtype.kind in "iu"
            and ((ids >= 0) & (ids < 32)).all(), "Invalid generated palette ids")
    require(previous[0] == 32 and np.array_equal(previous[1:], ids[:-1]), "Generated history differs")
    require(np.all(scores["color_cues"][1:][np.diff(ids) != 0]), "State changes outside cues")
    for name, value in scores.items():
        require(np.isfinite(value).all(), "Nonfinite saved model output: " + name)
    return {"status": "passed", "model_forward_calls": 0, "protected_slots_count": 469,
            "protected_bytes_compared": int(len(baseline) * 469), "changed_protected_bytes": 0,
            "baseline_protected_sha256": array_digest(baseline[:, ~allowed]),
            "overlay_protected_sha256": array_digest(o["overlay"][:, ~allowed]),
            "baseline_full_sha256": array_digest(baseline), "overlay_full_sha256": array_digest(o["overlay"]),
            "exact_RGB_and_dimmer_reconstruction": True, "no_invented_color_cues": True}


def full_forward(model, normalized, width, helpers):
    import numpy as np
    import torch
    require(model.training is False, "Inference requires eval mode")
    with torch.no_grad(), torch.autocast(device_type="cuda", enabled=False):
        output = model(torch.from_numpy(normalized[None]).to("cuda"))
        require(output["context10"].shape == (1, 2773, width), "Wrong architecture context")
        require(set(output) == {"accent_logit", "chromatic_change_logit", "context10"}, "Auxiliary output at inference")
        rollout = helpers["generated_history"](model, output["context10"], output["chromatic_change_logit"],
                                              initial_ids=None, state_ablation=False, cue_threshold=.5, initialize=True)
        logits = model.palette_logits(output["context10"], rollout["previous_ids"])
        to_np = lambda t: t.detach().cpu().numpy()
        scores = {k: to_np(v[0]) for k, v in {**output, **rollout}.items()}
        scores.update(accent_score=to_np(torch.sigmoid(output["accent_logit"])[0]),
                      color_score=to_np(torch.sigmoid(output["chromatic_change_logit"])[0]),
                      palette_logits=to_np(logits[0]), time40_s=np.arange(11092)/40,
                      time10_s=np.arange(2773)/10)
    for value in scores.values():
        require(np.isfinite(value).all(), "Nonfinite model result")
    return scores


def verify_saved(plan, arm, baseline, helpers):
    import numpy as np
    directory = Path(plan["derived_root"]) / arm["subdir"]
    inference = read_json(directory / "inference_saved.json")
    require(inference["recipe"] == arm["recipe"] and inference["checkpoint"] == arm["checkpoint"], "Saved arm differs")
    with np.load(checked(inference["model_outputs"]), allow_pickle=False) as z:
        scores = {k: z[k] for k in z.files}
    with np.load(checked(inference["overlay"]), allow_pickle=False) as z:
        overlay = {k: z[k] for k in z.files}
    proof = verify_mapping(scores, overlay, baseline, helpers["canonical_rig"]())
    return proof, inference


def infer_one(plan, arm, inputs, helpers, plan_ref, binding_ref):
    import numpy as np
    import torch
    directory = Path(plan["derived_root"]) / arm["subdir"]
    require(not directory.exists(), "Arm already exists; never repeat a forward")
    directory.mkdir()
    save_json(directory / "run_started.json", {"recipe": arm["recipe"], "plan": plan_ref,
              "authorization": binding_ref, "audio_route": arm["audio_route"],
              "ashuf_not_primary_donor_evaluation": arm["non_primary_donor_route"], "forward_attempt_reserved": True}, plan)
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(4)
    model, metadata = load_bound_model(arm, helpers["load_checkpoint"], "cuda")
    transforms, raw, normalized, baseline, times = inputs
    require(len(raw) <= model.cfg.max_frames40 and len(raw) % 4 == 0, "Full song outside model limit")
    save_json(directory / "checkpoint_metadata.json", metadata, plan)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    scores = full_forward(model, normalized, WIDTHS[arm["recipe"]], helpers)
    runtime = {"encoder_forward_calls": 1, "generated_palette_head_rollout": True,
               "elapsed_forward_and_rollout_s": time.monotonic() - started,
               "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "torch": torch.__version__,
               "numpy": np.__version__, "device": torch.cuda.get_device_name(),
               "float32": True, "autocast": False, "TF32": False, "model_eval": True,
               "chunking": False, "training": False, "visual_or_pose_input": False}
    # Save neural outputs before mapping. A mapping failure must not repeat the forward.
    save_npz(directory / "model_outputs.npz", scores, plan)
    del model
    torch.cuda.empty_cache()
    overlay = map_outputs(scores, baseline, times, transforms["palette_prototypes"], helpers)
    save_npz(directory / "overlay_dmx.npz", overlay, plan)
    save_json(directory / "inference_saved.json", {"recipe": arm["recipe"], "checkpoint": arm["checkpoint"],
              "plan": plan_ref, "authorization": binding_ref, "inputs": plan["inputs"],
              "source_files": plan["source_files"], "audio_route": arm["audio_route"],
              "ashuf_not_primary_donor_evaluation": arm["non_primary_donor_route"],
              "runtime": runtime, "model_outputs": reference(directory / "model_outputs.npz"),
              "overlay": reference(directory / "overlay_dmx.npz"),
              "counts": {"accent_events": len(overlay["accent_times_s"]),
                         "color_cues": int(scores["color_cues"].sum())}}, plan)
    proof, inference = verify_saved(plan, arm, baseline, helpers)
    save_json(directory / "protected_byte_equality.json", proof, plan)
    source_checks(plan["source_files"])
    for ref in plan["inputs"].values(): checked(ref)
    # Only a post-readback, source-checked run is complete. Exclusive atomic publication.
    save_json(directory / "complete.json", {"status": "complete", "recipe": arm["recipe"],
              "plan": plan_ref, "authorization": binding_ref, "inference": reference(directory / "inference_saved.json"),
              "proof": reference(directory / "protected_byte_equality.json"), "storage": guard(plan)}, plan)
    print(json.dumps({"status": "complete", "recipe": arm["recipe"], "derived_dir": str(directory)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("preflight", "infer", "verify"))
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--binding", type=Path, required=True)
    args = parser.parse_args()
    plan, binding = authorize(args.plan, args.binding, args.phase)
    helpers = load_helpers()
    inputs = load_inputs(plan, helpers)
    derived = Path(plan["derived_root"])
    plan_ref, binding_ref = reference(args.plan), reference(args.binding)
    if args.phase == "preflight":
        require(not derived.exists(), "Preflight refuses existing delivery directory")
        derived.mkdir()
        save_json(derived / "preflight.json", {"status": "passed", "model_forward_calls": 0,
                  "plan": plan_ref, "authorization": binding_ref, "storage": guard(plan),
                  "four_recipes": list(RECIPES), "common_transforms": True,
                  "reused_audio94": plan["inputs"]["audio94"]}, plan)
        print("PREFLIGHT_PASSED_NO_MODEL_CALLS", flush=True)
        return
    preflight = read_json(derived / "preflight.json")
    require(preflight["status"] == "passed" and preflight["plan"] == plan_ref, "Missing/changed preflight")
    if args.phase == "verify":
        for arm in plan["arms"]:
            proof, _ = verify_saved(plan, arm, inputs[3], helpers)
            print(json.dumps({"recipe": arm["recipe"], "readback": proof}), flush=True)
        return
    # Shared native lock; no overlapping models/renderers. Parent also verifies external GPU state.
    from lightsem.experiments.stage11_common import gpu_lock
    with gpu_lock():
        for arm in plan["arms"]:
            # Existing completed outputs are authenticated and skipped, never inferred again.
            directory = derived / arm["subdir"]
            if (directory / "complete.json").exists():
                complete = read_json(directory / "complete.json")
                require(complete["status"] == "complete" and complete["plan"] == plan_ref, "Existing completion differs")
                checked(complete["inference"]); checked(complete["proof"])
                verify_saved(plan, arm, inputs[3], helpers)
                continue
            guard(plan)
            infer_one(plan, arm, inputs, helpers, plan_ref, binding_ref)
    require(not (derived / "all_complete.json").exists(), "Aggregate already complete")
    save_json(derived / "all_complete.json", {"status": "complete", "plan": plan_ref,
              "arms": [reference(derived / a["subdir"] / "complete.json") for a in plan["arms"]],
              "anchor_not_rerun": True, "scope": "four fixed existing-checkpoint delivery videos; no scientific verdict",
              "storage": guard(plan)}, plan)


if __name__ == "__main__":
    main()
