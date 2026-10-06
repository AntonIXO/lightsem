"""D5: fixed-time, source-dependent upper-body trajectory refinement.

Musical targets are explicit controller hypotheses, not semantic annotations.
The pelvis and both complete leg chains remain fixed to the accepted source;
upper-body bone directions are optimized with fixed source bone lengths. There
is no clock, motion generator, resampling of source time, or fallback gesture.
The differentiable transfer term is the actual V3/01 requested-angle mapping.
It is only a surrogate for the actuator solver: decoded emitted DMX is checked
separately, once before and once after optimization, and never mislabeled as
the requested signal. Colors in that check are a fixed diagnostic placeholder;
the campaign renderer must use its independently frozen P2 stream.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np

from lightsem.conductor.body_transfer_v2.skeleton import PARENTS


REGIONS = ((3, 6, 9, 12, 15), (13, 16, 18, 20), (14, 17, 19, 21),
           (1, 4, 7, 10), (2, 5, 8, 11))
REGION_NAMES = ("torso", "left_arm", "right_arm", "left_leg", "right_leg")
MOVABLE = tuple(sorted(set(j for region in REGIONS[:3] for j in region)))
FIXED = tuple(j for j in range(22) if j not in MOVABLE)
FEET = (7, 8, 10, 11)


@dataclass(frozen=True)
class RefinementConfig:
    fps: float = 30.
    lighting_fps: float = 40.
    iterations: int = 80
    learning_rate: float = .008
    control_spacing_s: float = .20
    maximum_displacement_m: float = .12
    maximum_jerk_ratio: float = 1.05
    jerk_tolerance_m_s3: float = .5
    maximum_saturation_increase: float = .01
    maximum_emitted_effort_loss_ratio: float = 1.05
    bone_tolerance_m: float = .001
    effort_weight: float = 1.
    expansion_weight: float = .15
    source_weight: float = .025
    jerk_weight: float = .03
    settling_weight: float = .05
    transfer_weight: float = .08

    def validate(self):
        values = asdict(self)
        if not all(np.isfinite(v) for v in values.values()):
            raise ValueError("Refinement configuration must be finite")
        if self.fps != 30. or self.lighting_fps != 40.:
            raise ValueError("D5 contract is native 30 Hz and emitted 40 Hz")
        if (not isinstance(self.iterations, int) or self.iterations < 0
                or min(self.learning_rate, self.control_spacing_s,
                       self.maximum_displacement_m, self.bone_tolerance_m) <= 0
                or self.maximum_jerk_ratio < 1
                or self.maximum_emitted_effort_loss_ratio < 1
                or min(self.jerk_tolerance_m_s3, self.maximum_saturation_increase,
                       self.effort_weight, self.expansion_weight, self.source_weight,
                       self.jerk_weight, self.settling_weight, self.transfer_weight) < 0):
            raise ValueError("Invalid refinement bounds, weights, or iteration count")
        return self


def _derivative(value, fps):
    return np.diff(value, axis=0, prepend=value[:1]) * fps


def _regional_speed(world, fps):
    speed = np.linalg.norm(_derivative(world, fps), axis=-1)
    return np.stack([speed[:, region].mean(axis=1) for region in REGIONS], axis=1)


def _contacts(world, fps):
    """Same conservative adjacent-speed pseudo-label as native geometry."""
    foot = world[:, FEET]
    delta = np.linalg.norm(np.diff(foot, axis=0) * fps, axis=-1)
    speed = np.maximum(np.concatenate((delta[:1], delta)),
                       np.concatenate((delta, delta[-1:])))
    threshold = np.array([.12, .12, .055, .055])
    return ((foot[..., 1] <= threshold) & (foot[..., 1] >= -.04)
            & (speed <= .20)).astype(np.float32)


def build_refinement_targets(timeline, time):
    """Share D2's source/phrase evidence while applying it to geometry, not time."""
    from .retiming import musical_evidence
    evidence = musical_evidence(timeline, np.asarray(time, dtype=float))
    needed = ("attack", "sustained", "thinning", "phrase_preparation",
              "phrase_release", "confidence", "desired_rate")
    for key in needed:
        value = np.asarray(evidence[key], dtype=float)
        if value.shape != (len(time),) or not np.isfinite(value).all():
            raise ValueError(f"Invalid musical evidence: {key}")
    confidence = np.clip(evidence["confidence"], 0, 1)
    effort = 1 + confidence * (np.asarray(evidence["desired_rate"]) - 1)
    settling = confidence * np.clip(evidence["phrase_preparation"]
        + evidence["sustained"] * evidence["thinning"], 0, 1)
    expansion = 1 + .10 * confidence * np.clip(
        evidence["attack"] + evidence["phrase_release"]
        - evidence["phrase_preparation"] - evidence["sustained"] * evidence["thinning"], -1, 1)
    return {"effort": np.clip(effort, .35, 1.45),
            "expansion": expansion, "settling": settling, "evidence": evidence}


def _spline_basis(frames, fps, spacing):
    """Cubic B-spline deformation coordinates; these do not create motion."""
    phase = np.arange(frames, dtype=float) / (fps * spacing)
    base = np.floor(phase).astype(int)
    u = phase - base
    weights = np.stack(((1-u)**3, 3*u**3-6*u*u+4,
                        -3*u**3+3*u*u+3*u+1, u**3), axis=1) / 6
    indices = base[:, None] + np.arange(4)[None]
    return indices, weights, int(indices.max()) + 1


def _torch_requested_angles(world, outputs):
    """Differentiable exact V3/01 moving-fixture spatial readout in degrees."""
    import torch
    from lightsem.conductor.body_transfer_v2.common import REFERENCE
    ref = torch.as_tensor(REFERENCE, dtype=world.dtype, device=world.device)
    positions = world - world[:, :1] * world.new_tensor([1., 0., 1.])
    parent = np.maximum(PARENTS, 0)
    bones = positions - positions[:, parent]
    rb = ref - ref[parent]
    bones = torch.cat(((positions[:, 9] - positions[:, 0])[:, None], bones[:, 1:]), 1)
    rb = torch.cat(((ref[9] - ref[0])[None], rb[1:]), 0)
    unit = bones / torch.linalg.vector_norm(bones, dim=-1, keepdim=True).clamp_min(1e-8)
    neutral = rb / torch.linalg.vector_norm(rb, dim=-1, keepdim=True).clamp_min(1e-8)
    field = .9 * (unit - neutral) + .15 * (positions - ref)
    moving = [o for o in outputs if o["kind"] != "bulb"]
    if not moving:
        raise ValueError("Transfer surrogate needs moving fixtures")
    primary = [o["role_index"] for o in moving]
    secondary = [o["secondary_role"] for o in moving]
    blend = world.new_tensor([o["role_blend"] for o in moving])[None, :, None]
    point = (1-blend)*field[:, primary] + blend*field[:, secondary]
    sign = world.new_tensor([-1. if o["spatial_slot"] % 2 else 1. for o in moving])[None]
    transformed = torch.stack((point[..., 0] + .30*sign*point[..., 2],
        point[..., 1] + .20*sign*point[..., 0], point[..., 2]-.25*point[..., 1]), -1)
    raw = world.new_tensor([o["anchor"] for o in moving]) + transformed*world.new_tensor([2.6, 2.4, 2.2])
    targets = torch.stack((7.5*torch.tanh(raw[..., 0]/7.5),
        2.5+2.45*torch.tanh((raw[..., 1]-2.5)/2.45),
        -3.+2.8*torch.tanh((raw[..., 2]+3.)/2.8)), -1)
    vec = targets - world.new_tensor([o["position"] for o in moving])
    norm = torch.linalg.vector_norm(vec, dim=-1).clamp_min(1e-8)
    return torch.rad2deg(torch.stack((torch.atan2(vec[..., 0], -vec[..., 1]),
                                     torch.asin((vec[..., 2]/norm).clamp(-1+1e-7, 1-1e-7))), -1))


def evaluate_actual_transfer(world, time, config=None):
    """Run fixed map01 and actual actuator solver; return decoded motion traces.

    This evaluator does not qualify color or brightness. Its RGB placeholder
    has no effect on pan/tilt and is never a campaign-renderable P2 artifact.
    """
    from lightsem.conductor.body_transfer import TransferConfig, compile_intent, decode
    from lightsem.conductor.body_transfer_v2 import make_outputs, map_body
    from lightsem.experiments.beat_conditioned.beat_conditioned_render import _angles
    from .retiming import resample_body
    cfg = (config or RefinementConfig()).validate()
    time = np.asarray(time, dtype=float)
    count = round(len(time) / cfg.fps * cfg.lighting_fps)
    emitted_time = time[0] + np.arange(count) / cfg.lighting_fps
    # Use the campaign's same root/bone-geometry adapter. Linear XYZ blending
    # shortens rotating bones and therefore changes the map01 input geometry.
    source_time = np.minimum(emitted_time-time[0], (len(world)-1)/cfg.fps)
    body = resample_body(world, source_time, cfg.fps)
    outputs = make_outputs(beams=8, lasers=5, bulbs=50)
    intent = map_body(body, outputs, "01", fps=cfg.lighting_fps)
    color = np.zeros((count, 3), dtype=np.uint8)
    dmx, patch, _ = compile_intent(intent, outputs, color, color,
                                  TransferConfig(fps=cfg.lighting_fps))
    moving = [i for i, o in enumerate(outputs) if o["kind"] != "bulb"]
    vec = intent["targets"][:, moving] - np.asarray([outputs[i]["position"] for i in moving])
    requested = np.stack(_angles(vec), -1)
    emitted = decode(dmx, patch)["angles_deg"][:, moving]
    speed = np.linalg.norm(np.diff(emitted, axis=0) * cfg.lighting_fps, axis=-1)
    acceleration = np.linalg.norm(np.diff(emitted, n=2, axis=0) * cfg.lighting_fps**2, axis=-1)
    return {"time": emitted_time, "requested_angles": requested, "emitted_angles": emitted,
            "maximum_speed_deg_s": float(speed.max(initial=0)),
            "maximum_acceleration_deg_s2": float(acceleration.max(initial=0)),
            "saturation_fraction": float(np.mean(acceleration >= .95*360.)),
            "finite": bool(np.isfinite(requested).all() and np.isfinite(emitted).all()),
            "effort_fixture_mask": [outputs[i]["role_index"] in (0, *MOVABLE) for i in moving],
            "body_adapter": "shared retiming.resample_body: root linear, bone directions spherical, fixed bone lengths",
            "mapper": "V3/01", "rig": {"beams": 8, "lasers": 5, "bulbs": 50},
            "color_scope": "constant placeholder for motion-only diagnostic; not P2"}


def _transfer_summary(trace, baseline, time, effort):
    result = {k: v for k, v in trace.items() if k not in
              ("time", "requested_angles", "emitted_angles")}
    affected = np.asarray(trace["effort_fixture_mask"], dtype=float)[None, :, None]
    local_effort = 1 + (np.interp(trace["time"], time, effort)[1:, None, None]-1)*affected
    fps = 1 / np.median(np.diff(trace["time"]))
    for name in ("requested_angles", "emitted_angles"):
        reference = np.diff(baseline[name], axis=0) * fps
        actual = np.diff(trace[name], axis=0) * fps
        normalization = max(float(np.mean(reference**2)), .01)
        result[name.replace("_angles", "_effort_loss")] = float(
            np.mean((actual - reference*local_effort)**2) / normalization)
        result[name.replace("_angles", "_median_speed_deg_s")] = float(
            np.median(np.linalg.norm(actual, axis=-1)))
    return result


def _body_metrics(candidate, source, config):
    length = np.linalg.norm(candidate[:, 1:] - candidate[:, np.asarray(PARENTS)[1:]], axis=-1)
    original_length = np.linalg.norm(source[:, 1:] - source[:, np.asarray(PARENTS)[1:]], axis=-1)
    jerk = np.linalg.norm(np.diff(candidate, n=3, axis=0) * config.fps**3, axis=-1)
    original_jerk = np.linalg.norm(np.diff(source, n=3, axis=0) * config.fps**3, axis=-1)
    return {"maximum_displacement_m": float(np.linalg.norm(candidate-source, axis=-1).max()),
            "maximum_bone_length_error_m": float(np.abs(length-original_length).max()),
            "maximum_fixed_joint_displacement_m": float(np.abs(candidate[:, FIXED]-source[:, FIXED]).max()),
            "jerk_p99_m_s3": float(np.quantile(jerk, .99)),
            "source_jerk_p99_m_s3": float(np.quantile(original_jerk, .99)),
            "maximum_jerk_m_s3": float(jerk.max()),
            "source_maximum_jerk_m_s3": float(original_jerk.max())}


def refine_trajectory(body_world, timeline, *, timestamps=None, config=None,
                      transfer_evaluator: Callable | None = None, targets=None):
    """Return ``(DanceResult, DancePlan)``; every candidate retains source time.

    ``transfer_evaluator(world,time,config)`` is an injectable *actual-emission*
    evaluator with the same trace contract as :func:`evaluate_actual_transfer`.
    It cannot disable the emitted postgate. ``targets`` supports frozen target
    interventions and must contain finite length-T effort/expansion/settling.
    Failure is retained in metadata; no other arm or motion source is inserted.
    """
    import torch
    from .contracts import DancePlan, DanceResult
    from lightsem.conductor.body_transfer_v2 import make_outputs

    cfg = (config or RefinementConfig()).validate()
    source = np.asarray(body_world, dtype=np.float64)
    if source.ndim != 3 or source.shape[1:] != (22, 3) or len(source) < 5 or not np.isfinite(source).all():
        raise ValueError("Expected finite native body (T>=5,22,3)")
    time = np.arange(len(source))/cfg.fps if timestamps is None else np.asarray(timestamps, dtype=float)
    if (time.shape != (len(source),) or not np.isfinite(time).all() or time[0] < 0
            or not np.allclose(np.diff(time), 1/cfg.fps, atol=1e-6, rtol=0)):
        raise ValueError("D5 requires unchanged regular 30 Hz timestamps")
    parent = np.asarray(PARENTS)
    lengths = np.linalg.norm(source[:, 1:] - source[:, parent[1:]], axis=-1)
    if np.any(lengths < 1e-7) or np.max(np.ptp(lengths, axis=0)) > cfg.bone_tolerance_m:
        raise ValueError("D5 requires nondegenerate fixed-length source skeleton")
    target = build_refinement_targets(timeline, time) if targets is None else dict(targets)
    for key in ("effort", "expansion", "settling"):
        value = np.asarray(target[key], dtype=float)
        if value.shape != time.shape or not np.isfinite(value).all():
            raise ValueError(f"Expected finite per-frame refinement target {key}")
        target[key] = value
    if (np.any((target["effort"] < .35) | (target["effort"] > 1.45))
            or np.any((target["expansion"] < .85) | (target["expansion"] > 1.15))
            or np.any((target["settling"] < 0) | (target["settling"] > 1))):
        raise ValueError("Refinement targets exceed preregistered effort/expansion bounds")

    evaluator = transfer_evaluator or evaluate_actual_transfer
    before_trace = evaluator(source, time, cfg)
    before_transfer = _transfer_summary(before_trace, before_trace, time, target["effort"])
    src = torch.as_tensor(source, dtype=torch.float64)
    directions = (src[:, 1:] - src[:, parent[1:]]) / torch.as_tensor(lengths[..., None])
    source_direction = torch.cat((torch.zeros_like(src[:, :1]), directions), 1)
    ids, weights, knots = _spline_basis(len(source), cfg.fps, cfg.control_spacing_s)
    spline_ids = torch.as_tensor(ids)
    spline_weights = torch.as_tensor(weights, dtype=src.dtype)
    parameters = torch.nn.Parameter(torch.zeros((knots, len(MOVABLE), 3), dtype=src.dtype))
    optimizer = torch.optim.Adam([parameters], lr=cfg.learning_rate)
    source_v = torch.diff(src, dim=0) * cfg.fps
    speed_scale = max(float(torch.mean(source_v[:, MOVABLE]**2)), .0025)
    source_j = torch.diff(src, n=3, dim=0) * cfg.fps**3
    jerk_scale = max(float(torch.mean(source_j[:, MOVABLE]**2)), 1.)
    outputs = make_outputs()
    source_angles = _torch_requested_angles(src, outputs).detach()
    affected = torch.as_tensor([o["role_index"] in (0, *MOVABLE)
        for o in outputs if o["kind"] != "bulb"], dtype=src.dtype)[None, :, None]
    source_angle_v = torch.diff(source_angles, dim=0) * cfg.fps
    transfer_scale = max(float(torch.mean(source_angle_v**2)), .01)
    effort = torch.as_tensor(target["effort"], dtype=src.dtype)
    expansion = torch.as_tensor(target["expansion"], dtype=src.dtype)
    settling = torch.as_tensor(target["settling"], dtype=src.dtype)
    source_radius = torch.linalg.vector_norm(src[:, MOVABLE]-src[:, :1], dim=-1)
    # A local source-motion gate suppresses deformation during existing holds.
    motion = np.linalg.norm(_derivative(source, cfg.fps), axis=-1)[:, MOVABLE].mean(1)
    from scipy.ndimage import gaussian_filter1d
    gate = np.clip(gaussian_filter1d(motion, cfg.fps*.15) / .10, 0, 1)
    edge = np.minimum(np.arange(len(source)), np.arange(len(source))[::-1]) / max(1, cfg.fps)
    edge = np.clip(edge, 0, 1)
    gate *= edge*edge*(3-2*edge)
    gate = torch.as_tensor(gate, dtype=src.dtype)[:, None, None]

    def world_from_controls(scale=1.):
        correction = (parameters[spline_ids] * spline_weights[..., None, None]).sum(1) * gate * scale
        moved = source_direction[:, MOVABLE] + correction
        moved = moved / torch.linalg.vector_norm(moved, dim=-1, keepdim=True).clamp_min(1e-8)
        lookup = {joint: index for index, joint in enumerate(MOVABLE)}
        positions = [src[:, 0]]
        for joint in range(1, 22):
            if joint in lookup:
                positions.append(positions[parent[joint]] + moved[:, lookup[joint]]
                                 * torch.as_tensor(lengths[:, joint-1, None], dtype=src.dtype))
            else:
                positions.append(src[:, joint])
        return torch.stack(positions, 1)

    def objectives(world):
        velocity = torch.diff(world, dim=0)*cfg.fps
        effort_loss = torch.mean((velocity[:, MOVABLE]-source_v[:, MOVABLE]*effort[1:, None, None])**2) / speed_scale
        radius = torch.linalg.vector_norm(world[:, MOVABLE]-world[:, :1], dim=-1)
        expansion_loss = torch.mean((radius-source_radius*expansion[:, None])**2) / .01
        displacement = torch.linalg.vector_norm(world-src, dim=-1)
        source_loss = torch.mean((displacement/cfg.maximum_displacement_m)**2)
        bound_loss = 50*torch.mean(torch.relu(displacement/cfg.maximum_displacement_m-1)**2)
        jerk = torch.diff(world, n=3, dim=0)*cfg.fps**3
        jerk_loss = torch.mean(jerk[:, MOVABLE]**2) / jerk_scale
        acceleration = torch.diff(world, n=2, dim=0)*cfg.fps**2
        source_accel_scale = max(float(torch.mean(torch.diff(source_v, dim=0)**2)*cfg.fps**2), .01)
        settling_loss = torch.mean(acceleration[:, MOVABLE]**2*settling[2:, None, None]) / source_accel_scale
        angles = _torch_requested_angles(world, outputs)
        angular_velocity = torch.diff(angles, dim=0)*cfg.fps
        angular_acceleration = torch.diff(angles, n=2, dim=0)*cfg.fps**2
        angular_effort = 1 + (effort[1:, None, None]-1)*affected
        transfer_loss = torch.mean((angular_velocity-source_angle_v*angular_effort)**2) / transfer_scale
        transfer_loss = transfer_loss + torch.mean(torch.relu(torch.linalg.vector_norm(angular_acceleration, dim=-1)/360.-1)**2)
        components = {"effort": effort_loss, "expansion": expansion_loss,
                      "source": source_loss, "jerk": jerk_loss,
                      "settling": settling_loss, "transfer_surrogate": transfer_loss,
                      "displacement_bound": bound_loss}
        total = (cfg.effort_weight*effort_loss + cfg.expansion_weight*expansion_loss
                 + cfg.source_weight*source_loss + cfg.jerk_weight*jerk_loss
                 + cfg.settling_weight*settling_loss + cfg.transfer_weight*transfer_loss + bound_loss)
        return total, components

    def numeric_objectives(world):
        with torch.no_grad():
            total, values = objectives(torch.as_tensor(world, dtype=src.dtype))
        return {"total": float(total), **{key: float(value) for key, value in values.items()}}

    before_objectives = numeric_objectives(source)
    frozen = bool(np.array_equal(source, np.broadcast_to(source[:1], source.shape)))
    history = []
    if not frozen:
        for iteration in range(cfg.iterations):
            optimizer.zero_grad()
            loss, _ = objectives(world_from_controls())
            if not torch.isfinite(loss):
                raise RuntimeError("D5 produced a nonfinite optimizer objective")
            loss.backward()
            torch.nn.utils.clip_grad_norm_([parameters], 10.)
            optimizer.step()
            if iteration % 10 == 0 or iteration == cfg.iterations-1:
                history.append({"iteration": iteration, "objective": float(loss.detach())})

    # Geometric backtracking reduces the same candidate, never adds a fallback.
    failures = []
    chosen = source.copy()
    fraction = 0.
    for fraction_try in (() if frozen else (1., .5, .25, .125, .0625, .03125)):
        with torch.no_grad():
            proposal = world_from_controls(fraction_try).numpy()
        metrics = _body_metrics(proposal, source, cfg)
        jerk_ok = (metrics["jerk_p99_m_s3"] <= metrics["source_jerk_p99_m_s3"]*cfg.maximum_jerk_ratio+cfg.jerk_tolerance_m_s3
            and metrics["maximum_jerk_m_s3"] <= metrics["source_maximum_jerk_m_s3"]*cfg.maximum_jerk_ratio+cfg.jerk_tolerance_m_s3)
        if metrics["maximum_displacement_m"] <= cfg.maximum_displacement_m+1e-9 and jerk_ok:
            chosen, fraction = proposal, fraction_try
            break
    if not frozen and fraction == 0:
        failures.append("no nonzero optimizer step satisfied displacement and jerk bounds")
    after_objectives = numeric_objectives(chosen)
    after_trace = before_trace if np.array_equal(source, chosen) else evaluator(chosen, time, cfg)
    after_transfer = _transfer_summary(after_trace, before_trace, time, target["effort"])
    if not after_transfer["finite"]:
        failures.append("actual requested or emitted motion is nonfinite")
    if after_transfer["maximum_speed_deg_s"] > 90.+1e-6 or after_transfer["maximum_acceleration_deg_s2"] > 360.+1e-6:
        failures.append("actual emitted motion violated frozen actuator limits")
    if after_transfer["saturation_fraction"] > before_transfer["saturation_fraction"]+cfg.maximum_saturation_increase:
        failures.append("actual emitted acceleration saturation increased beyond allowance")
    if after_transfer["emitted_effort_loss"] > before_transfer["emitted_effort_loss"]*cfg.maximum_emitted_effort_loss_ratio+1e-8:
        failures.append("actual emitted effort target loss worsened beyond allowance")
    if not frozen and after_objectives["total"] >= before_objectives["total"]-1e-9:
        failures.append("fixed-time objective did not improve")
    if frozen:
        failures.append("frozen source: refinement correctly produced no autonomous movement")
    metadata = {"approach": "D5", "method": "fixed-time expressive trajectory refinement",
        "config": asdict(cfg), "time_policy": "identical source and output timestamps; no retiming",
        "musical_target_status": "source/phrase controller hypotheses, not ground-truth meaning",
        "fixed_joints": list(FIXED), "movable_joints": list(MOVABLE),
        "support_policy": "pelvis, hips, knees, ankles and feet exactly preserve accepted source",
        "bone_policy": "preserve source bone lengths at each frame; reject variable-length input",
        "frozen_source": frozen, "accepted_step_fraction": fraction,
        "objective_before": before_objectives, "objective_after": after_objectives,
        "transfer_before": before_transfer, "transfer_after": after_transfer,
        "body_constraints": _body_metrics(chosen, source, cfg), "optimizer_history": history,
        "postgate_passed": not failures, "infeasibility": failures,
        "qualification": "engineering candidate" if not failures else "failure diagnostic",
        "artistic_acceptance": "requires independent musical controls and user review"}
    effort_plan = np.broadcast_to(target["effort"][:, None], (len(source), len(REGIONS))).copy()
    effort_plan[:, 3:] = 1.
    events = []
    evidence = target.get("evidence", {})
    if evidence:
        from scipy.signal import find_peaks
        for name in ("phrase_preparation", "phrase_release"):
            values = np.asarray(evidence[name])
            peaks, _ = find_peaks(values, height=.1, distance=max(1, int(cfg.fps*.5)))
            events.extend({"time": float(time[i]), "kind": name,
                "evidence_strength": float(values[i]),
                "source": "measured source/texture change supporting inferred phrase boundary"}
                for i in peaks)
    plan = DancePlan(time=time.copy(), source_time=time.copy(), effort=effort_plan,
        anchors=[], events=sorted(events, key=lambda item: item["time"]),
        metadata={"approach": "D5", "region_names": list(REGION_NAMES),
            "targets": {k: target[k].tolist() for k in ("effort", "expansion", "settling")},
            "source_evidence": {k: np.asarray(v).tolist() for k, v in evidence.items()},
            "music_provenance": {} if timeline is None else timeline.metadata,
            "target_status": metadata["musical_target_status"], "time_policy": metadata["time_policy"]})
    result = DanceResult(time=time.copy(), world=chosen, contacts=_contacts(chosen, cfg.fps),
                         source_time=time.copy(), metadata=metadata)
    result.validate()
    plan.validate()
    return result, plan
