"""Auditable, source-preserving dynamic-dance interventions D1--D4.

No function creates an independent movement signal. Time maps select existing
poses, and recomposition selects fragments from the supplied song's own body.
World-space bone directions are spherically interpolated; native139 velocities
and rotation channels are never interpolated as an undifferentiated vector.
"""
from __future__ import annotations

import json
from typing import Any
import warnings

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from .contracts import DancePlan, DanceResult, MusicTimeline

PARENTS = np.asarray((-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19))
FEET = (7, 8, 10, 11)
REGIONS = ("support", "torso", "head", "left_arm", "right_arm")
# Collars use the arm clock; legs and pelvis never receive independent clocks.
JOINT_REGION = np.asarray((0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 2, 3, 4, 2, 3, 4, 3, 4, 3, 4))
SOURCE_REGIONS = {"drums": ["support"], "bass": ["support", "torso"],
                  "other": ["right_arm"], "vocals": ["head", "left_arm"]}


def _check_body(world: np.ndarray) -> np.ndarray:
    world = np.asarray(world, dtype=np.float64)
    if world.ndim != 3 or world.shape[1:] != (22, 3) or len(world) < 3:
        raise ValueError("Expected body [T,22,3], with at least three frames")
    if not np.isfinite(world).all():
        raise ValueError("Nonfinite source body")
    return world


def _constant(world: np.ndarray) -> bool:
    return bool(np.array_equal(world, np.broadcast_to(world[:1], world.shape)))


def _unit(value: np.ndarray) -> np.ndarray:
    return value / np.maximum(np.linalg.norm(value, axis=-1, keepdims=True), 1e-12)


def _slerp(a: np.ndarray, b: np.ndarray, fraction: np.ndarray) -> np.ndarray:
    """Unit direction interpolation, including a deterministic antipodal path."""
    a, b = _unit(a), _unit(b)
    f = np.asarray(fraction)[..., None]
    dot = np.clip(np.sum(a * b, axis=-1, keepdims=True), -1., 1.)
    tangent = b - dot * a
    norm = np.linalg.norm(tangent, axis=-1, keepdims=True)
    axis = np.eye(3)[np.argmin(np.abs(a), axis=-1)]
    fallback = _unit(axis - (axis * a).sum(-1, keepdims=True) * a)
    tangent = np.where(norm > 1e-9, tangent / np.maximum(norm, 1e-12), fallback)
    angle = np.arccos(dot)
    result = np.cos(angle * f) * a + np.sin(angle * f) * tangent
    linear = _unit((1. - f) * a + f * b)
    result = np.where(dot > .9995, linear, result)
    # Retain exact source values at integer samples, including degenerate bones.
    result = np.where(f <= 0., a, np.where(f >= 1., b, result))
    return result


def _project_rates(rates: np.ndarray, mean: float = 1.) -> np.ndarray:
    """Euclidean projection onto 0<=rate<=1.5 and a fixed total duration."""
    rates = np.asarray(rates, dtype=np.float64)
    if rates.size == 0 or not np.isfinite(rates).all():
        raise ValueError("Rates must be finite and nonempty")
    if mean < -1e-12 or mean > 1.5 + 1e-12:
        raise ValueError("Required duration cannot be represented within rate limits")
    mean = np.clip(mean, 0., 1.5)
    low, high = -float(np.max(rates)) - 2., 2. - float(np.min(rates))
    for _ in range(70):
        offset = (low + high) / 2.
        if np.mean(np.clip(rates + offset, 0., 1.5)) < mean:
            low = offset
        else:
            high = offset
    return np.clip(rates + (low + high) / 2., 0., 1.5)


def bounded_time_map(desired_rates: np.ndarray, fps: float = 30.,
                     end_time: float | None = None, smooth_seconds: float = .15):
    """Return source timestamps and rates with exact endpoints and bounded slope.

    ``desired_rates`` has one value per output frame. The last displayed rate is
    the last interval's rate. Duration conservation may weaken requested holds;
    the returned rates, never the proposals, are the actual intervention.
    """
    desired = np.asarray(desired_rates, dtype=np.float64)
    if desired.ndim != 1 or len(desired) < 3 or fps <= 0:
        raise ValueError("Expected at least three scalar rates and positive fps")
    if not np.isfinite(desired).all():
        raise ValueError("Nonfinite desired rates")
    duration = (len(desired) - 1) / fps
    end_time = duration if end_time is None else float(end_time)
    rates = desired[:-1].copy()
    if smooth_seconds > 0:
        rates = gaussian_filter1d(rates, smooth_seconds * fps, mode="nearest")
    rates = _project_rates(rates, end_time / duration)
    mapping = np.r_[0., np.cumsum(rates) / fps]
    mapping[-1] = end_time
    return mapping, np.r_[rates, rates[-1]]


def resample_body(world: np.ndarray, source_time: np.ndarray, fps: float = 30.) -> np.ndarray:
    """Resample decoded root and bone directions without changing bone lengths.

    Regional clocks use REGIONS order. Every lower-body bone and the root share
    the support clock. Fixed lengths use the source median per bone; historical
    decoder jitter in length is not turned into stretching during retiming.
    """
    world = _check_body(world)
    source_time = np.asarray(source_time, dtype=np.float64)
    if source_time.ndim not in (1, 2) or (source_time.ndim == 2 and source_time.shape[1] != 5):
        raise ValueError("source_time must have shape [T] or [T,5]")
    if not np.isfinite(source_time).all() or source_time.min() < -1e-9 or source_time.max() > (len(world)-1)/fps + 1e-8:
        raise ValueError("Source timestamps outside the actual body")
    if _constant(world):
        return np.broadcast_to(world[:1], (len(source_time), 22, 3)).copy()
    clock = source_time[:, None] if source_time.ndim == 1 else source_time
    pos = np.clip(clock * fps, 0., len(world) - 1.)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, len(world) - 1)
    frac = pos - lo
    output = np.empty((len(source_time), 22, 3), dtype=np.float64)
    output[:, 0] = ((1.-frac[:, :1]) * world[lo[:, 0], 0] + frac[:, :1] * world[hi[:, 0], 0])
    for joint in range(1, 22):
        parent = PARENTS[joint]
        vectors = world[:, joint] - world[:, parent]
        length = float(np.median(np.linalg.norm(vectors, axis=-1)))
        region = 0 if source_time.ndim == 1 else JOINT_REGION[joint]
        direction = _slerp(vectors[lo[:, region]], vectors[hi[:, region]], frac[:, region])
        output[:, joint] = output[:, parent] + length * direction
    return output


def contacts_from_body(world: np.ndarray, fps: float = 30.) -> np.ndarray:
    feet = np.asarray(world)[:, FEET]
    speed = np.linalg.norm(np.diff(feet, axis=0) * fps, axis=-1)
    speed = np.maximum(np.r_[speed[:1], speed], np.r_[speed, speed[-1:]])
    height = feet[:, :, 1]
    return ((height >= -.04) & (height <= np.asarray((.12, .12, .055, .055))) & (speed <= .20)).astype(np.float32)


def _sample(timeline: MusicTimeline, time: np.ndarray, values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    flat = values.reshape(len(values), -1)
    result = np.column_stack([np.interp(time, timeline.time, flat[:, i]) for i in range(flat.shape[1])])
    return result.reshape((len(time),) + values.shape[1:])


def _relative(value: np.ndarray, quantile: float = .9) -> np.ndarray:
    value = np.maximum(np.asarray(value, dtype=np.float64), 0.)
    scale = np.quantile(value, quantile, axis=0, keepdims=True)
    return np.clip(value / np.maximum(scale, 1e-10), 0., 2.)


def _source_observations(timeline: MusicTimeline, time: np.ndarray):
    source = np.asarray(timeline.sources, dtype=np.float64)
    rms = _relative(source[:, :, 0])
    onset = _relative(source[:, :, 1])
    flux = _relative(source[:, :, 2])
    # All normalizers use the full recording, not the eventual display crop.
    sustained = np.minimum(_relative(source[:, :, 3]), rms) * np.exp(-onset-flux*.5)
    return tuple(_sample(timeline, time, value) for value in (rms, onset, flux, sustained))


def musical_evidence(timeline: MusicTimeline, time: np.ndarray) -> dict[str, np.ndarray]:
    """Transparent musical evidence shared by pacing and trajectory refinement.

    Functional labels are reported but never directly converted to a speed.
    Section boundaries contribute only in proportion to observed source/texture
    change. These are dramaturgy hypotheses, not validated semantic labels.
    """
    time = np.asarray(time, dtype=np.float64)
    dt = float(np.median(np.diff(time)))
    rms, onset, flux, sustained_sources = _source_observations(timeline, time)
    source_sum = rms.sum(axis=1)
    proportions = rms / np.maximum(source_sum[:, None], 1e-10)
    texture = -np.sum(proportions * np.log(np.maximum(proportions, 1e-10)), axis=1) / np.log(4.)
    texture *= np.clip(source_sum / 2., 0., 1.)
    smooth = gaussian_filter1d(texture, max(.2 / dt, 1.))
    past = np.r_[np.repeat(smooth[0], max(1, round(.8 / dt))), smooth][:len(time)]
    thinning = np.clip((past - smooth) * 3., 0., 1.)
    # Low transient density with continuing pitched/vocal sources supports a
    # sustained movement; a low waveform RMS alone is insufficient evidence.
    attack = np.clip(np.mean(onset[:, :2] + .35*flux[:, :2], axis=1), 0., 1.5)
    sustained = np.clip(np.mean(sustained_sources[:, 1:], axis=1), 0., 1.)
    preparation = np.zeros(len(time))
    release = np.zeros(len(time))
    confidence = np.ones(len(time))
    for section in timeline.sections:
        boundary = float(section["start"])
        if boundary <= time[0] or boundary >= time[-1]:
            continue
        raw_confidence = section.get("confidence")
        conf = .5 if raw_confidence is None else float(np.clip(raw_confidence, 0., 1.))
        if conf < .25:
            continue
        before = (time >= boundary-1.5) & (time < boundary-.15)
        after = (time >= boundary+.15) & (time < boundary+1.5)
        if not before.any() or not after.any():
            continue
        texture_change = abs(float(np.mean(texture[after])-np.mean(texture[before])))
        balance_change = float(np.mean(np.abs(np.mean(proportions[after], axis=0)-np.mean(proportions[before], axis=0))))
        event_change = abs(float(np.mean(attack[after])-np.mean(attack[before])))
        evidence = conf * min(1., 2.*texture_change + 3.*balance_change + .5*event_change)
        preparation = np.maximum(preparation, evidence * np.exp(-.5*((time-(boundary-.7))/.55)**2))
        release = np.maximum(release, evidence * np.exp(-.5*((time-(boundary+.25))/.3)**2))
    variation = float(np.max(np.std(rms, axis=0))) + float(np.max(np.std(onset, axis=0)))
    if variation < 1e-8:
        confidence[:] = 0.
    desired_rate = 1. + confidence * (.32*attack - .45*sustained*thinning - .42*preparation + .30*release)
    return {"attack": attack, "sustained": sustained, "thinning": thinning,
            "texture": texture, "phrase_preparation": preparation,
            "phrase_release": release, "confidence": confidence,
            "desired_rate": desired_rate}


def _result(world: np.ndarray, result_world: np.ndarray, source_time: np.ndarray,
            timeline: MusicTimeline, method: str, fps: float, metadata: dict[str, Any],
            anchors=None, events=None, desired_rate=None):
    time = np.arange(len(world), dtype=np.float64)/fps
    actual = np.diff(source_time, axis=0)*fps
    actual = np.concatenate((actual, actual[-1:]), axis=0)
    if actual.ndim == 1:
        actual = actual[:, None]
    common = {"method": method, "fps": fps, "regions": list(REGIONS),
              "motion_origin": "provided same-song generated body", "independent_motion_source": False,
              "source_time_unit": "seconds", "source_order_preserved": method != "D4", **metadata}
    if desired_rate is not None:
        common["requested_rate_mean"] = float(np.mean(desired_rate))
        common["duration_conservation_rate_adjustment_mean_abs"] = float(np.mean(np.abs(actual[:, 0]-desired_rate)))
    plan = DancePlan(time=time, source_time=source_time, effort=actual.astype(np.float32),
                     anchors=[] if anchors is None else anchors, events=[] if events is None else events,
                     metadata=common)
    result = DanceResult(time=time, world=result_world.astype(np.float32),
                         contacts=contacts_from_body(result_world, fps), source_time=source_time,
                         metadata={**common, "source_audio": timeline.metadata.get("audio_path", timeline.metadata.get("audio", "unspecified")),
                                   "source_contact_fraction": float(contacts_from_body(world, fps).mean()),
                                   "output_contact_fraction": float(contacts_from_body(result_world, fps).mean())})
    return result, plan


def result_from_source_time(world: np.ndarray, source_time: np.ndarray,
                            timeline: MusicTimeline, method: str,
                            metadata: dict | None = None, fps: float = 30.):
    world = _check_body(world)
    output = resample_body(world, source_time, fps)
    return _result(world, output, source_time, timeline, method, fps, metadata or {})


def _control_rates(desired: np.ndarray, timeline: MusicTimeline, time: np.ndarray,
                    control: str | None) -> np.ndarray:
    control = {"identity": "native", "rms_only": "loudness", "shifted_schedule": "shuffle"}.get(control, control)
    if control is None:
        return desired
    if control in ("native", "frozen_music"):
        return np.ones_like(desired)
    if control == "shuffle":
        # Deterministic misplaced schedule; duration/rate histogram unchanged.
        return np.roll(desired, max(1, len(desired)//3))
    if control == "loudness":
        rms = _sample(timeline, time, _relative(timeline.mixture[:, 0]))
        return .6 + .7*rms
    raise ValueError(f"Unknown diagnostic control: {control}")


def d1_event_retiming(world: np.ndarray, timeline: MusicTimeline, fps: float = 30.,
                       control: str | None = None):
    """Match sparse pre-existing gesture-speed peaks to supported audio attacks."""
    world = _check_body(world)
    time = np.arange(len(world))/fps
    evidence = musical_evidence(timeline, time)
    speed = np.mean(np.linalg.norm(np.gradient(world[:, (15, 18, 19, 20, 21)], axis=0)*fps, axis=-1), axis=1)
    speed = gaussian_filter1d(speed, .07*fps)
    motion_peaks = find_peaks(speed, distance=max(1, int(.8*fps)),
                              prominence=max(float(np.std(speed))*.15, 1e-8))[0]
    attack = gaussian_filter1d(evidence["attack"], .04*fps)
    audio_peaks = find_peaks(attack, distance=max(1, int(.5*fps)),
                             prominence=max(float(np.std(attack))*.2, 1e-8))[0]
    # Beats are usable only with measured attacks, not an autonomous metronome.
    beat_indices = np.clip(np.rint(np.asarray(timeline.beats)*fps).astype(int), 0, len(time)-1)
    beat_indices = beat_indices[(np.asarray(timeline.beats) >= 0) & (np.asarray(timeline.beats) <= time[-1])]
    strong_beats = beat_indices[attack[beat_indices] > max(np.quantile(attack, .65), 1e-6)]
    audio_indices = np.unique(np.r_[audio_peaks, strong_beats])
    anchors = []
    accepted = [(0., 0.)]
    for source_index in motion_peaks:
        if len(audio_indices) == 0:
            break
        source = float(time[source_index])
        eligible = audio_indices[np.abs(time[audio_indices]-source) <= .65]
        if len(eligible) == 0:
            continue
        # Audio-strength preference is small relative to temporal proximity.
        costs = np.abs(time[eligible]-source) - .07*attack[eligible]
        target = float(time[eligible[np.argmin(costs)]])
        previous_target, previous_source = accepted[-1]
        status = "accepted"
        if target <= previous_target + .8:
            status = "rejected_anchor_spacing"
        elif source < previous_source or (source-previous_source)/(target-previous_target) > 1.5:
            status = "infeasible_rate_from_previous_anchor"
        elif target >= time[-1] or (time[-1]-source)/(time[-1]-target) > 1.5:
            status = "infeasible_endpoint_rate"
        if status == "accepted":
            accepted.append((target, source))
        anchors.append({"time": target, "source_time": source, "kind": "gesture_speed_apex_to_audio_attack",
                        "status": status, "audio_attack": float(np.interp(target, time, attack))})
    accepted.append((float(time[-1]), float(time[-1])))
    pairs = np.asarray(accepted)
    initial_map = np.interp(time, pairs[:, 0], pairs[:, 1])
    desired = np.r_[np.diff(initial_map)*fps, (initial_map[-1]-initial_map[-2])*fps]
    desired = _control_rates(desired, timeline, time, control)
    mapping, _ = bounded_time_map(desired, fps, smooth_seconds=.12)
    for anchor in anchors:
        anchor["actual_source_time"] = float(np.interp(anchor["time"], time, mapping))
        anchor["residual_seconds"] = anchor["actual_source_time"]-anchor["source_time"]
        if anchor["status"] == "accepted" and abs(anchor["residual_seconds"]) > .08:
            anchor["status"] = "unmet_after_rate_smoothing"
    metadata = {"intervention": "sparse event-anchored global retiming", "control": control,
                "accepted_anchors": sum(a["status"] == "accepted" for a in anchors),
                "infeasible_or_unmet_anchors": sum(a["status"] != "accepted" for a in anchors),
                "unselected_syncopation": "unchanged source ordering between sparse anchors"}
    return _result(world, resample_body(world, mapping, fps), mapping, timeline, "D1", fps,
                    metadata, anchors=anchors, desired_rate=desired)


def d2_phrase_retiming(world: np.ndarray, timeline: MusicTimeline, fps: float = 30.,
                       control: str | None = None):
    world = _check_body(world)
    time = np.arange(len(world))/fps
    evidence = musical_evidence(timeline, time)
    desired = _control_rates(evidence["desired_rate"], timeline, time, control)
    mapping, actual = bounded_time_map(desired, fps, smooth_seconds=.25)
    events = []
    for key in ("phrase_preparation", "phrase_release", "thinning"):
        peaks = find_peaks(evidence[key], height=.1, distance=max(1, int(.5*fps)))[0]
        for index in peaks:
            events.append({"time": float(time[index]), "kind": key, "evidence": float(evidence[key][index]),
                           "desired_rate": float(desired[index]), "actual_rate": float(actual[index])})
    return _result(world, resample_body(world, mapping, fps), mapping, timeline, "D2", fps,
                    {"intervention": "source texture and phrase preparation/release retiming", "control": control,
                     "labels_directly_set_speed": False, "events_are_hypotheses": True},
                    events=events, desired_rate=desired)


def _regional_mean_roundoff_bound(common_time: np.ndarray, start: int, end: int,
                                  fps: float) -> float:
    """Forward-error bound for a rate recovered from a bounded float64 clock.

    ``bounded_time_map`` accumulates rates <=1.5 and divides prefixes by fps.
    Interior prefix differences cancel earlier additions, leaving at most
    ``end-start`` additions, each rounded by half an accumulator ULP. Its final
    timestamp is fixed explicitly: there we must budget the whole earlier
    prefix plus the whole duration-projection sum, rather than cancellation.
    Endpoint division and the recovered-rate arithmetic are bounded separately.
    This is numerical uncertainty in an existing clock, not a larger rate cap.
    """
    frames = len(common_time)-1
    span = end-start
    prefix_steps = span if end < frames else start+frames
    accumulator_ulp = np.spacing(1.5*max(end,1))
    prefix_error = .5*prefix_steps*accumulator_ulp
    endpoint_error = .5*(abs(np.spacing(common_time[start])) +
                           abs(np.spacing(common_time[end])))*abs(fps)
    arithmetic_error = 4.*np.spacing(1.5)
    return float((prefix_error+endpoint_error)/span+arithmetic_error)


def _project_regional_rates(rates: np.ndarray, mean: float, common_time: np.ndarray,
                             start: int, end: int, fps: float) -> np.ndarray:
    """Keep successful projections identical; retry only rejected clock roundoff."""
    try:
        return _project_rates(rates, mean)
    except ValueError as error:
        if str(error) != "Required duration cannot be represented within rate limits":
            raise
        corrected_mean = float(np.clip(mean,0.,1.5))
        error_bound = _regional_mean_roundoff_bound(common_time,start,end,fps)
        if not np.isfinite(mean) or abs(mean-corrected_mean) > error_bound:
            raise
        record = {"event": "regional_clock_roundoff_retry", "start": int(start),
                  "end": int(end), "rejected_mean": float(mean),
                  "corrected_mean": corrected_mean, "analytic_ulp_bound": error_bound,
                  "mean_correction": float(corrected_mean-mean)}
        warnings.warn(json.dumps(record,sort_keys=True), RuntimeWarning,stacklevel=2)
        return _project_rates(rates,corrected_mean)


def regional_time_maps(desired_rates: np.ndarray, common_time: np.ndarray, fps: float = 30.,
                       anchor_indices: np.ndarray | None = None, max_offset_seconds: float = .35):
    """Bounded regional clocks coincide with support at every shared anchor."""
    desired = np.asarray(desired_rates, dtype=np.float64)
    common_time = np.asarray(common_time, dtype=np.float64)
    if desired.shape != (len(common_time), 5):
        raise ValueError("Expected five regional rate proposals")
    anchors = np.unique(np.r_[0, len(common_time)-1, [] if anchor_indices is None else anchor_indices]).astype(int)
    if anchors.min() < 0 or anchors.max() >= len(common_time):
        raise ValueError("Anchor outside source interval")
    mapping = np.repeat(common_time[:, None], 5, axis=1)
    for region in range(1, 5):
        proposal = gaussian_filter1d(desired[:, region], .12*fps, mode="nearest")
        for start, end in zip(anchors[:-1], anchors[1:]):
            mean = (common_time[end]-common_time[start])*fps/(end-start)
            projected = _project_regional_rates(proposal[start:end], mean, common_time, start, end, fps)
            mapping[start+1:end+1, region] = common_time[start] + np.cumsum(projected)/fps
        offset = mapping[:, region]-common_time
        alpha = min(1., max_offset_seconds/max(float(np.max(np.abs(offset))), 1e-12))
        mapping[:, region] = common_time + alpha*offset
    mapping[anchors] = common_time[anchors, None]
    return mapping


def d3_source_regional_retiming(world: np.ndarray, timeline: MusicTimeline, fps: float = 30.,
                                control: str | None = None):
    world = _check_body(world)
    time = np.arange(len(world))/fps
    rms, onset, flux, sustained = _source_observations(timeline, time)
    # Drums and bass govern the common support clock. Vocals and other-source
    # edits cannot move support or the independently assigned contralateral arm.
    source_rates = 1. + .40*np.clip(onset+.3*flux, 0., 1.5) - .25*np.clip(sustained, 0., 1.)
    common_proposal = np.mean(source_rates[:, :2], axis=1)
    common_proposal = _control_rates(common_proposal, timeline, time, control)
    common, common_rate = bounded_time_map(common_proposal, fps, smooth_seconds=.20)
    desired = np.column_stack((common_proposal, source_rates[:, 1], source_rates[:, 3], source_rates[:, 3], source_rates[:, 2]))
    if control is not None:
        desired = np.column_stack([_control_rates(desired[:, j], timeline, time, control) for j in range(5)])
    contact = contacts_from_body(world, fps)
    transitions = np.flatnonzero(np.any(contact[1:] != contact[:-1], axis=1))+1
    # Support changes are sampled in output time by the inverse common map.
    support_times = np.interp(transitions/fps, common, time)
    anchor_times = np.r_[np.asarray(timeline.downbeats), support_times,
                         [float(s["start"]) for s in timeline.sections if s.get("confidence") is None or float(s["confidence"]) >= .25]]
    anchor_times = anchor_times[(anchor_times >= 0) & (anchor_times <= time[-1])]
    anchors = np.unique(np.r_[0, np.rint(anchor_times*fps).astype(int), len(time)-1])
    mapping = regional_time_maps(desired, common, fps, anchors)
    details = [{"time": float(time[i]), "source_time": float(common[i]),
                "kind": "shared_support_or_phrase_anchor"} for i in anchors]
    return _result(world, resample_body(world, mapping, fps), mapping, timeline, "D3", fps,
                    {"intervention": "source-specific upper-body timing with shared support", "control": control,
                     "source_to_region_hypothesis": SOURCE_REGIONS, "shared_support_joints": [0,1,2,4,5,7,8,10,11],
                     "maximum_regional_offset_seconds": float(np.max(np.abs(mapping-common[:, None]))),
                     "regional_clock_anchor_count": len(anchors)}, anchors=details,
                    desired_rate=common_proposal)


def _blend_bodies(a: np.ndarray, b: np.ndarray, weight: np.ndarray) -> np.ndarray:
    weight = np.asarray(weight, dtype=np.float64)
    result = np.empty_like(a)
    result[:, 0] = (1.-weight[:, None])*a[:, 0] + weight[:, None]*b[:, 0]
    for joint in range(1, 22):
        parent = PARENTS[joint]
        av, bv = a[:, joint]-a[:, parent], b[:, joint]-b[:, parent]
        length = np.linalg.norm(av, axis=-1)
        result[:, joint] = result[:, parent] + _slerp(av, bv, weight)*length[:, None]
    return result


def d4_motif_recomposition(world: np.ndarray, timeline: MusicTimeline, fps: float = 30.,
                           control: str | None = None):
    """Recur own-song motion fragments through a contact-compatible pose graph.

    This is explicitly recomposition, not a monotone time warp. Fragment-local
    rates remain <=1.5; source jumps are disclosed and blended with directions
    from the adjacent selected source fragments. No foreign trajectory is used.
    """
    control = {"identity": "native", "rms_only": "loudness", "shifted_schedule": "shuffle"}.get(control, control)
    world = _check_body(world)
    time = np.arange(len(world))/fps
    identity = time.copy()
    contact = contacts_from_body(world, fps)
    sections = timeline.sections
    recurrence = np.asarray(timeline.recurrence, dtype=np.float64)
    if recurrence.shape != (len(sections), len(sections)):
        raise ValueError("Section recurrence matrix shape mismatch")
    metadata = {"intervention": "same-song contact-compatible recurrence fragment graph",
                "control": control, "source_order_preserved": False,
                "revision": "d4_floor_v2", "translation_axes": "XZ",
                "time_map_classification": "recomposition; jumps are not playback rates",
                "fragment_provenance": [], "rejected_transitions": [],
                "transition_seconds": .30, "independent_motion_source": False}
    if control in ("native", "frozen_music", "loudness") or _constant(world):
        metadata["diagnostic_reason"] = "constant source or requested identity comparator"
        return _result(world, world.copy(), identity, timeline, "D4", fps, metadata)
    if control not in (None, "shuffle"):
        raise ValueError(f"Unknown diagnostic control: {control}")
    beats = np.asarray(timeline.downbeats, dtype=np.float64)
    beats = beats[(beats > 0) & (beats < time[-1])]
    # Downbeat-to-downbeat nodes select only the supplied original body.
    boundaries = [0.]
    for beat in beats:
        if beat-boundaries[-1] >= 1.:
            boundaries.append(float(beat))
    boundaries += [time[-1]]
    if len(boundaries) == 2:
        metadata["diagnostic_reason"] = "no downbeat fragments; no invented rhythmic partition"
        return _result(world, world.copy(), identity, timeline, "D4", fps, metadata)
    frames = np.unique(np.rint(np.asarray(boundaries)*fps).astype(int))
    relative = world-world[:, :1]
    nodes = []
    for start, end in zip(frames[:-1], frames[1:]):
        midpoint = (time[start]+time[end])/2
        section_id = next((j for j,s in enumerate(sections) if float(s["start"]) <= midpoint < float(s["end"])), -1)
        nodes.append({"start": int(start), "end": int(end), "section": section_id})
    output = resample_body(world, identity, fps)
    source_time = identity.copy()
    selected = []
    changed = 0
    for index, node in enumerate(nodes):
        start, end, section_id = node["start"], node["end"], node["section"]
        chosen, confidence, why = index, None, "chronological"
        if 0 < index < len(nodes)-1 and section_id >= 0:
            section = sections[section_id]
            section_confidence = section.get("confidence")
            if section_confidence is None or float(section_confidence) >= .25:
                previous_sections = [j for j in range(section_id) if recurrence[section_id,j] >= .80]
                if control == "shuffle" and previous_sections:
                    previous_sections = [(j+max(1,len(sections)//2)) % len(sections) for j in previous_sections]
                fraction = (time[start]-float(section["start"]))/max(float(section["end"])-float(section["start"]), 1e-6)
                candidates = []
                for donor_index, donor in enumerate(nodes):
                    donor_section = donor["section"]
                    if donor_section not in previous_sections:
                        continue
                    ds, de = donor["start"], donor["end"]
                    rate = (de-ds)/(end-start)
                    if not .5 <= rate <= 1.5:
                        continue
                    # Both local support pattern and graph-edge contact must
                    # match; a rejected transition is retained as a diagnostic.
                    prev = nodes[selected[-1]]["end"] if selected else start
                    compatible = np.array_equal(contact[ds], contact[prev]) and np.array_equal(contact[de], contact[end])
                    pose_cost = float(np.sqrt(np.mean((relative[ds]-relative[prev])**2)))
                    if not compatible or pose_cost > .25:
                        metadata["rejected_transitions"].append({"target_fragment": index, "source_fragment": donor_index,
                                                                  "reason": "contact_mismatch" if not compatible else "pose_gap",
                                                                  "pose_rms_m": pose_cost})
                        continue
                    donor_s = sections[donor_section]
                    donor_fraction = (time[ds]-float(donor_s["start"]))/max(float(donor_s["end"])-float(donor_s["start"]), 1e-6)
                    cost = abs(donor_fraction-fraction)+pose_cost+abs(rate-1.)*.1
                    candidates.append((cost, donor_index, float(recurrence[section_id,donor_section])))
                if candidates:
                    _, chosen, confidence = min(candidates)
                    why = "recurring_section_contact_compatible_motif"
        selected.append(chosen)
        donor = nodes[chosen]
        ds, de = donor["start"], donor["end"]
        local_source = np.linspace(time[ds], time[de], end-start+1)
        segment = resample_body(world, local_source, fps)
        prior_translation = np.zeros(3)
        if start > 0:
            # Preserve the donor's world Y and floor. Align horizontal travel
            # only: accumulating crouch/standing height differences across
            # fragments would lift or sink every subsequent planted foot.
            translation = output[start, 0]-segment[0, 0]
            translation[1] = 0.
            segment += translation
            fade = min(round(.30*fps), len(segment)-1)
            if fade > 0:
                prev_node = nodes[selected[-2]]
                continuation = time[prev_node["end"]]+np.arange(fade+1)/fps
                continuation = np.clip(continuation, 0, time[-1])
                prior = resample_body(world, continuation, fps)
                prior_translation = output[start,0]-prior[0,0]
                prior_translation[1] = 0.
                prior += prior_translation
                weight = np.linspace(0., 1., fade+1)
                weight = weight*weight*(3.-2.*weight)
                segment[:fade+1] = _blend_bodies(prior, segment[:fade+1], weight)
        else:
            translation = np.zeros(3)
        output[start:end+1] = segment
        source_time[start:end+1] = local_source
        changed += int(chosen != index)
        metadata["fragment_provenance"].append({"target_fragment": index, "source_fragment": chosen,
                "target_seconds": [float(time[start]),float(time[end])], "source_seconds": [float(time[ds]),float(time[de])],
                "local_rate": float((de-ds)/(end-start)), "source_section": donor["section"],
                "target_section": section_id, "recurrence_similarity": confidence, "selection_reason": why,
                "root_translation_m": translation.tolist(),
                "prior_continuation_translation_m": prior_translation.tolist(),
                "transition_blend_from_fragment": selected[-2] if len(selected)>1 else None})
    metadata["recomposed_fragment_count"] = changed
    metadata["graph_node_count"] = len(nodes)
    metadata["source_time_jumps_are_fragment_boundaries"] = True
    # With no feasible recurrence preserve the untouched comparator exactly;
    # smoothing chronological transitions must not masquerade as recomposition.
    if changed == 0:
        output, source_time = world.copy(), identity
        metadata["diagnostic_reason"] = "no feasible recurring motif; exact baseline diagnostic"
    result, plan = _result(world, output, source_time, timeline, "D4", fps, metadata)
    # Finite differences of a jumping provenance clock are not physical speed.
    effort = np.ones((len(time),1), dtype=np.float32)
    for row in metadata["fragment_provenance"]:
        start, end = np.rint(np.asarray(row["target_seconds"])*fps).astype(int)
        effort[start:end+1,0] = row["local_rate"]
    plan.effort = effort
    return result, plan


METHODS = {"D1": d1_event_retiming, "D2": d2_phrase_retiming,
           "D3": d3_source_regional_retiming, "D4": d4_motif_recomposition}


def apply_method(identifier: str, world: np.ndarray, timeline: MusicTimeline,
                  fps: float = 30., control: str | None = None):
    """Common entry point returning (DanceResult, DancePlan)."""
    return METHODS[identifier.upper()](world, timeline, fps=fps, control=control)


def generate(method_id: str, body_world: np.ndarray, timeline: MusicTimeline,
             times: np.ndarray | None = None, control: str | None = None):
    """Uniform campaign entry point; source bodies must start at t=0."""
    fps = 30.
    if times is not None:
        times = np.asarray(times, dtype=np.float64)
        if times.shape != (len(body_world),) or not np.isclose(times[0], 0.):
            raise ValueError("Body timestamps must start at zero and match body length")
        intervals = np.diff(times)
        if not np.allclose(intervals, intervals[0], rtol=1e-6, atol=1e-8) or intervals[0] <= 0:
            raise ValueError("Body timestamps must be uniformly sampled")
        fps = 1./intervals[0]
    return apply_method(method_id, body_world, timeline, fps, control)


def retime_with_effort(world: np.ndarray, times: np.ndarray, effort: np.ndarray,
                      timeline: MusicTimeline, method: str = "D6", metadata: dict | None = None):
    """Convert learned global or five-region rate proposals to a valid body.

    Caller converts learned speed/duration predictions to rate proposals and
    supplies that provenance. This function adds no unlearned event schedule.
    """
    world = _check_body(world)
    times = np.asarray(times, dtype=np.float64)
    effort = np.asarray(effort, dtype=np.float64)
    if times.shape != (len(world),) or not np.isclose(times[0], 0.):
        raise ValueError("Expected body timestamps beginning at zero")
    fps = 1./float(np.median(np.diff(times)))
    if not np.allclose(np.diff(times), 1./fps, atol=1e-8):
        raise ValueError("Expected uniform native body timestamps")
    if effort.ndim == 1 or effort.shape == (len(world),1):
        desired = effort.reshape(-1)
        mapping, _ = bounded_time_map(desired, fps)
    elif effort.shape == (len(world),5):
        desired = effort[:,0]
        common, _ = bounded_time_map(desired, fps)
        contacts = contacts_from_body(world, fps)
        transitions = np.flatnonzero(np.any(contacts[1:] != contacts[:-1],axis=1))+1
        support_times = np.interp(transitions/fps, common, times)
        anchors = np.r_[support_times, np.asarray(timeline.downbeats)]
        anchors = anchors[(anchors >= 0) & (anchors <= times[-1])]
        mapping = regional_time_maps(effort, common, fps, np.rint(anchors*fps).astype(int))
    else:
        raise ValueError("Expected learned global [T] or regional [T,5] rate proposals")
    if len(desired) != len(world):
        raise ValueError("Effort length does not match body")
    return _result(world, resample_body(world,mapping,fps), mapping,timeline,method,fps,
                   {"intervention": "learned effort contact-aware retiming", **(metadata or {})}, desired_rate=desired)
