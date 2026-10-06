"""CPU checks of fixed-time D5, including actual constrained emitted commands."""
import json

import numpy as np
import pytest
import torch

from lightsem.conductor.body_transfer_v2 import make_outputs, map_body
from lightsem.conductor.body_transfer_v2.common import REFERENCE
from lightsem.conductor.body_transfer_v2.skeleton import PARENTS
from lightsem.experiments.beat_conditioned.beat_conditioned_render import _angles
from lightsem.experiments.dynamic_dance.refinement import (
    FIXED, RefinementConfig, _torch_requested_angles, build_refinement_targets, evaluate_actual_transfer,
    refine_trajectory,
)


def _body(frames=91):
    """A single smooth arm reach and return; synthetic unit-test input only."""
    body = np.broadcast_to(REFERENCE, (frames, 22, 3)).copy()
    t = np.linspace(0, 1, frames)
    reach = 64*t**3*(1-t)**3
    for shoulder, descendants, amplitude in ((16, [18, 20], .65), (17, [19, 21], -.3)):
        angle = reach*amplitude
        vec = body[:, descendants]-body[:, shoulder:shoulder+1]
        c, s = np.cos(angle)[:, None], np.sin(angle)[:, None]
        rotated = np.stack((c*vec[..., 0]-s*vec[..., 1],
                            s*vec[..., 0]+c*vec[..., 1], vec[..., 2]), -1)
        body[:, descendants] = body[:, shoulder:shoulder+1] + rotated
    return body


def _targets(frames, reverse=False):
    phase = np.linspace(0, 1, frames)
    effort = .65+.65*phase
    if reverse:
        effort = effort[::-1].copy()
    return {"effort": effort, "expansion": 1+.08*(effort-1),
            "settling": np.clip(1-effort, 0, 1)}


def test_transfer_surrogate_matches_fixed_v3_mapper_for_extended_rig():
    world = _body(25)
    outputs = make_outputs(beams=11, lasers=8, bulbs=2)
    actual = map_body(world, outputs, "01", fps=30.)
    moving = [i for i, o in enumerate(outputs) if o["kind"] != "bulb"]
    vector = actual["targets"][:, moving]-np.asarray([outputs[i]["position"] for i in moving])
    expected = np.stack(_angles(vector), -1)
    tensor = torch.tensor(world, dtype=torch.float64, requires_grad=True)
    calculated = _torch_requested_angles(tensor, outputs)
    np.testing.assert_allclose(calculated.detach().numpy(), expected, atol=1e-10, rtol=1e-10)
    calculated.square().mean().backward()
    assert torch.isfinite(tensor.grad).all()
    assert torch.linalg.vector_norm(tensor.grad) > 0


def test_actual_emission_probe_preserves_bones_during_30_to_40_hz_conversion(monkeypatch):
    import lightsem.conductor.body_transfer_v2 as transfer
    from lightsem.experiments.dynamic_dance.retiming import resample_body
    source = _body(13)
    source_time = 60.+np.arange(len(source))/30.
    seen = []
    original_map = transfer.map_body

    def inspect_geometry(world, outputs, algorithm_id, fps):
        seen.append(np.array(world))
        return original_map(world, outputs, algorithm_id, fps)

    monkeypatch.setattr(transfer, "map_body", inspect_geometry)
    trace = evaluate_actual_transfer(source, source_time)
    assert len(seen) == 1
    clock = np.minimum(trace["time"]-source_time[0], (len(source)-1)/30.)
    np.testing.assert_array_equal(seen[0], resample_body(source, clock, 30.))
    parent = np.asarray(PARENTS)[1:]
    expected = np.median(np.linalg.norm(source[:, 1:]-source[:, parent], axis=-1), axis=0)
    actual = np.linalg.norm(seen[0][:, 1:]-seen[0][:, parent], axis=-1)
    np.testing.assert_allclose(actual, np.broadcast_to(expected, actual.shape), atol=1e-12, rtol=0)
    assert trace["body_adapter"].startswith("shared retiming.resample_body")


def test_targets_use_shared_measured_source_phrase_evidence():
    from lightsem.experiments.dynamic_dance.contracts import MusicTimeline
    frames = 181
    time = np.arange(frames)/30.
    sources = np.zeros((frames, 4, 4))
    sources[:90, :, 0] = [.7, .1, .1, .1]
    sources[90:, :, 0] = [.1, .1, .1, .7]
    sources[:, :, 3] = sources[:, :, 0]
    sources[88:94, :, 1:3] = 1.
    timeline = MusicTimeline(time=time, muq=np.zeros((10, 1024)),
        muq_time=np.arange(10)/25., sources=sources, mixture=np.ones((frames, 4)),
        beats=np.arange(0., 6., .5), downbeats=np.array([0., 2., 4.]),
        sections=[{"start": 0., "end": 3., "label": "A", "confidence": None},
                  {"start": 3., "end": 6.1, "label": "B", "confidence": None}],
        recurrence=np.eye(2)).validate()
    targets = build_refinement_targets(timeline, time)
    assert np.ptp(targets["effort"]) > .1
    assert np.all((targets["effort"] >= .35) & (targets["effort"] <= 1.45))
    assert np.all((targets["expansion"] >= .9) & (targets["expansion"] <= 1.1))
    assert "phrase_preparation" in targets["evidence"]


def test_frozen_source_cannot_be_animated_by_musical_targets():
    source = np.repeat(_body()[40:41], 61, axis=0)
    original = source.copy()
    result, plan = refine_trajectory(source, None, targets=_targets(len(source)),
                                    config=RefinementConfig(iterations=8))
    np.testing.assert_array_equal(result.world, source)
    np.testing.assert_array_equal(source, original)
    np.testing.assert_array_equal(result.source_time, result.time)
    np.testing.assert_array_equal(plan.source_time, plan.time)
    assert result.metadata["frozen_source"]
    assert result.metadata["transfer_after"]["emitted_median_speed_deg_s"] == 0
    assert result.metadata["qualification"] == "failure diagnostic"
    json.dumps(result.metadata, allow_nan=False)


def test_fixed_time_refinement_preserves_support_lengths_and_hard_motion_bounds():
    source = _body()
    original = source.copy()
    time = 4.+np.arange(len(source))/30.
    result, plan = refine_trajectory(source, None, timestamps=time,
        targets=_targets(len(source)), config=RefinementConfig(iterations=20))
    np.testing.assert_array_equal(source, original)
    np.testing.assert_array_equal(result.time, time)
    np.testing.assert_array_equal(result.source_time, time)
    np.testing.assert_array_equal(result.world[:, FIXED], source[:, FIXED])
    parent = np.asarray(PARENTS)[1:]
    before = np.linalg.norm(source[:, 1:]-source[:, parent], axis=-1)
    after = np.linalg.norm(result.world[:, 1:]-result.world[:, parent], axis=-1)
    np.testing.assert_allclose(after, before, atol=1e-12, rtol=0)
    report = result.metadata
    assert report["body_constraints"]["maximum_displacement_m"] <= .12+1e-9
    assert report["transfer_after"]["maximum_speed_deg_s"] <= 90.+1e-6
    assert report["transfer_after"]["maximum_acceleration_deg_s2"] <= 360.+1e-6
    assert np.max(np.abs(result.world-source)) > 1e-5
    assert report["objective_after"]["total"] < report["objective_before"]["total"]
    assert np.all(plan.effort[:, 3:] == 1.)
    json.dumps(report, allow_nan=False)


def test_changing_musical_schedule_changes_fixed_time_refinement():
    source = _body(61)
    config = RefinementConfig(iterations=15)
    first, _ = refine_trajectory(source, None, targets=_targets(len(source)), config=config)
    second, _ = refine_trajectory(source, None, targets=_targets(len(source), reverse=True), config=config)
    assert np.sqrt(np.mean((first.world-second.world)**2)) > 1e-4
    np.testing.assert_array_equal(first.time, second.time)
    np.testing.assert_array_equal(first.source_time, second.source_time)


def test_actual_emission_gate_rejects_a_worsening_candidate_without_substitution():
    source = _body(61)
    calls = []

    def emitted_evaluator(world, time, config):
        trace = evaluate_actual_transfer(world, time, config)
        calls.append(world.copy())
        if len(calls) == 2:
            trace["saturation_fraction"] = 1.
        return trace

    result, _ = refine_trajectory(source, None, targets=_targets(len(source)),
        config=RefinementConfig(iterations=15), transfer_evaluator=emitted_evaluator)
    assert len(calls) == 2
    assert not result.metadata["postgate_passed"]
    assert result.metadata["qualification"] == "failure diagnostic"
    assert any("saturation increased" in item for item in result.metadata["infeasibility"])
    np.testing.assert_array_equal(result.world, calls[1])
    assert not np.array_equal(result.world, source)


@pytest.mark.parametrize("problem", ["bone", "timestamps", "target", "nonfinite"])
def test_invalid_inputs_fail_explicitly(problem):
    source = _body(31)
    time = np.arange(31)/30.
    target = _targets(31)
    if problem == "bone":
        source[15, 20, 0] += .1
    elif problem == "timestamps":
        time[15] += .01
    elif problem == "target":
        target["effort"][15] = 2.
    else:
        source[15, 20, 0] = np.nan
    with pytest.raises(ValueError):
        refine_trajectory(source, None, timestamps=time, targets=target,
                          config=RefinementConfig(iterations=1))
