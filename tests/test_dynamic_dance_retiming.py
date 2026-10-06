"""Scientific safeguards for source-preserving dynamic dance D1--D4."""
import copy
import json

import numpy as np
import pytest

from lightsem.experiments.dynamic_dance.contracts import MusicTimeline
from lightsem.experiments.dynamic_dance.retiming import (
    METHODS, PARENTS, JOINT_REGION, bounded_time_map, contacts_from_body,
    generate, musical_evidence, regional_time_maps, resample_body,
    retime_with_effort,
)
from lightsem.conductor.body_transfer_v2.skeleton import OFFSETS


def source_body(seconds=16, fps=30):
    """Moving fixed-bone fixture; test oscillations are not a product fallback."""
    time = np.arange(seconds*fps)/fps
    body = np.zeros((len(time),22,3),dtype=np.float64)
    body[:,0,0] = .06*time
    body[:,0,1] = 1.
    for joint in range(1,22):
        bone = np.broadcast_to(OFFSETS[joint],(len(time),3)).copy()
        if joint in (13,14,16,17,18,19,20,21):
            angle = .35*np.sin(time*(1.4 + .13*int(joint%2))) + .10*np.sin(time*3.1)
            x,y = bone[:,0].copy(),bone[:,1].copy()
            bone[:,0] = np.cos(angle)*x-np.sin(angle)*y
            bone[:,1] = np.sin(angle)*x+np.cos(angle)*y
        body[:,joint] = body[:,PARENTS[joint]]+bone
    return body.astype(np.float32)


def music(seconds=16, fps=30):
    time = np.arange(seconds*fps)/fps
    sources = np.zeros((len(time),4,4),dtype=np.float64)
    # Identical global RMS may contain changing instrumental composition.
    for source in range(4):
        sources[:,source,0] = .45+.18*np.sin(time*.8+source)
        sources[:,source,1] = sum(np.exp(-((time-(event+source*.13))/.09)**2)
                                      for event in np.arange(.5,seconds,.9))
        sources[:,source,2] = sources[:,source,1]*.7
        sources[:,source,3] = sources[:,source,0]
    sources[(time>5)&(time<8),0:2,0] *= .10
    sources[(time>5)&(time<8),0:2,1:3] *= .1
    mixture = np.zeros((len(time),4),dtype=np.float64)
    mixture[:,0] = .5
    mixture[:,1] = sources[:,:,1].sum(1)
    mixture[:,2] = 1000.
    mixture[:,3] = sources[:,:,2].sum(1)
    sections = [{"start":float(s),"end":float(min(s+4,seconds)),"label":"section_"+str(s//4),"confidence":.9}
                for s in range(0,seconds,4)]
    recurrence = np.eye(len(sections))
    if len(sections)>2:
        recurrence[0,2] = recurrence[2,0] = .97
    result = MusicTimeline(time=time,muq=np.zeros((len(time),1024),np.float32),muq_time=time,
            sources=sources,mixture=mixture,beats=np.arange(.5,seconds,.5),downbeats=np.arange(0,seconds,2.),
            sections=sections,recurrence=recurrence,metadata={"audio_path":"synthetic_test_song"})
    return result.validate()


def lengths(world):
    return np.linalg.norm(world[:,1:]-world[:,PARENTS[1:]],axis=-1)


def planted_crouch_body():
    """An initial 20 cm crouch-to-stand with fixed feet and constant bone lengths.

    Later music requests the initial fragment again. Aligning its root in XYZ
    incorrectly raises all later feet 20 cm; XZ alignment retains its floor.
    """
    body = source_body().astype(np.float64)
    time = np.arange(len(body))/30.
    progress = np.clip(time/2.,0.,1.)
    progress = progress*progress*(3.-2.*progress)
    body -= body[:,:1]
    root = np.zeros((len(body),3))
    root[:,1] = .75+.20*progress
    body += root[:,None]
    for hip,knee,ankle,toe in ((1,4,7,10),(2,5,8,11)):
        ankle_position = body[:,hip].copy()
        ankle_position[:,1] = .10
        difference = ankle_position-body[:,hip]
        distance = np.linalg.norm(difference,axis=1)
        upper = np.linalg.norm(OFFSETS[knee])
        lower = np.linalg.norm(OFFSETS[ankle])
        along = (upper*upper-lower*lower+distance*distance)/(2.*distance)
        perpendicular = np.sqrt(upper*upper-along*along)
        body[:,knee] = body[:,hip]+difference*(along/distance)[:,None]
        body[:,knee,2] += perpendicular
        body[:,ankle] = ankle_position
        body[:,toe] = ankle_position+OFFSETS[toe]
    return body.astype(np.float32)


@pytest.mark.parametrize("proposal", [np.r_[np.zeros(200),np.ones(280)*100], np.ones(480)*1e-9,
                                      np.r_[np.ones(400)*-100,np.ones(80)*100]])
def test_bounded_rates_conserve_entire_source(proposal):
    mapping, rates = bounded_time_map(proposal)
    assert mapping[0] == 0.
    assert mapping[-1] == (len(proposal)-1)/30
    assert np.diff(mapping).min() >= -1e-12
    assert np.diff(mapping).max()*30 <= 1.5+1e-10
    assert rates.min() >= 0 and rates.max() <= 1.5
    np.testing.assert_allclose(np.diff(mapping)*30,rates[:-1],atol=1e-10)


def test_retiming_invalid_input_rejected():
    with pytest.raises(ValueError):
        bounded_time_map(np.array([1.,np.nan,1.]))
    with pytest.raises(ValueError):
        bounded_time_map(np.ones(300),end_time=100)
    with pytest.raises(ValueError):
        resample_body(source_body(),np.array([-1.,0.,1.]))
    with pytest.raises(ValueError):
        generate("D1",source_body(),music(),times=np.arange(480)+1)


def test_spherical_sampling_has_fixed_lengths_including_antipodal_bones():
    body = source_body()
    # Antipodal wrists exercise the deterministic great-circle fallback.
    for joint in (20,21):
        body[1,joint] = 2*body[1,PARENTS[joint]]-body[0,joint]
    output = resample_body(body,np.linspace(0,(len(body)-1)/30,801))
    assert np.isfinite(output).all()
    np.testing.assert_allclose(lengths(output),np.broadcast_to(np.median(lengths(body),axis=0),lengths(output).shape),atol=2e-7)


@pytest.mark.parametrize("method",METHODS)
def test_frozen_body_never_creates_autonomous_motion(method):
    body = np.broadcast_to(source_body()[120:121],(480,22,3)).copy()
    result,plan = generate(method,body,music())
    np.testing.assert_array_equal(result.world,np.broadcast_to(result.world[:1],result.world.shape))
    np.testing.assert_array_equal(result.world,body)
    result.validate()
    plan.validate()
    json.dumps(result.metadata,allow_nan=False)


@pytest.mark.parametrize("method",("D1","D2","D3"))
def test_pure_retiming_geometry_endpoint_and_rates(method):
    body = source_body()
    result,plan = generate(method,body,music())
    mapping = result.source_time
    np.testing.assert_allclose(mapping[0],0,atol=1e-12)
    np.testing.assert_allclose(mapping[-1],479/30,atol=1e-12)
    assert np.diff(mapping,axis=0).min() >= -1e-10
    assert (np.diff(mapping,axis=0)*30).max() <= 1.5+1e-9
    np.testing.assert_allclose(lengths(result.world),np.broadcast_to(np.median(lengths(body),axis=0),lengths(body).shape),atol=3e-7)
    result.validate()
    plan.validate()


@pytest.mark.parametrize("method",METHODS)
def test_identity_control_preserves_source(method):
    body = source_body()
    result,_ = generate(method,body,music(),control="identity")
    np.testing.assert_allclose(result.world,body,atol=4e-7)


def test_event_retiming_responds_to_changed_attack_positions():
    body,one = source_body(),music()
    two = copy.deepcopy(one)
    two.sources[:,:,1:3] = np.roll(two.sources[:,:,1:3],11,axis=0)
    first,plan = generate("D1",body,one)
    second,_ = generate("D1",body,two)
    assert plan.anchors
    assert np.max(np.abs(first.source_time-second.source_time)) > .01
    for anchor in plan.anchors:
        assert "status" in anchor and "residual_seconds" in anchor


def test_phrase_timing_uses_source_texture_not_only_loudness_or_label():
    body,one = source_body(),music()
    two = copy.deepcopy(one)
    two.sources[:,0:2] = np.roll(two.sources[:,0:2],75,axis=0)
    np.testing.assert_array_equal(one.mixture[:,0],two.mixture[:,0])
    first,_ = generate("D2",body,one)
    second,_ = generate("D2",body,two)
    assert np.max(np.abs(first.source_time-second.source_time)) > .015
    labels = copy.deepcopy(one)
    for i,section in enumerate(labels.sections):
        section["label"] = "chorus" if i%2 else "verse"
    relabeled,_ = generate("D2",body,labels)
    np.testing.assert_array_equal(first.source_time,relabeled.source_time)


def test_global_audio_gain_does_not_change_non_loudness_schedule():
    body,one = source_body(),music()
    louder = copy.deepcopy(one)
    louder.sources *= 2.7
    louder.mixture *= 2.7
    for method in ("D1","D2","D3"):
        first,_ = generate(method,body,one)
        second,_ = generate(method,body,louder)
        np.testing.assert_allclose(first.source_time,second.source_time,atol=2e-10)


def test_frozen_music_observations_do_not_schedule_phrases():
    timeline = music()
    timeline.sources[:] = timeline.sources[0]
    evidence = musical_evidence(timeline,timeline.time)
    np.testing.assert_array_equal(evidence["desired_rate"],np.ones(len(timeline.time)))
    result,_ = generate("D2",source_body(),timeline)
    np.testing.assert_allclose(result.source_time,timeline.time,atol=1e-12)


def test_low_confidence_boundaries_do_not_schedule_phrase_impulses():
    timeline = music()
    for section in timeline.sections:
        section["confidence"] = .1
    evidence = musical_evidence(timeline,timeline.time)
    assert not np.any(evidence["phrase_preparation"])
    assert not np.any(evidence["phrase_release"])


def test_regional_clocks_preserve_support_and_obey_source_assignment():
    body,one = source_body(),music()
    two = copy.deepcopy(one)
    two.sources[:,3] = np.roll(two.sources[:,3],19,axis=0)
    first,plan = generate("D3",body,one)
    second,_ = generate("D3",body,two)
    np.testing.assert_array_equal(first.source_time[:,0],second.source_time[:,0])
    np.testing.assert_array_equal(first.source_time[:,4],second.source_time[:,4])
    assert np.max(np.abs(first.source_time[:,3]-second.source_time[:,3])) > .005
    # Actual reconstructed lower-body geometry comes from precisely one clock.
    common = resample_body(body,first.source_time[:,0])
    support_joints = np.flatnonzero(JOINT_REGION == 0)
    np.testing.assert_allclose(first.world[:,support_joints],common[:,support_joints],atol=2e-7)
    anchors = np.rint(np.asarray([a["time"] for a in plan.anchors])*30).astype(int)
    np.testing.assert_allclose(first.source_time[anchors],np.repeat(first.source_time[anchors,:1],5,axis=1),atol=1e-12)
    np.testing.assert_array_equal(first.world[:,21],second.world[:,21])


def test_regional_clock_projection_handles_extreme_proposals():
    n = 480
    common,_ = bounded_time_map(np.linspace(0,2,n))
    desired = np.tile(np.linspace(-2,3,n)[:,None],(1,5))
    desired[:,2] = desired[::-1,2]
    anchors = np.array([0,40,80,200,479])
    mapping = regional_time_maps(desired,common,anchor_indices=anchors)
    assert np.diff(mapping,axis=0).min() >= -1e-10
    assert np.diff(mapping,axis=0).max()*30 <= 1.5+1e-10
    assert np.max(np.abs(mapping-common[:,None])) <= .35+1e-10
    np.testing.assert_array_equal(mapping[anchors],np.repeat(common[anchors,None],5,axis=1))


def test_motifs_use_own_source_and_report_reordered_provenance():
    body,timeline = source_body(),music()
    result,plan = generate("D4",body,timeline)
    assert result.metadata["recomposed_fragment_count"] > 0
    assert result.metadata["source_order_preserved"] is False
    assert np.any(np.diff(result.source_time) < 0)
    assert not np.array_equal(result.world,body)
    np.testing.assert_allclose(lengths(result.world),np.broadcast_to(np.median(lengths(body),axis=0),lengths(body).shape),atol=4e-7)
    for row in result.metadata["fragment_provenance"]:
        assert 0 <= row["source_seconds"][0] < row["source_seconds"][1] <= timeline.time[-1]
        assert 0 <= row["local_rate"] <= 1.5
    result.validate()
    plan.validate()


def test_no_recurrence_is_exact_baseline_not_cosmetic_motion():
    body,timeline = source_body(),music()
    timeline.recurrence[:] = np.eye(len(timeline.sections))
    result,_ = generate("D4",body,timeline)
    np.testing.assert_array_equal(result.world,body)
    assert result.metadata["recomposed_fragment_count"] == 0
    assert "no feasible" in result.metadata["diagnostic_reason"]


def test_repeated_crouch_keeps_donor_floor_and_physical_contacts():
    body,timeline = planted_crouch_body(),music()
    result,plan = generate("D4",body,timeline)
    assert result.metadata["revision"] == "d4_floor_v2"
    assert result.metadata["translation_axes"] == "XZ"
    assert result.metadata["recomposed_fragment_count"] == 2
    rows = result.metadata["fragment_provenance"]
    # Frozen selection from the pre-repair graph: this repair changes placement,
    # not recurrence, fragment scoring, selected source motion, or its clock.
    assert [row["source_fragment"] for row in rows] == [0,1,2,3,0,1,6,7]
    assert contacts_from_body(body).mean() == 1.
    for row in rows:
        assert row["root_translation_m"][1] == 0.
        assert row["prior_continuation_translation_m"][1] == 0.
        start,end = np.rint(np.asarray(row["target_seconds"])*30).astype(int)
        ds,de = row["source_seconds"]
        # Outside the unchanged .30 s transition, actual source positions keep
        # world Y exactly (up to reconstruction precision), including the feet.
        target = np.arange(start+10,end)
        if len(target):
            source = np.linspace(ds,de,end-start+1)[10:-1]
            original = resample_body(body,source)
            np.testing.assert_allclose(result.world[target,:,1],original[:,:,1],atol=3e-7)
    # The historical XYZ translation left these final stationary feet +.20 m
    # high and contactless. Neither a contact label nor metadata can hide that.
    np.testing.assert_allclose(result.world[360:,(7,8,10,11),1],body[360:,(7,8,10,11),1],atol=3e-7)
    assert np.all(result.contacts[370:] == 1.)
    assert result.contacts.mean() > .95
    np.testing.assert_allclose(lengths(result.world),np.broadcast_to(np.median(lengths(body),axis=0),lengths(body).shape),atol=3e-7)
    result.validate()
    plan.validate()


def test_learned_effort_adapter_retains_constraints_and_frozen_body():
    body,timeline = source_body(),music()
    effort = np.tile(np.linspace(0,2,len(body))[:,None],(1,5))
    result,plan = retime_with_effort(body,timeline.time,effort,timeline,metadata={"checkpoint":"test_only"})
    assert result.metadata["checkpoint"] == "test_only"
    assert np.diff(result.source_time,axis=0).min() >= -1e-10
    assert np.diff(result.source_time,axis=0).max()*30 <= 1.5+1e-8
    frozen = np.broadcast_to(body[:1],body.shape).copy()
    unchanged,_ = retime_with_effort(frozen,timeline.time,effort,timeline)
    np.testing.assert_array_equal(unchanged.world,frozen)
    result.validate()
    plan.validate()


def test_contacts_are_recomputed_from_emitted_body():
    body,timeline = source_body(),music()
    result,_ = generate("D2",body,timeline)
    np.testing.assert_array_equal(result.contacts,contacts_from_body(result.world))
