"""CPU checks for full-song clock, source identity, brightness and media delivery."""
import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path
import subprocess

import numpy as np
import pytest

from lightsem.conductor.body_transfer import build_outputs
from lightsem.experiments.dynamic_dance.contracts import DanceResult, MusicTimeline
from scripts.rendering import render_d4max as render


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def campaign(tmp_path):
    duration = 2.013
    nb, nl, nv = (math.ceil(duration*hz) for hz in (30, 40, 30))
    audio = tmp_path/'source.wav'
    subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-n', '-f', 'lavfi', '-i',
        'sine=frequency=220:sample_rate=44100', '-t', str(duration), '-ac', '2', str(audio)], check=True)
    # Match sample-accurate source duration, not decimal formatting of -t.
    probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams', '-of', 'json', str(audio)], text=True))
    duration = float(probe['streams'][0]['duration'])
    nb, nl, nv = (math.ceil(duration*hz) for hz in (30, 40, 30))
    music_path = tmp_path/'music.npz'
    t = np.arange(nb)/30
    muqt = np.arange(math.ceil(duration*25))/25
    MusicTimeline(time=t, muq=np.zeros((len(muqt), 1024)), muq_time=muqt,
        sources=np.ones((nb, 4, 4))*.2, mixture=np.ones((nb, 4))*.3,
        beats=np.array([0., .5, 1., 1.5]), downbeats=np.array([0.]),
        sections=[{'start': 0., 'end': duration, 'label': 'estimated'}], recurrence=np.ones((1, 1)),
        metadata={'duration_s': duration, 'input_sha256':render.sha(audio)}).save(music_path)
    track = {'id': 'shpongle', 'title': 'Shpongle', 'audio': str(audio), 'duration_s': duration,
        'video_frames': nv, 'body_frames': nb, 'lighting_frames': nl, 'music_timeline': str(music_path)}
    algorithm = {'id': 'C3', 'name': 'D4 + D7 + D8'}
    plan = {'tracks': [track], 'algorithms': [algorithm], 'files': {str(audio): render.sha(audio)}}
    put(tmp_path/'PLAN.json', plan)
    body = np.zeros((nb, 22, 3))
    body[:, :, 1] = np.linspace(.1, 1.8, 22)
    body[:, :, 0] = np.linspace(-.4, .4, 22)
    body[:, 20, 0] += t*.2
    light_body = np.stack([np.interp(np.arange(nl)/40, t, body[:, j, d]) for j in range(22) for d in range(3)], axis=-1).reshape(nl, 22, 3)
    patch = build_outputs()
    address = 1
    for fixture in patch:
        fixture.update(address=address, universe=0, n_channels=4 if fixture['kind']=='bulb' else 8)
        address += fixture['n_channels']
    dmx = np.zeros((nl, 1, 512), dtype=np.uint8)
    for fixture in patch:
        start = fixture['address']-1
        if fixture['kind']!='bulb':
            dmx[:, 0, start:start+4] = [128, 0, 128, 0]
            start += 4
        dmx[:, 0, start:start+4] = [31, 50, 130, 220]
    for method in ('D7_reference', 'C3'):
        body_path = tmp_path/'bodies'/method/track['id']/'body.npz'
        DanceResult(t, body, np.zeros((nb, 4)), metadata={'method': method, 'newly_generated_motion': True}).save(body_path)
        folder = tmp_path/'timelines'/method/track['id']
        folder.mkdir(parents=True)
        np.savez_compressed(folder/'timeline.npz', time=np.arange(nl)/40, body=light_body, universes=dmx,
            source_time=np.arange(nl)/40, playback_rate=np.ones(nl), intervention_time=np.array([.5]))
        put(folder/'patch.json', patch)
        put(folder/'report.json', {'status': 'diagnostic', 'body_sha256': render.sha(body_path),
            'timeline_sha256': render.sha(folder/'timeline.npz'), 'patch_sha256': render.sha(folder/'patch.json'),
            'inputs': {'body_metadata_sha256': render.sha(body_path.with_suffix('.json'))}})
    inputs = {str(tmp_path/'PLAN.json'): render.sha(tmp_path/'PLAN.json')}
    for method in ('D7_reference', 'C3'):
        for filename in ('timeline.npz', 'patch.json', 'report.json'):
            path = tmp_path/'timelines'/method/track['id']/filename
            inputs[str(path)] = render.sha(path)
    put(tmp_path/'reports/comparisons/C3_shpongle.json', {'method':'C3', 'song': 'shpongle',
        'inputs': inputs, 'physical_engineering_status':'pass', 'physical_failure_reasons':[]})
    render.prepare_audio(tmp_path, plan, track)
    return tmp_path, plan, algorithm, track


@pytest.mark.parametrize('duration', [201.64, 277.3066666667, 1.001, .021, .025, .033333333334])
def test_fullsong_clock_never_trims_final_audio_or_samples_outside_timeline(duration):
    track={'duration_s':duration,'video_frames':math.ceil(duration*30),'body_frames':math.ceil(duration*30),
        'lighting_frames':math.ceil(duration*40)}
    times, indices=render.track_clock(track)
    assert len(indices)==math.ceil(duration*30)
    assert indices[0]==0 and indices[-1]<len(times)
    assert 0<=len(indices)/30-duration<1/30+1e-10
    assert np.all(np.diff(indices)>=0)


def test_complete_inputs_are_authenticated(campaign):
    reference, candidate, music, pins=render.load_comparison(*campaign)
    assert len(candidate.indices)==campaign[3]['video_frames']
    assert len(candidate.body)==campaign[3]['lighting_frames']
    assert len(music['time'])==campaign[3]['body_frames']
    assert reference.method=='D7_reference'
    assert pins['candidate']['body_sha256']
    assert render.prepare_audio(campaign[0],campaign[1],campaign[3])['playback_rate']==1
    folder=campaign[0]/'timelines/C3/shpongle'
    (folder/'timeline.npz').write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed after numeric'):
        render.load_comparison(*campaign)


def test_reference_and_candidate_compare_emitted_brightness_separately(campaign):
    reference,candidate,_,_=render.load_comparison(*campaign)
    first=render.trace_values(candidate)
    np.testing.assert_allclose(first['brightness'],31/255)
    controls={k:v.copy() for k,v in candidate.controls.items()}
    controls['dimmer'][:]=.8
    altered=render.trace_values(replace(candidate,controls=controls))
    np.testing.assert_allclose(altered['brightness'],.8)
    for key in ('upper','lower','emitted'):
        np.testing.assert_array_equal(first[key],altered[key])


def test_fragment_jumps_cannot_be_mislabeled_as_rate(campaign):
    root,plan,_,track=campaign
    folder=root/'timelines/C3/shpongle'
    with np.load(folder/'timeline.npz') as source:
        arrays={k:source[k] for k in source.files if k!='playback_rate'}
    np.savez_compressed(folder/'timeline.npz',**arrays)
    report=render.read(folder/'report.json')
    report['timeline_sha256']=render.sha(folder/'timeline.npz')
    put(folder/'report.json',report)
    with pytest.raises(ValueError,match='recorded fragment-local'):
        render.load_pair(root,plan,'C3',track)


def test_always_visible_bodies_dont_fake_stage_brightness(campaign):
    reference,candidate,music,_=render.load_comparison(*campaign)
    strip=render.TimelineStrip(reference,candidate,music,campaign[3])
    dark=(np.zeros((540,960,3),np.uint8),)*2
    image=render.compose(dark,reference,candidate,0,0.,campaign[2],campaign[3],strip)
    assert image.shape==(1080,1920,3)
    assert not np.any(image[400:800,:958])
    assert not np.any(image[400:800,962:])
    assert np.any(image[90:310,:])


def test_smoke_and_review_gate_require_actual_gpu_inspection(campaign):
    smoke=render.smoke_pair(*campaign,schematic=True)
    assert all(value>0 for value in smoke['fixture_class_visibility'].values())
    assert smoke['samples'][0]['time_s']==0
    assert smoke['optical_controls_excluded_from_production']
    assert render.smoke_pair(*campaign,schematic=True)==smoke
    with pytest.raises(ValueError,match='actual-DMX GPU smoke'):
        render.render_pair(*campaign)


def test_encoder_copies_authenticated_full_audio_without_retiming(campaign):
    root,_,_,track=campaign
    reference,_=render.audio_paths(root,track)
    command=render.command_for(root/'movie.mp4',reference,track,'libx264')
    assert command[command.index('-c:a')+1]=='copy'
    assert '-ss' not in command and '-af' not in command
    assert not any('atempo' in value for value in command)
    assert command[command.index('-frames:v')+1]==str(track['video_frames'])


def test_streamcopy_keeps_fullsong_reference_pcm_exact(campaign):
    root,_,_,track=campaign
    reference,record=render.audio_paths(root,track)
    movie=root/'audio_copy_test.mp4'
    subprocess.run(['ffmpeg','-v','error','-nostdin','-n','-f','lavfi','-i','color=black:s=64x64:r=30',
        '-i',str(reference),'-map','0:v:0','-map','1:a:0','-frames:v',str(track['video_frames']),
        '-c:v','libx264','-threads','1','-c:a','copy','-t',str(track['video_frames']/30),str(movie)],check=True)
    assert hashlib.sha256(render.base.decoded_audio(movie)).hexdigest()==render.read(record)['decoded_pcm_sha256']


def test_finite_failed_physical_output_is_explicit_diagnostic():
    assert render.evidence_summary({'physical_engineering_status':'fail','physical_failure_reasons':['sliding']})=='FAILURE DIAGNOSTIC | sliding'
    assert 'no baseline promotion' in render.evidence_summary({'physical_engineering_status':'pass'})


def test_exclusive_save_preserves_existing_evidence(tmp_path):
    path=tmp_path/'record.json'
    render.save(path,{'value':1})
    with pytest.raises(FileExistsError):
        render.save(path,{'value':2})
    assert render.read(path)=={'value':1}


def test_full_decode_audit_covers_last_frame_and_reference_audio(campaign):
    root,plan,algorithm,track=campaign
    *_,pins=render.load_comparison(*campaign)
    out=root/'media/C3/shpongle'
    out.mkdir(parents=True)
    movie=out/'full.mp4'
    reference,_=render.audio_paths(root,track)
    subprocess.run(['ffmpeg','-v','error','-nostdin','-n','-f','lavfi','-i','color=black:s=1920x1080:r=30',
        '-i',str(reference),'-map','0:v:0','-map','1:a:0','-frames:v',str(track['video_frames']),
        '-c:v','libx264','-preset','ultrafast','-threads','2','-c:a','copy','-t',str(track['video_frames']/30),str(movie)],check=True)
    put(out/'full_render.json',{'inputs':pins,'sha256':render.sha(movie),'codec':'libx264',
        'duration_s':track['video_frames']/30,'final_frame_padding_s':track['video_frames']/30-track['duration_s']})
    verified=render.audit_pair(*campaign)
    audit=render.read(out/'full_audit.json')
    assert verified['complete_full_song']
    assert audit['decoded_frames']==track['video_frames']
    assert audit['full_software_AV_decode'] and audit['pts_uniform'] and audit['audio_matches_full_reference']
    assert audit['decoded_samples'][0]['frame']==0
    assert audit['decoded_samples'][-1]['frame']==track['video_frames']-1
    assert Path(audit['contact_sheet']).is_file()


def test_motif_caption_respects_c4_retimed_c3_clock(campaign):
    _,candidate,_,_=render.load_comparison(*campaign)
    metadata={'d6_enabled':True,'fragment_clock_domain':'C3 source time, not final output time',
        'fragment_provenance':[{'target_seconds':[0.,1.],'source_seconds':[8.,9.],'source_fragment':4},
                               {'target_seconds':[1.,2.],'source_seconds':[20.,21.],'source_fragment':7}]}
    clocks=np.ones((len(candidate.times),5))*.6
    changed=replace(candidate,metadata=metadata,source_time=clocks)
    # Final time 1.5s is still inside C3's first fragment due to slower pacing.
    label=render.motif_label(changed,60)
    assert 'motif 4' in label and '8.0–9.0s' in label and 'via C3 root clock' in label


def test_c4_visual_reference_is_distinct_from_incremental_numerical_comparator():
    report={'comparator':'C3','physical_engineering_status':'pass','primary':{'magnitude_gate':'fail'}}
    assert render.comparison_note(report,'C4')=='Visual reference: D7. C4 numerical comparison: D4-max (D6 off).'
    assert render.evidence_summary(report).startswith('PROXY FAILURE DIAGNOSTIC')
