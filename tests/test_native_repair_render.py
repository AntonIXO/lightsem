"""Analytical unfiltered-jerk and honest pending-page checks; CPU only."""
from pathlib import Path
import json
import math
import hashlib
import subprocess
import numpy as np
import pytest
from scripts.rendering import native_repair_traces as traces
from scripts.rendering import render_native_repair_3d as render


def test_world_jerk_and_signed_root_height_are_not_relative_speed():
    time=np.arange(120)/30
    world=np.zeros((len(time),22,3));world[:,:,1]=(time**3-2)[:,None]
    result=traces.body_traces(world,time)
    np.testing.assert_allclose(result['root_y'][1],time**3-2)
    assert result['root_y'][1][0]<0
    np.testing.assert_allclose(result['body_jerk'][1],6.,atol=2e-9)
    np.testing.assert_allclose(result['root_vertical_jerk'][1],6.,atol=2e-9)
    np.testing.assert_allclose(result['body_jerk'][0],time[:-3]+1.5/30,atol=1e-14)
    for region in ('left_arm','right_arm','lower'):
        np.testing.assert_array_equal(result[region][1],0.)


def test_alternating_frame_tremor_is_preserved_by_forward_jerk():
    time=np.arange(120)/30;world=np.zeros((len(time),22,3))
    world[:,0,1]=.01*(-1.)**np.arange(len(time))
    values=traces.body_traces(world,time)
    assert np.min(np.abs(values['root_vertical_jerk'][1]))>2000
    assert values['root_vertical_jerk'][1].min()<0<values['root_vertical_jerk'][1].max()
    assert np.min(values['body_jerk'][1])>400


def test_raw_pixel_envelope_cannot_average_away_a_glitch():
    time=np.array([89.,90.,90.01,90.02,100.,150.,151.])
    value=np.array([99999.,0.,-9000.,11000.,2.,3.,99999.])
    low,high=traces.pixel_envelope(time,value,left=90,right=150,pixels=10)
    assert low[0]==-9000 and high[0]==11000
    assert high[-1]==3 and not np.any(high==99999.)
    assert np.isnan(high[3])


def test_single_joint_world_rms_jerk_definition():
    time=np.arange(120)/30;world=np.zeros((len(time),22,3));world[:,20,0]=time**3
    result=traces.body_traces(world,time)
    np.testing.assert_allclose(result['body_jerk'][1],6/np.sqrt(22),atol=2e-9)
    np.testing.assert_array_equal(result['root_vertical_jerk'][1],0)


def track_spec(song='shpongle',duration=277.3066666666667):
    return {'id':song,'duration_s':duration,'video_frames':math.ceil(duration*30),
            'body_frames':math.ceil(duration*30),'lighting_frames':math.ceil(duration*40)}


def minimal_plan(root):
    plan={'review':{'window':'full_song'},'tracks':[track_spec(),track_spec('cure_for_me',201.64)]}
    (root/'PLAN.json').write_text(json.dumps(plan))
    render.bind_plan(root,render.sha(root/'PLAN.json'))
    return plan


def test_plan_identity_must_be_supplied_not_inferred(tmp_path,monkeypatch):
    (tmp_path/'PLAN.json').write_text('{}');monkeypatch.setattr(render,'PLAN_SHA',None)
    with pytest.raises(ValueError,match='PLAN changed'):render.campaign(tmp_path,'shpongle')
    with pytest.raises(ValueError,match='digest differs'):render.bind_plan(tmp_path,'0'*64)


def test_pending_gallery_has_no_media_or_substitute_and_same_stable_page(tmp_path):
    minimal_plan(tmp_path)
    first=render.gallery(tmp_path,'pending_v1')
    assert first['verified_media']==0 and len(first['clips'])==6
    assert all(row['status']=='awaiting_generation' and 'src' not in row for row in first['clips'])
    before=(tmp_path/'review_3d_v1/revisions/pending_v1/index.html').read_bytes()
    (tmp_path/'bodies/C4/shpongle').mkdir(parents=True)
    (tmp_path/'bodies/C4/shpongle/body.npz').write_bytes(b'not a verified output')
    second=render.gallery(tmp_path,'pending_v2')
    row=next(row for row in second['clips'] if row['method']=='C4' and row['song']=='shpongle')
    assert row['status']=='awaiting_verification' and 'src' not in row
    assert second['path']==first['path']
    assert (tmp_path/'review_3d_v1/revisions/pending_v1/index.html').read_bytes()==before
    manifest=json.loads((tmp_path/'review_3d_v1/MANIFEST.json').read_text())
    assert all(row['status']=='planned' for row in manifest['variants'])
    assert manifest['roles']=={'D7_reference':'old_D7','C3':'broken_C','C4':'repaired_C'}
    with pytest.raises(FileExistsError):render.gallery(tmp_path,'pending_v1')


def test_missing_or_unsealed_scientific_evidence_cannot_claim_success(tmp_path):
    minimal_plan(tmp_path)
    assert render.review_evidence(tmp_path,'C4','shpongle')['status']=='pending'
    path=tmp_path/'reports/review/C4_shpongle.json';path.parent.mkdir(parents=True);path.write_text('{"status":"pass"}')
    assert render.review_evidence(tmp_path,'C4','shpongle')['status']=='unverified'
    seal=path.with_name(path.name+'.sha256.json');seal.write_text(json.dumps({'sha256':render.sha(path)}))
    assert render.review_evidence(tmp_path,'C4','shpongle',{'body_sha256':'x','timeline_sha256':'y'})['status']=='stale'


def test_guarded_gpu_and_original_audio_clock_remain_required(tmp_path):
    with pytest.raises(ValueError,match='sequential curator'):
        with render.renderers(tmp_path,[],False):pytest.fail('GPU must not open')
    command=render.encode_command(Path('/tmp/movie.mp4'),Path('/tmp/audio.m4a'),track_spec())
    assert command[command.index('-c:a')+1]=='copy'
    assert command[command.index('-frames:v')+1]=='8320'
    assert command[command.index('-t')+1]==str(8320/30)
    assert not any('atempo' in value or 'setpts' in value for value in command)


def test_raw_trace_export_preserves_native_time_and_peak_arrays(tmp_path,monkeypatch):
    time=np.arange(0,202,1/30);world=np.zeros((len(time),22,3));world[:,0,1]=.01*(-1.)**np.arange(len(time))
    rows=traces.body_traces(world,time)
    for name in ('requested','emitted','brightness'):rows[name]=(time,np.zeros(len(time)))
    data={'values':{method:rows for method in traces.METHODS},'scales':{},'waveform':{'time':np.arange(0,202,.025)+.0125,'bin_edges':np.arange(8081)/40,'minimum':np.zeros(8080),'maximum':np.zeros(8080)},'roles':traces.ROLES,'inputs':{},'interval_s':[0.,202.]}
    for name,_ in traces.ROWS:
        value=rows[name][1];lo,hi=float(value.min()),float(value.max());data['scales'][name]=[lo-1,hi+1]
    monkeypatch.setattr(traces,'song_traces',lambda *args:data)
    strip=traces.Strip(tmp_path,'C4','shpongle');path=tmp_path/'TRACE.npz';strip.export(path)
    with np.load(path) as result:
        rt,rv=rows['root_vertical_jerk'];selected=(rt>=0)&(rt<=202)
        np.testing.assert_array_equal(result['current_root_vertical_jerk_time'],rt[selected])
        np.testing.assert_array_equal(result['current_root_vertical_jerk_raw'],rv[selected])
        assert np.max(result['current_root_vertical_jerk_pixel_max'])==np.max(rv[selected])
        assert np.min(result['current_root_vertical_jerk_pixel_min'])==np.min(rv[selected])
    frame=render.compose([np.zeros((600,640,3),np.uint8)]*2,'C4',{'title':'Shpongle'},135.,strip)
    assert frame.shape==(1024,1280,3)


def test_waveform_is_original_stereo_mean_crop_and_preserves_peaks(tmp_path):
    import soundfile as sf
    rate=400;audio=np.zeros((151*rate+5,2),np.float32)
    audio[0]=[.6,.6];audio[-1]=[1.,1.]
    audio[100*rate+1]=[.8,-.4]
    audio[110*rate+2]=[-.8,.2]
    path=tmp_path/'original.wav';sf.write(path,audio,rate,subtype='FLOAT')
    result=traces.waveform({'audio':str(path),'duration_s':len(audio)/rate})
    assert len(result['time'])==6041 and result['time'][0]==.0125
    assert result['bin_edges'][-1]==151.0125
    assert result['maximum'][0]==pytest.approx(.6) and result['maximum'][-1]==1.
    assert result['maximum'].max()==1.
    assert result['minimum'].min()==pytest.approx(-.3)
    assert result['maximum'][100*40]==pytest.approx(.2)
    assert result['minimum'][110*40]==pytest.approx(-.3)


@pytest.mark.parametrize('song,duration,frames', [('shpongle',277.3066666666667,8320),('cure_for_me',201.64,6050)])
def test_complete_track_window_and_final_40hz_sample(song,duration,frames):
    track=track_spec(song,duration);value=render.window(track);body,ticks=render.frame_indices(track)
    assert value['frames']==frames and value['source_start_s']==0
    assert len(body)==frames and body[0]==0 and body[-1]==frames-1
    assert 0<=value['end_padding_s']<1/30
    assert value['duration_s']>=duration and body[-1]/30<duration
    assert ticks[-1]<track['lighting_frames'] and ticks[0]==0
    assert np.max(np.abs(body/30-ticks/40))<=1/80


def test_fullsong_envelope_retains_a_late_raw_glitch():
    time=np.arange(0,278,1/30);values=np.zeros(len(time));values[np.argmin(abs(time-271))]=9999
    lo,hi=traces.pixel_envelope(time,values,left=0,right=277.3066666666667)
    assert np.nanmax(hi)==9999
    _,zoom=traces.pixel_envelope(time,values,left=90,right=150)
    assert np.nanmax(zoom)==0


def audio_plan(root,monkeypatch):
    import soundfile as sf
    rate=44100;samples=round(2.013*rate);duration=samples/rate
    audio=root/'source.wav';signal=.2*np.sin(np.arange(samples)*2*np.pi*440/rate)
    sf.write(audio,np.column_stack((signal,signal)),rate)
    track={**track_spec('shpongle',duration),'audio':str(audio)}
    plan={'review':{'window':'full_song'},'tracks':[track],'files':{str(audio):render.sha(audio)}}
    (root/'PLAN.json').write_text(json.dumps(plan));render.bind_plan(root,render.sha(root/'PLAN.json'))
    monkeypatch.setattr(render.native,'resource_guard',lambda *args:None)
    return plan,track


def test_full_audio_encode_and_decode_audit_do_not_retain_sixty_second_assumptions(tmp_path,monkeypatch):
    plan,track=audio_plan(tmp_path,monkeypatch);reference_record=render.prepare_audio(tmp_path,'shpongle')
    assert reference_record['frames']==61 and reference_record['source_start_s']==0
    assert 0<reference_record['end_padding_s']<1/30
    assert not any('atempo' in word for word in reference_record['command'])
    assert render.prepare_audio(tmp_path,'shpongle')==reference_record
    reference,_=render.audio_paths(tmp_path,track)
    out=render.folder(tmp_path,'C4','shpongle');out.mkdir(parents=True);movie=out/'preview.mp4'
    command=render.encode_command(movie,reference,track)
    command[command.index('-f'):command.index('-i')+2]=['-f','lavfi','-i','color=black:s=1280x1024:r=30']
    subprocess.run(command,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    pins={'test_fixture':True,'audio':render.audio_pins(tmp_path,plan,track)}
    record={'inputs':pins,'sha256':render.sha(movie),**render.window(track)}
    (out/'RENDER.json').write_text(json.dumps(record))
    monkeypatch.setattr(render,'load',lambda *args:(plan,track,None,None,None,pins))
    monkeypatch.setattr(render,'require_smoke',lambda *args:'test_fixture')
    audited=render.audit(tmp_path,'C4','shpongle')
    assert audited['decoded_frames']==61 and audited['full_software_AV_decode'] and audited['pts_uniform']
    assert audited['duration_s']==61/30 and audited['source_duration_s']==track['duration_s']
    assert audited['pcm_sha256']==reference_record['decoded_pcm_sha256']
    assert audited['audio_matches_prepared_reference']


def test_exact_parent_full_reference_is_reused_with_new_plan_receipt(tmp_path,monkeypatch):
    parent=tmp_path/'parent';parent.mkdir();plan,track=audio_plan(parent,monkeypatch)
    first=render.prepare_audio(parent,'shpongle');reference,receipt=render.audio_paths(parent,track)
    old=parent/'audio/shpongle';old.mkdir(parents=True)
    (old/'reference.m4a').hardlink_to(reference)
    original={**first,'reference':str(old/'reference.m4a')};(old/'REFERENCE.json').write_text(json.dumps(original))
    child=parent/'repair';child.mkdir();(child/'PLAN.json').write_text(json.dumps({**plan,'new_review':'identity'}))
    render.bind_plan(child,render.sha(child/'PLAN.json'))
    reused=render.prepare_audio(child,'shpongle');new,_=render.audio_paths(child,track)
    assert 'Hardlink exact full-song' in reused['creation']
    assert reused['plan_sha256']!=first['plan_sha256']
    assert new.stat().st_ino==(old/'reference.m4a').stat().st_ino
    assert reused['decoded_pcm_sha256']==first['decoded_pcm_sha256']


def test_readable_summary_and_metrics_require_the_same_sealed_input_binding(tmp_path):
    minimal_plan(tmp_path)
    path=tmp_path/'reports/review/C4_shpongle.json';path.parent.mkdir(parents=True)
    metrics=[{'label':'Raw jerk','value':0,'unit':'m/s³','reference':723,'delta':-723},
             {'label':'HF residual ratio','value':1.307,'unit':'×'}]
    report={'plan_sha256':render.PLAN_SHA,'body_sha256':'actual-body','timeline_sha256':'actual-timeline',
            'summary':'Raw jerk improves, but the frozen high-frequency gate still fails.',
            'metrics':metrics,'failure_diagnostic':True}
    path.write_text(json.dumps(report));seal=path.with_name(path.name+'.sha256.json')
    seal.write_text(json.dumps({'sha256':render.sha(path)}))
    pins={'body_sha256':'actual-body','timeline_sha256':'actual-timeline'}
    evidence=render.review_evidence(tmp_path,'C4','shpongle',pins)
    assert evidence['status']=='authenticated' and evidence['summary']==report['summary']
    assert evidence['report']['metrics']==metrics and evidence['report']['failure_diagnostic']
    stale=render.review_evidence(tmp_path,'C4','shpongle',{**pins,'body_sha256':'another-body'})
    assert stale['status']=='stale' and 'summary' not in stale and 'report' not in stale
    path.write_text(json.dumps({**report,'summary':'Unsealed success claim'}))
    unsealed=render.review_evidence(tmp_path,'C4','shpongle',pins)
    assert unsealed['status']=='unverified' and 'summary' not in unsealed and 'report' not in unsealed
