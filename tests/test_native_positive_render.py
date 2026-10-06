"""CPU checks for truthful hybrid pacing, graph layout and reused audio."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from scripts.rendering import native_positive_traces as traces
from scripts.rendering import render_native_positive_3d as render


def fake_data():
    time=np.arange(80,161,.025); n=len(time); rows={}
    for method in traces.METHODS:
        rate=np.ones((n,1)) if method=='D7_reference' else np.where(time[:,None]<120,.3,1.4)
        raw={name:np.zeros(n) for name,_ in traces.ROWS}
        raw['rate']=rate.mean(1);raw['brightness'][:]=.4
        rows[method]={'raw':raw,'display':{k:v.copy() for k,v in raw.items()},
            'pacing':{'rate_channels':rate,'source_time':None if method=='D7_reference' else time[:,None]-10,
                      'motif_cuts':np.empty(0) if method=='D7_reference' else np.array([90.,120.,140.]),
                      'clock_scope':'native generated output' if method=='D7_reference' else 'saved D6 clock into D4-max'},
            'metadata':{'fragment_provenance':[{'target_seconds':[0,130],'motif_family':3},{'target_seconds':[130,180],'motif_family':9}]}}
    return {'time':time,'selected':(time>=90)&(time<=150),'values':rows,'scales':{k:1.5 for k,_ in traces.ROWS},
            'sources':np.zeros((n,4)),'source_scale':1.,'source_p95':np.ones(4),'sections':[],'inputs':{}}


def test_actual_rate_survives_source_clock_cut_without_differentiation():
    time=np.arange(12)/40; source=np.array([0.,.1,.2,.3,100.,100.1,100.2,100.3,0.,.1,.2,.3])
    rates=np.linspace(.05,1.5,12)[:,None]
    arrays={'source_time':source,'playback_rate':rates,'intervention_time':np.array([.1,.2])}
    result=traces.saved_pacing('C4',arrays,time)
    np.testing.assert_array_equal(result['rate_channels'],rates)
    np.testing.assert_array_equal(result['source_time'][:,0],source)
    assert rates.max()<=1.5  # The large source jumps never become false speed spikes.


def test_missing_recorded_rate_fails_instead_of_inventing_a_clock():
    with pytest.raises(ValueError,match='actual saved playback_rate'):
        traces.saved_pacing('C3',{'source_time':np.arange(20)/40},np.arange(20)/40)
    with pytest.raises(ValueError,match='motif-cut'):
        traces.saved_pacing('C4',{'playback_rate':np.ones(20)},np.arange(20)/40)


def test_native_reference_has_explicit_unit_clock_and_no_fake_motif_markers():
    pacing=traces.saved_pacing('D7_reference',{},np.arange(20)/40)
    np.testing.assert_array_equal(pacing['rate_channels'],np.ones((20,1)))
    assert pacing['source_time'] is None and len(pacing['motif_cuts'])==0


def test_hybrid_rate_caption_and_export_are_saved_data(monkeypatch,tmp_path):
    data=fake_data();monkeypatch.setattr(traces,'song_traces',lambda *args:data)
    strip=traces.Strip(tmp_path,'C4','shpongle')
    assert strip.image.size==(1280,304)
    text=strip.pacing_label(110.)
    assert 'Recomposed' in text and 'D6 rate 0.30×' in text and 'D4 source clock 100.00s' in text and 'motif 3' in text
    assert 'no retiming' not in text
    path=tmp_path/'TRACE.npz';strip.export(path)
    with np.load(path) as saved:
        np.testing.assert_array_equal(saved['current_saved_playback_rate'],data['values']['C4']['pacing']['rate_channels'][data['selected']])
        np.testing.assert_array_equal(saved['current_motif_cut_time'],[90,120,140])
        np.testing.assert_array_equal(saved['current_raw_rate'],saved['current_display_rate'])
        assert 'current_saved_source_time' in saved and 'reference_saved_source_time' not in saved
    frame=np.asarray(strip.frame(135.));assert frame.shape==(304,1280,3)


def test_3d_layout_keeps_saved_pacing_caption_and_fixture_views(monkeypatch,tmp_path):
    monkeypatch.setattr(traces,'song_traces',lambda *args:fake_data())
    strip=traces.Strip(tmp_path,'C3','shpongle')
    views=[np.zeros((600,640,3),np.uint8)]*2
    frame=render.compose(views,'C3',{'title':'Shpongle'},135.,strip)
    assert frame.shape==(1024,1280,3) and frame.dtype==np.uint8
    assert np.any(frame[720:])
    assert 'Historical' in render.METHODS['C3'] and 'Native C' in render.METHODS['D7_reference']


def test_gpu_access_still_requires_curator_guard(tmp_path):
    with pytest.raises(ValueError,match='sequential curator'):
        with render.renderers(tmp_path,[],False):
            pytest.fail('GPU renderer must not be reached')


def test_video_audio_clock_and_budget_unchanged():
    body,ticks=render.frame_indices();assert len(body)==1800 and body[0]==2700 and body[-1]==4499
    assert np.max(np.abs(body/30-ticks/40))<=1/80
    command=render.encode_command(Path('/tmp/hybrid.mp4'),Path('/tmp/audio.m4a'))
    assert command[command.index('-b:v')+1]=='1100k'
    assert command[command.index('-c:a')+1]=='copy'
    assert command[command.index('-t')+1]=='60'
    assert not any('atempo' in value or 'setpts' in value for value in command)


def test_existing_audio_is_hardlinked_with_truthful_new_receipt(tmp_path,monkeypatch):
    root=tmp_path/'hybrid_positive_v1';root.mkdir()
    audio=tmp_path/'source.wav';audio.write_bytes(b'original')
    track={'id':'shpongle','audio':str(audio)}
    monkeypatch.setattr(render,'campaign',lambda *args:({},track))
    monkeypatch.setattr(render.native,'verify_source',lambda *args:render.sha(audio))
    monkeypatch.setattr(render.native,'resource_guard',lambda *args:None)
    monkeypatch.setattr(render.native.base,'decoded_audio',lambda *args:b'pcm')
    old=tmp_path/'preview_3d_unlimited_v1/audio/shpongle';old.mkdir(parents=True)
    reference=old/'reference.m4a';reference.write_bytes(b'immutable AAC')
    receipt={'plan_sha256':'old-plan','source_audio':str(audio),'source_audio_sha256':render.sha(audio),'reference':str(reference),
        'reference_sha256':render.sha(reference),'source_start_s':90,'duration_s':60,'playback_rate':1.,
        'decoded_pcm_sha256':render.hashlib.sha256(b'pcm').hexdigest(),'command':['old','encoding']}
    (old/'REFERENCE.json').write_text(json.dumps(receipt))
    value=render.prepare_audio(root,'shpongle');new,_=render.audio_paths(root,track)
    assert new.stat().st_ino==reference.stat().st_ino
    assert value['plan_sha256']==render.PLAN_SHA and value['inherited_receipt']==str(old/'REFERENCE.json')
    assert 'no reencoding' in value['creation'] and value['original_encoding_command']==['old','encoding']
    assert render.prepare_audio(root,'shpongle')==value
