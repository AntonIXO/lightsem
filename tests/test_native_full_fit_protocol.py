import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from lightsem.experiments.native_dynamics.full_fit_admission import (
    iter_native_tracks, sha, verify_muq, verify_normalization, window_union,
)
from scripts.motion.build_native_full_fit_manifest_rtmpose_20261008 import build
from scripts.data.render_rtmpose_evidence import frames_at_pts


def fixture(tmp_path):
    n=600
    points=np.zeros((n,3,17,2),np.float32)
    points[...,0]=500; points[...,1]=300
    points[:,1:,11:13,1]=450
    points[:,2,:,0]=900
    box=np.tile([350,100,600,680],(n,3,1)).astype(np.float32)
    box[:,2,[0,2]]+=400
    # First track has degenerate torso geometry; other people remain eligible.
    raw=dict(t=np.arange(n)/30, K=points.reshape(-1,17,2),
             C=np.full((n*3,17),.8,np.float32), box=box.reshape(-1,4),
             track_id=np.tile(np.arange(3,dtype=np.int32),n),
             frame_index=np.repeat(np.arange(n,dtype=np.int32),3),candidate=np.ones(n*3,bool))
    source=tmp_path/'source.webm'; source.write_bytes(b'fixture-source')
    wav=tmp_path/'music.wav'
    sf.write(wav, np.full((44100*20,2),.01,np.float32),44100,subtype='FLOAT')
    row={'recording_id':'record','wav_path':str(wav),'wav_sha256':sha(wav),'duration_s':20.,
         'source_path':str(source),'source_sha256':sha(source),'classification':'normal'}
    audio=tmp_path/'audio.json';audio.write_text(json.dumps({'rows':[row]}))
    stage=tmp_path/'stage.json';stage.write_text(json.dumps({'recordings':[
        dict(recording_id='record',canonical_id='song',partition='test',source_path=str(source),source_sha256=sha(source))]}))
    legacy=tmp_path/'legacy.json';legacy.write_text(json.dumps({'rows':[]}))
    ids=tmp_path/'ids.txt';ids.write_text('record\n')
    raw['meta']=np.asarray(json.dumps(dict(source_sha256=sha(source),height=720,audio_start_s=0)))
    pose=tmp_path/'raw';pose.mkdir();np.savez(pose/'record.npz',**raw)
    return dict(stage_manifest=stage,audio_manifest=audio,ids_file=ids,pose_dir=pose,
                output_root=tmp_path/'out',legacy_map=legacy),raw,row


def test_manifest_admits_later_coaches_and_all_source_partitions(tmp_path):
    args,raw,_=fixture(tmp_path)
    report=build(**args)
    assert len(report['entries'])==2
    assert [r['track_id'] for r in report['entries']]==[1,2]
    assert all(r['partition']=='train' and r['source_partition']=='test' for r in report['entries'])
    assert report['coverage']['admitted_groups']==1
    assert report['entries'][0]['feature_key']==report['entries'][1]['feature_key']
    assert 'no asserted match' in report['entries'][0]['coach_identity_status']
    assert any(r.get('track_id')==0 for r in report['exclusions'])
    with pytest.raises(FileExistsError): build(**args)


def test_ragged_duplicates_fail_instead_of_averaging(tmp_path):
    _,raw,_=fixture(tmp_path)
    raw['track_id'][1]=0
    with pytest.raises(ValueError,match='Multiple detections'): list(iter_native_tracks(raw))


def test_window_union_does_not_double_count_overlap():
    union=window_union([0,15,30],600)
    assert union.sum()==510


def test_normalization_authenticates_samples_and_source_hash(tmp_path):
    _,_,row=fixture(tmp_path)
    original=Path(row['wav_path']);x,sr=sf.read(original,dtype='float32',always_2d=True)
    output=tmp_path/'gain.wav';sf.write(output,x*2,sr,subtype='FLOAT')
    receipt=tmp_path/'norm.json'
    norm=dict(recording_id='record',source_wav=str(original),source_wav_sha256=sha(original),
              output_wav=str(output),output_wav_sha256=sha(output),gain_db=20*np.log10(2))
    receipt.write_text(json.dumps(norm))
    row.update(wav_path=str(output),wav_sha256=sha(output),normalization_receipt=str(receipt))
    assert verify_normalization(row)['receipt_sha256']==sha(receipt)
    sf.write(output,x*3,sr,subtype='FLOAT')
    row['wav_sha256']=sha(output);norm['output_wav_sha256']=sha(output);receipt.write_text(json.dumps(norm))
    with pytest.raises(ValueError,match='scalar gain'): verify_normalization(row)


def muq_fixture():
    expected={'model_id':'MuQ','revision':'revision','weights_sha256':'weights',
              'layer':9,'native_hz':25,'chunk_seconds':8}
    metadata={'input_sha256':'wave','duration_s':8,'recipe':{
        'muq_layer':9,'muq_native_hz':25,'muq_chunk_seconds':8,
        'muq_boundary':'overlap-aware native25Hz grid; overlapping real8s crops are timestamp-aligned and averaged'},
        'muq':{'model_receipt':{'model_id':'MuQ','revision':'revision','artifacts':{
            'model.safetensors':{'sha256':'weights'}}},'layer':9,'frame_rate_hz':25}}
    return SimpleNamespace(metadata=metadata,muq=np.zeros((200,1024)),muq_time=np.arange(200)/25),expected


@pytest.mark.parametrize('changed',['waveform','revision','weights','layer','context','timestamps','nonfinite'])
def test_muq_cache_identity_rejects_drift(changed):
    timeline,expected=muq_fixture()
    assert verify_muq(timeline,expected,'wave')['frames']==200
    if changed=='waveform': timeline.metadata['input_sha256']='other'
    if changed=='revision': timeline.metadata['muq']['model_receipt']['revision']='other'
    if changed=='weights': timeline.metadata['muq']['model_receipt']['artifacts']['model.safetensors']['sha256']='other'
    if changed=='layer': timeline.metadata['muq']['layer']=6
    if changed=='context': timeline.metadata['recipe']['muq_chunk_seconds']=30
    if changed=='timestamps': timeline.muq_time[1]=timeline.muq_time[0]
    if changed=='nonfinite': timeline.muq[1,0]=np.nan
    with pytest.raises(ValueError): verify_muq(timeline,expected,'wave')


def test_evidence_uses_pts_when_source_has_twice_cache_fps():
    frames=[SimpleNamespace(time=i/60,to_ndarray=lambda format,i=i:np.full((1,1,3),i)) for i in range(8)]
    selected=frames_at_pts(frames,{1:2/60,3:6/60})
    assert selected[1][0][0,0,0]==2
    assert selected[3][0][0,0,0]==6


def test_production_sampler_balances_groups_not_number_of_tracks():
    from lightsem.experiments.native_dynamics.dataset import Records
    records=Records.__new__(Records)
    records.records={'jd':[dict(key=str(i),starts=np.array([0])) for i in range(11)]}
    records.groups={'jd':[[0],list(range(1,11))]}
    records.pack=lambda domain,chosen:chosen
    sample=records.sample('jd',5000,np.random.default_rng(42))
    first=sum(row['key']=='0' for row,_ in sample)
    assert 2300<first<2700


def test_audio_extraction_deduplicates_coaches_but_not_different_waveforms(tmp_path):
    from scripts.motion.native_dynamics_features import audio_work_items
    a={'feature_key':'jd_song','wav_path':str(tmp_path/'music.wav'),
       'wav_sha256':'wave','frames':600,'timeline_path':'song.npz','coach_id':0}
    b=dict(a,coach_id=1)
    assert len(audio_work_items([a,b]))==1
    with pytest.raises(ValueError,match='Conflicting waveform'):
        audio_work_items([a,dict(b,wav_sha256='other')])
    assert len(audio_work_items([a,dict(b,feature_key='jd_other',wav_sha256='other',timeline_path='other.npz')]))==2


def test_shared_feature_arrays_are_read_only_and_reused(monkeypatch,tmp_path):
    from lightsem.experiments.native_dynamics import dataset
    target=np.arange(30)/30
    path=tmp_path/'base.npz';np.savez(path,x30=np.ones((30,92),np.float32),time30=target)
    calls=[]
    monkeypatch.setattr(dataset.MusicTimeline,'load',lambda path:object())
    def features(timeline,t):
        calls.append(1)
        return np.ones((len(t),1061),np.float32),[]
    monkeypatch.setattr(dataset,'model_features',features)
    row=dict(timeline_sha256='timeline',base92_sha256='base',timeline_path='timeline.npz',base92_path=str(path))
    cache=dataset.SharedSongFeatures();first=cache.load(row,target);second=cache.load(row,target)
    assert first is second and len(calls)==1
    assert all(not x.flags.writeable for x in first)
    with pytest.raises(ValueError,match='clocks differ'):
        cache.load(row,target+.01)


def test_all_fit_refuses_legacy_fd_subset(tmp_path):
    from lightsem.experiments.native_dynamics.dataset import fd_manifest_path
    (tmp_path/'TRAINING_PROTOCOL.json').write_text(json.dumps({'require_all_finedance':True}))
    with pytest.raises(ValueError,match='no legacy subset'):fd_manifest_path(tmp_path)


def test_repair_recipe_is_the_selected_training_loss(monkeypatch):
    from lightsem.experiments.native_dynamics import training
    from lightsem.experiments.native_repair import training as repaired
    from lightsem.experiments.native_repair import losses
    monkeypatch.setattr(losses,'WEIGHTS',dict(losses.WEIGHTS))
    weights=dict(losses.WEIGHTS,raw_high_frequency=16.)
    assert training.training_loss_for_protocol({'repair_weights':weights}) is repaired.batch_losses
    assert losses.WEIGHTS['raw_high_frequency']==16.
    assert training.training_loss_for_protocol({}) is training.batch_losses


def test_loss_spec_binds_effective_weights_and_source_hashes():
    from lightsem.experiments.native_dynamics import training
    from lightsem.experiments.native_repair import losses

    weights=dict(losses.WEIGHTS,raw_high_frequency=16.)
    spec=training.loss_spec_for_protocol({'repair_weights':weights})
    assert spec['name']=='native_repair'
    assert spec['repair_weights']['raw_high_frequency']==16.
    assert spec['repair_weights']=={k:float(v) for k,v in sorted(weights.items())}
    assert all(path and len(digest)==64 for path,digest in spec['source_hashes'].items())
    assert any(path.endswith('/native_repair/losses.py') for path in spec['source_hashes'])


def test_loss_spec_without_repair_is_explicit_base_recipe():
    from lightsem.experiments.native_dynamics import training

    spec=training.loss_spec_for_protocol({})
    assert spec['name']=='native_dynamics'
    assert spec['repair_weights'] is None
    assert spec['base_weights']==training.LOSS_WEIGHTS
