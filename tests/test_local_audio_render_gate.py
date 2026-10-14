"""The diagnostic renderer must authenticate raw safety before mapping."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from lightsem.eval.training_health import motion_metrics
from scripts.motion import render_native_local_audio_candidate as renderer


def bundle(tmp_path, multiplier=1.):
    time=np.arange(120)/30.
    world=np.zeros((120,22,3),dtype=np.float32)
    world[:,:,0]=(.1*np.sin(time))[:,None]
    world[:,:,1]=1.+np.arange(22)[None]/100
    candidate=tmp_path/'candidate';candidate.mkdir()
    body=candidate/'shpongle.npz';ref=tmp_path/'reference.npz'
    np.savez(ref,time=time,body=world,native=np.zeros((120,139),np.float32))
    np.savez(body,time=time,body=world*multiplier,native=np.zeros((120,139),np.float32))
    audio=tmp_path/'input.wav';audio.write_bytes(b'source audio binding')
    checkpoint=tmp_path/'model.pt';checkpoint.write_bytes(b'checkpoint binding')
    a,b=motion_metrics(world*multiplier),motion_metrics(world)
    ratios={'speed':a['speed_p95']/b['speed_p95'],
            'acceleration':a['acceleration_raw']['p95']/b['acceleration_raw']['p95'],
            'jerk':a['jerk_raw']['p95']/b['jerk_raw']['p95']}
    receipt={'status':'diagnostic_only','finite':True,'body_path':str(body),'body_sha256':renderer.sha(body),
             'checkpoint':str(checkpoint),'checkpoint_sha256':renderer.sha(checkpoint),
             'baseline_body':str(ref),'baseline_body_sha256':renderer.sha(ref),
             'inputs':{'audio_sha256':renderer.sha(audio)},'raw_safety':ratios}
    audit=candidate/'shpongle.json';audit.write_text(json.dumps(receipt))
    return body,audio,checkpoint,audit,receipt


def test_valid_raw_gate_recomputed_from_authenticated_arrays(tmp_path):
    body,audio,ck,audit,_=bundle(tmp_path)
    verified=renderer.verify_raw_audit(body,audio,ck,audit)
    assert verified['status']=='pass' and verified['recomputed_ratios']=={'speed':1.,'acceleration':1.,'jerk':1.}


@pytest.mark.parametrize('bad', ['checkpoint','audio','body','baseline','rejected','forged_ratio'])
def test_bad_raw_gate_rejected_before_output_or_mapping(tmp_path,monkeypatch,bad):
    body,audio,ck,audit,receipt=bundle(tmp_path)
    if bad in ('checkpoint','audio','body','baseline'):
        path={'checkpoint':ck,'audio':audio,'body':body,'baseline':Path(receipt['baseline_body'])}[bad]
        path.write_bytes(path.read_bytes()+b'tamper')
    elif bad=='rejected':
        receipt['status']='rejected_raw_safety';audit.write_text(json.dumps(receipt))
    else:
        receipt['raw_safety']['jerk']=.1;audit.write_text(json.dumps(receipt))
    def forbidden(*a,**kw):pytest.fail('Mapper/render was called on invalid raw evidence')
    monkeypatch.setattr(renderer,'map_lights',forbidden)
    monkeypatch.setattr(renderer,'render_video',forbidden)
    out=tmp_path/'render'
    args=SimpleNamespace(candidate_root=body.parent,song='shpongle',out=out,audio=audio,checkpoint=ck)
    with pytest.raises(ValueError):renderer.run(args)
    assert not out.exists()


def test_honestly_recorded_exploding_candidate_is_rejected(tmp_path):
    body,audio,ck,audit,_=bundle(tmp_path,multiplier=3.)
    with pytest.raises(ValueError,match='rejected_raw_safety'):
        renderer.verify_raw_audit(body,audio,ck,audit)
