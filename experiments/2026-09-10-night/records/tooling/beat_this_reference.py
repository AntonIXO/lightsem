"""Published reference tracker on the unchanged GuitarSet cohort, GPU inference only."""
from pathlib import Path
import argparse, importlib.metadata, json, sys, time
import numpy as np
ROOT=Path('/mnt/lightsem-data/experiments/lightsem/night_expression_20260909')
sys.path.insert(0,str(ROOT/'source'))
from lightsem.experiments.stage11_common import authenticated_json,read_authenticated,digest,gpu_lock
FOLDER=ROOT/'rhythm/beat_this_reference'
WEIGHTS=ROOT/'reference_models/beat_this-final0.ckpt'

def freeze():
    import beat_this
    FOLDER.mkdir(exist_ok=True)
    files=[Path(__file__),WEIGHTS,ROOT/'rhythm/input_manifest.json']
    files+=sorted(Path(beat_this.__file__).parent.rglob('*.py'))
    value={'files':{str(p):digest(p) for p in files},'checkpoint':'final0','checkpoint_url':'https://cloud.cp.jku.at/public.php/dav/files/7ik4RrBKTS273gp/final0.ckpt',
        'versions':{p:importlib.metadata.version(p) for p in ('beat-this','torch','torchaudio','rotary-embedding-torch','numpy')},
        'float16':False,'dbn':False,'device':'cuda','cohort':358,'trim_s':5,'metric':'mir_eval default70ms F1 and continuity',
        'interpretation':'Published pretrained reference, not a new main arm. GuitarSet pretraining overlap unverified; not an independent held-out SOTA claim. Primary constant-grid versus PLP test is unchanged.'}
    path=FOLDER/'protocol.json'
    if path.exists():
        if read_authenticated(path)!=value:raise RuntimeError('Reference protocol drift')
    else:authenticated_json(path,value)

def run():
    import torch, soundfile as sf, mir_eval
    from beat_this.inference import Audio2Beats,load_checkpoint
    from beat_this.utils import replace_state_dict_key
    protocol=read_authenticated(FOLDER/'protocol.json')
    for path,sha in protocol['files'].items():
        if digest(path)!=sha:raise RuntimeError('Reference source/input drift: '+path)
    state=read_authenticated(ROOT/'pipeline_state.json')
    if not state.get('gpu_pipeline_complete'):raise RuntimeError('Main GPU pipeline must finish first')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    tracker=Audio2Beats(checkpoint_path=str(WEIGHTS),device='cuda',float16=False,dbn=False)
    expected=replace_state_dict_key(load_checkpoint(str(WEIGHTS),'cpu')['state_dict'],'model.','')
    actual=tracker.model.state_dict()
    if set(expected)!=set(actual) or any(not torch.equal(actual[k].cpu(),v) for k,v in expected.items()):raise RuntimeError('Reference checkpoint tensor mismatch')
    authenticated_json(FOLDER/'weight_audit.json',{'all_tensors_equal':True,'tensors':len(actual),'checkpoint_sha256':digest(WEIGHTS)})
    del expected,actual
    rows=[];manifest=read_authenticated(ROOT/'rhythm/input_manifest.json')
    wavs={p.stem.removesuffix('_mic'):p for p in (ROOT/'rhythm/guitarset').rglob('*_mic.wav')}
    for i,row in enumerate(manifest['rows']):
        key=row['key'];rp=FOLDER/(key+'.json')
        if rp.exists():
            r=read_authenticated(rp)
            if r['protocol_sha256']!=digest(FOLDER/'protocol.json') or digest(r['path'])!=r['sha256']:raise RuntimeError('Saved reference result drift')
            rows.append(r);continue
        audio_path=wavs[key]
        if digest(audio_path)!=row['audio_sha256'] or digest(row['path'])!=row['sha256']:raise RuntimeError('Reference input drift')
        wave,sr=sf.read(audio_path,dtype='float32',always_2d=True)
        torch.cuda.synchronize();begin=time.monotonic()
        beats,downbeats=tracker(wave,sr)
        torch.cuda.synchronize();elapsed=time.monotonic()-begin
        beats=np.asarray(beats);downbeats=np.asarray(downbeats)
        if not np.isfinite(beats).all() or not np.isfinite(downbeats).all():raise ValueError('Reference nonfinite events')
        path=FOLDER/(key+'.npz');np.savez_compressed(path,beats=beats,downbeats=downbeats)
        scores={}
        with np.load(row['path']) as z:
            for mode,pred in [('beats',beats),('downbeats',downbeats)]:
                truth=mir_eval.beat.trim_beats(z[mode]);pred=mir_eval.beat.trim_beats(pred);c=mir_eval.beat.continuity(truth,pred)
                scores[mode]={'f1':float(mir_eval.beat.f_measure(truth,pred)),'cmlt':float(c[1]),'amlt':float(c[3]),'reference_events':len(truth),'predicted_events':len(pred)}
        r={'key':key,'player':row['player'],'lead_sheet':row['lead_sheet'],'scores':scores,'path':str(path),'sha256':digest(path),
            'protocol_sha256':digest(FOLDER/'protocol.json'),'gpu_task_seconds':elapsed}
        authenticated_json(rp,r);rows.append(r)
        if (i+1)%30==0:print(json.dumps({'stage':'beat_this_reference','clips':i+1}),flush=True)
    if len(rows)!=358 or {r['key'] for r in rows}!={r['key'] for r in manifest['rows']}:raise ValueError('Incomplete reference cohort')
    baseline=read_authenticated(ROOT/'rhythm/report.json');base={r['key']:r for r in baseline['rows']}
    from lightsem.experiments.beat_conditioned_metrics import bootstrap_mean
    grouped={}
    for row in rows:grouped.setdefault(row['lead_sheet'],[]).append(row)
    differences={m:bootstrap_mean([np.mean([r['scores']['beats'][m]-base[r['key']]['scores']['constant_grid'][m] for r in group]) for group in grouped.values()]) for m in ('f1','cmlt','amlt')}
    authenticated_json(FOLDER/'report.json',{'rows':rows,'means':{mode:{m:float(np.mean([r['scores'][mode][m] for r in rows])) for m in ('f1','cmlt','amlt')} for mode in ('beats','downbeats')},
        'difference_from_constant_grid':differences,'gpu_task_seconds':sum(r['gpu_task_seconds'] for r in rows),
        'protocol_sha256':digest(FOLDER/'protocol.json'),'interpretation':protocol['interpretation']})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['freeze','run']);a=ap.parse_args()
    if a.stage=='freeze':freeze()
    else:
        with gpu_lock(ROOT):run()
