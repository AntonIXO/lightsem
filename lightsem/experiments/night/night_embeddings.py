"""Pinned, frozen MuQ features on original waveforms; run in encoder environment."""
from pathlib import Path
import argparse
import json
import time
import numpy as np
from .night_common import ROOT, BASE, STAGE, MUQ_REVISION, initial_setup
from ..stage11.stage11_common import authenticated_json, read_authenticated, digest, artifact_manifest, gpu_lock

MODEL_ID='OpenMuQ/MuQ-large-msd-iter'
LAYERS=(3,6,9,12)
MODEL_CACHE=Path('/mnt/lightsem-data/models/night_encoders')

def download(root=ROOT):
    from huggingface_hub import snapshot_download
    path=snapshot_download(MODEL_ID,revision=MUQ_REVISION,cache_dir=str(MODEL_CACHE),
                           allow_patterns=['config.json','model.safetensors','README.md'])
    cfg=json.loads((Path(path)/'config.json').read_text())
    assert cfg['encoder_depth']==12 and cfg['label_rate']==25
    result={'model_id':MODEL_ID,'revision':MUQ_REVISION,'path':path,
            'artifacts':artifact_manifest([Path(path)/'config.json',Path(path)/'model.safetensors']),
            'native_hz':25,'sample_rate':24000,'layers':list(LAYERS),'precision':'fp32',
            'code_version':'muq==0.1.0','weights_license':'CC-BY-NC-4.0'}
    authenticated_json(Path(root)/'features/muq_model.json',result);return result

def extract_waveform(model,waveform):
    import torch
    sr=24000;rate=25;core=20*sr;context=5*sr;size=30*sr
    waveform=np.asarray(waveform,np.float32)
    if waveform.ndim!=1 or len(waveform)<sr or not np.isfinite(waveform).all():raise ValueError('Bad waveform')
    padded=np.pad(waveform,(context,size),mode='reflect')
    n=int(np.ceil(len(waveform)/sr*rate));out={k:[] for k in LAYERS}
    with torch.inference_mode():
        for start in range(0,len(waveform),core):
            x=torch.from_numpy(padded[start:start+size])[None].cuda()
            result=model(x,output_hidden_states=True)
            if len(result.hidden_states)!=13:raise ValueError('Unexpected MuQ layer indexing')
            take=min(20*rate,n-start//(sr//rate))
            for layer in LAYERS:
                z=result.hidden_states[layer][0,5*rate:5*rate+take]
                if z.shape!=(take,1024) or not torch.isfinite(z).all():raise FloatingPointError('MuQ output invalid')
                out[layer].append(z.cpu().numpy().astype(np.float16))
    return {k:np.concatenate(v) for k,v in out.items()}

def extract(root=ROOT):
    import torch,soundfile as sf,soxr
    from muq import MuQ
    from ..stage11.stage11_data import load_manifest
    m,n,_=initial_setup(root);root=Path(root);folder=root/'features/muq_native';folder.mkdir(exist_ok=True)
    info=read_authenticated(root/'features/muq_model.json')
    from ..stage11.stage11_common import verify_artifacts
    verify_artifacts(info['artifacts'])
    torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=MuQ.from_pretrained(info['path']).cuda().float().eval()
    if model.config.encoder_depth!=12 or len(model.model.conformer.layers)!=12:raise ValueError('Model config drift')
    records={r['recording_id']:r for r in load_manifest(STAGE)['recordings']}
    selected=sorted({e['recording_id'] for e in m['entries']})
    rows=[];started=time.monotonic();gpu_seconds=0.
    for index,key in enumerate(selected):
        r=records[key]
        if r['partition'] not in ('train','val'):raise RuntimeError('TEST waveform prohibited')
        rp=folder/(key+'.json')
        if rp.exists():
            result=read_authenticated(rp);verify_artifacts(result['artifacts']);rows.append(result);continue
        wav=Path(r['wav_path']);x,sr=sf.read(wav,dtype='float32',always_2d=True);x=x.mean(1)
        if sr!=24000:x=soxr.resample(x,sr,24000)
        torch.cuda.synchronize();t=time.monotonic();features=extract_waveform(model,x);torch.cuda.synchronize();elapsed=time.monotonic()-t
        gpu_seconds+=elapsed;paths=[]
        for layer,z in features.items():
            p=folder/(key+f'_layer{layer}.npy')
            if p.exists():raise RuntimeError('Unverified partial embedding exists: '+str(p))
            np.save(p,z);paths.append(p)
        result={'recording_id':key,'canonical_id':r['canonical_id'],'partition':r['partition'],
                'source_waveform':str(wav),'waveform_sha256':digest(wav),'frames':len(features[3]),
                'native_hz':25.,'time_origin_s':0.,'context_s':30,'core_s':20,'boundary':'reflect5s left / enough right, retain middle20s',
                'gpu_seconds':elapsed,'model_revision':MUQ_REVISION,'artifacts':artifact_manifest(paths)}
        authenticated_json(rp,result);rows.append(result)
        if index%20==0:print(json.dumps({'stage':'muq_extract','recordings':index+1,'total':len(selected),
                                        'gpu_seconds':gpu_seconds,'elapsed_s':time.monotonic()-started}),flush=True)
    report={'rows':rows,'gpu_seconds':sum(r['gpu_seconds'] for r in rows),'elapsed_s':time.monotonic()-started,
            'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'model':info,'source_sha256':digest(__file__)}
    authenticated_json(folder/'report.json',report)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['download','extract']);p.add_argument('--root',type=Path,default=ROOT)
    a=p.parse_args()
    if a.stage=='download':download(a.root)
    else:
        with gpu_lock(a.root):extract(a.root)
