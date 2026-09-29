"""Bounded TRAIN-only MERT/complex-phase controls; not another main training arm."""
from pathlib import Path
import argparse,json,time
import numpy as np
from scipy.signal import resample_poly
from scipy.ndimage import uniform_filter1d
from .night_common import ROOT,STAGE,base_context
from .night_auxiliary import auxiliary_for_entry,interpolate_native
from .night_synthetic import complex_novelty
from ..stage11.stage11_common import authenticated_json,read_authenticated,digest,gpu_lock

MERT=Path('/mnt/lightsem-data/models/hf/models--m-a-p--MERT-v1-330M/snapshots/5240c2708a5acaee1007f43fb9735c7dcd0b78c9')

def cohort():
    m,n=base_context();unique={}
    for e in sorted(m['entries'],key=lambda e:(e['canonical_id'],e['key'])):
        if e['partition']=='train':unique.setdefault(e['canonical_id'],e)
    entries=list(unique.values());ix=np.linspace(0,len(entries)-1,80,dtype=int)
    return [entries[i] for i in ix]

def extract(root=ROOT):
    import torch,soundfile as sf,soxr
    from transformers import AutoModel,AutoConfig,Wav2Vec2FeatureExtractor
    from ..stage11.stage11_data import load_manifest
    root=Path(root);folder=root/'diagnostics/embedding_controls';folder.mkdir(exist_ok=True)
    records={r['recording_id']:r for r in load_manifest(STAGE)['recordings']};rows=[]
    config=AutoConfig.from_pretrained(str(MERT),trust_remote_code=True,local_files_only=True)
    model=AutoModel.from_config(config,trust_remote_code=True)
    state=torch.load(MERT/'pytorch_model.bin',map_location='cpu',weights_only=True)
    converted={};key_mapping={};expected=set(model.state_dict())
    for key,value in state.items():
        new=key
        if key not in expected and key.endswith('.weight_g'):new=key.removesuffix('.weight_g')+'.parametrizations.weight.original0'
        if key not in expected and key.endswith('.weight_v'):new=key.removesuffix('.weight_v')+'.parametrizations.weight.original1'
        if new!=key:key_mapping[key]=new
        converted[new]=value
    model.load_state_dict(converted,strict=True)
    if any(not torch.equal(v,converted[k]) for k,v in model.state_dict().items()):raise RuntimeError('MERT checkpoint tensor mismatch')
    weight_audit={'all_tensors_equal':True,'tensor_count':len(converted),'key_mapping':key_mapping,'checkpoint_sha256':digest(MERT/'pytorch_model.bin')}
    authenticated_json(folder/'weight_audit.json',weight_audit)
    del state,converted
    model=model.cuda().float().eval()
    processor=Wav2Vec2FeatureExtractor.from_pretrained(str(MERT),local_files_only=True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    step=1;receptive=1
    for k,s in zip(model.config.conv_kernel,model.config.conv_stride):receptive+=(k-1)*step;step*=s
    center=(receptive-1)/2/24000;rate=24000/step
    if rate!=75:raise ValueError('Unexpected MERT frame rate')
    begun=time.monotonic()
    for i,e in enumerate(cohort()):
        key=e['key'];rp=folder/(key+'.json')
        if rp.exists():rows.append(read_authenticated(rp));continue
        start=(e['n_frames']-2400)//4;times=e['offset_s']+(start+np.arange(2400))/40.
        x,sr=sf.read(records[e['recording_id']]['wav_path'],dtype='float32',always_2d=True);x=x.mean(1)
        native,sample_rate=x,sr
        if sr!=24000:x=soxr.resample(x,sr,24000)
        padded=np.pad(x,(5*24000,30*24000),mode='reflect');out=np.empty((2400,1024),np.float32)
        with torch.inference_mode():
            for block in sorted(set(np.floor(times/20).astype(int))):
                s=block*20*24000;wave=padded[s:s+30*24000]
                inp=processor(wave,sampling_rate=24000,return_tensors='pt')
                h=model(**{k:v.cuda() for k,v in inp.items()},output_hidden_states=True).hidden_states
                if len(h)!=25:raise ValueError('MERT hidden-state indexing changed')
                z=(.5*(h[15][0]+h[16][0])).cpu().numpy()
                z=resample_poly(z,8,15,axis=0,window=('kaiser',8.6),padtype='line')
                which=np.floor(times/20).astype(int)==block
                local=times[which]-(block*20-5+center)
                out[which]=interpolate_native(z,local,rate=40.)
        segment=native[round(times[0]*sample_rate):round((times[0]+60)*sample_rate)]
        complex_,ct=complex_novelty(segment,sample_rate)
        complex_=np.stack([np.interp(np.arange(2400)/40,ct,complex_[:,j]) for j in range(4)],1)
        path=folder/(key+'.npz');np.savez_compressed(path,mert=out.astype(np.float16),complex_novelty=complex_.astype(np.float32))
        row={'key':key,'canonical_id':e['canonical_id'],'start_frame':start,'entry':e,'path':str(path),'sha256':digest(path)}
        authenticated_json(rp,row);rows.append(row)
        if i%10==0:print(json.dumps({'stage':'mert_control','songs':i+1,'elapsed_s':time.monotonic()-begun}),flush=True)
    authenticated_json(folder/'input_manifest.json',{'rows':rows,'model':str(MERT),'layers':[15,16],
        'preprocessor':'official Wav2Vec2FeatureExtractor on each30s context; center20s retained',
        'anti_alias':'75Hz to40Hz resample_poly8/15, convolution receptive-center timestamp retained',
        'partition':'train','weight_audit':weight_audit,'gpu_task_seconds':time.monotonic()-begun,'source_sha256':digest(__file__)})

def evaluate(root=ROOT):
    import torch
    root=Path(root);folder=root/'diagnostics/embedding_controls';manifest=read_authenticated(folder/'input_manifest.json');m,n=base_context()
    names=['com_speed','vert_bounce_amp','qom','hand_speed','foot_speed'];ii=[n['target_names'].index(k) for k in names]
    data=[];rng=np.random.default_rng(4421);projection=rng.standard_normal((92,1024)).astype(np.float32)/np.sqrt(92)
    for row in manifest['rows']:
        e=row['entry'];s=row['start_frame']
        if e['partition']!='train' or digest(row['path'])!=row['sha256']:raise RuntimeError('Audit input drift')
        with np.load(e['path']) as z:base=z['x'][s:s+2400];target=uniform_filter1d(z['y'][:,ii],40,axis=0,mode='nearest')[s:s+2400]
        with np.load(row['path']) as z:mert=z['mert'].astype(np.float32);phase=z['complex_novelty']
        features={'base':base,'fourier':np.concatenate((base,auxiliary_for_entry('fourier',e,root,s,2400)),1),
            'muq':np.concatenate((base,auxiliary_for_entry('muq',e,root,s,2400)),1),'mert':np.concatenate((base,mert),1),
            'complex_phase':np.concatenate((base,phase),1),'width_control':np.concatenate((base,base@projection),1)}
        # Matching sampling and target scales for all feature families.
        data.append({'key':e['canonical_id'],'target':target[40:-40:40],
                     'features':{k:v[40:-40:40] for k,v in features.items()}})
    results={};begun=time.monotonic();truth=np.concatenate([d['target'] for d in data]);keys=np.concatenate([np.repeat(d['key'],len(d['target'])) for d in data])
    unique=sorted(set(keys));order=np.random.default_rng(4421).permutation(unique);fold_map={k:i%5 for i,k in enumerate(order)};fold=np.array([fold_map[k] for k in keys])
    for family in data[0]['features']:
        x=torch.from_numpy(np.concatenate([d['features'][family] for d in data]).astype(np.float32)).cuda();y=torch.tensor(truth,dtype=torch.float32,device='cuda');pred=np.empty_like(truth)
        for f in range(5):
            tr=torch.from_numpy(fold!=f).cuda();te=~tr;mu=x[tr].mean(0);sd=x[tr].std(0).clamp_min(1e-5)
            a=(x[tr]-mu)/sd;ym=y[tr].mean(0);ys=y[tr].std(0).clamp_min(1e-5);b=(y[tr]-ym)/ys
            w=torch.linalg.solve(a.T@a/len(a)+.1*torch.eye(a.shape[1],device='cuda'),a.T@b/len(a))
            pred[fold==f]=(((x[te]-mu)/sd@w)*ys+ym).cpu().numpy()
        rows=[]
        for key in unique:
            keep=keys==key;t=truth[keep];p=pred[keep];var=t.var(0);ok=var>1e-8
            rows.append({'canonical_id':key,'r2':float((1-np.square(t-p).mean(0)[ok]/var[ok]).mean()),
                         'within_song_centered_r2':float((1-np.square((t-t.mean(0))-(p-p.mean(0))).mean(0)[ok]/var[ok]).mean())})
        results[family]={'rows':rows,'macro_r2':float(np.mean([r['r2'] for r in rows]))}
        np.savez_compressed(folder/(family+'_predictions.npz'),prediction=pred,truth=truth,canonical_ids=keys,fold=fold)
        print(json.dumps({'stage':'embedding_control_probe','family':family,'r2':results[family]['macro_r2']}),flush=True)
        del x,y,a,b,w;torch.cuda.empty_cache()
    authenticated_json(folder/'report.json',{'results':results,'targets':names,'songs':len(unique),'partition':'train',
        'interpretation':'Development probes on80predeclaredTRAINsongs, not new held-out confirmation and not a main-arm selection rule',
        'gpu_task_seconds':time.monotonic()-begun,'source_sha256':digest(__file__)})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['extract','evaluate']);a=p.parse_args()
    with gpu_lock(ROOT):{'extract':extract,'evaluate':evaluate}[a.stage]()
