"""TRAIN-only layer selection and memory-mapped auxiliary training arrays."""
from pathlib import Path
import argparse,json,time
import numpy as np
from scipy.ndimage import uniform_filter1d
from .night_common import ROOT,base_context,feature_path
from .night_embeddings import LAYERS
from ..stage11.stage11_common import authenticated_json,read_authenticated,digest,gpu_lock
from ..dance_no_clock.dance_no_clock_data import hierarchical_weights

def interpolate_native(array,times,rate=25.):
    u=np.clip(np.asarray(times,dtype=np.float64)*rate,0,len(array)-1)
    lo=np.floor(u).astype(np.int64);hi=np.minimum(lo+1,len(array)-1);frac=(u-lo).astype(np.float32)
    out=np.empty((len(times),array.shape[1]),np.float32)
    for start in range(0,len(times),2048):
        end=min(start+2048,len(times));a=np.asarray(array[lo[start:end]],np.float32);b=np.asarray(array[hi[start:end]],np.float32)
        out[start:end]=a+(b-a)*frac[start:end,None]
    return out

def auxiliary_for_entry(family,e,root=ROOT,start=0,frames=None,layer=None):
    root=Path(root);frames=e['n_frames']-start if frames is None else frames
    if family=='base':return np.zeros((frames,0),np.float32)
    if family=='fourier':return np.asarray(np.load(feature_path(family,e['key'],root),mmap_mode='r')[start:start+frames],np.float32)
    if layer is None:layer=read_authenticated(root/'features/muq/layer_selection.json')['selected_layer']
    array=np.load(root/'features/muq_native'/(e['recording_id']+f'_layer{layer}.npy'),mmap_mode='r')
    times=e['offset_s']+(start+np.arange(frames))/40.
    if times[0]<0 or times[-1]>(len(array)-1)/25+.041:raise ValueError('Auxiliary timestamps outside recording')
    return interpolate_native(array,times)

def select_layer(root=ROOT):
    import torch
    root=Path(root);m,n=base_context();dest=root/'features/muq';dest.mkdir(exist_ok=True)
    if (dest/'layer_selection.json').exists():return read_authenticated(dest/'layer_selection.json')
    train=[e for e in m['entries'] if e['partition']=='train'];channels=['com_speed','vert_bounce_amp','qom','hand_speed','foot_speed']
    indices=[n['target_names'].index(k) for k in channels];samples=[];labels=[];song_keys=[];entry_slices=[];offset=0
    for e in train:
        # One-second means, at 2s spacing, interior only. Exact same rows across layers.
        ii=np.arange(40,e['n_frames']-40,80)
        with np.load(e['path']) as z:y=uniform_filter1d(z['y'][:,indices],40,axis=0,mode='nearest')[ii]
        labels.append(y);samples.append((e,ii));song_keys.extend([e['canonical_id']]*len(ii));entry_slices.append((offset,offset+len(ii)));offset+=len(ii)
    truth=np.concatenate(labels);song_keys=np.asarray(song_keys);unique=sorted(set(song_keys));rng=np.random.default_rng(991)
    hierarchy=hierarchical_weights(train)
    row_weights=np.concatenate([np.full(len(ii),hierarchy[e['key']]/len(ii),np.float32) for e,ii in samples])
    perm=rng.permutation(unique);fold_map={k:i%5 for i,k in enumerate(perm)};fold=np.asarray([fold_map[k] for k in song_keys]);results=[]
    device='cuda';begun=time.monotonic()
    for layer in LAYERS:
        features=[]
        for e,ii in samples:
            a=np.load(root/'features/muq_native'/(e['recording_id']+f'_layer{layer}.npy'),mmap_mode='r')
            features.append(interpolate_native(a,e['offset_s']+ii/40.))
        x=torch.from_numpy(np.concatenate(features)).cuda();y=torch.from_numpy(truth).cuda();pred=np.empty_like(truth)
        weights=torch.from_numpy(row_weights).cuda()
        del features
        for f in range(5):
            tr=torch.from_numpy(fold!=f).cuda();te=~tr
            w=weights[tr];w=w/w.sum()
            mean=(x[tr]*w[:,None]).sum(0);std=((x[tr]-mean).square()*w[:,None]).sum(0).sqrt().clamp_min(1e-5);a=(x[tr]-mean)/std
            ym=(y[tr]*w[:,None]).sum(0);ys=((y[tr]-ym).square()*w[:,None]).sum(0).sqrt().clamp_min(1e-5);b=(y[tr]-ym)/ys
            coef=torch.linalg.solve(a.T@(a*w[:,None])+.01*torch.eye(a.shape[1],device=device),a.T@(b*w[:,None]))
            pred[fold==f]=(((x[te]-mean)/std@coef)*ys+ym).cpu().numpy()
        rows=[]
        for key in unique:
            mask=song_keys==key;yy=truth[mask];pp=pred[mask];variance=np.square(yy-yy.mean(0)).mean(0)
            supported=variance>1e-8
            r2=1-np.square(yy-pp).mean(0)[supported]/variance[supported]
            centered=1-np.square((yy-yy.mean(0))-(pp-pp.mean(0))).mean(0)[supported]/variance[supported]
            rows.append({'canonical_id':key,'r2':float(r2.mean()) if len(r2) else None,
                         'within_song_centered_r2':float(centered.mean()) if len(centered) else None})
        score=float(np.mean([r['r2'] for r in rows if r['r2'] is not None]))
        results.append({'layer':layer,'score':score,'rows':rows});np.savez_compressed(dest/f'layer{layer}_oof.npz',prediction=pred,truth=truth,song_keys=song_keys,fold=fold)
        print(json.dumps({'stage':'layer_selection','layer':layer,'oof_macro_song_r2':score}),flush=True)
        del x,y,a,b,coef,tr,te,weights;torch.cuda.empty_cache()
    best=max(results,key=lambda r:(r['score'],-r['layer']))['layer']
    result={'selected_layer':best,'candidates':list(LAYERS),'selection':'highest TRAIN five-fold canonical-song macro R2; earlier layer breaks ties',
            'target_channels':channels,'rows':results,'fit_partition':'train','val_used':False,'test_used':False,
            'gpu_seconds':time.monotonic()-begun,'source_sha256':digest(__file__)}
    authenticated_json(dest/'layer_selection.json',result);return result

def consolidate(family,root=ROOT):
    root=Path(root);m,n=base_context();entries=[e for e in m['entries'] if e['partition']=='train' and e['n_frames']>=640]
    folder=root/'features'/family;folder.mkdir(exist_ok=True);rp=folder/'training_array.json'
    if rp.exists():return read_authenticated(rp)
    dimension=294 if family=='fourier' else 1024;count=sum(e['n_frames'] for e in entries)
    path=folder/'training.npy'
    if path.exists():raise RuntimeError('Unverified partial consolidated array: '+str(path))
    out=np.lib.format.open_memmap(path,mode='w+',dtype=np.float16,shape=(count,dimension))
    weights=hierarchical_weights(entries);mean=np.zeros(dimension);square=mean.copy();offset=0;begin=time.monotonic()
    sources=[]
    for i,e in enumerate(entries):
        f=auxiliary_for_entry(family,e,root).astype(np.float16)
        if not np.isfinite(f).all():raise FloatingPointError('Auxiliary quantization overflow')
        out[offset:offset+len(f)]=f;offset+=len(f);w=weights[e['key']]
        mean+=w*f.astype(np.float64).mean(0);square+=w*np.square(f.astype(np.float64)).mean(0)
        sources.append({'key':e['key'],'source_tensor_sha256':e['sha256']})
        if i%100==0:print(json.dumps({'stage':'consolidate_'+family,'entries':i+1,'elapsed_s':time.monotonic()-begin}),flush=True)
    out.flush();del out
    result={'family':family,'path':str(path),'sha256':digest(path),'shape':[count,dimension],'dtype':'float16',
            'mean':mean.tolist(),'std':np.sqrt(np.maximum(square-mean**2,1e-10)).tolist(),
            'fit_partition':'train','weighting':'equal canonical song / recording / coach / frame',
            'entry_keys':[e['key'] for e in entries],'sources':sources,'source_sha256':digest(__file__)}
    if family=='muq':result['layer_selection_sha256']=digest(folder/'layer_selection.json')
    else:result['feature_report_sha256']=digest(folder/'report.json')
    authenticated_json(rp,result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['select','fourier','muq']);a=p.parse_args()
    if a.stage=='select':
        with gpu_lock(ROOT):select_layer()
    else:consolidate(a.stage)
