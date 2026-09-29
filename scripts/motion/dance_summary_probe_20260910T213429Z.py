"""Read-only exploratory TRAIN ridge / held-out song VAL summary diagnostic."""
import os,json,hashlib,time
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
import torch
from lightsem.experiments.motion_study.music_data import motion_summary,SUMMARY_NAMES

torch.set_num_threads(2)
HERE=Path(__file__); STEM=HERE.stem
ROOT=Path('/mnt/lightsem-data/experiments/lightsem/motion_native_audio_20260910')
PLAN=json.loads(HERE.with_name(STEM+'_plan.json').read_text())
CACHE=ROOT/'diagnostics'/STEM
CACHE.mkdir(parents=True,exist_ok=False)
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
manifest=json.loads((ROOT/'data/audio_manifest.json').read_text())
entries=sorted([e for e in manifest['entries'] if e['partition'] in ('train','val') and e['n_frames']>=960],key=lambda e:e['key'])
groups={s:{e['canonical_id'] for e in entries if e['partition']==s} for s in ('train','val')}
assert not groups['train']&groups['val']
data=[]; start=time.monotonic()
for e in entries:
    assert e['partition'] in ('train','val')
    assert digest(e['path'])==e['sha256'] and digest(e['audio_path'])==e['audio_sha256']
    with np.load(e['path']) as d: y=d['features'].copy()
    with np.load(e['audio_path']) as d:x=d['x'].copy()
    assert len(x)==len(y)==e['n_frames'] and x.shape[1]==92
    with torch.inference_mode(): y=motion_summary(torch.from_numpy(y)).numpy().astype('float64')
    x=x[:len(y)*15].reshape(len(y),15,92).mean(1).astype('float64')
    assert np.isfinite(x).all() and np.isfinite(y).all()
    np.savez_compressed(CACHE/(e['key']+'.npz'),x=x,y=y)
    data.append(dict(key=e['key'],song=e['canonical_id'],split=e['partition'],x=x,y=y[8:-8]))
print('EXTRACTED',len(data),'seconds',time.monotonic()-start,flush=True)
def weights(rows):
    counts=Counter(r['song'] for r in rows)
    return [1/(len(counts)*counts[r['song']]) for r in rows]
def mom(arr,w):
    mean=sum(ww*a.mean(0) for a,ww in zip(arr,w))
    var=sum(ww*(a*a).mean(0) for a,ww in zip(arr,w))-mean*mean
    return mean,np.sqrt(np.maximum(var,1e-8))
train=[r for r in data if r['split']=='train']; tw=weights(train)
ym,ys=mom([r['y'] for r in train],tw);ys=np.maximum(ys,.01)
def make_x(r,lag):
    return r['x'][8+lag:len(r['x'])-8+lag]
def fit(lag,centered):
    xx=[make_x(r,lag) for r in train]; yy=[r['y'] for r in train]
    if centered:
        xx=[a-a.mean(0) for a in xx]; yy=[a-a.mean(0) for a in yy]
    xm,xs=mom(xx,tw);xs=np.maximum(xs,.01)
    ymean=np.zeros_like(ym) if centered else ym
    cov=np.zeros((92,92));cross=np.zeros((92,18))
    for x,y,w in zip(xx,yy,tw):
        x=(x-xm)/xs;y=(y-ymean)/ys
        cov+=w*x.T@x/len(x);cross+=w*x.T@y/len(x)
    coef=np.linalg.solve(cov+.1*np.eye(92),cross)
    return xm,xs,ymean,coef

def metrics(y,pred):
    err=pred-y
    return {'mse':float(np.mean((err/ys)**2)),'mae_channels':np.mean(np.abs(err),0).tolist(),
            'mse_channels':np.mean(err**2,0).tolist()}
def aggregate(rows):
    grouped=defaultdict(list)
    for r in rows:grouped[r['song']].append(r)
    songs={}
    for song,rr in grouped.items():
        songs[song]={cond:{metric:np.mean([r['conditions'][cond][metric] for r in rr],axis=0).tolist() for metric in rr[0]['conditions'][cond]} for cond in rr[0]['conditions']}
    agg={cond:{metric:np.mean([r[cond][metric] for r in songs.values()],axis=0).tolist() for metric in next(iter(songs.values()))[cond]} for cond in next(iter(songs.values()))}
    return {'aggregate':agg,'per_song':songs,'records':rows}

results={};models={}
for lag_sec in PLAN['lags_seconds']:
    lag=int(lag_sec*2);label=str(lag_sec)
    xm,xs,yymean,coef=fit(lag,True)
    models['within_'+label]=dict(x_mean=xm.tolist(),x_std=xs.tolist(),coefficient=coef.tolist())
    results[label]={}
    for split in ('train','val'):
        records=[r for r in data if r['split']==split]; rows=[]
        for idx,r in enumerate(records):
            y=r['y']-r['y'].mean(0); x=make_x(r,lag);x=x-x.mean(0)
            pred=((x-xm)/xs)@coef*ys
            donor=next(records[(idx+j)%len(records)] for j in range(1,len(records)) if records[(idx+j)%len(records)]['song']!=r['song'])
            dx=make_x(donor,lag); dx=dx-dx.mean(0)
            # Deterministic normalized-time resampling; no target-based donor selection.
            dx=np.stack([np.interp(np.linspace(0,1,len(x)),np.linspace(0,1,len(dx)),dx[:,c]) for c in range(92)],axis=1)
            dx-=dx.mean(0)
            dp=((dx-xm)/xs)@coef*ys
            cond={'correct':metrics(y,pred),'oracle_track_constant':metrics(y,np.zeros_like(y)),'donor':metrics(y,dp)}
            sh=[metrics(y,np.roll(pred,int(round(frac*len(pred))),axis=0)) for frac in (.25,.5,.75)]
            cond['within_shift']={k:np.mean([a[k] for a in sh],axis=0).tolist() for k in sh[0]}
            rows.append({'key':r['key'],'song':r['song'],'bins':len(y),'donor':donor['key'],'conditions':cond})
        results[label][split]=aggregate(rows)
    print('LAG',lag_sec,{s:results[label][s]['aggregate']['correct']['mse'] for s in ('train','val')},flush=True)

# Raw global fit is solely the zero-lag complementary identity-sensitive diagnostic.
xm,xs,yymean,coef=fit(0,False)
models['raw_zero']=dict(x_mean=xm.tolist(),x_std=xs.tolist(),y_mean=yymean.tolist(),coefficient=coef.tolist())
raw={}; spread={}
for split in ('train','val'):
    records=[r for r in data if r['split']==split];rows=[]
    ww=weights(records); yy=[r['y'] for r in records]
    mean,sd=mom(yy,ww)
    within=np.sqrt(sum(w*((a-a.mean(0))**2).mean(0) for a,w in zip(yy,ww)))
    spread[split]={'mean':mean.tolist(),'global_sd':sd.tolist(),'within_record_sd':within.tolist()}
    for r in records:
        x=make_x(r,0); y=r['y']
        pred=((x-xm)/xs)@coef*ys+yymean
        const=((x.mean(0)-xm)/xs)@coef*ys+yymean
        conditions={name:metrics(y,a) for name,a in [('correct',pred),('train_mean',ym),('oracle_track_constant',y.mean(0)),('audio_track_constant',const)]}
        rows.append({'key':r['key'],'song':r['song'],'conditions':conditions})
    raw[split]=aggregate(rows)

# Identical bootstrap draws across comparisons. Intervals are exploratory, unadjusted.
song_ids=sorted(results['0']['val']['per_song'])
rng=np.random.default_rng(20260910);boot=rng.integers(0,len(song_ids),(10000,len(song_ids)))
def ci(values):
    a=np.asarray(values);return {'mean':float(a.mean()),'ci95':np.quantile(a[boot].mean(1),[.025,.975]).tolist(),'songs_positive':int((a>0).sum())}
comparisons={}
for lag in results:
    songs=results[lag]['val']['per_song']
    comparisons[lag]={null:ci([songs[g][null]['mse']-songs[g]['correct']['mse'] for g in song_ids]) for null in ('oracle_track_constant','donor','within_shift')}
    comparisons[lag]['zero_lag_minus_this_lag']=ci([results['0']['val']['per_song'][g]['correct']['mse']-songs[g]['correct']['mse'] for g in song_ids])
summary={'plan':PLAN,'plan_sha256':digest(HERE.with_name(STEM+'_plan.json')),'script_sha256':digest(HERE),
 'input_manifest_sha256':digest(ROOT/'data/audio_manifest.json'),'summary_source_sha256':digest(Path('/root/lightsem/lightsem/experiments/motion_study/music_data.py')),
 'names':list(SUMMARY_NAMES),'target_train_mean':ym.tolist(),'target_train_std':ys.tolist(),'cohort':{s:{'records':sum(r['split']==s for r in data),'songs':len(groups[s]),'bins':sum(len(r['y']) for r in data if r['split']==s)} for s in ('train','val')},
 'excluded_short':[{'key':e['key'],'split':e['partition'],'frames':e['n_frames']} for e in manifest['entries'] if e['partition'] in ('train','val') and e['n_frames']<960],
 'results':results,'raw_zero_lag':raw,'spread':spread,'bootstrap':comparisons,
 'per_song_descriptive_best_lag':{g:min(results,key=lambda lag:results[lag]['val']['per_song'][g]['correct']['mse']) for g in song_ids},
 'cache':str(CACHE),'seconds':time.monotonic()-start,'test_access':False,'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES')}
HERE.with_suffix('.json').write_text(json.dumps(summary,indent=2)+'\n')
(CACHE/'models.json').write_text(json.dumps(models))
print('DONE',HERE.with_suffix('.json'),'seconds',summary['seconds'],flush=True)
