"""Paired3D FineDance and observed2D JustDance remain distinct batch types."""
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from .data import ROOT,read,write,sha,save_npz
from .kinematics import project_native,coarse_targets
from ..dynamic_dance.contracts import MusicTimeline
from ..dynamic_dance.audio import model_features

OLD=Path('/mnt/lightsem-data/experiments/lightsem/dynamic_dance_20261005_v1')
CONTEXT_BINS=1024


def prepare_supervision(root=ROOT):
    root=Path(root);fd=read(OLD/'learning/cache/manifest.json');jd=read(root/'data/MANIFEST.json');rows=[]
    for domain,entries in [('fd',fd['entries']),('jd',jd['entries'])]:
        for entry in entries:
            key=entry['key'];path=root/'data/supervision'/f'{domain}_{key}.npz'
            if domain=='fd':
                with np.load(entry['path']) as z:native=z['features'].copy()
                with torch.no_grad():pose,_=project_native(torch.from_numpy(native)[None],torch.ones(1,len(native),dtype=torch.bool))
                pose=pose[0].numpy();mask=np.ones(pose.shape[:2],bool)
            else:
                with np.load(entry['pose_cache']) as z:pose=z['pose'].copy();mask=z['mask'].copy()
            result=coarse_targets(pose,mask)
            if len(result['time'])>CONTEXT_BINS:raise ValueError('Whole song exceeds declared context; no silent truncation')
            save_npz(path,**result)
            rows.append({'domain':domain,'key':key,'partition':entry['partition'],'path':str(path),'sha256':sha(path),'bins':len(result['time']),'source_path':entry['path'] if domain=='fd' else entry['pose_cache'],'source_sha256':entry['sha256'] if domain=='fd' else entry['pose_cache_sha256']})
            print('SUPERVISION',domain,key,flush=True)
    report={'entries':rows,'fd_manifest_sha256':sha(OLD/'learning/cache/manifest.json'),'jd_manifest_sha256':sha(root/'data/MANIFEST.json'),'kinematics_sha256':sha(Path(__file__).with_name('kinematics.py')),'context_bins':CONTEXT_BINS,'target_units':'projected torso-lengths/s, original30Hz timestamps; no metric3D claims forJD','targets_are_motion_only':True}
    write(root/'data/SUPERVISION.json',report);return report


class Records:
    """Read immutable raw features once; normalize only sampled GPU tensors."""
    def __init__(self,root=ROOT,partition='train',frames=480):
        self.root=Path(root);self.frames=frames;self.records={'fd':[],'jd':[]};self.groups={};self.inputs={}
        fd=read(OLD/'learning/cache/manifest.json');jd=read(self.root/'data/MANIFEST.json');sup=read(self.root/'data/SUPERVISION.json');features=read(self.root/'features/COMPLETE.json')
        supervised={(r['domain'],r['key']):r for r in sup['entries']};feature_rows={r['key']:r for r in features['entries']}
        for domain,entries in [('fd',fd['entries']),('jd',jd['entries'])]:
            for row in entries:
                if row['partition']!=partition:continue
                key=row['key'];s=supervised[domain,key]
                if sha(s['path'])!=s['sha256']:raise ValueError('Changed dynamics supervision')
                with np.load(s['path']) as z:labels={k:z[k].copy() for k in z.files}
                if domain=='fd':
                    for path,digest in [(row['path'],row['sha256']),(row['audio_path'],row['audio_sha256']),(row['cache_path'],row['cache_sha256'])]:
                        if sha(path)!=digest:raise ValueError('Changed pairedFineDanceinput')
                        self.inputs[path]=digest
                    with np.load(row['path']) as z:native=z['features'].copy().astype(np.float32)
                    with np.load(row['audio_path']) as z:base=z['x'].copy().astype(np.float32)
                    with np.load(row['cache_path']) as z:extra=z['extra'].copy().astype(np.float16)
                    pose=None;mask=None;starts=np.arange(max(len(native)-frames+1,1),dtype=np.int32)[::15];group=row['canonical_id']
                else:
                    f=feature_rows[key]
                    for path,digest in [(row['pose_cache'],row['pose_cache_sha256']),(f['timeline_path'],f['timeline_sha256']),(f['base92_path'],f['base92_sha256'])]:
                        if sha(path)!=digest:raise ValueError('Changed JD input')
                        self.inputs[path]=digest
                    with np.load(row['pose_cache']) as z:pose=z['pose'].copy();mask=z['mask'].copy();starts=z['starts'].copy();target=z['time'].copy()
                    timeline=MusicTimeline.load(f['timeline_path']);extra,_=model_features(timeline,target);extra=extra.astype(np.float16)
                    with np.load(f['base92_path']) as z:base=z['x30'].copy().astype(np.float32)
                    native=None;group=row['group']
                length=len(base)
                if extra.shape!=(length,1061) or base.shape!=(length,92):raise ValueError('Fixed paired audio schema required')
                if domain=='fd' and len(native)!=length:raise ValueError('Native3D/audio lengths differ')
                if domain=='jd' and len(pose)!=length:raise ValueError('Observed2D/audio lengths differ')
                bins=(length+14)//15;pad=bins*15-length
                padded=np.pad(extra.astype(np.float32),((0,pad),(0,0)))
                coarse=padded.reshape(bins,15,1061).sum(1)/np.minimum(15,length-np.arange(bins)*15)[:,None]
                if bins>CONTEXT_BINS or len(labels['time'])!=bins:raise ValueError('Whole-song context/labels mismatch')
                self.records[domain].append({'key':key,'group':group,'base':base,'extra':extra,'coarse':coarse.astype(np.float16),'native':native,'pose':pose,'mask':mask,'starts':starts,'labels':labels,'length':length,'bins':bins})
            grouped=defaultdict(list)
            for i,row in enumerate(self.records[domain]):grouped[row['group']].append(i)
            self.groups[domain]=[grouped[k] for k in sorted(grouped)]
        if any(not v for v in self.records.values()):raise ValueError('Both real-data domains required')
    def sample(self,domain,batch,rng):
        groups=self.groups[domain];chosen=[]
        for _ in range(batch):
            group=groups[int(rng.integers(len(groups)))];record=self.records[domain][group[int(rng.integers(len(group)))]];start=int(record['starts'][int(rng.integers(len(record['starts'])))])
            chosen.append((record,start))
        return self.pack(domain,chosen)
    def pack(self,domain,chosen):
        b=len(chosen);t=self.frames
        values={'base':np.zeros((b,t,92),np.float32),'extra':np.zeros((b,t,1061),np.float16),'valid':np.zeros((b,t),bool),'indices':np.zeros((b,t),np.int64),'coarse':np.zeros((b,CONTEXT_BINS,1061),np.float16),'coarse_valid':np.zeros((b,CONTEXT_BINS),bool),'plan_target':np.zeros((b,CONTEXT_BINS,48),np.float32),'plan_valid':np.zeros((b,CONTEXT_BINS,6,3),bool),'pose_context':np.zeros((b,CONTEXT_BINS,26),np.float32),'pose_context_valid':np.zeros((b,CONTEXT_BINS),bool)}
        if domain=='fd':values['native']=np.zeros((b,t,139),np.float32)
        else:values['pose']=np.zeros((b,t,13,2),np.float32);values['pose_mask']=np.zeros((b,t,13),bool)
        examples=[]
        for i,(r,start) in enumerate(chosen):
            n=min(t,r['length']-start);bins=r['bins'];examples.append((r['key'],start,n))
            for name in ('base','extra'):values[name][i,:n]=r[name][start:start+n]
            values['valid'][i,:n]=True;values['indices'][i]=np.minimum(start+np.arange(t),r['length']-1)
            values['coarse'][i,:bins]=r['coarse'];values['coarse_valid'][i,:bins]=True
            for dst,src in [('plan_target','plan'),('plan_valid','plan_valid'),('pose_context','pose_context'),('pose_context_valid','pose_context_valid')]:values[dst][i,:bins]=r['labels'][src]
            if domain=='fd':values['native'][i,:n]=r['native'][start:start+n]
            else:
                values['pose'][i,:n]=r['pose'][start:start+n];values['pose_mask'][i,:n]=r['mask'][start:start+n]
        return {'domain':domain,'examples':examples,**values}


def to_device(batch,base_norms,extra_norms,device='cuda'):
    result={k:(torch.from_numpy(v).to(device,non_blocking=True) if isinstance(v,np.ndarray) else v) for k,v in batch.items()}
    base=(result.pop('base').float()-base_norms[0])/base_norms[1]
    extra=(result.pop('extra').float()-extra_norms[0])/extra_norms[1]
    result['audio']=torch.where(result['valid'][...,None],torch.cat((base,extra),-1),0.)
    result['coarse']=torch.where(result['coarse_valid'][...,None],(result['coarse'].float()-extra_norms[0])/extra_norms[1],0.)
    return result

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);a=p.parse_args();torch.set_num_threads(2);prepare_supervision(a.root)
