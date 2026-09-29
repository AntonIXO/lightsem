"""GPU reconstruction probes distinguish descriptor loss from audio predictability."""
from pathlib import Path
import argparse,gc,json,math,time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .night_common import ROOT,BASE,base_context
from .night_training import NightResident
from .night_auxiliary import auxiliary_for_entry
from ..stage11.stage11_common import (authenticated_json,read_authenticated,atomic_torch,digest,artifact_manifest,
                            seed_all,gpu_lock,verify_artifacts)
from ..dance_no_clock.dance_no_clock_data import hierarchical_weights

PROBES=('descriptor128','descriptor256','pose_code24','audio_base','audio_fourier','audio_muq')

class PoseDecoder(nn.Module):
    def __init__(self,input_dim,width=128,autoencoder=False):
        super().__init__();self.autoencoder=autoencoder;self.width=width;self.input_dim=input_dim
        self.pose_code=nn.Sequential(nn.Linear(34,128),nn.SiLU(),nn.Linear(128,24)) if autoencoder else None
        self.input=nn.Linear(24 if autoencoder else input_dim,width)
        self.blocks=nn.ModuleList([nn.Conv1d(width,width,5,padding=2*d,dilation=d) for d in (1,2,4,8,16)])
        self.norms=nn.ModuleList([nn.LayerNorm(width) for _ in self.blocks]);self.output=nn.Linear(width,34)

    def forward(self,x):
        if self.pose_code is not None:x=self.pose_code(x)
        h=self.input(x)
        for norm,conv in zip(self.norms,self.blocks):h=h+F.silu(conv(norm(h).transpose(1,2)).transpose(1,2))
        return self.output(h)

    @torch.no_grad()
    def predict(self,x,chunk=640):
        if self.input.weight.device.type!='cuda':raise RuntimeError('GPU probe inference required')
        radius=62;out=[];self.eval()
        for start in range(0,x.shape[1],chunk):
            end=min(start+chunk,x.shape[1]);lo=max(0,start-radius);hi=min(x.shape[1],end+radius)
            out.append(self(x[:,lo:hi].cuda())[:,start-lo:end-lo].cpu())
        return torch.cat(out,1)

def load_pose_training(data,root=ROOT):
    root=Path(root);weights=hierarchical_weights(data.entries);ys=[];ms=[];mean=np.zeros(34);sq=mean.copy();support=mean.copy()
    for e in data.entries:
        with np.load(root/'features/pose'/(e['key']+'.npz')) as z:y=z['pose'];m=z['mask']
        count=m.sum(0);ok=count>0;w=weights[e['key']]
        mean[ok]+=w*(y*m).sum(0,dtype=np.float64)[ok]/count[ok]
        sq[ok]+=w*(y.astype(float)**2*m).sum(0)[ok]/count[ok];support[ok]+=w
        ys.append(y);ms.append(m)
    mean/=np.maximum(support,1e-9);sq/=np.maximum(support,1e-9);std=np.sqrt(np.maximum(sq-mean**2,1e-4))
    y=torch.from_numpy(np.concatenate(ys)).cuda();m=torch.from_numpy(np.concatenate(ms)).cuda()
    y=(y-torch.tensor(mean,device='cuda',dtype=torch.float32))/torch.tensor(std,device='cuda',dtype=torch.float32)
    norms={'mean':mean.tolist(),'std':std.tolist(),'fit_partition':'train','support':support.tolist()}
    return y,m,norms

def input_for(name,x,descriptors,events,pose):
    if name.startswith('descriptor'):return torch.cat((descriptors,events),dim=-1)
    if name=='pose_code24':return pose
    return x

def pose_errors(pred,truth,mask):
    pred=np.asarray(pred);truth=np.asarray(truth);mask=np.asarray(mask,bool)
    joint=np.linalg.norm((pred-truth).reshape(-1,17,2),axis=2);valid=mask.reshape(-1,17,2).all(2)
    v=np.linalg.norm((np.diff(pred,axis=0)-np.diff(truth,axis=0)).reshape(-1,17,2)*40,axis=2)
    vm=valid[1:]&valid[:-1]
    from lightsem.motion.skeleton import LHIP,RHIP
    root_pred=.5*(pred.reshape(-1,17,2)[:,LHIP]+pred.reshape(-1,17,2)[:,RHIP])
    root_true=.5*(truth.reshape(-1,17,2)[:,LHIP]+truth.reshape(-1,17,2)[:,RHIP])
    rv=valid[:,LHIP]&valid[:,RHIP]
    edges=((5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16))
    pp=pred.reshape(-1,17,2);tt=truth.reshape(-1,17,2);bone=[]
    for a,b in edges:
        mm=valid[:,a]&valid[:,b]
        if mm.any():bone.extend(abs(np.linalg.norm(pp[:,a]-pp[:,b],axis=1)-np.linalg.norm(tt[:,a]-tt[:,b],axis=1))[mm])
    return {'joint_error_tl':float(joint[valid].mean()) if valid.any() else None,
            'velocity_error_tl_s':float(v[vm].mean()) if vm.any() else None,
            'root_error_tl':float(np.linalg.norm(root_pred-root_true,axis=1)[rv].mean()) if rv.any() else None,
            'bone_length_error_tl':float(np.mean(bone)) if bone else None,'valid_joints':int(valid.sum()),
            'prediction_std':float(pred.std(0).mean()),'truth_std':float(truth.std(0).mean())}

def evaluation_rows():
    return read_authenticated(BASE/'evaluation/input_manifest.json')['rows']

def evaluate_probe(model,name,seed,pose_norms,root=ROOT):
    root=Path(root);m,n=base_context();folder=root/'diagnostics/pose_probes'/name/f'seed_{seed}';rows=[]
    mu=np.asarray(pose_norms['mean'],np.float32);std=np.asarray(pose_norms['std'],np.float32)
    family=name.removeprefix('audio_') if name.startswith('audio_') else 'base'
    auxnorm=read_authenticated(root/'features'/family/'training_array.json') if family!='base' else None
    for row in evaluation_rows():
        e=row['entry'];key=e['key'];start=row['start_frame'];count=row['frames']
        with np.load(row['path']) as z:
            raw=z['x'];d=(z['y']-n['y_center'])/n['y_scale'];b=z['b']
        with np.load(root/'features/pose'/(key+'.npz')) as z:truth=z['pose'][start:start+count];mask=z['mask'][start:start+count]
        x=(raw-n['x_mean'])/n['x_std']
        if family!='base':
            a=auxiliary_for_entry(family,e,root,start,count).astype(np.float16).astype(np.float32)
            x=np.concatenate((x,(a-auxnorm['mean'])/auxnorm['std']),axis=1)
        tensors=[torch.tensor(v,dtype=torch.float32)[None] for v in (x,d,b,(truth-mu)/std)]
        inp=input_for(name,*tensors)
        pred=model.predict(inp)[0].numpy()*std+mu
        modes={'full':pred}
        if name!='pose_code24':
            shifted=model.predict(torch.roll(inp,400,1))[0].numpy()*std+mu
            modes['shift10s']=shifted
        if name.startswith('descriptor'):
            with np.load(BASE/'evaluation/candidate'/key/'full.npz') as z:pd=z['standardized'];pb=z['event_scores']
            modes['incumbent_audio_to_descriptors']=model.predict(torch.tensor(np.concatenate((pd,pb),1),dtype=torch.float32)[None])[0].numpy()*std+mu
        errors={k:pose_errors(v,truth,mask) for k,v in modes.items()};errors['train_mean']=pose_errors(np.broadcast_to(mu,truth.shape),truth,mask)
        path=folder/(key+'.npz');np.savez_compressed(path,truth=truth,mask=mask,**modes)
        rows.append({'key':key,'canonical_id':e['canonical_id'],'errors':errors,'path':str(path),'sha256':digest(path)})
    result={'name':name,'seed':seed,'rows':rows,'pose_normalizers':pose_norms,'source_sha256':digest(__file__)}
    authenticated_json(folder/'evaluation.json',result);return result

def run_probes(root=ROOT):
    root=Path(root);protocol=read_authenticated(root/'diagnostics/protocol.json');m,n=base_context()
    for path,expected in protocol['code_hashes'].items():
        if digest(path)!=expected:raise RuntimeError('Diagnostic source drift')
    for path,expected in protocol['input_hashes'].items():
        if digest(path)!=expected:raise RuntimeError('Diagnostic input drift')
    for row in read_authenticated(root/'features/pose/report.json')['rows']:
        if digest(row['path'])!=row['sha256']:raise RuntimeError('Pose input changed')
    data=NightResident([e for e in m['entries'] if e['partition']=='train'],n,320,root)
    pose,mask,pose_norms=load_pose_training(data,root);reports=[];steps=protocol['pose_probe_steps']
    for seed in (0,1):
        for name in PROBES:
            folder=root/'diagnostics/pose_probes'/name/f'seed_{seed}';folder.mkdir(parents=True,exist_ok=True)
            if (folder/'report.json').exists():
                r=read_authenticated(folder/'report.json');verify_artifacts(r['artifacts'])
                adopted=protocol.get('adopted_unchanged_reports',{}).get(str(folder/'report.json'))
                if adopted is not None:
                    if digest(folder/'report.json')!=adopted:raise RuntimeError('Adopted report changed')
                elif r.get('protocol_sha256')!=digest(root/'diagnostics/protocol.json'):
                    raise RuntimeError('Completed probe belongs to another protocol')
                if r['name']!=name or r['seed']!=seed or r['steps']!=steps:raise RuntimeError('Probe identity mismatch')
                reports.append(r);continue
            if any(folder.iterdir()):raise RuntimeError('Partial scientific probe must be preserved before recovery')
            family=name.removeprefix('audio_') if name.startswith('audio_') else 'base';data.set_family(family)
            seed_all(seed);width=256 if name=='descriptor256' else 128
            input_dim=24 if name.startswith('descriptor') else (34 if name=='pose_code24' else 92+data.aux_dim)
            model=PoseDecoder(input_dim,width,name=='pose_code24').cuda();opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
            atomic_torch(folder/'initial.pt',{'model':model.state_dict(),'name':name,'seed':seed,'step':0})
            rng=np.random.default_rng(17000+seed);begin=time.monotonic();elapsed=0.;model.train()
            with (folder/'training.jsonl').open('w') as log:
                for step in range(steps):
                    torch.cuda.synchronize();t=time.monotonic();x,d,b,ix=data.batch(32,320,rng,return_index=True);y=pose[ix];mm=mask[ix]
                    inp=input_for(name,x,d,b,y);pred=model(inp);loss=((pred-y).square()*mm).sum()/mm.sum().clamp_min(1)
                    vm=mm[:,1:]&mm[:,:-1];vel=((pred[:,1:]-pred[:,:-1]-y[:,1:]+y[:,:-1]).square()*vm).sum()/vm.sum().clamp_min(1)
                    total=loss+.1*vel;opt.zero_grad(set_to_none=True);total.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                    if not torch.isfinite(total) or not torch.isfinite(grad):raise FloatingPointError('Probe nonfinite')
                    opt.step();torch.cuda.synchronize();elapsed+=time.monotonic()-t
                    if step==0 or (step+1)%1000==0:
                        v={'name':name,'seed':seed,'step':step+1,'mse':float(loss.detach()),'velocity_mse':float(vel.detach()),'gpu_seconds':elapsed}
                        log.write(json.dumps(v)+'\n');log.flush();print(json.dumps(v),flush=True)
            atomic_torch(folder/'final.pt',{'model':model.state_dict(),'name':name,'seed':seed,'steps':steps,'input_dim':input_dim,'width':width,
                                           'pose_normalizers':pose_norms,'protocol_sha256':digest(root/'diagnostics/protocol.json')})
            ev=evaluate_probe(model,name,seed,pose_norms,root)
            r={'name':name,'seed':seed,'steps':steps,'gpu_training_seconds':elapsed,'wall_seconds':time.monotonic()-begin,
               'protocol_sha256':digest(root/'diagnostics/protocol.json'),
               'parameters':sum(p.numel() for p in model.parameters()),'gpu':torch.cuda.get_device_name(),'precision':'fp32',
               'artifacts':artifact_manifest([folder/'initial.pt',folder/'final.pt',folder/'training.jsonl',folder/'evaluation.json'])}
            authenticated_json(folder/'report.json',r);reports.append(r)
            del model,opt,inp,pred,total,loss,vel,x,d,b,y,mm,ix;gc.collect();torch.cuda.empty_cache()
    authenticated_json(root/'diagnostics/pose_probe_report.json',{'rows':reports,'gpu_training_seconds':sum(r['gpu_training_seconds'] for r in reports),
        'interpretation':'Reconstruction and conditional-mean probes. A negative audio pose regression is not a universal information bound.'})

if __name__=='__main__':
    with gpu_lock(ROOT):run_probes()
