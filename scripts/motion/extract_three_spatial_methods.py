from __future__ import annotations
import argparse, json, hashlib
from pathlib import Path
import numpy as np
import torch
from scipy.signal import resample_poly
import soundfile as sf

ART=Path('/mnt/lightsem-data/experiments/lightsem/three_spatial_methods_20261011_v1')
SHOW=Path('/root/lightsem/experiments/native_bf16_v7b')
SONGS={'shpongle':Path('/root/shpongle.flac'),'cure_for_me':SHOW/'demo_audio/Cure_For_Me.wav'}

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read_audio(path,sr=16000):
 x,rate=sf.read(path,dtype='float32',always_2d=True)
 if rate!=sr:x=np.stack([resample_poly(x[:,c],sr,rate) for c in range(x.shape[1])],1)
 return x.T.astype('float32'),sr

def target_time(song):
 with np.load(SHOW/'demo_features/features/timelines'/f'{song}.npz',allow_pickle=False) as z:return z['time'].astype('float64')
def interp_context(time,features,target):
 time=np.asarray(time,float); features=np.asarray(features,np.float32)
 if features.ndim!=2 or features.shape[1]!=64:raise ValueError(features.shape)
 order=np.argsort(time); time=time[order];features=features[order]; keep=np.r_[True,np.diff(time)>1e-9];time=time[keep];features=features[keep]
 out=np.stack([np.interp(target,time,features[:,i]) for i in range(64)],1).astype('float32')
 if not np.isfinite(out).all():raise ValueError('nonfinite interpolated context')
 return out

def extract_spain(checkpoint,device):
 from lightsem.experiments.audio_stereo.spain_pilot import SpaInNet,SpaInNetConfig
 raw=torch.load(checkpoint,map_location='cpu',weights_only=False); cfgd=raw.get('config',{}).get('config',{})
 cfg=SpaInNetConfig(**{k:cfgd[k] for k in ('sample_rate','n_fft','hop_length','win_length','width','position_dim','num_layers','sources') if k in cfgd})
 model=SpaInNet(cfg); model.load_state_dict(raw['state_dict'],strict=True);model.to(device).eval()
 rows={}; angles=torch.tensor([[-45.,-15.,15.,45.]],device=device)
 for song,path in SONGS.items():
  wav,_=read_audio(path,cfg.sample_rate); n=wav.shape[-1]; chunk=cfg.sample_rate*8; all_t=[];all_f=[]
  for start in range(0,n,chunk):
   core=wav[:,start:min(n,start+chunk)]; pad=chunk-core.shape[-1]
   if pad: core=np.pad(core,((0,0),(0,pad)))
   with torch.inference_mode(): _,feat=model(torch.from_numpy(core)[None].to(device),angles)
   f=feat['native_descriptor'][0].cpu().numpy(); t=start/cfg.sample_rate+np.arange(len(f))/30.; keep=t < min(n,cfg.sample_rate*999)/cfg.sample_rate; all_t.append(t[keep]);all_f.append(f[keep])
  target=target_time(song); fused=interp_context(np.concatenate(all_t),np.concatenate(all_f),target)
  out=ART/'contexts/spain'/f'{song}.npz';out.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(out,time30=target,fused30=fused)
  rows[song]={'context_path':str(out),'context_sha256':sha(out),'angles_deg':[-45.,-15.,15.,45.],'checkpoint_sha256':sha(checkpoint),'source_audio_sha256':sha(path),'descriptor':'SpaIn-Net D1-CAT native_descriptor'}
 return rows

def extract_crw(checkpoint,device):
 from lightsem.experiments.audio_stereo.crw_pilot import StereoCRW,StereoCRWConfig
 raw=torch.load(checkpoint,map_location='cpu',weights_only=False);model=StereoCRW(StereoCRWConfig());model.load_state_dict(raw['state_dict'],strict=True);model.to(device).eval();cfg=model.config;rows={}
 for song,path in SONGS.items():
  wav,_=read_audio(path,cfg.sample_rate); step=round(cfg.sample_rate/30); req=cfg.required_samples; starts=np.arange(0,max(1,wav.shape[-1]-req+1),step,dtype=int);vals=[];times=[]
  for i in range(0,len(starts),64):
   ss=starts[i:i+64]; batch=np.stack([wav[:,s:s+req] for s in ss]);
   if batch.shape[-1]<req: batch=np.pad(batch,((0,0),(0,0),(0,req-batch.shape[-1])))
   with torch.inference_mode(): out=model(torch.from_numpy(batch).to(device))['native_features'].cpu().numpy()
   vals.append(out);times.append(ss/cfg.sample_rate)
  target=target_time(song);fused=interp_context(np.concatenate(times),np.concatenate(vals),target)
  outp=ART/'contexts/crw'/f'{song}.npz';outp.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(outp,time30=target,fused30=fused)
  rows[song]={'context_path':str(outp),'context_sha256':sha(outp),'checkpoint_sha256':sha(checkpoint),'source_audio_sha256':sha(path),'descriptor':'StereoCRW native correspondence/confidence features','pretrained_fma_weights':False}
 return rows

def extract_bsrnn(checkpoint,source_root,device):
 from lightsem.experiments.audio_stereo.bsrnn_teacher import load_teacher,extract_streamed,context_64
 teacher=load_teacher(checkpoint,device,source_root=source_root);rows={}
 for song,path in SONGS.items():
  f=extract_streamed(path,teacher,split='showcase',chunk_seconds=10,hop_seconds=9,fps=40);target=target_time(song);raw=f.context_64;fused=interp_context(f.time_40,raw,target)
  out=ART/'contexts/bsrnn_simo'/f'{song}.npz';out.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(out,time30=target,fused30=fused)
  rows[song]={'context_path':str(out),'context_sha256':sha(out),'checkpoint_sha256':sha(checkpoint),'source_audio_sha256':sha(path),'descriptor':'oBSRNN-SIMO streamed source stereo descriptors','metadata':f.metadata}
 return rows

def main():
 p=argparse.ArgumentParser();p.add_argument('--method',choices=['spain','crw','bsrnn'],required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--source-root',type=Path,default=Path('/mnt/lightsem-data/tools/bsrnn'));p.add_argument('--out-manifest',type=Path,required=True);p.add_argument('--device',default='cuda');a=p.parse_args();
 if a.out_manifest.exists():raise FileExistsError(a.out_manifest)
 if not torch.cuda.is_available() and a.device.startswith('cuda'):raise RuntimeError('GPU required; refusing fallback')
 device=torch.device(a.device)
 rows=(extract_spain(a.checkpoint,device) if a.method=='spain' else extract_crw(a.checkpoint,device) if a.method=='crw' else extract_bsrnn(a.checkpoint,a.source_root,device))
 payload={'schema_version':1,'status':'complete','method':a.method,'rows':[{'id':s,**r} for s,r in rows.items()],'checkpoint':str(a.checkpoint),'checkpoint_sha256':sha(a.checkpoint),'source_root_commit':None if a.method!='bsrnn' else __import__('subprocess').check_output(['git','-c',f'safe.directory={a.source_root}','-C',str(a.source_root),'rev-parse','HEAD'],text=True).strip(),'split':'showcase','test_access':False,'native_feature_dim':64,'scope':'offline method feature extraction; no full stem cache; raw body safety is downstream'}
 a.out_manifest.parent.mkdir(parents=True,exist_ok=True);a.out_manifest.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n');print(json.dumps({'method':a.method,'manifest':str(a.out_manifest),'rows':list(rows)}))
if __name__=='__main__':main()
