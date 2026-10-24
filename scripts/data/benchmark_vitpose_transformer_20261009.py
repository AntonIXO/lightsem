from __future__ import annotations
import argparse,av,json,time,subprocess,threading,hashlib,fcntl
from pathlib import Path
import numpy as np
LOCKS=['/tmp/lightsem-audio-understanding-gpu.lock','/mnt/lightsem-data/experiments/lightsem/stage11_20260908/gpu.lock','/mnt/lightsem-data/experiments/lightsem/native_dynamics_full_all_20261007_seed0_v1/gpu.lock']
def sha(p):return hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()
def decode(path,start,count):
 with av.open(str(path)) as c:
  s=c.streams.video[0];s.thread_type='AUTO';stride=max(1,round(float(s.average_rate)/30));prev=-1e9;out=[]
  for i,f in enumerate(c.decode(s)):
   if f.time is None or f.time<=prev:raise ValueError('bad PTS')
   prev=float(f.time)
   if prev<start or i%stride:continue
   out.append((i,int(f.pts),prev,f.to_ndarray(format='bgr24')))
   if len(out)>=count:break
 return out
class Samp:
 def __init__(self):self.stop=False;self.v=[]
 def run(self):
  while not self.stop:
   try:
    x=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True);self.v.append(float(x.strip()))
   except:pass
   time.sleep(.1)
def main(a):
 root=Path(a.root);p=json.loads((root/'PROTOCOL.json').read_text());
 for lock in LOCKS:
  h=open(lock,'a+');fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('GPU occupied')
 from rtmlib import YOLOX,ViTPose
 d=YOLOX(p['detector'],model_input_size=(640,640),backend='onnxruntime',device='cuda',score_thr=.7)
 pose=ViTPose(p['pose'],model_input_size=(192,256),backend='onnxruntime',device='cuda',to_openpose=False)
 import onnxruntime as ort
 for m in [d,pose]:
  if m.session.get_providers()[0]!='CUDAExecutionProvider':raise RuntimeError('CPU fallback')
 out=root/'results';out.mkdir(exist_ok=True);samp=Samp();th=threading.Thread(target=samp.run);th.start();rows=[]
 for rec in p['recordings']:
  fs=decode(Path(rec['path']),p['start_s'],p['frames']);lat=[];n=0;inst=[];finite=[];begin=time.perf_counter()
  for _,_,_,im in fs:
   t=time.perf_counter();b=np.asarray(d(im),np.float32).reshape(-1,4);k,c=pose(im,bboxes=b);lat.append(time.perf_counter()-t);n+=len(k);inst.append(len(k));finite.append(np.isfinite(k).all())
  elapsed=time.perf_counter()-begin;row={'recording_id':rec['id'],'frames':len(fs),'fps_end_to_end':len(fs)/elapsed,'fps_inference':len(fs)/sum(lat),'mean_instances':float(np.mean(inst)),'finite_frame_fraction':float(np.mean(finite)),'peak_vram_mib':max(samp.v,default=None),'latency_p50_ms':float(np.quantile(lat,.5)*1000),'latency_p95_ms':float(np.quantile(lat,.95)*1000)};rows.append(row);print(json.dumps(row),flush=True)
 samp.stop=True;th.join();(out/'RESULT.json').write_text(json.dumps({'candidate':'ViTPose-S body17 ONNX + YOLOX-m HumanArt','rows':rows,'model_sha256':{'pose':sha(p['pose']),'detector':sha(p['detector'])},'protocol_sha256':sha(root/'PROTOCOL.json'),'gpu_samples':samp.v},indent=2)+'\n')
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);main(ap.parse_args())
