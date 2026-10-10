"""Isolated pose benchmark. Raw PTS/detections; shared production admission downstream."""
from __future__ import annotations
import argparse, hashlib, json, time, subprocess, os, shutil, threading
from pathlib import Path
from contextlib import ExitStack
import fcntl
import av
import cv2
import numpy as np
from lightsem.experiments.native_dynamics.rtmpose_cache import CoachTracker,coach_candidates,sha,write_new
from scripts.data.extract_rtmpose_native import dedupe_geometry,save_new
LOCKS=['/tmp/lightsem-audio-understanding-gpu.lock','/mnt/lightsem-data/experiments/lightsem/stage11_20260908/gpu.lock','/mnt/lightsem-data/experiments/lightsem/native_dynamics_full_all_20261007_seed0_v1/gpu.lock']

def decode(path):
 with av.open(str(path)) as c:
  st=c.streams.video[0];st.thread_type='AUTO';fps=float(st.average_rate);stride=max(1,round(fps/30));prev=-float('inf')
  for i,f in enumerate(c.decode(st)):
   if f.pts is None or f.time is None or f.time<=prev:raise ValueError('missing/nonmonotonic PTS')
   prev=f.time
   if i%stride==0:yield i,f.pts,float(f.time),f.to_ndarray(format='bgr24'),fps,stride,str(f.time_base)

class Model:
 def __init__(self,cfg,profile):
  import onnxruntime as ort
  from rtmlib import YOLOX,RTMPose,RTMO
  self.cfg=cfg; self.tools=[]
  if cfg['kind']=='rtmo':
   self.pose=RTMO(cfg['pose'],model_input_size=(640,640),backend='onnxruntime',device='cuda',score_thr=.7);self.tools=[self.pose]
  else:
   self.det=YOLOX(cfg['det'],model_input_size=(640,640),backend='onnxruntime',device='cuda',score_thr=.7)
   self.pose=RTMPose(cfg['pose'],model_input_size=(192,256),backend='onnxruntime',device='cuda',to_openpose=False);self.tools=[self.det,self.pose]
  self.profiles=[]
  for i,m in enumerate(self.tools):
   o=ort.SessionOptions();o.intra_op_num_threads=2;o.inter_op_num_threads=1;o.log_severity_level=3;o.enable_profiling=True;o.profile_file_prefix=str(profile/f'session_{i}')
   m.session=ort.InferenceSession(m.onnx_model,sess_options=o,providers=['CUDAExecutionProvider','CPUExecutionProvider'])
   if m.session.get_providers()[0]!='CUDAExecutionProvider':raise RuntimeError('CPU fallback')
 def __call__(self,im):
  if self.cfg['kind']=='rtmo':
   from rtmlib.tools.object_detection.post_processings import multiclass_nms
   x,ratio=self.pose.preprocess(im);det,pose=self.pose.inference(x)
   b=det[0,:,:4]/ratio;ds=det[0,:,4]
   _,keep=multiclass_nms(b,ds[:,None],nms_thr=.45,score_thr=.7)
   if keep is None:return np.empty((0,4),np.float32),np.empty((0,17,2),np.float32),np.empty((0,17),np.float32)
   return b[keep],pose[0,keep,:,:2]/ratio,pose[0,keep,:,2]
  b=np.asarray(self.det(im),np.float32).reshape(-1,4)
  if not len(b):return b,np.empty((0,self.cfg['joints'],2),np.float32),np.empty((0,self.cfg['joints']),np.float32)
  k,c=self.pose(im,bboxes=b);return b,k,c
 def finish_profile(self):
  receipts=[]
  for m in self.tools:
   p=Path(m.session.end_profiling());e=json.loads(p.read_text());nodes=[x.get('args',{}) for x in e if x.get('cat')=='Node'];cuda=sum(x.get('provider')=='CUDAExecutionProvider' for x in nodes);conv=sum(x.get('provider')=='CUDAExecutionProvider' and 'Conv' in x.get('op_name','') for x in nodes)
   if not conv:raise RuntimeError('no profiled CUDA convolution')
   receipts.append({'path':str(p),'sha256':sha(p),'cuda_node_events':cuda,'cuda_conv_events':conv})
  return receipts

def run(a):
 root=a.root;p=json.loads((root/'PROTOCOL.json').read_text());cfg=p['candidates'][a.candidate]
 for path,digest in p['code_hashes'].items():
  if sha(path)!=digest:raise RuntimeError('source drift '+path)
 for key in ['pose','det']:
  if key in cfg and sha(cfg[key])!=cfg[key+'_sha256']:raise RuntimeError('model drift')
 od=root/'raw'/a.candidate;od.mkdir(exist_ok=False);(od/'profiles').mkdir()
 with ExitStack() as st:
  for lock in LOCKS:
   f=st.enter_context(open(lock,'a+'));fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('GPU occupied')
  vals=[];end=threading.Event()
  def sampler():
   while not end.wait(.25):
    try:vals.append([time.time(),float(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())])
    except:pass
  th=threading.Thread(target=sampler);th.start();model=Model(cfg,od/'profiles')
  # Engineering warmup only, profile then closed to bound disk usage.
  source=p['recordings'][0]['path']
  for i,row in enumerate(decode(source)):
   model(row[3])
   if i==7:break
  profile=model.finish_profile();results=[]
  for rec in p['recordings']:
   if shutil.disk_usage('/mnt').free<20*2**30:raise RuntimeError('disk floor')
   if sha(rec['path'])!=rec['source_sha256']:raise RuntimeError('source changed')
   begun=time.perf_counter();v0=len(vals);times=[];pts=[];indices=[];kall=[];call=[];ball=[];ids=[];candidates=[];fi=[];tracker=CoachTracker();lat=[];pixels=hashlib.sha256()
   for count,(frame_no,stamp,t,im,fps,stride,tb) in enumerate(decode(rec['path'])):
    h,w=im.shape[:2];t0=time.perf_counter();b,k,c=model(im);lat.append(time.perf_counter()-t0)
    b=np.asarray(b,np.float32).reshape(-1,4);k=np.asarray(k,np.float32).reshape(-1,cfg['joints'],2);c=np.asarray(c,np.float32).reshape(-1,cfg['joints'])
    if len(k)!=len(b):raise RuntimeError('box/pose cardinality')
    candidate=coach_candidates(b,w,h);sel=np.flatnonzero(candidate);sel=sel[dedupe_geometry(b[sel])]
    tid=np.full(len(b),-1,np.int32);tid[sel]=tracker.step(t,b[sel],[np.zeros(48,np.float32) for _ in sel])
    times.append(t);pts.append(stamp);indices.append(frame_no);kall.append(k);call.append(c);ball.append(b);ids.append(tid);candidates.append(candidate);fi.append(np.full(len(b),count,np.int32))
    # Hash sparse source pixels to authenticate identical decoder sampling across arms.
    if count%30==0:pixels.update(im.tobytes())
   elapsed=time.perf_counter()-begun
   meta={'candidate':a.candidate,'source':rec,'fps_src':fps,'stride':stride,'time_base':tb,'width':w,'height':h,'n_frames':len(times),'track_segments':tracker.next_id,'inference_seconds':sum(lat),'elapsed_seconds_including_decode_track':elapsed,'fps_inference':len(times)/sum(lat),'fps_end_to_end':len(times)/elapsed,'latency_p50_ms':float(np.quantile(lat,.5)*1000),'latency_p95_ms':float(np.quantile(lat,.95)*1000),'vram_total_mib_peak':max([x[1] for x in vals[v0:]],default=None),'decoded_sparse_pixels_sha256':pixels.hexdigest(),'max_keypoint_score_threshold':None,'missing_contract':'absent instances not filled; finite estimates do not prove visibility','protocol_sha256':sha(root/'PROTOCOL.json')}
   arrays=dict(t=np.array(times),source_pts=np.array(pts,dtype=np.int64),source_frame=np.array(indices),K=np.concatenate(kall),C=np.concatenate(call),box=np.concatenate(ball),track_id=np.concatenate(ids),candidate=np.concatenate(candidates),frame_index=np.concatenate(fi),meta=np.array(json.dumps(meta)))
   dest=od/(rec['id']+'.npz');save_new(dest,arrays);results.append({'id':rec['id'],'path':str(dest),'sha256':sha(dest),**meta});print(json.dumps({'candidate':a.candidate,'recording':rec['id'],'frames':len(times),'tracks':tracker.next_id,'fps':meta['fps_end_to_end']}),flush=True)
  end.set();th.join();write_new(od/'COMPLETE.json',{'rows':results,'profiles':profile,'protocol_sha256':sha(root/'PROTOCOL.json'),'vram_samples':vals})
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--candidate',required=True);run(ap.parse_args())
