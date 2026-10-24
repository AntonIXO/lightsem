import av,json,time,hashlib,subprocess,threading
from pathlib import Path
import numpy as np
from cuda.bindings import runtime
import tensorrt as trt
def bbox_xyxy2cs(bbox,padding=1.0):
 bbox=np.asarray(bbox); c=(bbox[:2]+bbox[2:])*0.5; scale=(bbox[2:]-bbox[:2])*padding; return c,scale
def top_down_affine(input_size,bbox_scale,bbox_center,img):
 import cv2
 w,h=input_size;aspect=w/h;bw,bh=bbox_scale
 if bw>bh*aspect: scale=np.array([bw,bw/aspect],np.float32)
 else: scale=np.array([bh*aspect,bh],np.float32)
 src=np.array([bbox_center,bbox_center+np.array([0.,-scale[0]*.5]),bbox_center+np.array([scale[1]*.5,0.])],np.float32);dst=np.array([[w*.5,h*.5],[w*.5,0.],[w,h*.5]],np.float32);M=cv2.getAffineTransform(src,dst);return cv2.warpAffine(img,M,(int(w),int(h)),flags=cv2.INTER_LINEAR),scale
ROOT=Path('/root/lightsem/experiments/pose_bf16_20261009'); V3=Path('/mnt/lightsem-data/experiments/lightsem/pose_extraction_research_20261009_v3');
PROTO=json.loads((V3/'PROTOCOL.json').read_text());
rec_ids=['bl399rfzD5s','-G-6DNPGEfk','kQzKKB1E9ZA','zWnvxM5cs8g']
REC={r['id']:r for r in PROTO['recordings']}

def bfloat_to_f32(a):return (a.astype(np.uint32)<<16).view(np.float32)
class Runner:
 def __init__(self):
  self.engine=trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine((ROOT/'rtmpose_m_autocast_bf16.plan').read_bytes());self.ctx=self.engine.create_execution_context();self.ctx.set_input_shape('input',(1,3,256,192));
  self.stream=runtime.cudaStreamCreate()[1];self.ptr={};self.host={}
  for n in ['input','simcc_x','simcc_y']:
   shape=tuple(self.ctx.get_tensor_shape(n));arr=np.empty(shape,np.float32);self.host[n]=arr;self.ptr[n]=runtime.cudaMalloc(arr.nbytes)[1];self.ctx.set_tensor_address(n,self.ptr[n])
 def one(self,x):
  inp=np.ascontiguousarray(x,dtype=np.float32).transpose(2,0,1)[None];self.host['input'][:]=inp
  runtime.cudaMemcpyAsync(self.ptr['input'],self.host['input'].ctypes.data,self.host['input'].nbytes,runtime.cudaMemcpyKind.cudaMemcpyHostToDevice,self.stream);self.ctx.execute_async_v3(self.stream)
  for n in ['simcc_x','simcc_y']:runtime.cudaMemcpyAsync(self.host[n].ctypes.data,self.ptr[n],self.host[n].nbytes,runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost,self.stream)
  runtime.cudaStreamSynchronize(self.stream);sx=self.host['simcc_x'];sy=self.host['simcc_y'];ix=np.argmax(sx,axis=-1);iy=np.argmax(sy,axis=-1);score=(np.max(sx,axis=-1)+np.max(sy,axis=-1))/2;return np.stack([ix/2,iy/2],-1),score
 def close(self):
  for p in self.ptr.values():runtime.cudaFree(p)
  runtime.cudaStreamDestroy(self.stream)
def decode(path,start=60,count=60):
 out=[]
 with av.open(str(path)) as c:
  s=c.streams.video[0];s.thread_type='AUTO';stride=max(1,round(float(s.average_rate)/30));prev=-1e9;sample=-1
  for i,f in enumerate(c.decode(s)):
   if f.time is None or f.time<=prev:raise ValueError('PTS')
   prev=float(f.time)
   if i%stride:continue
   sample+=1
   if prev<start:continue
   out.append((sample,prev,f.to_ndarray(format='bgr24')))
   if len(out)>=count:break
 return out
rows=[];r=Runner();
for rid in rec_ids:
 npz=V3/'raw'/'rtmpose_m'/f'{rid}.npz'
 with np.load(npz,allow_pickle=False) as z:
  fi=z['frame_index'].astype(int);boxes=z['box'].astype(np.float32); groups={}
  for j,f in enumerate(fi):groups.setdefault(int(f),[]).append(boxes[j])
 frames=decode(REC[rid]['path']);times=[];finite=[];instances=[];begin=time.perf_counter();calls=0
 for f,t,im in frames:
  bs=groups.get(f,[]); instances.append(len(bs));
  for box in bs:
   center,scale=bbox_xyxy2cs(np.asarray(box),padding=1.25);crop,scale=top_down_affine((192,256),scale,center,im);crop=(crop-np.array((123.675,116.28,103.53),np.float32))/np.array((58.395,57.12,57.375),np.float32);k,s=r.one(crop);finite.append(bool(np.isfinite(k).all() and np.isfinite(s).all()));calls+=1
  times.append(t)
 elapsed=time.perf_counter()-begin;rows.append({'recording_id':rid,'frames':len(frames),'pose_calls':calls,'fps_source_frames':len(frames)/elapsed,'pose_calls_per_sec':calls/elapsed,'mean_instances':float(np.mean(instances)),'finite_pose_fraction':float(np.mean(finite)) if finite else None,'first_pts':times[0],'last_pts':times[-1]});print(rows[-1],flush=True)
r.close();(ROOT/'BF16_RESULT.json').write_text(json.dumps({'status':'real_bf16_tensorrt_pose_engine','engine_sha256':hashlib.file_digest((ROOT/'rtmpose_m_bf16.plan').open('rb'),'sha256').hexdigest(),'rows':rows,'trt_version':trt.__version__,'protocol_sha256':hashlib.file_digest((V3/'PROTOCOL.json').open('rb'),'sha256').hexdigest()},indent=2)+'\n')
