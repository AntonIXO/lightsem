from __future__ import annotations
import argparse, av, cv2, hashlib, json, os, subprocess, threading, time
from pathlib import Path
from contextlib import ExitStack
import fcntl
import numpy as np

LOCKS=['/tmp/lightsem-audio-understanding-gpu.lock','/mnt/lightsem-data/experiments/lightsem/stage11_20260908/gpu.lock','/mnt/lightsem-data/experiments/lightsem/native_dynamics_full_all_20261007_seed0_v1/gpu.lock']

def sha(p): return hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()

def decode(path, start=60., count=240):
    with av.open(str(path)) as c:
        s=c.streams.video[0]; s.thread_type='AUTO'; fps=float(s.average_rate); stride=max(1,round(fps/30.)); prev=-float('inf'); out=[]
        for i,f in enumerate(c.decode(s)):
            if f.time is None or f.time<=prev: raise ValueError('invalid PTS')
            prev=float(f.time)
            if prev<start or i%stride: continue
            out.append((i,int(f.pts),prev,f.to_ndarray(format='bgr24')))
            if len(out)>=count:break
    if len(out)<count: raise ValueError('short clip')
    return out

class GPUSampler:
    def __init__(self): self.stop=False; self.samples=[]
    def run(self):
        while not self.stop:
            try:
                x=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(',')
                self.samples.append([float(x[0]),float(x[1])])
            except Exception: pass
            time.sleep(.1)

def make_fp16(src,dst):
    if dst.exists(): return
    import onnx
    from onnxruntime.transformers.float16 import convert_float_to_float16
    model=onnx.load(str(src)); model=convert_float_to_float16(model,keep_io_types=True,disable_shape_infer=False)
    onnx.checker.check_model(model); onnx.save(model,str(dst))

class Runner:
    def __init__(self, root, variant):
        import onnxruntime as ort
        from rtmlib import YOLOX,RTMPose
        cfg=json.loads((root/'PROTOCOL.json').read_text())['variants'][variant]; self.cfg=cfg; self.variant=variant
        det=cfg['detector']; pose=cfg['pose']; self.interval=int(cfg.get('detector_interval',1)); self.thread_count=int(cfg.get('intra_op_num_threads',2))
        self.det=YOLOX(str(det),model_input_size=(640,640),backend='onnxruntime',device='cuda',score_thr=.7)
        self.pose=RTMPose(str(pose),model_input_size=(192,256),backend='onnxruntime',device='cuda',to_openpose=False)
        for name,tool in [('det',self.det),('pose',self.pose)]:
            o=ort.SessionOptions();o.intra_op_num_threads=self.thread_count;o.inter_op_num_threads=1;o.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL;o.log_severity_level=3
            tool.session=ort.InferenceSession(tool.onnx_model,sess_options=o,providers=['CUDAExecutionProvider','CPUExecutionProvider'])
            if tool.session.get_providers()[0]!='CUDAExecutionProvider': raise RuntimeError(name+' CPU fallback')
    def run(self,frames):
        bs=[];ks=[];cs=[];det_calls=0;lat=[];prev=None
        for j,(_,_,_,im) in enumerate(frames):
            t=time.perf_counter()
            if prev is None or j%self.interval==0:
                b=np.asarray(self.det(im),np.float32).reshape(-1,4);det_calls+=1;prev=b.copy()
            else:
                b=prev.copy()
            if len(b): k,c=self.pose(im,bboxes=b)
            else: k=np.empty((0,17,2),np.float32);c=np.empty((0,17),np.float32)
            bs.append(b);ks.append(np.asarray(k,np.float32));cs.append(np.asarray(c,np.float32));lat.append(time.perf_counter()-t)
        return {'boxes':bs,'keypoints':ks,'scores':cs,'detector_calls':det_calls,'latency':lat}

def compare(base,cur):
    finite=[];centers=[];counts=[];diff=[]
    for b,c in zip(base['keypoints'],cur['keypoints']):
        finite.append(float(np.isfinite(c).all()))
        counts.append(len(c))
        if len(b) and len(c):
            n=min(len(b),len(c)); diff.append(float(np.nanmean(np.linalg.norm(b[:n]-c[:n],axis=-1))))
    return {'finite_frame_fraction':float(np.mean(finite)),'mean_instances':float(np.mean(counts)),'mean_abs_keypoint_diff_px_vs_baseline':float(np.mean(diff)) if diff else None}

def main(a):
    lock_handles=[]
    for lock in LOCKS:
        h=open(lock,'a+'); fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB); lock_handles.append(h)
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(): raise RuntimeError('GPU occupied')
    root=Path(a.root); p=json.loads((root/'PROTOCOL.json').read_text()); variant=a.variant
    cfg=p['variants'][variant]
    for x in [cfg['detector'],cfg['pose']]:
        if not Path(x).exists(): raise FileNotFoundError(x)
        if sha(x)!=cfg['sha256'][str(x)]: raise RuntimeError('model hash mismatch')
    frames=[]
    for rec in p['recordings']:
        frames.append((rec['id'],decode(Path(rec['path']),p['start_s'],p['frames'])))
    out=root/'results'/variant; out.mkdir(parents=False,exist_ok=False)
    sampler=GPUSampler(); th=threading.Thread(target=sampler.run); th.start(); runner=Runner(root,variant); rows=[]; baseline=None
    for rid,fs in frames:
        t=time.perf_counter(); result=runner.run(fs); elapsed=time.perf_counter()-t
        metrics={
            'variant':variant,'recording_id':rid,'frames':len(fs),
            'fps_end_to_end':len(fs)/elapsed,'fps_inference':len(fs)/sum(result['latency']),
            'detector_calls':result['detector_calls'],'detector_interval':runner.interval,
            'intra_op_num_threads':runner.thread_count,
            'peak_vram_mib':max([x[0] for x in sampler.samples],default=None),
            'mean_instances':float(np.mean([len(x) for x in result['boxes']])),
            'finite_keypoint_frames':int(sum(np.isfinite(x).all() for x in result['keypoints'])),
            'finite_frame_fraction':float(np.mean([np.isfinite(x).all() for x in result['keypoints']])),
            'latency_p50_ms':float(np.quantile(result['latency'],.5)*1000),
            'latency_p95_ms':float(np.quantile(result['latency'],.95)*1000)}
        if variant!='baseline':
            bp=root/'results'/'baseline'/(rid+'.npz')
            if bp.exists():
                with np.load(bp,allow_pickle=True) as bz: baseline={'boxes':list(bz['boxes']),'keypoints':list(bz['keypoints']),'scores':list(bz['scores'])}
                metrics.update(compare(baseline,result))
            else: metrics['comparison_missing_baseline']=True
        np.savez_compressed(out/f'{rid}.npz',t=np.array([x[2] for x in fs]),source_pts=np.array([x[1] for x in fs]),boxes=np.array(result['boxes'],dtype=object),keypoints=np.array(result['keypoints'],dtype=object),scores=np.array(result['scores'],dtype=object),meta=json.dumps(metrics))
        rows.append(metrics); print(json.dumps(metrics),flush=True)
    sampler.stop=True;th.join(); receipt={'variant':variant,'rows':rows,'model_hashes':{str(cfg['detector']):sha(cfg['detector']),str(cfg['pose']):sha(cfg['pose'])},'gpu_samples':sampler.samples,'protocol_sha256':sha(root/'PROTOCOL.json')}
    (out/'RESULT.json').write_text(json.dumps(receipt,indent=2)+'\n')
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--variant',required=True);main(ap.parse_args())
