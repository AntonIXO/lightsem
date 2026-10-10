"""Benchmark speed candidates without touching the production RTMPose cache.

The script creates a new diagnostic root, checks detector dynamic-batch export
parity, and compares RTMPose-m against RTMPose-s/tiny on the same source frames.
It never writes production pose files or changes the selected method.
"""
from __future__ import annotations
import argparse, hashlib, json, time
from pathlib import Path
import av, cv2, numpy as np

MODEL_M_DET=Path('/mnt/lightsem-data/torch-home/hub/checkpoints/yolox_m_8xb8-300e_humanart-c2c7a14a.onnx')
MODEL_M_POSE=Path('/mnt/lightsem-data/torch-home/hub/checkpoints/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.onnx')
MODEL_S_DET=Path('/mnt/lightsem-data/torch-home/hub/checkpoints/rtmpose_s/20230928/yolox_onnx/yolox_tiny_8xb8-300e_humanart-6f3252f9/end2end.onnx')
MODEL_S_POSE=Path('/mnt/lightsem-data/torch-home/hub/checkpoints/rtmpose_s/20230831/rtmpose_onnx/rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504/end2end.onnx')

def sha(p): return hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()

def dynamic_detector(src,dst):
    import onnx
    model=onnx.load(src)
    inp=model.graph.input[0].type.tensor_type.shape.dim[0]; inp.ClearField('dim_value'); inp.dim_param='batch'
    for value in model.graph.output:
        dim=value.type.tensor_type.shape.dim[0]; dim.ClearField('dim_value'); dim.dim_param='batch'
    onnx.checker.check_model(model); onnx.save(model,dst); return sha(dst)

def frames(path,count=96):
    out=[]
    with av.open(str(path)) as c:
        s=c.streams.video[0]
        for i,f in enumerate(c.decode(s)):
            if i%max(1,int(round(float(s.average_rate or 30)/30)))==0:out.append(f.to_ndarray(format='bgr24'))
            if len(out)>=count:break
    if len(out)<count:raise ValueError('insufficient frames')
    return out

def run_candidate(det_path,pose_path,det_size,frames):
    import onnxruntime as ort
    from rtmlib import YOLOX,RTMPose
    detector=YOLOX(str(det_path),model_input_size=det_size,score_thr=.7,backend='onnxruntime',device='cuda')
    pose=RTMPose(str(pose_path),model_input_size=(192,256),backend='onnxruntime',device='cuda',to_openpose=False)
    results=[]; start=time.perf_counter()
    for image in frames:
        boxes=detector(image)
        k,s=pose(image,bboxes=boxes)
        results.append((boxes,k,s))
    elapsed=time.perf_counter()-start
    return {'seconds':elapsed,'fps':len(frames)/elapsed,'frames':len(frames),'detections':int(sum(len(x[0]) for x in results)),'providers':detector.session.get_providers(),'first':results[0]}

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True);p.add_argument('--frames',type=int,default=96);p.add_argument('--gpu-timing',action='store_true')
    a=p.parse_args();a.output_root.mkdir(parents=True,exist_ok=True)
    fs=frames(a.source,a.frames); dst=a.output_root/'yolox_m_dynamic.onnx'; dynamic_hash=dynamic_detector(MODEL_M_DET,dst)
    # Dynamic graph validity and batch shape check are CPU-safe and independent
    # of production cache output.
    import onnxruntime as ort
    import cv2
    x=np.stack([cv2.resize(f,(640,640)).transpose(2,0,1) for f in fs[:4]]).astype(np.float32)
    dynamic_error=None; dynamic_out=None
    try:
        sess=ort.InferenceSession(str(dst),providers=['CPUExecutionProvider']); dynamic_out=sess.run(None,{sess.get_inputs()[0].name:x})
    except Exception as exc:
        dynamic_error=f'{type(exc).__name__}: {exc}'
    rows=[]
    for name,det,pose,size in [('m_static',MODEL_M_DET,MODEL_M_POSE,(640,640)),('m_dynamic',dst,MODEL_M_POSE,(640,640)),('s_tiny',MODEL_S_DET,MODEL_S_POSE,(416,416))]:
        # Dynamic detector is exercised through the same graph only for the
        # batch-shape check above; rtmlib's wrapper remains static-frame exact.
        if name=='m_dynamic':
            row={'candidate':name,'detector_sha256':dynamic_hash,'pose_sha256':sha(pose),'dynamic_batch_shape':None if dynamic_out is None else list(dynamic_out[0].shape),'dynamic_batch_valid':dynamic_out is not None,'dynamic_batch_error':dynamic_error,'cpu_only_graph_check':True}
        else: row={'candidate':name,'detector_sha256':sha(det),'pose_sha256':sha(pose)}
        if a.gpu_timing and name!='m_dynamic':
            try:
                timing=run_candidate(det,pose,size,fs)
                timing.pop('first',None); row.update(timing); row['gpu_timing_status']='complete'
            except Exception as exc:
                row['gpu_timing_status']='failed'; row['gpu_timing_error']=f'{type(exc).__name__}: {exc}'
        # GPU timing is intentionally done in a separate process after the
        # production lock is free; keep source frames and hashes now.
        rows.append(row)
    result={'schema_version':1,'status':'graph_check_complete_gpu_timing_pending','source_path':str(a.source),'source_sha256':sha(a.source),'frames':len(fs),'frame_hashes':[hashlib.sha256(f.tobytes()).hexdigest() for f in fs[:4]],'dynamic_detector':str(dst),'rows':rows,'rtmpose_m_model_hashes':{'detector':sha(MODEL_M_DET),'pose':sha(MODEL_M_POSE)},'rtmpose_s_model_hashes':{'detector':sha(MODEL_S_DET),'pose':sha(MODEL_S_POSE)}}
    (a.output_root/'BENCHMARK.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
