#!/usr/bin/env python3
"""Frozen learned-rhythm extraction. Outputs are features, not accuracy evidence."""
from pathlib import Path
import argparse, fcntl, hashlib, json, os, shutil, time
import numpy as np
import soundfile as sf
import torch
from beat_this.inference import Audio2Frames
from beat_this.model.postprocessor import Postprocessor

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',required=True);ap.add_argument('--output',required=True)
    a=ap.parse_args(); manifest_path=Path(a.manifest); out=Path(a.output)
    manifest=json.loads(manifest_path.read_text()); out.mkdir(parents=True,exist_ok=True)
    lock=(out/'gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required; no CPU fallback')
    if sha(manifest['checkpoint'])!=manifest['checkpoint_sha256']:raise RuntimeError('Checkpoint drift')
    if Path(out/'complete.json').exists():raise RuntimeError('Completed extraction must not be overwritten')
    torch.set_num_threads(4); torch.cuda.reset_peak_memory_stats()
    predictor=Audio2Frames(checkpoint_path=manifest['checkpoint'],device='cuda:0',float16=False)
    post=Postprocessor(type='minimal')
    started=time.monotonic(); records=[]
    print(json.dumps({'stage':'loaded','pid':os.getpid(),'device':torch.cuda.get_device_name(),'torch':torch.__version__,'manifest_sha256':sha(manifest_path),'records':len(manifest['inputs'])}),flush=True)
    for i,row in enumerate(manifest['inputs']):
        if shutil.disk_usage(out).free<manifest['free_disk_floor_bytes']:raise RuntimeError('Free-disk floor reached; no deletion permitted')
        path=out/(row['recording_id']+'.npz')
        if path.exists():raise RuntimeError('Refuse overwrite: '+str(path))
        if sha(row['audio_path'])!=row['audio_sha256']:raise RuntimeError('Audio changed: '+row['audio_path'])
        audio,sr=sf.read(row['audio_path'],dtype='float32',always_2d=False)
        if not np.isfinite(audio).all() or len(audio)==0:raise RuntimeError('Invalid source audio')
        t0=time.monotonic()
        beat,down=predictor(audio,sr)
        if not torch.isfinite(beat).all() or not torch.isfinite(down).all():raise RuntimeError('Nonfinite predictions')
        beats,downbeats=post(beat,down)
        beat_np=beat.detach().cpu().numpy().astype(np.float32);down_np=down.detach().cpu().numpy().astype(np.float32)
        if len(beat_np)!=len(down_np) or abs(len(beat_np)/50-len(audio)/sr)>.1:raise RuntimeError('Output duration mismatch')
        np.savez_compressed(path,times_s=np.arange(len(beat_np),dtype=np.float64)/50,beat_logit=beat_np,downbeat_logit=down_np,beats_s=np.asarray(beats),downbeats_s=np.asarray(downbeats))
        rec={**row,'path':str(path),'sha256':sha(path),'frames':len(beat_np),'audio_seconds':len(audio)/sr,'elapsed_s':time.monotonic()-t0,'beat_logit_std':float(beat_np.std()),'downbeat_logit_std':float(down_np.std()),'detected_beats':len(beats),'detected_downbeats':len(downbeats),'peak_vram_bytes':torch.cuda.max_memory_allocated()}
        records.append(rec)
        with (out/'progress.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
        print(json.dumps({'stage':'extracted','index':i+1,'total':len(manifest['inputs']),**rec}),flush=True)
        if i+1==manifest['pilot_rows']:
            if not all(r['frames']>0 and r['beat_logit_std']>1e-6 for r in records):raise RuntimeError('Smoke capture collapsed')
            (out/'pilot_complete.json').write_text(json.dumps({'scope':'runtime/capture integrity only; not beat accuracy','rows':len(records),'manifest_sha256':sha(manifest_path),'elapsed_s':time.monotonic()-started},indent=2))
            print(json.dumps({'stage':'pilot_passed','accuracy_measured':False}),flush=True)
    (out/'complete.json').write_text(json.dumps({'scope':manifest['scope'],'accuracy_measured':False,'rows':len(records),'manifest_sha256':sha(manifest_path),'elapsed_s':time.monotonic()-started,'peak_vram_bytes':torch.cuda.max_memory_allocated(),'records':records},indent=2))
    print(json.dumps({'stage':'complete','rows':len(records),'elapsed_s':time.monotonic()-started,'accuracy_measured':False}),flush=True)
if __name__=='__main__': main()
