#!/usr/bin/env python3
"""Explicit experimental preview; accepted motion/angles are not regenerated."""
import argparse, hashlib, json, shutil, subprocess, time
from pathlib import Path
from dataclasses import asdict
import numpy as np
from run_rhythm_companion import arrays, render_config, optical, encode_command, camera_project, audit_movie
from lightsem.conductor.rig import canonical_rig
from lightsem.sim.renderer import StageRenderer
from lightsem.experiments.stage11_common import gpu_lock

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()

def save(path,obj):
    with path.open('x') as f:json.dump(obj,f,indent=2,allow_nan=False)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('mode',choices=['render','audit'])
    ap.add_argument('--streams',type=Path,required=True)
    ap.add_argument('--key',default='candidate')
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--audio',type=Path,default=Path('/root/shpongle.flac'))
    ap.add_argument('--start',type=float,default=0)
    ap.add_argument('--duration',type=float,default=277.3)
    args=ap.parse_args()
    out=args.output.parent
    if args.mode=='audit':
        audit_movie(out,args.output)
        return
    if args.output.exists():raise RuntimeError('Refuse overwrite')
    if shutil.disk_usage(out).free < 1024**3+400*1024**2:raise RuntimeError('Media storage reserve')
    raw=arrays(args.streams)
    frames=raw[args.key]
    if frames.dtype!=np.uint8 or frames.ndim!=2 or frames.shape[1]!=512:raise RuntimeError('DMX shape/type')
    n=round(args.duration*30); start=round(args.start*30)
    indices=np.rint(np.arange(start,start+n)*40/30).astype(int)
    if indices[-1]>=len(frames):raise RuntimeError('No padding of DMX permitted')
    rig=canonical_rig();cfg=render_config()
    # Same fixed renderer-only beam zoom convention as previous companion delivery.
    d=optical(frames,rig)
    yy,xx=np.mgrid[:1080:4,:1920:4]
    mask=np.ones(yy.shape,bool)
    for x,y in camera_project([f.position for f in rig.fixtures],cfg):
        mask &= (xx-x)**2+(yy-y)**2>60**2
    cmd=encode_command(args.output,{'audio':str(args.audio)},1920,1080,start,n)
    cmd[cmd.index('-c:v'):cmd.index('-c:v')]=['-vf',"drawtext=text='EXPERIMENTAL - learned color and accents - unchanged motion':fontcolor=white:fontsize=22:x=24:y=18:box=1:boxcolor=black@0.6"]
    started=time.monotonic();metrics=[]
    log=out/(args.output.stem+'_encode.log')
    with gpu_lock():
        renderer=StageRenderer(rig,cfg)
        print('GPU',renderer.renderer_name,flush=True)
        try:
            with log.open('xb') as err:
                proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=err)
                try:
                    for i,src in enumerate(indices):
                        if i%30==0:
                            if shutil.disk_usage(out).free<1024**3+32*1024**2:raise RuntimeError('Free disk floor')
                            if sum(p.stat().st_size for p in out.rglob('*') if p.is_file())>512*1024**2:raise RuntimeError('Demo media512MiB cap')
                        rgb=renderer.render_frame(d[src],0.)
                        if rgb.shape!=(1080,1920,3) or rgb.dtype!=np.uint8:raise RuntimeError('Native output shape')
                        # Per-frame spatial subsample only; NOT a full-pixel clipping certification.
                        sample=rgb[::4,::4];lum=sample[:,:,0]*.2126+sample[:,:,1]*.7152+sample[:,:,2]*.0722
                        metrics.append((float(lum[mask].mean()),float((sample.max(2)[mask]>=250).mean())))
                        proc.stdin.write(rgb.tobytes())
                        if i%300==0:print('RENDER',i,n,'elapsed_s',round(time.monotonic()-started,2),flush=True)
                    proc.stdin.close()
                    if proc.wait():raise RuntimeError(log.read_text())
                except BaseException:
                    try:proc.stdin.close()
                    except (BrokenPipeError,OSError):pass
                    proc.terminate();proc.wait();raise
        finally:
            renderer.ctx.release()
    metric=np.asarray(metrics)
    np.savez_compressed(out/(args.output.stem+'_sampled_pixels.npz'),metrics=metric)
    receipt={'path':str(args.output),'frames':n,'start_frame':start,'duration_s':n/30,
        'video_sha256':sha(args.output),'dmx':str(args.streams),'dmx_sha256':sha(args.streams),
        'dmx_key':args.key,'source_audio':str(args.audio),'source_audio_sha256':sha(args.audio),
        'render_source_sha256':sha(__file__),'config':asdict(cfg),'command':cmd,
        'seconds':time.monotonic()-started,'native_resolution':[1920,1080],
        'sampled_outside_bulb_mean_luma_255':float(metric[:,0].mean()),
        'sampled_outside_bulb_clipping_mean':float(metric[:,1].mean()),
        'pixel_scope':'Every frame, every fourth row/column, excluding bulb disks; not full-pixel clipping or visual acceptance',
        'scope':'Experimental learned nonangular overlay; unchanged saved motion and pan/tilt; no quality/generalization claim'}
    save(out/(args.output.stem+'_render.json'),receipt)
    print('RENDER_COMPLETE',args.output,flush=True)

if __name__=='__main__':main()
