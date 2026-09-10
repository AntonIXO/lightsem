"""Frozen correction of the rejected surface collapse. Reuses motion; no inference."""
import argparse,json,math,subprocess,time
from pathlib import Path
import numpy as np
from ..stage11_common import authenticated_json,read_authenticated,digest,freeze_sources,verify_sources,gpu_lock
from .spatial_observer import spatial_series,summary,gates,THRESHOLDS
from ...conductor.rig import canonical_rig

ROOT=Path('/mnt/lightsem-data/experiments/lightsem/motion_native_audio_20260910')
OLD=ROOT/'delivery/final_diagnostic'
OUT=ROOT/'delivery/surface_revision_01'


def freeze():
    if (OUT/'protocol.json').exists():raise ValueError('Never overwrite a freeze')
    design=json.loads((OUT/'design_preregistered.json').read_text())
    assert (OUT/'registration.json').exists()
    source=freeze_sources(OUT)
    tests={str(p):digest(p) for p in Path('/root/lightsem/tests').glob('test_*mapping.py')}
    tests.update({str(Path('/root/lightsem/tests/test_motion_study.py')):digest('/root/lightsem/tests/test_motion_study.py')})
    for path in tests:
        dest=OUT/'source/tests'/Path(path).name;dest.parent.mkdir(exist_ok=True);dest.write_bytes(Path(path).read_bytes());dest.chmod(0o444)
    p={'design':design,'source_sha256':source['source_sha256'],'design_sha256':digest(OUT/'design_preregistered.json'),
       'tests':tests,'renderer':read_authenticated(OLD/'protocol.json')['renderer'],
       'duration':read_authenticated(OLD/'protocol.json')['entries'][0]['duration'],
       'audio':'/root/shpongle.flac','audio_sha256':digest('/root/shpongle.flac')}
    authenticated_json(OUT/'protocol.json',p)
    return p


def context():
    p=read_authenticated(OUT/'protocol.json');s=verify_sources(OUT)
    if not Path(__file__).resolve().is_relative_to(Path(s['snapshot'])):raise ValueError('Use frozen launcher')
    for name,sha in p['design']['inputs'].items():
        if digest(name)!=sha:raise ValueError('Input drift '+name)
    if digest(OUT/'design_preregistered.json')!=p['design_sha256']:raise ValueError('Design drift')
    if digest(p['audio'])!=p['audio_sha256']:raise ValueError('Audio drift')
    return p


def mechanics(dmx,lo,hi):
    angles=[]
    for f in canonical_rig().by_kind('moving_head'):
        a=f.address-1
        ticks=np.stack((dmx[:,a].astype(float)*256+dmx[:,a+1],dmx[:,a+2].astype(float)*256+dmx[:,a+3]),-1)
        angles.append((ticks/65535-.5)*[f.pan_range_deg,f.tilt_range_deg])
    a=np.stack(angles,1);v=np.diff(a,axis=0)*40;speed=np.linalg.norm(v,axis=-1)
    segment=speed[round(lo*40):round(hi*40)-1]
    return {'median':float(np.median(segment)),'p95':float(np.quantile(segment,.95)),
            'max':float(speed.max()),'acceleration_max':float(np.linalg.norm(np.diff(v,axis=0)*40,axis=-1).max())}


def map_lights():
    import torch
    from .lighting_map import compose,LightingConfig
    p=context();torch.set_num_threads(4);rig=canonical_rig()
    if (OUT/'measurements.json').exists():raise ValueError('No overwrite')
    with np.load(OLD/'inference/shpongle.npz') as z:features=z['flow0']
    with np.load(ROOT/'data/shpongle/loudness.npz') as z:loud={k:z[k] for k in z.files}
    with np.load(OLD/'lighting/hybrid.npz') as z:baseline=z['dmx']
    old,_,_=compose(features,loud,p['duration'])
    if not np.array_equal(old,baseline):raise ValueError('Legacy DMX changed')
    mask=np.ones(512,bool)
    for f in rig.by_kind('moving_head'):mask[f.address-1:f.address+3]=False
    rows={}
    for name in p['design']['arms']:
        frames,trace,report=compose(features,loud,p['duration'],config=LightingConfig(**p['design'].get('mapper_config',{'target_space':name})))
        if not np.array_equal(frames[:,mask],baseline[:,mask]):raise ValueError('Photometry confound')
        np.savez_compressed(OUT/(name+'.npz'),dmx=frames)
        np.savez_compressed(OUT/(name+'_trace.npz'),**trace)
        s=spatial_series(frames,rig);np.savez_compressed(OUT/(name+'_spatial.npz'),**s)
        segments={}
        for label,(lo,hi) in p['design']['segments'].items():
            r=summary(s,lo,hi);segments[label]={'summary':r,'gates':gates(r,active=label not in ('quiet','entry','full')),
                                             'mechanics':mechanics(frames,lo,hi)}
        # Every complete 10-second block is reported, not only selected active.
        blocks=[]
        for lo in range(0,270,10):
            r=summary(s,lo,lo+10);blocks.append({'start':lo,'summary':r,'gates':gates(r)})
        rows[name]={'segments':segments,'blocks':blocks,'mapping':report,'photometric_bytes_identical':True,
                    'dmx_sha256':digest(OUT/(name+'.npz'))}
        print(name,json.dumps(segments),flush=True)
    authenticated_json(OUT/'measurements.json',{'rows':rows,'legacy_dmx_identical':True,'protocol_sha256':digest(OUT/'protocol.json')})


def render_lights():
    import cv2
    from .exposure_study import masks
    from ..quality_observer import image_pair_motion
    from ...sim.renderer import StageRenderer,RenderConfig
    p=context();measure=read_authenticated(OUT/'measurements.json')
    for name in p['design']['arms']:
        path=OUT/(name+'.mp4')
        if path.exists():
            r=read_authenticated(path.with_suffix('.video.json'))
            if digest(path)!=r['sha256']:raise ValueError('Video drift')
            continue
        if digest(OUT/(name+'.npz'))!=measure['rows'][name]['dmx_sha256']:raise ValueError('DMX drift')
        with np.load(OUT/(name+'.npz')) as z:dmx=z['dmx']
        cfg=RenderConfig(**p['renderer']);rig=canonical_rig();renderer=StageRenderer(rig,cfg)
        regions=masks(rig,cfg,list(range(6)));yy,xx=np.mgrid[:cfg.height,:cfg.width]
        union=np.logical_or.reduce(regions);border=(xx<cfg.width*.1)|(xx>=cfg.width*.9)|(yy<cfg.height*.1)|(yy>=cfg.height*.9)
        cmd=['ffmpeg','-v','error','-nostdin','-n','-f','rawvideo','-pix_fmt','rgb24','-s',f'{cfg.width}x{cfg.height}',
             '-r','30','-i','pipe:0','-i',p['audio'],'-map','0:v:0','-map','1:a:0','-c:v','av1_nvenc','-preset','p5','-cq','25',
             '-b:v','0','-pix_fmt','yuv420p','-c:a','aac','-b:a','192k','-t',str(p['duration']),'-shortest',str(path)]
        enc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
        rows=[];previous=None;start=time.monotonic()
        try:
            for i in range(math.ceil(p['duration']*30)):
                rgb=renderer.render_frame(dmx[min(round(i*40/30),len(dmx)-1)],0.)
                enc.stdin.write(rgb.tobytes())
                # Every rendered frame has exposure/visibility checks, not5Hz.
                luma=(rgb@np.array([.2126,.7152,.0722],np.float32))/255
                row={'time':i/30,'bulb_luma':float(np.mean([luma[m].mean() for m in regions])),
                     'frame_luma':float(luma.mean()),'frame_clipped':float((rgb.min(-1)>=250).mean()),
                     'border_luma':float(luma[border].mean())}
                # Adjacent-frame screen motion is a diagnostic; full tracks have
                # additional slower-scale trajectory and normalized image checks.
                small=cv2.resize(luma,(320,180),interpolation=cv2.INTER_AREA)
                if previous is not None:row['frame_abs_change']=float(np.abs(small-previous).mean())
                previous=small
                rows.append(row)
                if i%300==0:print(json.dumps({'render':name,'frame':i,'seconds':i/30,'elapsed':time.monotonic()-start}),flush=True)
        finally:
            enc.stdin.close();error=enc.stderr.read().decode();code=enc.wait();renderer.ctx.release()
        if code:raise RuntimeError(error)
        authenticated_json(path.with_suffix('.video.json'),{'path':str(path),'sha256':digest(path),'observations':rows,
            'frames':len(rows),'wall_seconds':time.monotonic()-start,'protocol_sha256':digest(OUT/'protocol.json')})
        print('COMPLETE '+str(path),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['freeze','map','render']);args=ap.parse_args()
    if args.stage=='freeze':freeze()
    elif args.stage=='map':map_lights()
    else:
        with gpu_lock(ROOT):render_lights()
