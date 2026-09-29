"""Registered optics controls. CPU imports only; EGL exists only in render/match.
Run with bash run.sh scripts/motion/run_dance_optics_full.py --help.
All output stages are create-only. Failed stages remain as evidence; do not retry
in-place. No vault access. Geometry is NOT tracked pixels or music acceptance.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from decimal import Decimal, ROUND_CEILING
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np

REPO = Path('/root/lightsem')
NATIVE = Path('/mnt/lightsem-data/experiments/lightsem/motion_native_audio_20260910')
DEFAULT = Path('/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/optics_full_controls_01')
AUDIO = Path('/root/shpongle.flac')
PAIR = ('closer_native_legacy', 'closer_zero_angular_flux')
LUMA = np.array([.2126, .7152, .0722])
SCOPE = 'Geometry proxy and raw display pixels only; no music or visual acceptance.'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def array_digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    """Exclusive creation, durable flush, read-only receipt and digest sidecar."""
    path = Path(path)
    with path.open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    path.chmod(0o444)
    with Path(str(path)+'.sha256').open('x') as f:
        f.write(digest(path)+'\n')
    Path(str(path)+'.sha256').chmod(0o444)


def authenticated(path):
    if digest(path) != Path(str(path)+'.sha256').read_text().strip():
        raise RuntimeError(f'Receipt changed: {path}')
    return read_json(path)


def copy_new(src, dst):
    dst = Path(dst); dst.parent.mkdir(parents=True, exist_ok=True)
    with Path(src).open('rb') as a, dst.open('xb') as b:
        shutil.copyfileobj(a, b)
    dst.chmod(0o444)
    if digest(src) != digest(dst):
        raise RuntimeError(f'Copy changed during freeze: {src}')


def validate_design(d, reg):
    if json.loads(reg['hypothesis']['plan']) != d or json.loads(reg['experiment']['config_note']) != d:
        raise ValueError('Design disagrees with registration')
    expected = {f'{c}_{z}_{e}' for c,z,e in itertools.product(
        ('legacy','closer'), ('native','zero'), ('legacy','angular_peak','angular_flux'))}
    expected.add('closer_zero_static_bright_foil')
    if set(d['arms']) != expected:
        raise ValueError('Expected exact 12-arm factorial and static foil')
    if d['intervals']['active'] != [70.,90.] or d['intervals']['full'] != [0.,277.3]:
        raise ValueError('Unexpected registered time intervals')
    for arm in d['arms'].values():
        c = arm['renderer']
        if (c['width'],c['height'],c['fps'],c['cam_orbit']) != (640,360,30.,0.):
            raise ValueError('Unexpected renderer dimensions/rate/orbit')


def validate_engineering(e):
    if e.get('maximum_byte_difference') != 0 or e.get('nonhead_edge_mode_parity') is not True:
        raise ValueError('Default and nonhead GPU parity must pass before freezing')
    if not e.get('new_controls_execute') or e.get('config_preview',{}).get('beam_flux_reference_deg') != 14.:
        raise ValueError('Preflight must include current flux control')


def validate_engineering_binding(e,d):
    if not e.get('before_after_source_parity') or not e.get('source_hashes'):
        raise ValueError('Engineering receipt must bind tested source/input bytes before and after GPU check')
    required = {str(p) for p in (REPO/'lightsem').rglob('*') if p.is_file() and p.suffix in ('.py','.json')}
    required.add(d['source'])
    if not required.issubset(e['source_hashes']):
        raise ValueError('Engineering receipt omits current Python/JSON dependency or source DMX')
    for name,sha in e['source_hashes'].items():
        if digest(name) != sha:raise ValueError('Source/input changed since engineering: '+name)


def freeze(out, engineering):
    d, reg, e = read_json(out/'design.json'), read_json(out/'registration.json'), read_json(engineering)
    validate_design(d,reg); validate_engineering(e); validate_engineering_binding(e,d)
    if (out/'freeze.json').exists() or (out/'frozen').exists():
        raise FileExistsError('Freeze already exists or failed; no overwrite')
    root = out/'frozen'; root.mkdir()
    inputs = {'design.json':out/'design.json', 'registration.json':out/'registration.json',
              'engineering.json':engineering, 'candidate.npz':Path(d['source']),
              'run_dance_optics_full.py':Path(__file__).resolve()}
    files = {}
    for name,p in inputs.items():
        copy_new(p,root/name); files[name] = digest(root/name)
    # Freeze all native Python/JSON dependency sources, not only shaders.
    for p in sorted((REPO/'lightsem').rglob('*')):
        if p.is_file() and p.suffix in ('.py','.json') and '__pycache__' not in p.parts:
            rel = 'source/'+str(p.relative_to(REPO))
            copy_new(p,root/rel); files[rel] = digest(root/rel)
    # Validate complete stream without interpreting any motion/pixel metrics.
    with np.load(root/'candidate.npz', allow_pickle=False) as z:
        stream = z['dmx']
        if stream.dtype != np.uint8 or stream.shape != (11092,512):
            raise ValueError(f'Unexpected DMX dtype/shape: {stream.dtype}, {stream.shape}')
    write_json(out/'freeze.json', {'schema':1, 'created_unix':time.time(), 'files':files,
        'audio_path':str(AUDIO), 'audio_sha256':digest(AUDIO), 'source_path':d['source'],
        'engineering_path':str(engineering), 'native_gpu_lock_root':str(NATIVE),
        'scope':SCOPE, 'engineering_source_binding':
        'GPU receipt hashes Python/JSON dependencies, source DMX, archived comparator and preflight script before/after execution; all verified at freeze.'})
    print('FROZEN',out,flush=True)


def load_frozen(out):
    receipt = authenticated(out/'freeze.json'); root = out/'frozen'
    for rel,sha in receipt['files'].items():
        if digest(root/rel) != sha: raise RuntimeError(f'Frozen bytes changed: {rel}')
    if digest(__file__) != receipt['files']['run_dance_optics_full.py']:
        raise RuntimeError('Run the frozen script; implementation differs from freeze')
    for name in ('design.json','registration.json'):
        if digest(out/name) != receipt['files'][name]: raise RuntimeError(f'Live {name} changed')
    if digest(receipt['source_path']) != receipt['files']['candidate.npz']:
        raise RuntimeError('Original source candidate bytes changed')
    if digest(receipt['audio_path']) != receipt['audio_sha256']: raise RuntimeError('Audio changed')
    if any(k == 'lightsem' or k.startswith('lightsem.') for k in sys.modules):
        raise RuntimeError('Native project imported before frozen source activation')
    sys.path.insert(0,str(root/'source'))
    from lightsem.conductor.rig import canonical_rig
    from lightsem.sim.renderer import RenderConfig
    rig = canonical_rig()
    d=read_json(root/'design.json')
    validate_design(d,read_json(root/'registration.json'))
    with np.load(root/'candidate.npz',allow_pickle=False) as z: stream=z['dmx'].copy()
    return receipt,d,stream,rig,RenderConfig


def transform_dmx(source, rig, arm):
    """Only registered head zoom and foil head pan/tilt bytes may differ."""
    if source.dtype != np.uint8 or source.ndim != 2 or source.shape[1] != 512:
        raise ValueError('DMX must be uint8 [N,512]')
    d=source.copy(); allowed=np.zeros(512,bool); angles=[]; zoom=[]
    for f in rig.by_kind('moving_head'):
        a=f.address-1; angles.extend(range(a,a+4)); zoom.append(a+9)
    if arm['zoom'] == 'zero': d[:,zoom]=0; allowed[zoom]=True
    elif arm['zoom'] != 'native': raise ValueError('Unknown zoom')
    if arm.get('frozen_angles'):
        i=round(arm['freeze_time_s']*40)
        if not 0<=i<len(d): raise ValueError('Foil freeze time outside source')
        d[:,angles]=source[i,angles]; allowed[angles]=True
    if not np.array_equal(d[:,~allowed],source[:,~allowed]):
        raise AssertionError('Unregistered source-byte mutation')
    if not arm.get('frozen_angles') and not np.array_equal(d[:,angles],source[:,angles]):
        raise AssertionError('Moving-arm angles changed')
    return d, {'source_dmx_sha256':array_digest(source),'arm_dmx_sha256':array_digest(d),
        'source_angle_sha256':array_digest(source[:,angles]),'arm_angle_sha256':array_digest(d[:,angles]),
        'nonallowed_bytes_identical':True, 'allowed_zero_based_channels':np.flatnonzero(allowed).tolist(),
        'changed_zero_based_channels':np.flatnonzero(np.any(d!=source,axis=0)).tolist(),
        'changed_bytes':int(np.count_nonzero(d!=source))}


def project(points,cfg):
    cam=np.asarray(cfg.cam_pos,float); forward=np.asarray(cfg.cam_target,float)-cam
    forward/=np.linalg.norm(forward)
    right=np.cross(forward,[0.,1.,0.]);right/=np.linalg.norm(right);up=np.cross(right,forward)
    delta=np.asarray(points)-cam;depth=delta@forward
    with np.errstate(divide='ignore',invalid='ignore'):
        x=(delta@right)/depth/cfg.camera_scale*cfg.height/2+cfg.width/2-.5
        y=cfg.height/2-(delta@up)/depth/cfg.camera_scale*cfg.height/2-.5
    return np.where((depth>0)[...,None],np.stack((x,y),-1),np.nan)


def first_hits(directions,positions):
    """Analytic axis intersections with shader finite planes AND 3 truss boxes.
    Not camera sphere tracing, light occlusion, or a measured illuminated spot.
    Wall matches sdScene's y<9 bound (no invented y>=0 condition).
    """
    v=np.asarray(directions,float);p=np.broadcast_to(positions,v.shape)
    with np.errstate(divide='ignore',invalid='ignore'):
        tf=-p[...,1]/v[...,1];tw=(-6-p[...,2])/v[...,2]
        floor=p+tf[...,None]*v;wall=p+tw[...,None]*v
    tf=np.where((tf>1e-7)&(abs(floor[...,0])<11)&(floor[...,2]>-7)&(floor[...,2]<9),tf,np.inf)
    tw=np.where((tw>1e-7)&(abs(wall[...,0])<9)&(wall[...,1]<9),tw,np.inf)
    distances=[tf,tw]
    for center,half in [([0,6.3,-1],[6.2,.16,.16]),([-6,3.2,-1],[.14,3.2,.14]),([6,3.2,-1],[.14,3.2,.14])]:
        lower=np.array(center)-half;upper=np.array(center)+half
        parallel=abs(v)<1e-12
        with np.errstate(divide='ignore',invalid='ignore'):
            a=(lower-p)/v;b=(upper-p)/v
        near=np.where(parallel,-np.inf,np.minimum(a,b)).max(-1)
        far=np.where(parallel,np.inf,np.maximum(a,b)).min(-1)
        outside=np.any(parallel&((p<lower)|(p>upper)),axis=-1)
        t=np.where(near>1e-7,near,far)
        distances.append(np.where((~outside)&(far>=np.maximum(near,0))&(t>1e-7),t,np.inf))
    ds=np.stack(distances,-1);kind=np.argmin(ds,axis=-1);distance=ds.min(-1)
    valid=np.isfinite(distance)
    hit=p+np.where(valid,distance,0)[...,None]*v
    return np.where(valid[...,None],hit,np.nan),np.where(valid,kind,-1),distance


def half_peak_angle(nominal_half_rad, profile, softness_deg=.6):
    """Invert smoothstep at HALF ITS ACTUAL ON-AXIS PEAK, not absolute .5."""
    theta=np.asarray(nominal_half_rad,float)
    if profile=='legacy': lower=np.cos(theta)-.018;upper=np.cos(theta)+.0072
    elif profile=='angular':
        soft=np.deg2rad(softness_deg);lower=np.cos(theta+soft);upper=np.cos(np.maximum(theta-soft,0))
    else: raise ValueError(profile)
    peak_t=np.clip((1-lower)/(upper-lower),0,1)
    target=.5*peak_t**2*(3-2*peak_t)
    lo=np.zeros_like(theta);hi=np.ones_like(theta)
    for _ in range(55):
        t=(lo+hi)/2;value=t*t*(3-2*t)
        lo=np.where(value<target,t,lo);hi=np.where(value>=target,t,hi)
    return np.arccos(np.clip(lower+(upper-lower)*(lo+hi)/2,-1,1))


def circle_radius(centers,directions,world_radius,cfg):
    """Equivalent projected area radius from EXACTLY 64 polygon vertices."""
    v=np.asarray(directions,float)
    ref=np.broadcast_to([0.,1.,0.],v.shape).copy()
    ref[abs(v[...,1])>.9]=[1.,0.,0.]
    u=np.cross(v,ref);u/=np.linalg.norm(u,axis=-1,keepdims=True);w=np.cross(v,u)
    phi=np.arange(64)*2*np.pi/64
    ring=centers[...,None,:]+world_radius[...,None,None]*(
        np.cos(phi)[...,None]*u[...,None,:]+np.sin(phi)[...,None]*w[...,None,:])
    xy=project(ring,cfg)
    area=abs(np.sum(xy[...,0]*np.roll(xy[...,1],-1,axis=-1)-xy[...,1]*np.roll(xy[...,0],-1,axis=-1),axis=-1))/2
    return np.sqrt(area/np.pi)


def band_rms(x,fs=40.):
    from scipy.signal import welch
    x=np.asarray(x)
    if not np.isfinite(x).all(): raise ValueError('Invalid projections must not be dropped')
    n=min(400,len(x))
    f,p=welch(x,fs=fs,window='hann',nperseg=n,noverlap=n//2,detrend='linear',axis=0,scaling='density')
    m=(f>=1)&(f<2)
    integrate=np.trapezoid if hasattr(np,'trapezoid') else np.trapz
    return np.sqrt(np.maximum(0,integrate(p[m],f[m],axis=0).sum(axis=-1)))


def geometry_arm(dmx,rig,cfg,intervals):
    heads=rig.by_kind('moving_head');positions=np.array([f.position for f in heads])
    v=[];half=[]
    for f in heads:
        a=f.address-1;d=dmx.astype(np.float64,copy=False)
        pan=np.deg2rad(((d[:,a]*256+d[:,a+1])/65535-.5)*f.pan_range_deg)
        tilt=np.deg2rad(((d[:,a+2]*256+d[:,a+3])/65535-.5)*f.tilt_range_deg)
        v.append(np.stack((np.sin(pan)*np.cos(tilt),-np.cos(pan)*np.cos(tilt),np.sin(tilt)),-1))
        half.append(np.deg2rad((f.beam_min_deg+d[:,a+9]/255*(f.beam_max_deg-f.beam_min_deg))/2))
    v=np.stack(v,1);half=np.stack(half,1)
    hits,surface,distance=first_hits(v,positions)
    mid=positions+(hits-positions)/2
    theta=half_peak_angle(half,cfg.beam_edge_profile,cfg.beam_edge_softness_deg)
    radius=circle_radius(mid,v,distance/2*np.tan(theta),cfg)
    midpx=project(mid,cfg);hitpx=project(hits,cfg)
    result={}
    for name,(lo,hi) in intervals.items():
        sl=slice(round(lo*40),round(hi*40));valid=np.isfinite(radius[sl])&(radius[sl]>0)&np.isfinite(midpx[sl]).all(-1)
        if not valid.all():
            result[name]={'valid':False,'invalid_per_head':(~valid).sum(0).tolist(),'primary':None}
            continue
        rms=band_rms(midpx[sl]);hit_rms=band_rms(hitpx[sl]);r=radius[sl].mean(0);ratios=rms/r
        result[name]={'valid':True,'frames':len(midpx[sl]),'welch_nperseg':min(400,len(midpx[sl])),
            'primary':float(ratios.mean()),'ratio_of_means_supplemental':float(rms.mean()/r.mean()),
            'heads':[{ 'name':f.name,'midshaft_band_rms_px':float(rms[i]),'first_hit_band_rms_px':float(hit_rms[i]),
                'mean_half_peak_circle_radius_px':float(r[i]),'motion_radius_ratio':float(ratios[i]),
                'mean_half_peak_angle_deg':float(np.rad2deg(theta[sl,i]).mean()),
                'surface_counts':{str(k):int(np.count_nonzero(surface[sl,i]==k)) for k in range(-1,5)}} for i,f in enumerate(heads)]}
    arrays={'midshaft_px':midpx,'first_hit_px':hitpx,'half_peak_radius_px':radius,'half_peak_angle_rad':theta,'first_hit_world':hits,'surface':surface}
    return result,arrays


def contrasts(results):
    out=[]
    for factor,levels in [('camera',('legacy','closer')),('zoom',('native','zero')),('cone',('legacy','angular_peak','angular_flux'))]:
        axes=[('legacy','closer'),('native','zero'),('legacy','angular_peak','angular_flux')]
        axis={'camera':0,'zoom':1,'cone':2}[factor]
        others=[i for i in range(3) if i!=axis]
        for fixed in itertools.product(*(axes[i] for i in others)):
            for low,high in itertools.combinations(levels,2):
                names=[]
                for level in (low,high):
                    values=[None]*3;values[axis]=level
                    for i,value in zip(others,fixed):values[i]=value
                    names.append('_'.join(values))
                a,b=(results[n]['active'] for n in names)
                row={'factor':factor,'baseline':names[0],'candidate':names[1]}
                if a['valid'] and b['valid']:
                    row.update(difference=b['primary']-a['primary'],ratio=b['primary']/a['primary'] if a['primary'] else None,
                        per_head=[{'name':x['name'],'difference':y['motion_radius_ratio']-x['motion_radius_ratio'],
                                   'ratio':y['motion_radius_ratio']/x['motion_radius_ratio'] if x['motion_radius_ratio'] else None}
                                  for x,y in zip(a['heads'],b['heads'])])
                else:row['invalid']=True
                out.append(row)
    return out


def geometry(out,loaded):
    _,d,source,rig,RC=loaded;target=out/'geometry';target.mkdir()
    results={};parity={}
    for name,arm in d['arms'].items():
        stream,parity[name]=transform_dmx(source,rig,arm)
        results[name],arrays=geometry_arm(stream,rig,RC(**arm['renderer']),d['intervals'])
        np.savez_compressed(target/f'{name}.npz',**arrays)
    a=results['legacy_native_legacy']['active'];b=results['closer_zero_angular_flux']['active']
    ratio=b['primary']/a['primary'] if a['valid'] and b['valid'] and a['primary'] else None
    foil=results['closer_zero_static_bright_foil']['active']
    foil_comparison={'baseline':'closer_zero_angular_flux','candidate':'closer_zero_static_bright_foil',
        'scope':'Frozen80s head angles; brightness affects renderer only, not this geometry'}
    if b['valid'] and foil['valid']:
        foil_comparison.update(ratio=foil['primary']/b['primary'] if b['primary'] else None,
            per_head=[{'name':x['name'],'moving_ratio':x['motion_radius_ratio'],'foil_ratio':y['motion_radius_ratio'],
                       'difference':y['motion_radius_ratio']-x['motion_radius_ratio']}
                      for x,y in zip(b['heads'],foil['heads'])])
    else:foil_comparison['invalid']=True
    artifacts={p.name:digest(p) for p in target.glob('*.npz')}
    write_json(target/'geometry.json',{'scope':SCOPE,'instrument':
        '40Hz emitted axes; finite shader floor/wall and truss first hit; half actual hit distance; '
        '64-vertex perpendicular half-on-axis-peak circle projected area; 400 Hann Welch linear detrend '
        '50% overlap density trapezoid 1<=f<2, x+y power. Primary mean of six RMS/mean-radius ratios. '
        'No brightness weighting, camera occlusion, pixel tracking or illuminated spot claim.',
        'arms':results,'parity':parity,'single_factor_contrasts':contrasts(results),
        'combined_vs_baseline_ratio':ratio,'proxy_gate':ratio is not None and ratio>=2,
        'static_foil_contrast':foil_comparison,'artifacts':artifacts})
    print('GEOMETRY COMPLETE',flush=True)


def pixel_masks(rig,cfg):
    yy,xx=np.indices((cfg.height,cfg.width));body=yy>=28;bulbs=np.zeros_like(body)
    xy=project(np.array([f.position for f in rig.fixtures[:16]]),cfg)
    for x,y in xy:
        if np.isfinite(x+y):bulbs|=(xx-x)**2+(yy-y)**2<=20**2
    masks={'outside_caption':body,'outside_bulbs':body&~bulbs,'bulbs_only':body&bulbs}
    if any(not m.any() for m in masks.values()):raise ValueError('Empty pixel region')
    return masks,xy


def pixel_stats(rgb,masks):
    luma=rgb@LUMA;white=np.all(rgb>=250,axis=-1);anyclip=np.any(rgb>=250,axis=-1)
    result={}
    for name,m in masks.items():
        result[name+'_mean_luma_255']=float(luma[m].mean())
        result[name+'_support_gt5']=float(np.mean(luma[m]>5))
        result[name+'_white_ge250']=float(white[m].mean())
        result[name+'_anychannel_ge250']=float(anyclip[m].mean())
    return result


def brightness_scalars(means):
    means=np.asarray(means,float)
    if len(means)!=2 or not np.isfinite(means).all() or np.any(means<=0):
        raise ValueError('Two finite positive raw means required')
    return means.min()/means


def dim_rgb(rgb,scalar):
    if not math.isfinite(scalar) or not 0<=scalar<=1:raise ValueError('Only global dimming allowed')
    return np.rint(rgb.astype(np.float64)*scalar).clip(0,255).astype(np.uint8)


def frame_count(duration,fps):
    return int((Decimal(str(duration))*Decimal(str(fps))).to_integral_value(rounding=ROUND_CEILING))


@contextmanager
def encoder(path,width,height,duration,audio,gpu=0):
    cmd=['ffmpeg','-v','error','-nostdin','-n','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}',
        '-r','30','-i','pipe:0','-i',str(audio),'-map','0:v:0','-map','1:a:0',
        '-c:v','av1_nvenc','-gpu',str(gpu),'-preset','p5','-cq','23','-b:v','0','-pix_fmt','yuv420p',
        '-c:a','aac','-b:a','192k','-t',str(duration),'-shortest',str(path)]
    # stderr on disk avoids pipe deadlock and retains encoder diagnostics.
    with path.with_suffix('.ffmpeg.log').open('xb') as log:
        proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=log)
        try:
            yield proc.stdin,cmd
            proc.stdin.close()
            rc=proc.wait()
            if rc:raise RuntimeError(f'ffmpeg exit {rc}; see {log.name}')
        except BaseException:
            if not proc.stdin.closed:proc.stdin.close()
            if proc.poll() is None:proc.terminate()
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
            raise


class Sheets:
    def __init__(self,path,label):
        self.path=path;self.label=label;self.tiles=[];self.times=[];self.pages=[]
    def add(self,rgb,t):
        import cv2
        tile=cv2.resize(rgb,(320,180),interpolation=cv2.INTER_AREA)
        cv2.putText(tile,f'{t:06.1f}s',(5,174),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1,cv2.LINE_AA)
        self.tiles.append(tile);self.times.append(float(t))
        if len(self.tiles)==30:self.flush()
    def flush(self):
        import cv2
        if not self.tiles:return
        count=len(self.tiles);start=self.times[-count]
        tiles=self.tiles+[np.zeros_like(self.tiles[0])]*(30-count)
        sheet=np.concatenate([np.concatenate(tiles[i:i+3],axis=1) for i in range(0,30,3)],axis=0)
        path=self.path/f'review_{int(start):03d}.jpg'
        if path.exists():raise FileExistsError(path)
        if not cv2.imwrite(str(path),cv2.cvtColor(sheet,cv2.COLOR_RGB2BGR)):raise OSError(path)
        self.pages.append({'path':path.name,'first_second':start,'samples':count,'sha256':digest(path)})
        self.tiles=[]


def render_one(out,loaded,name,matched=False):
    receipt,d,source,rig,RC=loaded
    if name not in d['arms']:raise ValueError(f'Unknown arm {name}')
    scalar=1.;match_info=None
    if matched:
        if name not in PAIR:raise ValueError(f'Matched arms must be {PAIR}')
        # Do not choose the brightness control before all original arms finish.
        all_receipts={n:authenticated(out/'renders'/n/'render.json') for n in d['arms']}
        means=[all_receipts[n]['pixels']['outside_bulbs_mean_luma_255']['mean'] for n in PAIR]
        scalars=brightness_scalars(means);scalar=float(scalars[PAIR.index(name)])
        match_info={'pair':list(PAIR),'raw_means_luma_255':means,'scalars':scalars.tolist(),
                    'target_mean_luma_255':min(means),'scalar_policy':'one global RGB multiplier; nearest-even uint8 rounding; no framewise scaling',
                    'original_receipts':{n:digest(out/'renders'/n/'render.json') for n in d['arms']}}
    target=out/('matched' if matched else 'renders')/name
    target.mkdir(parents=True,exist_ok=False)
    arm=d['arms'][name];cfg=RC(**arm['renderer']);stream,parity=transform_dmx(source,rig,arm)
    masks,bulbxy=pixel_masks(rig,cfg);duration=d['intervals']['full'][1];n=frame_count(duration,cfg.fps)
    write_json(target/'started.json',{'started_unix':time.time(),'arm':name,'matched':matched,'scalar':scalar,
        'freeze_sha256':digest(out/'freeze.json'),'renderer_registered':arm['renderer'],'parity':parity,
        'planned_generated_frames':n,'duration_s':duration,'scope':SCOPE})
    from dataclasses import asdict
    from lightsem.sim.renderer import StageRenderer
    from lightsem.experiments.stage11.stage11_common import gpu_lock
    import cv2
    stats=[];sheets=Sheets(target,name);raw_hash=hashlib.sha256();post_hash=hashlib.sha256()
    frame_hashes=[];source_indices=[]
    with gpu_lock(NATIVE):
        renderer=StageRenderer(rig,cfg)
        try:
            with encoder(target/'video.mp4',640,360,duration,receipt['audio_path'],cfg.gpu_device) as (pipe,cmd):
                for i in range(n):
                    idx=min(round(i*40/30),len(stream)-1);source_indices.append(idx)
                    raw=renderer.render_frame(stream[idx],0.).copy()
                    raw_hash.update(raw.tobytes());frame_hashes.append(array_digest(raw))
                    rgb=dim_rgb(raw,scalar) if matched else raw
                    post_hash.update(rgb.tobytes());stats.append(pixel_stats(rgb,masks))
                    # Measurements are BEFORE caption overlay and encoding.
                    decorated=rgb.copy()
                    cv2.putText(decorated,name+(' MATCHED' if matched else ''),(8,19),cv2.FONT_HERSHEY_SIMPLEX,.42,(255,255,255),1,cv2.LINE_AA)
                    pipe.write(decorated.tobytes())
                    if i%30==0:sheets.add(decorated,i/30)
                    if i%900==0:print(name,'generated',i+1,'/',n,flush=True)
            device=renderer.renderer_name
        finally:renderer.ctx.release()
    sheets.flush()
    columns={k:np.array([row[k] for row in stats]) for k in stats[0]}
    np.savez_compressed(target/'raw_pixel_metrics.npz',time_s=np.arange(n)/30,
        source_index=np.array(source_indices),raw_frame_sha256=np.asarray(frame_hashes,dtype='S64'),**columns)
    summary={k:{'mean':float(a.mean()),'max':float(a.max()),'min':float(a.min())} for k,a in columns.items()}
    if matched:
        original=all_receipts[name]
        if raw_hash.hexdigest()!=original['raw_rgb_stream_sha256']:
            write_json(target/'failed.json',{'reason':'Frozen raw re-render not byte identical to original',
                'expected':original['raw_rgb_stream_sha256'],'actual':raw_hash.hexdigest()})
            raise RuntimeError('Brightness re-render raw bytes differ; do not use matched pair')
        actual=summary['outside_bulbs_mean_luma_255']['mean']
        match_info.update(actual_matched_mean_luma_255=actual,
            residual_quantization_luma_255=actual-match_info['target_mean_luma_255'],raw_rerender_byte_identical=True)
    clipping={kind:summary['outside_bulbs_'+kind]['mean']<.01 and summary['outside_bulbs_'+kind]['max']<.05
              for kind in ('white_ge250','anychannel_ge250')}
    files={p.name:digest(p) for p in target.iterdir() if p.is_file() and not p.name.endswith('.sha256')}
    write_json(target/'render.json',{'arm':name,'scope':SCOPE,'renderer_registered':arm['renderer'],
        'renderer_resolved':asdict(cfg),'device':device,'time_uniform':0.,'parity':parity,
        'duration_s':duration,'generated_frames':n,'encoded_frames':None,'decoder_audit':'PENDING external full decode/audio audit',
        'raw_rgb_stream_sha256':raw_hash.hexdigest(),'display_rgb_before_caption_sha256':post_hash.hexdigest(),
        'pixels':summary,'pixel_instrument':'Every raw uint8 RGB frame before caption/encoding; Rec709 weighted display luma in 0..255; support >5 descriptive, not a quality gate; clipping >=250.',
        'mask_pixel_counts':{k:int(v.sum()) for k,v in masks.items()},'bulb_centers_px':bulbxy.tolist(),
        'mask_instrument':'Top28 excluded; 20px circles around all rendered fixture origins; bulb-only excludes top28 too. No head-segment attribution.',
        'clipping_gates':clipping,'brightness_match':match_info,'review_sample_hz':1,
        'review_sample_seconds':sheets.times,'review_pages':sheets.pages,'review_status':'PENDING human chronological review',
        'ffmpeg_command':cmd,'artifacts':files,'completed_unix':time.time()})
    print('COMPLETE',name,'GENERATED',n,flush=True)



def assemble_pair(out,loaded):
    """CPU-only side-by-side delivery from the two matched member videos.
    The extra transcode is a review artifact, not the raw-pixel instrument.
    """
    _,d,_,_,_=loaded
    members=[out/'matched'/name for name in PAIR]
    receipts=[authenticated(p/'render.json') for p in members]
    for p,r in zip(members,receipts):
        if digest(p/'video.mp4')!=r['artifacts']['video.mp4']:
            raise RuntimeError('Matched member video changed')
    target=out/'matched'/'pair';target.mkdir(exist_ok=False)
    duration=d['intervals']['full'][1]
    cmd=['ffmpeg','-v','error','-nostdin','-n','-hwaccel','none','-i',str(members[0]/'video.mp4'),
         '-hwaccel','none','-i',str(members[1]/'video.mp4'),
         '-filter_complex','[0:v:0][1:v:0]hstack=inputs=2:shortest=1[v]',
         '-map','[v]','-map','0:a:0','-c:v','libx264','-preset','fast','-crf','18',
         '-threads','4','-pix_fmt','yuv420p','-c:a','copy','-t',str(duration),str(target/'video.mp4')]
    write_json(target/'started.json',{'command':cmd,'members':[str(p) for p in members],
        'scope':'CPU decoded/reencoded matched review pair, not raw pixel metrics or acceptance'})
    with (target/'ffmpeg.log').open('xb') as log:
        completed=subprocess.run(cmd,stdout=log,stderr=log,check=False)
    if completed.returncode:raise RuntimeError('Pair encoding failed; see ffmpeg.log')
    write_json(target/'pair.json',{'command':cmd,'member_render_sha256':[digest(p/'render.json') for p in members],
        'pair_sha256':digest(target/'video.mp4'),'encoded_frames':None,'decoder_audio_audit':'PENDING',
        'scope':'1280x360 CPU-reencoded review derivative. Raw means/scalars belong to separate640x360 members.'})
    print('MATCHED REVIEW PAIR COMPLETE; external decoder audit still required',flush=True)


def report(out,loaded):
    _,d,_,_,_=loaded
    geo=authenticated(out/'geometry'/'geometry.json')
    def checked(path):
        value=authenticated(path)
        for name,sha in value['artifacts'].items():
            if digest(path.parent/name)!=sha:raise RuntimeError(f'Artifact changed: {name}')
        return value
    arms={name:checked(out/'renders'/name/'render.json') for name in d['arms']}
    matches={name:checked(out/'matched'/name/'render.json') for name in PAIR}
    for name,sha in geo['artifacts'].items():
        if digest(out/'geometry'/name)!=sha:raise RuntimeError('Geometry artifact changed')
    candidate=arms['closer_zero_angular_flux']
    pixel_contrasts=[]
    control_pairs=[(c['factor'],c['baseline'],c['candidate']) for c in geo['single_factor_contrasts']]
    control_pairs += [('static_foil','closer_zero_angular_flux','closer_zero_static_bright_foil'),
                      ('combined','legacy_native_legacy','closer_zero_angular_flux')]
    for factor,base,cand in control_pairs:
        pixel_contrasts.append({'factor':factor,'baseline':base,'candidate':cand,
            'scope':'Fulltrack perframe mean raw pixel statistics; not optical motion',
            'metrics':{k:{'baseline_mean':v['mean'],'candidate_mean':arms[cand]['pixels'][k]['mean'],
                'difference':arms[cand]['pixels'][k]['mean']-v['mean'],
                'ratio':arms[cand]['pixels'][k]['mean']/v['mean'] if v['mean'] else None}
                for k,v in arms[base]['pixels'].items()}})
    result={'scope':SCOPE,'geometry':geo,'raw_pixels':{k:v['pixels'] for k,v in arms.items()},
        'matched_pair':{k:v['brightness_match'] for k,v in matches.items()},
        'raw_pixel_control_contrasts':pixel_contrasts,
        'proxy_gate':geo['proxy_gate'],'candidate_clipping_gates':candidate['clipping_gates'],
        'generated_frame_counts':{k:v['generated_frames'] for k,v in arms.items()},
        'unassessed_gates':['actual decoded frame/audio coverage (external parent audit)',
            'human chronological 1Hz review and synced video comparison','music acceptance is outside this experiment'],
        'coverage_gate':None,'visual_acceptance':None,'music_acceptance':None,
        'input_receipts':{str(p.relative_to(out)):digest(p) for p in [out/'geometry/geometry.json']+
            [out/'renders'/k/'render.json' for k in arms]+[out/'matched'/k/'render.json' for k in matches]}}
    write_json(out/'report.json',result)
    print('REPORT COMPLETE; decode, audio and visual review remain unassessed',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=DEFAULT)
    sub=parser.add_subparsers(dest='stage',required=True)
    p=sub.add_parser('freeze');p.add_argument('--engineering',type=Path,default=DEFAULT.parent/'optics_flux_engineering_03/engineering.json')
    sub.add_parser('geometry')
    for name in ('render','match'):
        p=sub.add_parser(name);p.add_argument('--arm',required=True)
    sub.add_parser('pair')
    sub.add_parser('report')
    args=parser.parse_args();out=args.out.resolve()
    if args.stage=='freeze':freeze(out,args.engineering);return
    loaded=load_frozen(out)
    if args.stage=='geometry':geometry(out,loaded)
    elif args.stage in ('render','match'):render_one(out,loaded,args.arm,args.stage=='match')
    elif args.stage=='pair':assemble_pair(out,loaded)
    else:report(out,loaded)


if __name__=='__main__':main()
