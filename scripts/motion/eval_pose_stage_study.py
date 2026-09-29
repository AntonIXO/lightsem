"""Preregistered pose_stage mapper study. No real work without registration+freeze.
Use bash run_torch.sh scripts/motion/eval_pose_stage_study.py --help. Synthetic tests
may import pure utilities; no import here opens bodies, torch, models or EGL.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import importlib.metadata
import io
import json
import os
from pathlib import Path
import shutil
import sys
import time
import numpy as np

REPO=Path('/root/lightsem')
OPTICS=Path('/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/optics_full_controls_01')
NATIVE=Path('/mnt/lightsem-data/experiments/lightsem/motion_native_audio_20260910')
SUPPORTED_PLAN_HASH='9879f1a2af77f7e17eae12fdcda52901c623e91c8713c465eaabfbc5dabb0e9f'
FREE_RESERVE_BYTES=1024**3
STUDY_LIMIT_BYTES=7*1024**3//4
BOOKKEEPING_MARGIN_BYTES=4*1024**2
_ACTIVE_OUTPUT=None

class ResourceBlocked(RuntimeError):
    def __init__(self,message,observation=None):
        self.observation=observation or {'message':message,'scientific_refutation':False}
        super().__init__(message)

def storage_state(out):
    out=Path(out)
    total=sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
    return {'filesystem_free_bytes':shutil.disk_usage(out).free,'study_artifact_bytes':total,
        'remaining_study_budget_bytes':STUDY_LIMIT_BYTES-total,
        'free_reserve_bytes':FREE_RESERVE_BYTES,'study_limit_bytes':STUDY_LIMIT_BYTES,
        'bookkeeping_margin_bytes':BOOKKEEPING_MARGIN_BYTES}

def check_storage(out,next_bytes=0,bookkeeping=False):
    s=storage_state(out);margin=0 if bookkeeping else BOOKKEEPING_MARGIN_BYTES
    require(type(next_bytes) is int and next_bytes>=0,'Invalid storage write estimate')
    if s['filesystem_free_bytes']-next_bytes<FREE_RESERVE_BYTES+margin or s['study_artifact_bytes']+next_bytes>STUDY_LIMIT_BYTES-margin:
        observation=dict(s,next_write_bytes=next_bytes,bookkeeping=bookkeeping,scientific_refutation=False)
        raise ResourceBlocked('Storage resource blocked; preserve partial evidence, ask parent; '+json.dumps(observation,sort_keys=True),observation)
    return s

def resource_preflight(out,stage,estimate):
    before=storage_state(out);error=None
    try:check_storage(out,int(estimate))
    except ResourceBlocked as exc:error=exc
    d=out/'resource_preflight';d.mkdir(exist_ok=True)
    index=len(list(d.glob(stage+'_*.json')));p=d/f'{stage}_{index:03d}.json'
    row={'stage':stage,'estimated_output_bytes':int(estimate),'estimate_is_hard_bound':False,
        'before':before,'status':'resource_blocked' if error else 'available',
        'scope':'infrastructure only; not a scientific gate','error':str(error) if error else None}
    try:savejson(p,row,bookkeeping=True)
    except BaseException as recording_error:
        if error is None:raise
        error.add_note('Resource preflight receipt failed: '+repr(recording_error))
    if error:raise error
    return row


def require(ok,msg):
    if not ok: raise ValueError(msg)
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def jhash(x):return hashlib.sha256(canonical(x)).hexdigest()
def checked(p,sha):
    require(digest(p)==sha,'Hash changed: '+str(p));return Path(p)
def rawjson(p):return json.loads(Path(p).read_text())
def loadjson(p):
    x=rawjson(p);require(jhash(x['payload'])==x['sha256'],'Corrupt receipt '+str(p));return x['payload']
def writebytes(p,b,bookkeeping=False):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    bookkeeping=bookkeeping or p.name in ('failed.json','numeric_report.json','render_report.json')
    if _ACTIVE_OUTPUT is not None and p.resolve().is_relative_to(_ACTIVE_OUTPUT):check_storage(_ACTIVE_OUTPUT,len(b),bookkeeping)
    with p.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
    p.chmod(0o444)
def savejson(p,x,bookkeeping=False):writebytes(p,canonical({'payload':x,'sha256':jhash(x)})+b'\n',bookkeeping)
def savenpz(p,**z):
    # Geometry may contain explicitly masked invalid samples; never silently interpolate.
    b=io.BytesIO();np.savez_compressed(b,**z);writebytes(p,b.getvalue())
def arrays(p,keys=None):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in (z.files if keys is None else keys)}
@contextmanager
def lock(out):
    import fcntl
    global _ACTIVE_OUTPUT
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    with (out/'.pose_stage.lock').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        previous=_ACTIVE_OUTPUT;_ACTIVE_OUTPUT=out.resolve()
        try:yield
        finally:
            _ACTIVE_OUTPUT=previous;fcntl.flock(f,fcntl.LOCK_UN)
def copy_new(src,dst):
    before=digest(src);writebytes(dst,Path(src).read_bytes());checked(src,before);checked(dst,before)

def validate_registration(plan,reg):
    require(jhash(plan)==SUPPORTED_PLAN_HASH,'Plan differs from implemented protocol; review implementation before freeze')
    require(reg.get('registered_by')=='parent' and reg.get('registered_before_candidate_execution') is True,'Parent preregistration required')
    for key,field,idkey in [('hypothesis','plan','hypothesis_id'),('experiment','config_note','experiment_id')]:
        row=reg[key];value=row[field];value=json.loads(value) if isinstance(value,str) else value
        require(row.get('id')==plan[idkey] and value==plan,'Registration disagrees with plan: '+key)

def source_manifest(plan):
    """Authenticate existing complete maps; no source model arrays or scores opened."""
    root=Path(plan['source_review_directory']);freeze=loadjson(root/'freeze.json');protocol=loadjson(root/'protocol.json')
    checked(root/'protocol.json',freeze['protocol_sha256'])
    require(protocol['mapping']==plan['baseline'],'Cached baseline config is not the matched stage02 configuration')
    bindings={str(root/'freeze.json'):digest(root/'freeze.json'),str(root/'protocol.json'):digest(root/'protocol.json')}
    sources={}
    for item in plan['items']:
        target=root/'mapped'/item['source_review_id'];receipt=loadjson(target/'map.json')
        expected={k:v for k,v in item.items() if k not in ('source_arm','source_review_id')};expected['id']=item['source_review_id']
        require(receipt['item']==expected and receipt['whole_recording_frames']==item['full_frames'],'Cached map item/frame identity')
        require(receipt['freeze_sha256']==digest(root/'freeze.json'),'Cached map freeze identity')
        require({'full_world.npz','full_dmx.npz'}.issubset(receipt['files']),'Incomplete cached mapping')
        bindings[str(target/'map.json')]=digest(target/'map.json')
        for rel,sha in receipt['files'].items():
            require(not Path(rel).is_absolute() and '..' not in Path(rel).parts,'Unsafe cached relative path')
            checked(target/rel,sha);bindings[str(target/rel)]=sha
        if 'adopted_from' in receipt:
            a=receipt['adopted_from'];checked(a['path'],a['sha256']);bindings[a['path']]=a['sha256']
        checked(item['audio_path'],receipt['audio_sha256']);bindings[item['audio_path']]=receipt['audio_sha256']
        if item['key']=='shpongle':
            curve=protocol['evaluation']['shpongle']['loudness']['path']
            require(curve in freeze['inputs'],'Source loudness absent from original freeze')
            checked(curve,freeze['inputs'][curve]);bindings[curve]=freeze['inputs'][curve]
        else:
            curve=str(target/'loudness/loudness.npz')
            require('loudness/loudness.npz' in receipt['files'],'Cached source loudness unavailable; never recompute')
        sources[item['id']]={'world':str(target/'full_world.npz'),'dmx':str(target/'full_dmx.npz'),'loudness':curve,'map':str(target/'map.json')}
    return bindings,sources,freeze

def runtime_identity():
    # Actual installed files, not merely version strings/RECORD claims. No package import/GPU.
    distributions=('numpy','scipy','torch','clarabel','Pillow','moderngl','glcontext','opencv-python')
    files={str(Path(sys.executable).resolve()):digest(Path(sys.executable).resolve())};versions={}
    for name in distributions:
        d=importlib.metadata.distribution(name);versions[name]=d.version
        for rel in d.files or []:
            p=Path(d.locate_file(rel)).resolve()
            if p.is_file() and p.suffix not in ('.pyc','.pyo'):
                files[str(p)]=digest(p)
    # Also bind the loader/OS libraries and EGL driver files used by renderer and
    # software codec. ldd is metadata inspection, never an EGL context or workload.
    import subprocess
    for exe in (sys.executable,shutil.which('ffmpeg'),shutil.which('ffprobe')):
        text=subprocess.check_output(['ldd',str(exe)],text=True)
        for token in text.split():
            p=Path(token)
            if token.startswith('/') and p.is_file():files[str(p)]=digest(p)
    for pattern in ('libEGL*','libGL*','libOpenGL*','libglapi*','libgbm*','libdrm*','libnvidia*'):
        for p in Path('/usr/lib').glob(pattern):
            if p.is_file():files[str(p)]=digest(p)
    for line in Path('/proc/self/maps').read_text().splitlines():
        name=line.split()[-1]
        if name.startswith('/') and Path(name).is_file():files[str(Path(name).resolve())]=digest(name)
    ambient={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','PYOPENGL_PLATFORM','EGL_PLATFORM','LIBGL_DRIVERS_PATH','__EGL_VENDOR_LIBRARY_FILENAMES')}
    return {'python':sys.version,'prefix':sys.prefix,'versions':versions,'files':files,'ambient':ambient}

def freeze(out,plan_path,registration):
    plan=rawjson(plan_path);reg=rawjson(registration);validate_registration(plan,reg)
    require(Path(plan['artifact_directory']).resolve()==out.resolve(),'Output differs from registered directory')
    require(not (out/'frozen').exists() and not (out/'freeze.json').exists(),'Existing/incomplete freeze; never repair in place')
    inputs,sources,old=source_manifest(plan)
    # Helpers come from previously frozen studies, never the mutable outputs scripts.
    review=Path(plan['source_review_directory'])/'reviewer.py';checked(review,old['script_sha256'])
    of=rawjson(OPTICS/'freeze.json');checked(OPTICS/'freeze.json',Path(str(OPTICS/'freeze.json')+'.sha256').read_text().strip())
    optics=OPTICS/'frozen/run_dance_optics_full.py';checked(optics,of['files']['run_dance_optics_full.py'])
    inputs[str(OPTICS/'freeze.json')]=digest(OPTICS/'freeze.json')
    # All implicit non-angle/default controls must also match the cached mapper.
    import ast
    def defaults(path):
        node=next(x for x in ast.parse(Path(path).read_text()).body if isinstance(x,ast.ClassDef) and x.name=='LightingConfig')
        return {x.target.id:ast.literal_eval(x.value) for x in node.body if isinstance(x,ast.AnnAssign)}
    mapper_rel='lightsem/experiments/motion_study/lighting_map.py'
    old_mapper=Path(plan['source_review_directory'])/'source'/mapper_rel
    checked(old_mapper,old['source_files'][mapper_rel]);inputs[str(old_mapper)]=digest(old_mapper)
    require(defaults(old_mapper)==defaults(REPO/mapper_rel),'Implicit baseline/nonangle configuration defaults changed')
    for rel in ('lightsem/sim/renderer.py','lightsem/sim/shaders.py','lightsem/conductor/rig.py'):
        require(digest(REPO/rel)==old['source_files'][rel]==of['files']['source/'+rel],'Frozen optics/rig mismatch: '+rel)
    font=Path(old['font_path']);checked(font,old['font_sha256'])
    executables={str(Path(shutil.which(n)).resolve()):digest(Path(shutil.which(n)).resolve()) for n in ('ffmpeg','ffprobe')}
    runtime=runtime_identity()
    estimate=sum(p.stat().st_size for p in (REPO/'lightsem').rglob('*') if p.is_file() and p.suffix in ('.py','.json'))+32*1024**2
    resource_preflight(out,'freeze',estimate)
    frozen=out/'frozen';frozen.mkdir();files={}
    tocopy={'plan.json':Path(plan_path),'registration.json':Path(registration),
        'eval_pose_stage_study.py':Path(__file__).resolve(),'render_pose_stage_study.py':REPO/'scripts/motion/render_pose_stage_study.py',
        'source_helpers/review.py':review,'source_helpers/optics.py':optics,
        'run_torch.sh':REPO/'run_torch.sh','run.sh':REPO/'run.sh',
        'implementation.md':REPO/'experiments/2026-09-10-motion/outputs/loose/dance_pose_stage_study_implementation.md',
        'pre_freeze_engineering_decisions.md':REPO/'experiments/2026-09-10-motion/outputs/loose/pose_stage_pre_freeze_review_decisions.md',
        'source_tests/test_pose_stage_study.py':REPO/'tests/test_pose_stage_study.py',
        'source_tests/test_pose_stage_renderer.py':REPO/'tests/test_pose_stage_renderer.py',
        'source_tests/pose_stage_namespace_smoke.py':REPO/'tests/pose_stage_namespace_smoke.py'}
    for p in (REPO/'lightsem').rglob('*'):
        if p.is_file() and p.suffix in ('.py','.json'):tocopy['source/'+str(p.relative_to(REPO))]=p
    for rel,p in sorted(tocopy.items()):copy_new(p,frozen/rel);files[rel]=digest(frozen/rel)
    for p in (plan_path,registration):inputs[str(Path(p).resolve())]=digest(p)
    savejson(out/'freeze.json',{'schema':1,'created_unix':time.time(),'files':files,'inputs':inputs,'sources':sources,
        'runtime':runtime,'executables':executables,'font_path':str(font),'font_sha256':digest(font),
        'native_gpu_lock_root':str(NATIVE),'helper_bindings':{'review':{'path':str(review),'sha256':digest(review)},'optics':{'path':str(optics),'sha256':digest(optics)}},
        'scope':plan['claim_scope'],'plan_sha256':jhash(plan)})

def load_helper(out,name):
    require(name in ('review','optics'),'Unknown helper')
    fr=loadjson(Path(out)/'freeze.json');p=Path(out)/'frozen/source_helpers'/f'{name}.py'
    checked(p,fr['files']['source_helpers/'+name+'.py'])
    spec=importlib.util.spec_from_file_location('_pose_stage_frozen_'+name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def context(out,activate=False):
    out=Path(out);fr=loadjson(out/'freeze.json');frozen=out/'frozen'
    checked(__file__,fr['files']['eval_pose_stage_study.py'])
    for rel,sha in fr['files'].items():checked(frozen/rel,sha)
    for p,sha in fr['inputs'].items():checked(p,sha)
    for b in fr['helper_bindings'].values():checked(b['path'],b['sha256'])
    for p,sha in fr['executables'].items():checked(p,sha)
    checked(fr['font_path'],fr['font_sha256'])
    runtime=fr['runtime'];require(runtime['python']==sys.version and runtime['prefix']==sys.prefix,'Runtime interpreter changed; use frozen run_torch.sh environment')
    for p,sha in runtime['files'].items():checked(p,sha)
    require(all(os.environ.get(k)==v for k,v in runtime['ambient'].items()),'Frozen numerical/EGL runtime environment changed')
    for name,version in runtime['versions'].items():require(importlib.metadata.version(name)==version,'Runtime distribution changed')
    plan=rawjson(frozen/'plan.json');validate_registration(plan,rawjson(frozen/'registration.json'))
    ctx={'plan':plan,'freeze':fr,'freeze_sha256':digest(out/'freeze.json'),'helpers':{name:load_helper(out,name) for name in ('review','optics')}}
    if activate:
        bindings=activate_source(frozen/'source',fr['files'])
        from lightsem.conductor.rig import canonical_rig
        from lightsem.sim.renderer import RenderConfig
        ctx.update(rig=canonical_rig(),cfg=RenderConfig(**plan['renderer']['renderer']),namespace_bindings=bindings)
    return ctx

def activate_source(source,files):
    require(not any(k=='lightsem' or k.startswith('lightsem.') for k in sys.modules),'Mixed mutable/frozen project imports')
    source=Path(source).resolve();sys.path.insert(0,str(source))
    from lightsem.experiments.motion_study import lighting_map,pose_stage
    from lightsem.sim.renderer import RenderConfig
    bindings={}
    for name,m in list(sys.modules.items()):
        if name=='lightsem' or name.startswith('lightsem.'):
            path=Path(m.__file__).resolve()
            require(path.is_relative_to(source),'Old helper namespace shadowed new frozen source: '+name)
            rel='source/'+str(path.relative_to(source));require(rel in files,'Unbound frozen project module: '+name)
            checked(path,files[rel]);bindings[name]={'path':str(path),'sha256':files[rel]}
    require(lighting_map.LightingConfig(target_space='pose_stage').target_space=='pose_stage','New mapper dispatch unavailable')
    require(Path(pose_stage.__file__).resolve()==source/'lightsem/experiments/motion_study/pose_stage.py','Wrong pose_stage implementation')
    return bindings


def validate_arrays(world,dmx,times,dmx_time,n):
    require(world.shape==(n,22,3) and np.isfinite(world).all(),'Incomplete/nonfinite full world')
    require(dmx.dtype==np.uint8 and dmx.shape==(round(n/30*40),512),'Incomplete/corrupt full DMX')
    require(np.array_equal(times,np.arange(n)/30) and np.array_equal(dmx_time,np.arange(len(dmx))/40),'Incomplete/noncanonical time coverage')

def verify_parity(cached,recomputed):
    require(cached.dtype==recomputed.dtype and cached.shape==recomputed.shape and np.array_equal(cached,recomputed),'Cached baseline recomputation parity failed; no repair or alternate baseline')

def nonangle_equal(a,b,rig):
    mask=np.ones(512,bool)
    for f in rig.by_kind('moving_head'):mask[f.address-1:f.address+3]=False
    return bool(a.shape==b.shape and np.array_equal(a[:,mask],b[:,mask]))

def decoded(dmx,rig):
    require(dmx.dtype==np.uint8 and dmx.ndim==2 and dmx.shape[1]==512,'Invalid DMX')
    d=dmx.astype(float);angles=[];vectors=[];levels=[]
    for f in rig.by_kind('moving_head'):
        a=f.address-1;pt=np.stack((d[:,a]*256+d[:,a+1],d[:,a+2]*256+d[:,a+3]),-1)/65535-.5
        pt*=np.array([f.pan_range_deg,f.tilt_range_deg]);angles.append(pt)
        p,t=np.deg2rad(pt).T;vectors.append(np.stack((np.sin(p)*np.cos(t),-np.cos(p)*np.cos(t),np.sin(t)),-1))
        levels.append(d[:,a+4]/255*(d[:,a+6:a+9]/255@np.array([.2126,.7152,.0722])))
    require(len(angles)==6,'Study requires six heads')
    return np.stack(angles,1),np.stack(vectors,1),np.stack(levels,1)

def mechanics(dmx,rig):
    a,v,level=decoded(dmx,rig);speed=np.linalg.norm(np.diff(a,axis=0)*40,axis=-1);acc=np.linalg.norm(np.diff(a,n=2,axis=0)*1600,axis=-1)
    return {'max_speed_deg_s':float(speed.max(initial=0)),'max_acceleration_deg_s2':float(acc.max(initial=0)),
        'pass':bool(speed.max(initial=0)<=90+1e-8 and acc.max(initial=0)<=360+1e-8),'instrument':'decoded pan/tilt Euclidean norms; every full-record difference'},v,level

def band_rms(x,band):
    from scipy.signal import welch
    require(len(x)>=400 and np.isfinite(x).all(),'Invalid or too-short broadband coverage; no dropped samples')
    f,p=welch(x,fs=40,window='hann',nperseg=400,noverlap=200,detrend='linear',axis=0,scaling='density')
    # Exact piecewise-linear PSD edges, not legacy half-open-bin trapezoids.
    require(0<=band[0]<band[1]<=f[-1],'Invalid spectral band')
    grid=np.r_[band[0],f[(f>band[0])&(f<band[1])],band[1]]
    flat=p.reshape(len(f),-1)
    psd=np.stack([np.interp(grid,f,flat[:,j]) for j in range(flat.shape[1])],-1).reshape((len(grid),)+p.shape[1:])
    integrate=np.trapezoid if hasattr(np,'trapezoid') else np.trapz
    return np.sqrt(np.maximum(0,integrate(psd,grid,axis=0).sum(-1)))

def longest_run(mask,fps=40):
    change=np.diff(np.r_[False,np.asarray(mask,bool),False].astype(int));return float(np.max(np.flatnonzero(change==-1)-np.flatnonzero(change==1),initial=0)/fps)
def stats(a):
    a=np.asarray(a);good=np.isfinite(a);v=a[good]
    return {'samples':int(a.size),'invalid':int((~good).sum()),'min':float(v.min()) if v.size else None,'median':float(np.median(v)) if v.size else None,'p10':float(np.quantile(v,.1)) if v.size else None,'p90':float(np.quantile(v,.9)) if v.size else None,'max':float(v.max()) if v.size else None}

def cloud(first_hit_px,first_hit_radius):
    valid=np.isfinite(first_hit_px).all((1,2))&np.isfinite(first_hit_radius).all(1)&(first_hit_radius>0).all(1)
    spread=np.sqrt(np.mean(np.sum((first_hit_px-first_hit_px.mean(1,keepdims=True))**2,-1),1))
    ratio=spread/first_hit_radius.mean(1);below=(ratio<1)&valid
    return {'valid_all':bool(valid.all()),'invalid_frames':int((~valid).sum()),'status':'measured' if valid.all() else 'unmeasurable_coverage_not_collapse',
        'ratio':stats(ratio),'fraction_below':float(below.mean()),'fraction_below_denominator':'all scheduled frames; invalid excluded from below numerator; gate cannot pass missing samples',
        'fraction_below_valid_only':float(below[valid].mean()) if valid.any() else None,'longest_below_s':longest_run(below),
        'pass':bool(valid.all() and below.mean()<=.01 and longest_run(below)<=.5)},ratio,spread

def hull_area(points):
    # Exact monotone-chain area; collinear/duplicate clouds yield zero, no jitter repair.
    pts=sorted(set(map(tuple,np.asarray(points).tolist())))
    if len(pts)<3:return 0.
    def cross(o,a,b):return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])
    lower=[];upper=[]
    for seq,out in ((pts,lower),(reversed(pts),upper)):
        for p in seq:
            while len(out)>=2 and cross(out[-2],out[-1],p)<=0:out.pop()
            out.append(p)
    p=np.array(lower[:-1]+upper[:-1]);return float(abs(np.sum(p[:,0]*np.roll(p[:,1],-1)-p[:,1]*np.roll(p[:,0],-1)))/2)

def geometry(dmx,rig,cfg,optics,interval):
    # Common zero-zoom override ONLY after original-byte nonangle test.
    stream=dmx.copy();heads=rig.by_kind('moving_head')
    for f in heads:stream[:,f.address-1+9]=0
    _,v,level=decoded(stream,rig);pos=np.array([f.position for f in heads]);hits,surface,dist=optics.first_hits(v,pos)
    theta=optics.half_peak_angle(np.array([np.deg2rad(f.beam_min_deg/2) for f in heads]),cfg.beam_edge_profile,cfg.beam_edge_softness_deg)
    mid=pos+(hits-pos)/2;midpx=optics.project(mid,cfg);hitpx=optics.project(hits,cfg)
    radius=optics.circle_radius(mid,v,dist/2*np.tan(theta),cfg)
    hitradius=optics.circle_radius(hits,v,dist*np.tan(theta),cfg)
    hitvalid=np.isfinite(hits).all(-1)&(surface>=0);viewport=np.isfinite(hitpx).all(-1)&(hitpx[...,0]>=0)&(hitpx[...,0]<cfg.width)&(hitpx[...,1]>=0)&(hitpx[...,1]<cfg.height)
    lo,hi=interval;sl=slice(round(lo*40),round(hi*40));require(sl.stop<=len(dmx) and sl.stop>sl.start,'Missing analysis coverage')
    valid=np.isfinite(midpx[sl]).all() and np.isfinite(radius[sl]).all() and bool((radius[sl]>0).all())
    bands={}
    for band in ((.5,4),(.5,1),(1,2),(2,4)):
        if valid:
            rms=band_rms(midpx[sl],band);rad=radius[sl].mean(0);ratios=rms/rad
            bands[str(list(band))]={'mean_head_ratio':float(ratios.mean()),'per_head_ratio':ratios.tolist(),'per_head_rms_px':rms.tolist(),'per_head_mean_radius_px':rad.tolist(),'ratio_of_means_supplemental':float(rms.mean()/rad.mean())}
        else:bands[str(list(band))]={'mean_head_ratio':None,'invalid_geometry':True}
    cl,ratio,spread=cloud(hitpx[sl],hitradius[sl]);pair=np.triu_indices(6,1)
    worlddist=np.linalg.norm(hits[sl,:,None]-hits[sl,None,:],axis=-1)[:,pair[0],pair[1]]
    pixdist=np.linalg.norm(hitpx[sl,:,None]-hitpx[sl,None,:],axis=-1)[:,pair[0],pair[1]]
    coneratio=pixdist/(hitradius[sl,pair[0]]+hitradius[sl,pair[1]])
    chart=np.stack((hits[sl,:,0],np.where(surface[sl]==0,1-hits[sl,:,2],7+hits[sl,:,1])),-1)
    chartvalid=np.isin(surface[sl],[0,1]).all(1)&np.isfinite(chart).all((1,2))
    area=np.full(len(chart),np.nan)
    for i in np.flatnonzero(chartvalid):area[i]=hull_area(chart[i])
    blocks=[{'start_s':i/40,'end_s':min(i+400,len(dmx))/40,'valid_head_fraction':float(hitvalid[i:i+400].mean()),'viewport_fraction':float(viewport[i:i+400].mean())} for i in range(0,len(dmx),400)]
    out={'interval_s':list(interval),'bands':bands,'primary':bands[str([.5,4])]['mean_head_ratio'],'geometry_valid':bool(valid),
        'full_valid_head_fraction':float(hitvalid.mean()),'full_viewport_fraction':float(viewport.mean()),'blocks10s':blocks,'collapse':cl,
        'surface_fractions':{str(k):float((surface[sl]==k).mean()) for k in range(-1,5)},'camera_spread_px':stats(spread),
        'first_hit_x_span_m':stats(np.ptp(hits[sl,:,0],axis=1)),'intrinsic_floor_wall_hull_area_m2':stats(area),'intrinsic_chart_supported_frames':int(chartvalid.sum()),
        'intrinsic_chart_unsupported_frames':int((~chartvalid).sum()),'intrinsic_chart_truss_frames':int(np.isin(surface[sl],[2,3,4]).any(1).sum()),
        'pair_first_hit_distance_m':stats(worlddist),'pair_camera_distance_px':stats(pixdist),'pair_projected_circle_overlap_ratio':stats(coneratio),
        'pair_projected_circle_overlap_fraction_valid_pairs':float((coneratio[np.isfinite(coneratio)]<1).mean()) if np.isfinite(coneratio).any() else None,
        'pair_valid_samples':int(np.isfinite(coneratio).sum()),'pair_invalid_samples':int((~np.isfinite(coneratio)).sum()),
        'pair_scope':'Pair first-hit separations and projected equivalent-circle overlaps; NOT all-head collapse or exact illuminated cone/pixel overlap'}
    data={'first_hit_world':hits,'surface':surface,'first_hit_px':hitpx,'midshaft_px':midpx,'midshaft_radius_px':radius,'first_hit_radius_px':hitradius,'collapse_interval_ratio':ratio,'interval_chart_area_m2':area}
    return out,data

def map_gates(reports,item,equal):
    c=reports['candidate'];b=reports['baseline'];m=c['mechanics'];cl=c['collapse']
    require(type(equal) is bool,'Nonangle comparison must be an explicit boolean')
    gates={'measurement_coverage':c['geometry_valid'] and b['geometry_valid'] and c['collapse']['valid_all'] and b['collapse']['valid_all'],
        'mechanics':m['max_speed_deg_s']<=90+1e-8 and m['max_acceleration_deg_s2']<=360+1e-8,
        'valid':c['full_valid_head_fraction']>=.99,'nonangle':equal,
        'collapse':cl['valid_all'] and cl['fraction_below']<=.01 and cl['longest_below_s']<=.5}
    if item['key']=='shpongle':
        gates.update(primary=c['primary'] is not None and b['primary'] is not None and b['primary']>0 and c['primary']>=2*b['primary'],
            surfaces=c['surface_fractions']['0']>=.10 and c['surface_fractions']['1']>=.10,
            quiet=c['quiet']['coordinate_speed_median_all_heads_deg_s']<=16.3488)
    return {k:bool(v) for k,v in gates.items()}


def map_item(out,ctx,item):
    target=out/'mapped'/item['id'];existed=target.exists()
    state={'phase':'validate_source_cache','completed_gates':{},'completed_checks':{},'completed_mechanics':{}}
    try:return _map_item(out,ctx,item,state)
    except BaseException as error:
        if not existed:
            try:
                target.mkdir(parents=True,exist_ok=True)
                files={str(p.relative_to(target)):digest(p) for p in target.rglob('*') if p.is_file()}
                savejson(target/'failed.json',{'item':item,'freeze_sha256':ctx['freeze_sha256'],
                    'error_type':type(error).__name__,'error':str(error),'files':files,'operation':state,
                    'status':'resource_blocked' if isinstance(error,ResourceBlocked) else 'infrastructure_or_validation_failure_not_a_scientific_gate_verdict',
                    'partial_artifacts_retained':True,'retry_in_place':False})
            except BaseException as recording_error:
                error.add_note('Failure receipt could not be completed: '+repr(recording_error))
                print('SECONDARY FAILURE RECORDING ERROR:',repr(recording_error),file=sys.stderr,flush=True)
        raise


def _map_item(out,ctx,item,state):
    target=out/'mapped'/item['id']
    if target.exists():return mapped(out,ctx,item)[1]
    from lightsem.experiments.motion_study.lighting_map import LightingConfig,compose_world_joints
    import torch
    torch.set_num_threads(4)
    src=ctx['freeze']['sources'][item['id']];worlds=arrays(src['world'],[item['source_arm'],'times']);cached=arrays(src['dmx'],[item['source_arm'],'time']);curve=arrays(src['loudness'])
    world=worlds[item['source_arm']];baseline=cached[item['source_arm']];n=item['full_frames']
    validate_arrays(world,baseline,worlds['times'],cached['time'],n)
    target.mkdir(parents=True,exist_ok=False);savejson(target/'started.json',{'item':item,'freeze_sha256':ctx['freeze_sha256'],'started_unix':time.time()})
    state['phase']='baseline_full_recomputation'
    original=world.copy();recomputed,_,br=compose_world_joints(world,curve,n/30,config=LightingConfig(**ctx['plan']['baseline']),dynamics=True)
    state['phase']='baseline_exact_parity'
    verify_parity(baseline,recomputed);state['completed_checks']['baseline_exact_parity']=True
    state['phase']='candidate_full_mapping'
    candidate,trace,cr=compose_world_joints(world,curve,n/30,config=LightingConfig(**dict(ctx['plan']['baseline'],**ctx['plan']['candidate_change'])),dynamics=True)
    require(np.array_equal(original,world),'Mapper modified saved world; no root reintegration allowed')
    validate_arrays(world,candidate,worlds['times'],cached['time'],n)
    state['phase']='save_full_streams';state['completed_gates']['nonangle']=nonangle_equal(baseline,candidate,ctx['rig'])
    savenpz(target/'full_world.npz',world=world,times=worlds['times'])
    savenpz(target/'full_dmx.npz',baseline=baseline,candidate=candidate,time=cached['time'])
    reports={};interval=(70,90) if item['key']=='shpongle' else (item['start']/30,(item['start']+item['frames'])/30)
    for arm,dmx in [('baseline',baseline),('candidate',candidate)]:
        state['phase']=arm+'_mechanics'
        mech,v,levels=mechanics(dmx,ctx['rig']);state['completed_mechanics'][arm]=mech
        if arm=='candidate':state['completed_gates']['mechanics']=mech['pass']
        state['phase']=arm+'_geometry'
        g,arr=geometry(dmx,ctx['rig'],ctx['cfg'],ctx['helpers']['optics'],interval)
        g['mechanics']=mech
        if item['key']=='shpongle':
            angles,_,_=decoded(dmx,ctx['rig'])
            speed=np.linalg.norm(np.diff(angles[:800],axis=0)*40,axis=-1)
            g['quiet']={'coordinate_speed_median_all_heads_deg_s':float(np.median(speed)),
                'instrument':'decoded pan/tilt first differences; Euclidean norm; 0-20s six heads; unweighted, no lit gating'}
        reports[arm]=g;state['phase']=arm+'_save_numeric'
        savejson(target/(arm+'_numeric.json'),{'item':item,'arm':arm,'freeze_sha256':ctx['freeze_sha256'],'report':g})
        savenpz(target/(arm+'_geometry.npz'),**arr)
    if item['key']=='shpongle':
        reports['candidate']['quiet']['candidate_minus_matched_baseline_deg_s']=reports['candidate']['quiet']['coordinate_speed_median_all_heads_deg_s']-reports['baseline']['quiet']['coordinate_speed_median_all_heads_deg_s']
    equal=nonangle_equal(baseline,candidate,ctx['rig']);gates=map_gates(reports,item,equal)
    state['phase']='save_complete_map_receipt';state['completed_gates']=gates
    files={str(p.relative_to(target)):digest(p) for p in target.rglob('*') if p.is_file()}
    row={'item':item,'freeze_sha256':ctx['freeze_sha256'],'files':files,'reports':reports,'gates':gates,'pass':bool(all(gates.values())),
        'baseline_recomputation_exact':True,'nonangle_equal':equal,'source_bindings':src,'mapping_reports':{'baseline':br,'candidate':cr},'world_unchanged':True,'namespace_bindings':ctx['namespace_bindings']}
    savejson(target/'map.json',row);return row

def mapped(out,ctx,item):
    target=Path(out)/'mapped'/item['id'];row=loadjson(target/'map.json')
    require(row['item']==item and row['freeze_sha256']==ctx['freeze_sha256'],'Cached result wrong identity')
    require({'full_world.npz','full_dmx.npz','baseline_geometry.npz','candidate_geometry.npz','started.json'}.issubset(row['files']),'Incomplete mapping result')
    for rel,sha in row['files'].items():checked(target/rel,sha)
    gates=map_gates(row['reports'],item,row['nonangle_equal'])
    require(row['gates']==gates and row['pass']==all(gates.values()) and row['baseline_recomputation_exact'] is True,'Inconsistent result gates/parity')
    return target,row

def oracle_wins(rows,plan):
    spec=plan['oracle_transfer_selection'];selected=[r for r in rows if r['item']['category'] in spec['categories']]
    require(len(selected)==spec['total'],'Oracle gain denominator must be exact eight travel/coordination windows')
    return sum(r['reports']['candidate']['primary'] is not None and r['reports']['baseline']['primary'] is not None and r['reports']['candidate']['primary']>r['reports']['baseline']['primary'] for r in selected)


def stage_receipt(out,ctx,name):
    path=Path(out)/name/'stage.json';row=loadjson(path)
    require(row['freeze_sha256']==ctx['freeze_sha256'] and row['stage']==name,'Stage identity')
    for item_id,sha in row['map_receipts'].items():
        item=next(x for x in ctx['plan']['items'] if x['id']==item_id);target,r=mapped(out,ctx,item);checked(target/'map.json',sha)
    expected=[x['id'] for x in ctx['plan']['items'] if (x['key']=='shpongle')==(name=='native')]
    require(set(row['map_receipts'])==set(expected),'Incomplete stage coverage')
    rows=[mapped(out,ctx,next(x for x in ctx['plan']['items'] if x['id']==i))[1] for i in expected]
    gates={'all_item_gates':all(r['pass'] for r in rows)}
    if name=='references':
        wins=oracle_wins(rows,ctx['plan'])
        gates['oracle_transfer']=wins>=ctx['plan']['oracle_transfer_selection']['minimum_improved']
        require(row['wins']==wins,'Cached oracle win count changed')
    require(row['gates']==gates and row['pass']==all(gates.values()),'Cached stage gate verdict inconsistent')
    return row

def evaluate_stage(out,ctx,name):
    require(name in ('native','references'),'Unknown numerical stage')
    if name=='references':require(stage_receipt(out,ctx,'native')['pass'],'Native early stop forbids reference candidate evaluation')
    path=out/name/'stage.json'
    if path.exists():return stage_receipt(out,ctx,name)
    selected=[x for x in ctx['plan']['items'] if (x['key']=='shpongle')==(name=='native')]
    estimate=sum(item['full_frames']*8192 for item in selected if not (out/'mapped'/item['id']).exists())
    resource_preflight(out,name,estimate)
    rows=[map_item(out,ctx,item) for item in selected]
    gates={'all_item_gates':all(r['pass'] for r in rows)}
    if name=='references':
        wins=oracle_wins(rows,ctx['plan'])
        gates['oracle_transfer']=wins>=ctx['plan']['oracle_transfer_selection']['minimum_improved']
    row={'stage':name,'freeze_sha256':ctx['freeze_sha256'],'map_receipts':{r['item']['id']:digest(out/'mapped'/r['item']['id']/'map.json') for r in rows},'gates':gates,'pass':all(gates.values()),'wins':wins if name=='references' else None}
    savejson(path,row);return row

def render_schedule(out,ctx):
    n=stage_receipt(out,ctx,'native');items={x['id']:x for x in ctx['plan']['items']};shp=items['shpongle_prior_flow0']
    def view(item,start,frames,suffix,matched=False):return {'id':item['id']+'_'+suffix,'item_id':item['id'],'start':start,'frames':frames,'matched':matched}
    if not n['pass']:return [view(shp,lo*30,600,f'diagnostic_{lo}_{lo+20}') for lo in (0,70,190)]
    r=stage_receipt(out,ctx,'references')
    if not r['pass']:
        result=[view(items[i],items[i]['start'],600,'diagnostic') for i in ('holds_083_1200_reference','travel_027_4200_reference','coordination_057_1200_reference')]
        return result+[view(shp,70*30,600,'diagnostic_70_90')]
    result=[view(item,item['start'],item['frames'],'planned') for item in ctx['plan']['items']]
    return result+[view(shp,0,shp['full_frames'],'global_matched',True)]

def numerical_report(out,ctx):
    """Terminal create-only report supports failed, corrupt and unrun coverage."""
    rows=[]
    for item in ctx['plan']['items']:
        target=out/'mapped'/item['id'];row={'item':item,'status':'unrun','gates':None}
        if (target/'map.json').exists():
            try:
                _,m=mapped(out,ctx,item)
                row.update(status='complete',gates=m['gates'],passed=m['pass'],receipt_sha256=digest(target/'map.json'))
            except (ValueError,KeyError,FileNotFoundError) as error:
                row.update(status='corrupt_result',error=str(error))
        elif (target/'failed.json').exists():
            try:
                failed=loadjson(target/'failed.json')
                require(failed['item']==item and failed['freeze_sha256']==ctx['freeze_sha256'],'Failure identity')
                for rel,sha in failed['files'].items():checked(target/rel,sha)
                row.update(status='resource_blocked' if failed.get('status')=='resource_blocked' else 'failed',failure=failed,gates=failed.get('operation',{}).get('completed_gates',{}),receipt_sha256=digest(target/'failed.json'))
            except (ValueError,KeyError,FileNotFoundError) as error:
                row.update(status='corrupt_failure',error=str(error))
        elif target.exists():
            row.update(status='incomplete',files={str(p.relative_to(target)):digest(p) for p in target.rglob('*') if p.is_file()})
        rows.append(row)
    stages={}
    for name in ('native','references'):
        if (out/name/'stage.json').exists():
            try:stages[name]={'status':'complete','receipt':stage_receipt(out,ctx,name),'sha256':digest(out/name/'stage.json')}
            except (ValueError,KeyError,FileNotFoundError) as error:stages[name]={'status':'corrupt','error':str(error)}
        else:stages[name]={'status':'unrun_or_incomplete'}
    passed=all(s['status']=='complete' and s['receipt']['pass'] for s in stages.values())
    result={'freeze_sha256':ctx['freeze_sha256'],'storage_now':storage_state(out),
        'resource_preflights':[loadjson(p) for p in sorted((out/'resource_preflight').glob('*.json'))],
        'rows':rows,'stages':stages,'all_numeric_gates_pass':passed,
        'all_maps_complete':all(r['status']=='complete' for r in rows),'unassessed_items':[r['item']['id'] for r in rows if r['status']!='complete'],
        'terminal_create_only':True,'scope':ctx['plan']['claim_scope'],'human_music_fit_acceptance':None,'pipeline_qualified':False}
    savejson(out/'numeric_report.json',result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('freeze','verify','native','references','schedule','report'))
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--plan',type=Path);parser.add_argument('--registration',type=Path)
    a=parser.parse_args();a.output=a.output.resolve()
    with lock(a.output):
        if a.stage=='freeze':
            require(a.plan is not None and a.registration is not None,'Freeze needs explicit --plan --registration');freeze(a.output,a.plan,a.registration);return
        ctx=context(a.output,activate=a.stage in ('native','references'))
        if a.stage in ('native','references'):print(json.dumps(evaluate_stage(a.output,ctx,a.stage),indent=2))
        elif a.stage=='schedule':print(json.dumps(render_schedule(a.output,ctx),indent=2))
        elif a.stage=='report':print(json.dumps(numerical_report(a.output,ctx),indent=2))
        else:print('VERIFIED frozen sources, inputs, helpers, runtime and parent registration; no real stage executed')
if __name__=='__main__':main()
