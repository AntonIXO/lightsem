"""Observed Just Dance poses on original timestamps; no synthetic 3D labels."""
from pathlib import Path
from dataclasses import dataclass,asdict
import hashlib,json,re
import numpy as np
from scipy.ndimage import gaussian_filter1d

ROOT=Path('/mnt/lightsem-data/experiments/lightsem/native_dynamics_jd_20261005_v1')
CACHE=Path('/mnt/lightsem-data/cache')
COCO=np.array([0,5,6,7,8,9,10,11,12,13,14,15,16])
SMPL=np.array([15,16,17,18,19,20,21,1,2,4,5,7,8])
REGIONS={'lower_body':[7,8,9,10,11,12],'torso':[1,2,7,8],'head':[0], 'left_arm':[1,3,5],'right_arm':[2,4,6],'whole_articulation':list(range(13))}

@dataclass(frozen=True)
class Quality:
    confidence:float=.60
    torso_confidence:float=.70
    minimum_duration_s:float=90.
    maximum_duration_s:float=420.
    minimum_coverage:float=.80
    minimum_box_height_fraction:float=.30
    maximum_interpolation_gap_s:float=.060
    maximum_av_duration_difference_s:float=.25
    maximum_joint_step_torso:float=1.25
    duplicate_run_frames:int=6
    train_groups:int=72
    dev_groups:int=24
    validation_fraction:float=.25
    seed:int=7001
    frames:int=480
    fps:float=30.

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    with Path(str(path)+'.sha256.json').open('x') as f:json.dump({'sha256':sha(path)},f)
def save_npz(path,**arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not all(np.isfinite(a).all() for a in arrays.values()):raise ValueError('Nonfinite cache value')
    with path.open('xb') as f:np.savez_compressed(f,**arrays)

def pose_observations(path,duration,cfg=Quality(),*,coach_index=None):
    with np.load(path,allow_pickle=False) as z:
        k=z['K'].astype(np.float64);confidence=z['C'].astype(np.float64);box=z['box'].astype(np.float64);time=z['t'].astype(np.float64);meta=json.loads(z['meta'].item())
    if k.ndim!=4 or k.shape[2:]!=(17,2) or len(time)!=len(k) or np.any(np.diff(time)<=0):raise ValueError('Invalid pose/timestamp layout')
    coaches=k.shape[1]
    if coach_index is None:
        if coaches!=1:raise ValueError('Single-coach pilot; multicoach recordings excluded before selection')
        coach_index=0
    if isinstance(coach_index,bool) or not isinstance(coach_index,(int,np.integer)) or not 0<=coach_index<coaches:
        raise ValueError('Invalid coach index')
    if confidence.shape!=k.shape[:-1] or box.shape!=(len(k),coaches,4):
        raise ValueError('Invalid confidence/box layout')
    # Expanded admission selects an existing tracked coach, never averages or
    # concatenates different people. The historical single-coach call is fixed.
    k=k[:,coach_index];confidence=confidence[:,coach_index];box=box[:,coach_index];height=float(meta['height'])
    valid=np.isfinite(k).all(-1)&np.isfinite(confidence)&(confidence>=cfg.confidence)
    # Native boxes are xyxy; tiny UI people cannot become a dance target.
    visible=np.isfinite(box).all(-1)&((box[:,3]-box[:,1])>=cfg.minimum_box_height_fraction*height)
    valid&=visible[:,None]
    torso_ids=[5,6,11,12];torso_valid=valid[:,torso_ids].all(1)&(confidence[:,torso_ids]>=cfg.torso_confidence).all(1)
    shoulder=(k[:,5]+k[:,6])/2;hip=(k[:,11]+k[:,12])/2;length=np.linalg.norm(shoulder-hip,axis=-1)
    eligible=torso_valid&np.isfinite(length)&(length>.05*height)
    if eligible.sum()<30:raise ValueError('Insufficient reliable torso observations')
    scale=float(np.median(length[eligible]))
    # Audit exact repeated observations. Raw extractor leaves missing detections
    # NaN (no forward fill), so a confident repeated pose is NOT automatically
    # an estimator failure; preserve real held poses and flag runs for review.
    clean=np.where(np.isfinite(k),k,0.)
    repeats=np.r_[False,np.all(clean[1:]==clean[:-1],axis=(1,2)) & valid[1:].all(1)&valid[:-1].all(1)]
    edges=np.diff(np.r_[False,repeats,False].astype(int));starts=np.flatnonzero(edges==1);ends=np.flatnonzero(edges==-1)
    frozen=np.zeros(len(k),bool)
    for a,b in zip(starts,ends):
        if b-a>=cfg.duplicate_run_frames:frozen[max(0,a-1):b]=True
    # Tracking jumps are missing observations, not fast gestures to learn.
    steps=np.linalg.norm(np.diff(clean,axis=0),axis=-1)/scale
    jumps=steps>cfg.maximum_joint_step_torso
    valid[1:]&=~jumps;valid[:-1]&=~jumps
    hip_valid=valid[:,[11,12]].all(1);valid&=hip_valid[:,None]
    q=(clean-hip[:,None])/scale;q[...,1]*=-1
    q=np.where(valid[...,None],q,0.)[:,COCO];valid=valid[:,COCO]
    target=np.arange(int(np.ceil(duration*cfg.fps)))/cfg.fps
    hi=np.searchsorted(time,target,side='left');hi=np.clip(hi,1,len(time)-1);lo=hi-1
    fraction=(target-time[lo])/np.maximum(time[hi]-time[lo],1e-12)
    ok=(target>=time[0])&(target<=time[-1])&((time[hi]-time[lo])<=cfg.maximum_interpolation_gap_s)
    mask=valid[lo]&valid[hi]&ok[:,None]
    value=(1-fraction[:,None,None])*q[lo]+fraction[:,None,None]*q[hi]
    # Exact native samples need only that observation, not a missing neighbour.
    exact=np.isclose(target,time[hi],atol=1e-8,rtol=0)
    value[exact]=q[hi[exact]];mask[exact]=valid[hi[exact]]&ok[exact,None]
    exact0=np.isclose(target,time[0],atol=1e-8,rtol=0)
    value[exact0]=q[0];mask[exact0]=valid[0]
    value=np.where(mask[...,None],value,0.).astype(np.float32)
    coverage=float((mask.sum(1)>=10).mean())
    # Eligible 16s windows must have enough observed joints; holes remain masked.
    n=cfg.frames;good=(mask.sum(1)>=10).astype(float)
    window=np.convolve(good,np.ones(n)/n,'valid') if len(good)>=n else np.array([])
    starts=np.flatnonzero(window>=cfg.minimum_coverage)[::15].astype(np.int32)
    return {'time':target,'pose':value,'mask':mask,'starts':starts}, {'coverage':coverage,'torso_scale_px':scale,'constant_pose_frames_for_review':int(frozen.sum()),'tracking_jump_edges':int(jumps.any(1).sum()),'native_frames':len(time),'native_duration_s':float(time[-1]),'coach_count':coaches,'coach_index':int(coach_index),'pose_meta':meta,'eligible_windows':len(starts)}

def prepare_manifest(root=ROOT,cfg=Quality()):
    import soundfile as sf
    root=Path(root);mapping=read(CACHE/'justdance_songs.json');audited=[];rejected=[]
    # The old development groups are protected.  Every other duplicate-resolved
    # group with a paired waveform and an eligible masked window is admitted to
    # TRAIN; there is deliberately no historical 72-group cap.
    old_path=root/'data/MANIFEST.json'
    if not old_path.exists() and (root/'records/MANIFEST_20261005_original.json').exists():
        old_path=root/'records/MANIFEST_20261005_original.json'
    old=read(old_path) if old_path.exists() else {'entries':[]}
    protected_groups={r['group'] for r in old.get('entries',[]) if r.get('partition')!='train'}
    protected_keys={r['key'] for r in old.get('entries',[]) if r.get('partition')!='train'}
    for item in sorted(mapping.get('rows',[]),key=lambda r:r['yt_id']):
        key=item['yt_id'];pose=CACHE/'justdance_pose'/f'{key}.npz';wav=CACHE/'justdance_wav'/f'{key}.wav'
        if item.get('excluded') or not pose.exists() or not wav.exists():
            rejected.append({'key':key,'reason':'excluded or missing paired pose/audio'});continue
        title=item.get('title_norm','').lower();group=item.get('song_key')
        if not group:
            rejected.append({'key':key,'reason':'unresolved duplicate identity'});continue
        if 'cure for me' in title or 'shpongle' in title:
            rejected.append({'key':key,'reason':'showcase quarantine'});continue
        try:
            duration=sf.info(wav).frames/sf.info(wav).samplerate
            if not cfg.minimum_duration_s<=duration<=cfg.maximum_duration_s:raise ValueError('duration outside 90-420s')
            with np.load(pose,allow_pickle=False) as z:coaches=z['K'].shape[1]
            for coach in range(coaches):
                observations,quality=pose_observations(pose,duration,cfg,coach_index=coach)
                if abs(quality['native_duration_s']-duration)>cfg.maximum_av_duration_difference_s:raise ValueError('pose/audio duration binding mismatch')
                if quality['coverage']<cfg.minimum_coverage or not quality['eligible_windows']:raise ValueError('insufficient reliable original-time pose coverage')
                audited.append({'key':key,'coach_index':coach,'group':group,'title':title,'pose_path':str(pose),'wav_path':str(wav),'duration_s':duration,'frames':len(observations['time']),'quality':{k:v for k,v in quality.items() if k!='pose_meta'},'pose_sha256':sha(pose),'wav_sha256':sha(wav),'protected':key in protected_keys or group in protected_groups})
        except (ValueError,KeyError) as error:rejected.append({'key':key,'reason':str(error)})
    groups={}
    for row in audited:groups.setdefault(row['group'],[]).append(row)
    selected=[]
    for group,rows in groups.items():
        # Preserve the exact historical dev recording where possible; otherwise
        # choose the highest-coverage coach deterministically.
        old_keys={r['key'] for r in old.get('entries',[]) if r.get('group')==group and r.get('partition')!='train'}
        candidates=[r for r in rows if r['key'] in old_keys] or rows
        row=sorted(candidates,key=lambda x:(-x['quality']['coverage'],x['key'],x['coach_index']))[0]
        row['partition']='val' if group in protected_groups else 'train';selected.append(row)
    selected.sort(key=lambda x:(x['partition']!='val',hashlib.sha256((str(cfg.seed)+x['group']).encode()).hexdigest()))
    if len({r['group'] for r in selected if r['partition']=='val'})<len(protected_groups):
        raise ValueError('Protected development groups could not be reconstructed')
    for row in selected:
        path=root/'data/poses'/f"{row['key']}.npz";observations,_=pose_observations(row['pose_path'],row['duration_s'],cfg,coach_index=row['coach_index']);save_npz(path,**observations)
        row['pose_cache']=str(path);row['pose_cache_sha256']=sha(path)
        row['timeline_path']=str(root/'features/timelines'/f"jd_{row['key']}.npz")
        row['feature_key']='jd_'+row['key']
    report={'schema_version':2,'quality':asdict(cfg),'entries':selected,'qualified_groups_admitted':len(groups),'qualified_recordings':len(audited),'rejected':rejected,'selection':'All duplicate-resolved, label-independent eligible groups admitted to TRAIN; prior development groups remain VAL; one best coach/recording per group. Showcase and unresolved identities remain excluded.','grouping_source':str(CACHE/'justdance_songs.json'),'grouping_sha256':sha(CACHE/'justdance_songs.json'),'native_motion_3D_labels':False,'seconds':{p:sum(x['duration_s'] for x in selected if x['partition']==p) for p in ['train','val']},'code_sha256':sha(__file__)}
    write(root/'data/MANIFEST.json',report);return report

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);a=p.parse_args();r=prepare_manifest(a.root);print(json.dumps({k:r[k] for k in ['qualified_groups_before_bounded_selection','qualified_recordings','seconds']}))
