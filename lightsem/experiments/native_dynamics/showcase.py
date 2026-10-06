"""Full-song native generation and fixed actual-DMX transfer; no time warp."""
from pathlib import Path
import math,hashlib
import numpy as np
import torch
from .data import ROOT,read,write,sha,save_npz
from .evaluation import setup,load_final,cpu_noise
from .dataset import CONTEXT_BINS
from ..dynamic_dance.contracts import MusicTimeline,DanceResult
from ..dynamic_dance.audio import model_features
from ..dynamic_dance.training import validate_audio_binding
from ..motion_study.geometry import joints_from_features
from ..dynamic_dance_combo.campaign import map_one,body_path
COMBO=Path('/mnt/lightsem-data/experiments/lightsem/dynamic_dance_combo_20261005_v1')
ALIASES={'continued':'D7_reference','dynamics':'C3','joint':'C4'}
CONDITIONS=('correct','repeat','gain','shift','mute_drums','mute_vocals','donor','reconstruction')

def prepare(root=ROOT):
    root=Path(root);old=read(COMBO/'PLAN.json');donors=read(COMBO/'records/DONOR_DURATION_BINDINGS_v1.json');tracks=old['tracks'];pins={}
    for t in tracks:
        t['waveform_controls']['donor']={k:donors['tracks'][t['id']][k] for k in ('base92','music_timeline')}
        for p in [t['audio'],t['loudness_receipt'],t['base92'],t['music_timeline']]+[v[k] for v in t['waveform_controls'].values() for k in ('base92','music_timeline')]:
            pins[p]=sha(p)
            if Path(p).suffix=='.npz' and Path(p).with_suffix('.json').exists():pins[str(Path(p).with_suffix('.json'))]=sha(Path(p).with_suffix('.json'))
    pins[str(ROOT/'STUDY_PLAN.json')]=sha(ROOT/'STUDY_PLAN.json')
    plan={'id':'native_dynamics_jd_20261005_v1','schema_version':1,'tracks':tracks,'algorithms':[{'id':'C3','name':'Native dynamics losses (B)'},{'id':'C4','name':'Joint dynamics architecture (C)'}],'internal_aliases':{'D7_reference':'A matched continued training, NOT unchanged D7','C3':'B native dynamics losses','C4':'C jointly trained native planner'},'reference':'D7_reference','generation_seed':7001,'display_training_seed':0,'body_fps':30,'fps':40,'render_fps':30,'render_size':[1920,1080],'counts':old['counts'],'transfer':old['transfer'],'minimum_free_bytes':20*2**30,'product_baseline_changed':False,'files':pins,'brightness':old['brightness'],'colors':old['colors'],'composition':'Native learned30Hz trajectories. No retiming, rate floor, fragment recomposition or post-generation dynamics controller.','precision':'FP32 deterministic inference','controls':list(CONDITIONS)+['plan_shift'],'physical_criteria':'Existing fixedGATE_PARAMETERS; paired comparison plus original A reference. Missing evidence remains unqualified.','primary_criteria':'Frozen JD observational validation; exposed showcases are descriptive, not independent semantic labels.'}
    write(root/'PLAN.json',plan)
    (root/'colors').symlink_to(COMBO/'colors',target_is_directory=True)
    from ..dynamic_dance_combo.diagnostics import freeze_release_intervals
    for t in tracks:freeze_release_intervals(root,t)
    return plan


def input_arrays(track,condition):
    paths=track if condition in ('correct','repeat','plan_shift') else track['waveform_controls'][condition]
    timeline=MusicTimeline.load(paths['music_timeline']);binding=validate_audio_binding(timeline,paths['base92'])
    count=track['body_frames'];time=np.arange(count)/30
    if timeline.time[-1]<time[-1]-1/30-1e-8:raise ValueError('Audio context does not cover full song')
    extra,names=model_features(timeline,time)
    with np.load(paths['base92']) as z:
        base=z['x30'] if 'x30' in z else z['x'];bt=z['time30'] if 'time30' in z else z['time']
        if bt[-1]<time[-1]-1/30-1e-8:raise ValueError('Base92 incomplete')
        base=np.stack([np.interp(time,bt,base[:,i]) for i in range(92)],-1).astype(np.float32)
    bins=(count+14)//15;pad=bins*15-count
    coarse=np.pad(extra.astype(np.float32),((0,pad),(0,0))).reshape(bins,15,1061).sum(1)/np.minimum(15,count-np.arange(bins)*15)[:,None]
    if bins>CONTEXT_BINS:raise ValueError('Whole-song planning context overflow')
    return base,extra,coarse.astype(np.float32),{'timeline':paths['music_timeline'],'timeline_sha256':sha(paths['music_timeline']),'base92':paths['base92'],'base92_sha256':sha(paths['base92']),'binding':binding,'feature_names':names}


@torch.inference_mode()
def generate(root,arm,seed,song):
    setup();root=Path(root);plan=read(root/'PLAN.json');track=next(t for t in plan['tracks'] if t['id']==song)
    model,norms,identity=load_final(root,arm,seed);maxframes=max(t['body_frames'] for t in plan['tracks']);n=track['body_frames'];time=np.arange(n)/30
    shared_noise=cpu_noise(model.flow,maxframes,7001)[:,:n].contiguous().to('cuda')
    conditions=CONDITIONS+(('plan_shift',) if arm=='joint' else ())
    # Seed1 validates complete native generation and fixed mapping numerically;
    # all waveform interventions use predeclared displayseed0, no seedselection.
    if seed==1:conditions=('correct',)
    folder=root/'generation'/arm/f'seed_{seed}'/song;folder.mkdir(parents=True,exist_ok=False)
    for condition in conditions:
        base,extra,coarse,binding=input_arrays(track,condition)
        fine=torch.cat(((torch.from_numpy(base).to('cuda')-norms[0][0])/norms[0][1],(torch.from_numpy(extra).to('cuda')-norms[1][0])/norms[1][1]),-1)[None]
        c=torch.zeros((1,CONTEXT_BINS,1061),device='cuda');valid=torch.zeros((1,CONTEXT_BINS),device='cuda',dtype=torch.bool);bins=len(coarse)
        c[0,:bins]=(torch.from_numpy(coarse).to('cuda')-norms[1][0])/norms[1][1];valid[0,:bins]=True;indices=torch.arange(n,device='cuda')[None]
        override=None
        if condition=='plan_shift':
            p=model.planner(c,valid);override=p['conditioning'].clone();override[:,:bins]=torch.roll(override[:,:bins],bins//2,1)
        native,p=model.generate(fine,c,valid,indices,shared_noise,plan_override=override)
        world=joints_from_features(native)[0].cpu().numpy();x=native[0].cpu().numpy()
        if not np.isfinite(x).all() or not np.isfinite(world).all():raise ValueError('Nonfinite fullsong generated body')
        method=ALIASES[arm]+('' if seed==0 else '_seed1')+('' if condition=='correct' else '__'+condition)
        metadata={'arm':arm,'training_seed':seed,'draw_seed':7001,'method':method,'song':song,'condition':condition,'checkpoint':identity,'audio':binding,'noise_sha256':hashlib.sha256(shared_noise.cpu().numpy().tobytes()).hexdigest(),'fps':30.,'motion_origin':'new audio-conditioned native flow with motion-data learning','newly_generated_motion':True,'source_body_used':False,'retiming':False,'motif_recomposition':False,'rate_floor':False,'product_baseline':False,'native_time':'original 30Hz;32Euler flow steps are not musical playback time','planner_connected':arm=='joint','planner_override':condition=='plan_shift','precision':'FP32'}
        destination=body_path(root,method,song)
        DanceResult(time=time,world=world,native=x,contacts=x[:,-4:],metadata=metadata).save(destination)
        save_npz(folder/(condition+'_planner.npz'),time=(np.arange(bins)*15+7)/30,packed=p['packed'][0,:bins].cpu().numpy(),embedding=p['embedding'][0,:bins].cpu().numpy(),conditioning=(p['conditioning'] if override is None else override)[0,:bins].cpu().numpy())
        if condition=='correct':map_one(root,song,method)
        else:map_compact_control(root,track,method)
        print(arm,seed,song,condition,'generated_and_mapped',flush=True)
    write(folder/'COMPLETE.json',{'arm':arm,'seed':seed,'song':song,'conditions':list(conditions),'checkpoint':identity,'no_posthoc_motion_control':True,'body_artifacts':{str(body_path(root,ALIASES[arm]+('' if seed==0 else '_seed1')+('' if condition=='correct' else '__'+condition),song)):sha(body_path(root,ALIASES[arm]+('' if seed==0 else '_seed1')+('' if condition=='correct' else '__'+condition),song)) for condition in conditions}})


def map_compact_control(root,track,method):
    """Retain exact emitted controls without redundant per-bulb field geometry."""
    from ..dynamic_dance_combo.campaign import compile_world
    from ..dynamic_dance_combo.brightness import load_verified_loudness
    from ..dynamic_dance.retiming import resample_body
    from lightsem.conductor.body_transfer_v2 import make_outputs
    from lightsem.conductor.body_transfer import decode
    root=Path(root);p=body_path(root,method,track['id']);body=DanceResult.load(p);plan=read(root/'PLAN.json')
    time=np.arange(track['lighting_frames'])/40;world=resample_body(body.world,np.minimum(time,body.time[-1]),30.)
    with np.load(root/'colors'/track['id']/'P2.npz') as z:left=z['left'].copy();right=z['right'].copy()
    loud,_=load_verified_loudness(track['loudness_receipt'],track['audio']);outputs=make_outputs(**plan['counts'])
    universes,patch,intent,brightness,hard=compile_world(world,time,outputs,left,right,loud,plan['transfer'])
    moving=np.asarray([p['kind']!='bulb' for p in patch]);vectors=intent['targets'][:,moving]-np.asarray([p['position'] for p in patch])[moving]
    length=np.linalg.norm(vectors,axis=-1);requested=np.stack((np.rad2deg(np.arctan2(vectors[...,0],-vectors[...,1])),np.rad2deg(np.arcsin(np.clip(vectors[...,2]/length,-1,1)))),-1)
    folder=root/'control_emission'/method/track['id'];save_npz(folder/'emission.npz',time=time,universes=universes,requested=requested)
    write(folder/'patch.json',patch)
    write(folder/'REPORT.json',{'body_sha256':sha(p),'emission_sha256':sha(folder/'emission.npz'),'patch_sha256':sha(folder/'patch.json'),'hard_checks':hard,'storage':'ExactDMX andrequestedangles; redundant fieldpoints omitted, no approximation of emitted controls.','plan_sha256':sha(root/'PLAN.json')})
