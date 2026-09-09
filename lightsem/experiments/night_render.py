"""Music-conditioned group composition and expressive brightness, without clocks."""
from dataclasses import dataclass, asdict
import numpy as np
from scipy.ndimage import uniform_filter1d
from .beat_conditioned_render import controls_to_dmx as old_map, _angles, _limited_ticks
from .night_common import base_context
from lightsem.conductor.rig import canonical_rig

@dataclass(frozen=True)
class ExpressionConfig:
    composition: str = 'auto'
    mix_window_frames: int = 41
    activity_low_z: float = -.45
    activity_high_z: float = .70
    dimmer_power: float = 1.25
    event_mix: float = .20
    wash_gain: float = .60
    blinder_gain: float = .25
    strobe_gain: float = .25
    projector_gain: float = .70
    pan_speed_deg_s: float = 180.
    tilt_speed_deg_s: float = 120.

def _smoothstep(x):
    x=np.clip(x,0,1);return x*x*(3-2*x)

def group_targets(controls, names, rig, cfg):
    """Mirror targets around world x=0; stable fixture identity selects style draws."""
    g=np.asarray(controls,float)
    if g.ndim!=3 or g.shape[2]!=len(names) or not np.isfinite(g).all():raise ValueError('Expected (G,T,C)')
    names=list(names);heads=sorted(rig.by_kind('moving_head'),key=lambda f:(f.position[0],f.name))
    if len(heads)%2:raise ValueError('This version requires an even number of heads')
    if cfg.composition not in ('auto','mirror','independent'):raise ValueError('Unknown composition')
    symmetry=g[:,:,names.index('symmetry')].mean(0)
    # This is a policy on predicted side-activity balance, not a geometric-pose label.
    mixture=_smoothstep(.5+.6*uniform_filter1d(symmetry,cfg.mix_window_frames,mode='nearest'))
    if cfg.composition=='mirror':mixture[:]=1
    if cfg.composition=='independent':mixture[:]=0
    nf=len(rig.fixtures);out=np.zeros((g.shape[1],nf,3));styles=np.zeros(nf,int)
    targets={}
    def target(f,z):
        lateral=np.tanh(1.2*z[:,names.index('lateral_pos')]);bounce=np.tanh(1.2*z[:,names.index('vert_bounce')])
        spread=.5+.5*np.tanh(z[:,names.index('spread')]);hand=.5+.5*np.tanh(z[:,names.index('hand_spread')])
        side=f.position[0]/max(1.,max(abs(h.position[0]) for h in heads))
        return np.column_stack((np.clip(.75*f.position[0]+2.4*lateral+.8*side*(2*hand-1),-5.4,5.4),
                                1.7+1.4*bounce,-3.2+1.5*bounce))
    for pair in range(len(heads)//2):
        left,right=heads[pair],heads[-pair-1]
        li=(2*pair)%len(g);ri=(2*pair+1)%len(g)
        lt=target(left,g[li]);rt=target(right,g[ri]);mirror=lt.copy();mirror[:,0]*=-1
        targets[left.name]=lt;targets[right.name]=mixture[:,None]*mirror+(1-mixture[:,None])*rt
        for f,style in ((left,li),(right,ri)):
            ix=next(i for i,x in enumerate(rig.fixtures) if x.name==f.name);styles[ix]=style;out[:,ix]=targets[f.name]
    return out,mixture,styles

def compose_scene(group_controls, target_names, duration_s, *, event_scores=None, rig=None, config=None):
    cfg=config or ExpressionConfig();rig=rig or canonical_rig();names=list(target_names)
    g=np.asarray(group_controls,float)
    if g.ndim==2:g=g[None]
    if g.shape[1]!=round(duration_s*40):raise ValueError('Control duration mismatch')
    shared=g.mean(0)
    frames,old_trace,old_report=old_map(shared,names,duration_s,rig=rig,event_scores=event_scores)
    targets,mixture,styles=group_targets(g,names,rig,cfg)
    dimmers=np.asarray(old_trace['dimmer']).copy();events=np.zeros((len(shared),3)) if event_scores is None else np.asarray(event_scores,float)
    if events.shape!=(len(shared),3) or not np.isfinite(events).all() or np.any((events<0)|(events>1)):
        raise ValueError('Invalid event scores')
    accent=np.maximum(2*events-1,0).max(1);panels=[]
    def brightness(f,style):
        z=g[style]
        activity=.65*z[:,names.index('qom')]+.35*z[:,names.index('weight')]
        level=_smoothstep((activity-cfg.activity_low_z)/(cfg.activity_high_z-cfg.activity_low_z))**cfg.dimmer_power
        balance=.75+.25*np.sign(f.position[0])*np.tanh(z[:,names.index('laterality')])
        return np.clip((1-cfg.event_mix)*level*balance+cfg.event_mix*accent*balance,0,1)
    for i,f in enumerate(rig.fixtures):
        a=f.address-1
        if f.kind=='moving_head':
            dimmers[:,i]=brightness(f,styles[i])
            # In mirror mode, brightness is mirrored as well as geometry.
            partner=min(rig.by_kind('moving_head'),key=lambda h:abs(h.position[0]+f.position[0]))
            if f.position[0]>0:
                j=next(j for j,h in enumerate(rig.fixtures) if h.name==partner.name)
                dimmers[:,i]=mixture*brightness(partner,styles[j])+(1-mixture)*dimmers[:,i]
            pan,tilt=_angles(targets[:,i]-np.asarray(f.position))
            pt=_limited_ticks(pan,f.pan_range_deg,cfg.pan_speed_deg_s,40.)
            tt=_limited_ticks(tilt,f.tilt_range_deg,cfg.tilt_speed_deg_s,40.)
            frames[:,a]=(pt>>8).astype(np.uint8);frames[:,a+1]=(pt&255).astype(np.uint8)
            frames[:,a+2]=(tt>>8).astype(np.uint8);frames[:,a+3]=(tt&255).astype(np.uint8)
            frames[:,a+4]=np.rint(dimmers[:,i]*255).astype(np.uint8)
            panels.append({'name':f.name,'style_draw':int(styles[i]),'pan_max_deg_s':float(abs(np.diff(pt)).max(initial=0)*f.pan_range_deg*40/65535),
                           'tilt_max_deg_s':float(abs(np.diff(tt)).max(initial=0)*f.tilt_range_deg*40/65535)})
        else:
            gain={'wash':cfg.wash_gain,'blinder':cfg.blinder_gain,'strobe':cfg.strobe_gain,'projector':cfg.projector_gain}[f.kind]
            dimmers[:,i]*=gain;frames[:,a]=np.rint(np.clip(dimmers[:,i],0,1)*255).astype(np.uint8)
    trace={'dimmer':dimmers,'targets':targets,'symmetry_mix':mixture,'style_index':styles,
           'shared_controls':shared}
    report={'mapping':'group_expression_v4','config':asdict(cfg),'heads':panels,
            'independent_temporal_generators':False,'trained_fixture_head':False,
            'input_channels_used':sorted(set(old_report['input_channels_used']+['symmetry'])),
            'selection':'predicted side-activity balance, independent conditional style draws, fixed rig geometry',
            'brightness':'TRAIN-standardized activity plus learned events; no constant dimmer floor or per-track quota',
            'exposure':'fixed role gains; no song-position or ending input'}
    return frames,trace,report

def isolate_role(frames,kind,rig=None):
    rig=rig or canonical_rig();out=np.asarray(frames).copy()
    for f in rig.fixtures:
        if f.kind!=kind:out[:,f.address-1:f.address-1+f.n_channels]=0
    return out

def composition_observation(frames,trace,rig=None):
    rig=rig or canonical_rig(); heads=sorted([i for i,f in enumerate(rig.fixtures) if f.kind=='moving_head'],key=lambda i:(rig.fixtures[i].position[0],rig.fixtures[i].name))
    dim=np.asarray(trace['dimmer'])[:,heads]
    spans=np.quantile(dim,.95,axis=0)-np.quantile(dim,.05,axis=0)
    vs=[]
    for i in heads:
        f=rig.fixtures[i];a=f.address-1
        p=(frames[:,a].astype(float)*256+frames[:,a+1])*f.pan_range_deg/65535
        t=(frames[:,a+2].astype(float)*256+frames[:,a+3])*f.tilt_range_deg/65535
        vs.append(np.column_stack((np.diff(p),np.diff(t))).reshape(-1))
    corr=np.corrcoef(vs)
    # Exclude intended mirror partners from the unrelated-pair statistic.
    vals=[abs(corr[i,j]) for i in range(len(heads)) for j in range(i+1,len(heads)) if j!=len(heads)-i-1]
    return {'head_dimmer_p95_p05':spans.tolist(),'heads_with_span_ge_point4':int((spans>=.4).sum()),
            'head_off_fraction':(dim<=1/255).mean(0).tolist(),'head_on_fraction':(dim>.05).mean(0).tolist(),
            'unrelated_velocity_abs_correlation':float(np.nanmedian(vals)) if np.isfinite(vals).any() else None,
            'symmetry_mix_quantiles':np.quantile(trace['symmetry_mix'],[0,.05,.5,.95,1]).tolist()}

def render_video(frames,audio_path,video_path,duration_s,*,rig=None,width=854,height=480):
    """GPU raster and AV1 NVENC, static camera/shader, audio preserved."""
    import math,subprocess,time
    from pathlib import Path
    from lightsem.sim.renderer import StageRenderer,RenderConfig
    frames=np.asarray(frames);path=Path(video_path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(path)
    if frames.dtype!=np.uint8 or frames.ndim!=2 or frames.shape[1]!=512:raise ValueError('Expected DMX')
    cfg=RenderConfig(width=width,height=height,fps=30.,haze=.25,cam_orbit=0.,intensity_gain=2.4)
    renderer=StageRenderer(rig or canonical_rig(),cfg);name=renderer.renderer_name
    if 'nvidia' not in name.lower():raise RuntimeError('NVIDIA renderer required')
    command=['ffmpeg','-v','error','-nostdin','-n','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}',
             '-r','30','-i','pipe:0','-i',str(audio_path),'-map','0:v:0','-map','1:a:0',
             '-c:v','av1_nvenc','-preset','p5','-cq','28','-b:v','0','-pix_fmt','yuv420p',
             '-c:a','aac','-b:a','192k','-t',str(duration_s),'-shortest',str(path)]
    process=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=subprocess.PIPE);begun=time.monotonic()
    try:
        for i in range(math.ceil(duration_s*30)):
            frame=renderer.render_frame(frames[min(round(i*40/30),len(frames)-1)],0.)
            process.stdin.write(frame.tobytes())
    finally:
        process.stdin.close();error=process.stderr.read().decode(errors='replace');rc=process.wait()
        process.stderr.close();renderer.ctx.release()
    if rc:raise RuntimeError(error)
    return {'renderer':name,'config':asdict(cfg),'codec':'av1_nvenc','shader_time':0.,'camera_orbit':0.,
            'gpu_task_seconds':time.monotonic()-begun,'argv':command}
