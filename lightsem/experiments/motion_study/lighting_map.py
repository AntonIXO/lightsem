"""One generated body, three regional fixture pairs, measured audio dynamics."""
import colorsys
from dataclasses import asdict,dataclass

import numpy as np
import torch

from .geometry import joints_from_features,rotation_6d_to_matrix
from ..beat_conditioned_render import _angles
from ...conductor.rig import canonical_rig


@dataclass(frozen=True)
class LightingConfig:
    fps: float = 40.
    quiet_dimmer: float = .12
    active_dimmer: float = .88
    loudness_range_lu: float = 12.
    target_smoothing_s: float = .55
    palette_smoothing_s: float = 10.
    maximum_speed_deg_s: float = 30.
    maximum_acceleration_deg_s2: float = 80.
    quiet_target_speed_deg_s: float = 3.
    wash_gain: float = .18
    blinder_gain: float = .06
    strobe_gain: float = .025
    projector_gain: float = .20


def ema(values,seconds,fps):
    x = np.asarray(values,dtype=np.float64)
    out = np.empty_like(x)
    out[0] = x[0]
    alpha = -np.expm1(-1/(max(seconds,1e-6)*fps))
    for i in range(1,len(x)):
        out[i] = out[i-1]+alpha*(x[i]-out[i-1])
    return out


def smoothstep(value):
    x = np.clip(value,0,1)
    return x*x*(3-2*x)


def limited_angles(targets,ranges,activity,cfg):
    """Choose integer DMX velocities subject to emitted speed AND acceleration.

    Previous velocity is always a candidate. Quantization is included in the
    constraints; rounding a continuous acceleration limit is insufficient.
    """
    targets,ranges = np.asarray(targets,float),np.asarray(ranges,float)
    target_ticks = np.rint((.5+targets/ranges)*65535).astype(np.int64)
    if np.any((target_ticks<0)|(target_ticks>65535)):
        raise ValueError('Target exceeds fixture mechanics')
    positions = np.empty_like(target_ticks)
    positions[0] = target_ticks[0]
    velocity = np.zeros(2,np.int64)
    tick_speed = ranges/65535*cfg.fps
    neighbours = np.asarray([(i,j) for i in (-1,0,1) for j in (-1,0,1)],np.int64)
    for t in range(1,len(targets)):
        error = (target_ticks[t]-positions[t-1])*ranges/65535
        desired = error/ .35
        local_limit = cfg.quiet_target_speed_deg_s+(cfg.maximum_speed_deg_s-cfg.quiet_target_speed_deg_s)*activity[t]**2
        desired *= min(1.,local_limit/max(np.linalg.norm(desired),1e-12))
        old_speed = velocity*tick_speed
        change = desired-old_speed
        proposed = old_speed+change*min(1.,cfg.maximum_acceleration_deg_s2/cfg.fps/max(np.linalg.norm(change),1e-12))
        center = np.rint(proposed/tick_speed).astype(np.int64)
        candidates = np.concatenate((center[None]+neighbours,velocity[None]),axis=0)
        speeds = candidates*tick_speed
        acceleration = np.linalg.norm((speeds-old_speed)*cfg.fps,axis=1)
        next_positions = positions[t-1]+candidates
        valid = ((np.linalg.norm(speeds,axis=1)<=cfg.maximum_speed_deg_s+1e-12)&
                 (acceleration<=cfg.maximum_acceleration_deg_s2+1e-12)&
                 ((next_positions>=0)&(next_positions<=65535)).all(1))
        if not valid.any():
            raise ValueError('No mechanically valid quantized motion step')
        costs = ((speeds-proposed)**2).sum(1)
        costs[~valid] = np.inf
        velocity = candidates[int(np.argmin(costs))]
        positions[t] = positions[t-1]+velocity
    return positions


def compose(features,loudness,duration_s,*,config=None,rig=None,dynamics=True,independent_features=None):
    """Audio is the only external input; features are internally generated motion.

    `independent_features` is an explicit ablation with the same generator. The
    normal path always shares ONE body; fixture regions do not select new draws.
    """
    cfg,rig = config or LightingConfig(),rig or canonical_rig()
    features = np.asarray(features,np.float32)
    if features.ndim!=2 or features.shape[1]!=139 or not np.isfinite(features).all():
        raise ValueError('Expected one finite native motion sequence')
    if abs(len(features)/30-duration_s)>1/30+1e-6:
        raise ValueError('Motion and audio duration differ')
    times = np.arange(round(duration_s*cfg.fps))/cfg.fps
    native_times = np.arange(len(features))/30
    def resample(value):
        shape = value.shape[1:]
        flat = value.reshape(len(value),-1)
        return np.stack([np.interp(times,native_times,flat[:,i]) for i in range(flat.shape[1])],axis=1).reshape(len(times),*shape)
    with torch.inference_mode():
        joints = resample(joints_from_features(torch.from_numpy(features)).numpy())
        rotations = rotation_6d_to_matrix(torch.from_numpy(features[:,3:9])).numpy()
    if independent_features is not None:
        extra = np.asarray(independent_features,np.float32)
        if extra.shape!=(6,*features.shape) or not np.isfinite(extra).all():
            raise ValueError('Independent ablation requires exactly six full generated sequences')
        with torch.inference_mode():
            alternatives = [resample(joints_from_features(torch.from_numpy(v)).numpy()) for v in extra]
    else:
        alternatives = [joints]*6
    momentary = np.interp(times,loudness['time'],loudness['momentary'])
    short = np.interp(times,loudness['time'],loudness['short_term'])
    valid = np.asarray(loudness['short_term'])> -70
    anchor = float(np.quantile(np.asarray(loudness['short_term'])[valid],.90)) if valid.any() else -23.
    raw_level = smoothstep((.7*momentary+.3*short-(anchor-cfg.loudness_range_lu))/cfg.loudness_range_lu)
    measured_activity = ema(raw_level,.30,cfg.fps)
    # The ablation holds dynamics constant; it does not read target/source motion.
    activity = measured_activity if dynamics else np.full(len(times),.7)
    level = cfg.quiet_dimmer+(cfg.active_dimmer-cfg.quiet_dimmer)*activity**1.35
    palette_level = ema(measured_activity,cfg.palette_smoothing_s,cfg.fps) if dynamics else np.full(len(times),.7)
    orientation = resample(rotations[:,[0,2],2])
    orientation = ema(orientation,3.,cfg.fps)
    hue = .60+.16*palette_level+.035*np.tanh(orientation[:,0])
    heads = sorted(rig.by_kind('moving_head'),key=lambda f:(f.position[0],f.name))
    if len(heads)!=6:
        raise ValueError('This canonical mapping uses three fixture pairs')
    frames = np.zeros((len(times),512),np.uint8)
    targets = np.zeros((len(times),len(rig.fixtures),3))
    dimmers = np.zeros((len(times),len(rig.fixtures)))
    motion_speed = np.linalg.norm(np.diff(joints,axis=0,prepend=joints[:1])*cfg.fps,axis=-1)
    accent = ema(np.clip((motion_speed[:,[20,21,10,11]].mean(1)-.35)/2.,0,1),.15,cfg.fps)
    reports = []
    for rank,f in enumerate(heads):
        side = -1 if rank<3 else 1
        pair = min(rank,5-rank)
        body = alternatives[rank]
        root = body[:,0]
        slow_root = ema(root,12.,cfg.fps)
        travel = root-slow_root
        if pair==0:
            regional = body[:,20 if side<0 else 21]-root
        elif pair==1:
            regional = .65*(body[:,10 if side<0 else 11]-root)+.35*(body[:,4 if side<0 else 5]-root)
        else:
            regional = .65*(body[:,12]-root)+.35*(body[:,15]-root)
            regional[:,0] += side*.12
        amplitude = .25+.75*activity
        target = np.column_stack((
            .45*f.position[0]+1.6*np.tanh(slow_root[:,0]/5)+1.8*np.tanh(travel[:,0]/1.2)+2.4*amplitude*np.tanh(regional[:,0]/.55),
            1.65+amplitude*(1.15*np.tanh(regional[:,1]/.6)+.45*np.tanh((root[:,1]-.9)/.3)),
            -3.6+1.3*np.tanh(travel[:,2]/1.5)+.8*amplitude*np.tanh(regional[:,2]/.5)))
        target = np.clip(target,[-5.4,.45,-5.5],[5.4,3.6,-1.4])
        target = ema(target,cfg.target_smoothing_s,cfg.fps)
        pan,tilt = _angles(target-np.asarray(f.position))
        ticks = limited_angles(np.column_stack((pan,tilt)),[f.pan_range_deg,f.tilt_range_deg],activity,cfg)
        index = next(i for i,h in enumerate(rig.fixtures) if h.name==f.name)
        targets[:,index] = target
        regional_accent = .025*accent*activity
        brightness = np.clip(level*(.94+.06*side*np.tanh(regional[:,0]))+regional_accent,0,.95)
        dimmers[:,index] = brightness
        rgb = np.asarray([colorsys.hsv_to_rgb(float(v)%1,.86,1.) for v in hue+(pair-1)*.045])
        a = f.address-1
        frames[:,a] = (ticks[:,0]>>8).astype(np.uint8)
        frames[:,a+1] = (ticks[:,0]&255).astype(np.uint8)
        frames[:,a+2] = (ticks[:,1]>>8).astype(np.uint8)
        frames[:,a+3] = (ticks[:,1]&255).astype(np.uint8)
        frames[:,a+4] = np.rint(brightness*255).astype(np.uint8)
        frames[:,a+6:a+9] = np.rint(rgb*255).astype(np.uint8)
        frames[:,a+9] = np.rint((.22+.12*palette_level)*255).astype(np.uint8)
        angle = (ticks/65535-.5)*np.asarray([f.pan_range_deg,f.tilt_range_deg])
        velocity = np.diff(angle,axis=0)*cfg.fps
        acceleration = np.diff(velocity,axis=0)*cfg.fps
        reports.append({'fixture':f.name,'region':('hands','legs','torso')[pair],
            'speed_max_deg_s':float(np.linalg.norm(velocity,axis=1).max(initial=0)),
            'acceleration_max_deg_s2':float(np.linalg.norm(acceleration,axis=1).max(initial=0))})
    for i,f in enumerate(rig.fixtures):
        if f.kind=='moving_head':
            continue
        a = f.address-1
        gain = {'wash':cfg.wash_gain,'blinder':cfg.blinder_gain,'strobe':cfg.strobe_gain,'projector':cfg.projector_gain}[f.kind]
        brightness = gain*(level if f.kind in ('wash','projector') else activity*accent)
        dimmers[:,i] = brightness
        frames[:,a] = np.rint(brightness*255).astype(np.uint8)
        if f.kind in ('wash','projector'):
            rgb = np.asarray([colorsys.hsv_to_rgb(float(v+.08)%1,.80,1.) for v in hue])
            frames[:,a+2:a+5] = np.rint(rgb*255).astype(np.uint8)
            frames[:,a+5] = 90
            if f.kind=='projector':
                frames[:,a+8:a+10] = 128
    trace = {'time':times,'momentary_lufs':momentary,'short_term_lufs':short,'activity':activity,
             'measured_activity':measured_activity,'dimmer':dimmers,'targets':targets,'palette_hue':hue,'motion_accent':accent}
    report = {'config':asdict(cfg),'shared_body':independent_features is None,'direct_audio_dynamics':dynamics,
              'loudness_anchor_lufs':anchor,'head_limits':reports,'root':'native full integration once before bounded rig mapping',
              'composition':'outer hands, middle legs, inner torso; fixed regional differences from one body',
              'ending_rule':None,'rng':None}
    return frames,trace,report
