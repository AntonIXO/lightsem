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
    target_space: str = "legacy"  # surface modes remain opt-in pending visual acceptance
    quiet_dimmer: float = .12
    active_dimmer: float = .88
    loudness_range_lu: float = 12.
    target_smoothing_s: float = .55
    tracking_s: float = .35
    target_smoothing_mode: str = "causal"
    angular_feedforward: bool = False
    angular_controller: str = "chase"
    stage_base_unfolded_m: tuple = (5.7, 3.2, 5.7)
    stage_lean_gain_m: float = 0.
    surface_gain: float = 1.
    torso_gain: float = 1.
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


def smooth_targets(values,seconds,fps,mode="causal"):
    """Offline zero-phase option; no periodic clock or new target trajectory.

    Reflection padding reduces endpoint transients. Constant targets stay
    constant. This option requires the full sequence, unlike the causal EMA.
    """
    if mode == "causal":
        return ema(values,seconds,fps)
    if mode != "zero_phase":
        raise ValueError("Unknown target smoothing mode")
    x = np.asarray(values,dtype=np.float64)
    if len(x)<2 or seconds<=1e-6:
        return x.copy()
    from scipy.signal import filtfilt
    alpha = -np.expm1(-1/(seconds*fps))
    return filtfilt([alpha],[1.,alpha-1.],x,axis=0,
                   padlen=min(len(x)-1,int(np.ceil(6*seconds*fps))))


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
    target_velocity = (np.gradient(targets,1/cfg.fps,axis=0)
                       if cfg.angular_feedforward and len(targets)>1
                       else np.zeros_like(targets))
    positions = np.empty_like(target_ticks)
    positions[0] = target_ticks[0]
    velocity = np.zeros(2,np.int64)
    tick_speed = ranges/65535*cfg.fps
    neighbours = np.asarray([(i,j) for i in (-1,0,1) for j in (-1,0,1)],np.int64)
    for t in range(1,len(targets)):
        error = (target_ticks[t]-positions[t-1])*ranges/65535
        desired = error / cfg.tracking_s + target_velocity[t]
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


def surface_targets(body, fixture_x, rank, activity, cfg):
    """World motion -> bounded floor illumination, never an invisible mid-air point.

    Anatomical LEFT is +x in the neutral canonical skeleton. Negative-x
    fixtures keep anatomical right; positive-x keep left even during turns.
    No per-frame side swapping or mirroring: world articulation and root turns
    stay in world axes. Subtract the fixed neutral pose, NOT a rolling mean or
    root-rotated reference that would remove sustained gestures or turns.
    """
    from .geometry import OFFSETS, PARENTS
    neutral = np.zeros((22,3))
    for i in range(1,22):
        neutral[i] = neutral[PARENTS[i]] + OFFSETS[i]
    side = -1 if rank<3 else 1
    pair = min(rank,5-rank)
    root = body[:,0]
    slow_root = ema(root,12.,cfg.fps)
    travel = root-slow_root
    if pair==0:
        ids,weights = ([21] if side<0 else [20]),np.array([1.])
    elif pair==1:
        ids,weights = ([11,5] if side<0 else [10,4]),np.array([.65,.35])
    else:
        # Shoulder/head triangle keeps the two torso fixtures related but
        # distinct; torso yaw/lean moves this triangle, not a shared centroid.
        ids,weights = ([17,15,12] if side<0 else [16,15,12]),np.array([.5,.3,.2])
    regional = np.sum((body[:,ids]-root[:,None])*weights[None,:,None],axis=1)
    rest = np.sum(neutral[ids]*weights[:,None],axis=0)
    delta = regional-rest
    amplitude = (.25+.75*activity)*cfg.surface_gain*(cfg.torso_gain if pair==2 else 1.)
    # Fixture-scale floor lanes carry the composition. Shared travel is bounded
    # separately so accumulated native root drift cannot take over the stage.
    shared_x = .5*np.tanh(slow_root[:,0]/5)+.9*np.tanh(travel[:,0]/1.2)
    shared_z = .7*np.tanh(travel[:,2]/1.5)
    target = np.column_stack((
        .95*fixture_x+shared_x+amplitude*2.8*delta[:,0],
        np.zeros(len(body)),
        (-2.,-4.1,-2.6)[pair]+shared_z+amplitude*(2.6*delta[:,2]+2.2*delta[:,1]+.7*np.tanh((root[:,1]-.9)/.3))))
    target = np.clip(target,[-7.5,0.,-5.3],[7.5,0.,1.])
    return target,delta


def compose(features,loudness,duration_s,*,config=None,rig=None,dynamics=True,independent_features=None,world_joints_override=None):
    """Audio is the only external input; features are internally generated motion.

    `independent_features` is an explicit ablation with the same generator. The
    normal path always shares ONE body; fixture regions do not select new draws.
    `world_joints_override` is an opt-in review seam for a full (N,22,3)
    canonical body. It bypasses FK and infers root orientation from hip/spine
    anchors for every comparable world-body input. It cannot mix independent
    native draws. None preserves the original native path.
    """
    cfg,rig = config or LightingConfig(),rig or canonical_rig()
    if cfg.target_space not in ("legacy", "floor_reanchor", "articulated_surface", "articulated_stage"):
        raise ValueError("Unknown illumination target space")
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
    if world_joints_override is None:
        with torch.inference_mode():
            joints = resample(joints_from_features(torch.from_numpy(features)).numpy())
            rotations = rotation_6d_to_matrix(torch.from_numpy(features[:,3:9])).numpy()
    else:
        # Review seam: saved FULL world coordinates, never reintegrate a crop.
        # Every world-body arm uses the SAME hip/spine orientation inference.
        world = np.asarray(world_joints_override)
        if world.shape != (len(features),22,3) or not np.isfinite(world).all():
            raise ValueError('Expected finite full world joints (N,22,3)')
        if independent_features is not None:
            raise ValueError('World override cannot mix independent native bodies')
        from .articulated_stage import infer_root_rotation
        rotations = infer_root_rotation(world)
        joints = resample(world)
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
    raw_targets = np.zeros_like(targets)
    regional_trace = np.zeros((len(times),6,3))
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
        if cfg.target_space == 'floor_reanchor':
            # Mechanism ablation: old x gestures and z signals, same smoothing,
            # but no elevated-point extrapolation or compressed fixture anchors.
            target[:,0] = np.clip(target[:,0]+.5*f.position[0],-7.5,7.5)
            target[:,1] = 0.
        elif cfg.target_space == 'articulated_surface':
            target,regional_trace[:,rank] = surface_targets(body,f.position[0],rank,activity,cfg)
        elif cfg.target_space == 'articulated_stage':
            if cfg.target_smoothing_s > 1e-6:
                raise ValueError("Stage chart requires no Cartesian smoothing across the floor/wall seam")
            from .articulated_stage import stage_targets
            target,regional_trace[:,rank] = stage_targets(body,f.position[0],rank,activity,cfg)
        else:
            regional_trace[:,rank] = regional
        index = next(i for i,h in enumerate(rig.fixtures) if h.name==f.name)
        raw_targets[:,index] = target
        target = smooth_targets(target,cfg.target_smoothing_s,cfg.fps,cfg.target_smoothing_mode)
        pan,tilt = _angles(target-np.asarray(f.position))
        if cfg.angular_controller == "chase":
            controller = limited_angles
        elif cfg.angular_controller == "offline_projection":
            from .offline_angles import project_angles
            controller = project_angles
        else:
            raise ValueError("Unknown angular controller")
        ticks = controller(np.column_stack((pan,tilt)),[f.pan_range_deg,f.tilt_range_deg],activity,cfg)
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
             'measured_activity':measured_activity,'dimmer':dimmers,'targets':targets,'raw_targets':raw_targets,
             'regional':regional_trace,'palette_hue':hue,'motion_accent':accent}
    report = {'config':asdict(cfg),'shared_body':independent_features is None,'direct_audio_dynamics':dynamics,
              'loudness_anchor_lufs':anchor,'head_limits':reports,'root':'native full integration once before bounded rig mapping',
              'composition':'outer hands, middle legs, inner torso; fixed regional differences from one body',
              'geometry_anatomy':'surface mode: anatomical right to negative-x, left to positive-x, fixed identities through world turns',
              'photometry':'legacy regional brightness signal retained unchanged for matched geometry intervention',
              'ending_rule':None,'rng':None}
    if world_joints_override is not None:
        report['root'] = 'saved full world joints; no FK/root integration in mapper'
        report['orientation'] = 'same infer_root_rotation hip/spine anchors for every world-body arm'
    return frames,trace,report


def compose_world_joints(world_joints,loudness,duration_s,*,config=None,rig=None,dynamics=True):
    """Map saved complete canonical world bodies; no FK or root integration.

    Uses infer_root_rotation for all arms, not native rotation metadata. Default
    compose remains unchanged. Call on the complete recording and slice DMX later.
    """
    world = np.asarray(world_joints)
    if world.ndim != 3 or world.shape[1:] != (22,3) or len(world)<2 or not np.isfinite(world).all():
        raise ValueError('Expected finite full world joints (N>=2,22,3)')
    return compose(np.zeros((len(world),139),np.float32),loudness,duration_s,
                   config=config,rig=rig,dynamics=dynamics,world_joints_override=world)
