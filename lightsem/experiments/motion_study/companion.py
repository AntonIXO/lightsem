"""Rhythm-led lighting, with motion as a spatial-style diagnostic only.

A companion, not six joint trackers. One body supplies slow openness, height,
lean and direction. A shared audio clock schedules independent paired light
phrases at 2/4/8 beats. The saved body is never retimed or changed. Full sequence
only: smoothing and controller history must not restart at review crops.

The wrapper preserves the incumbent non-angle DMX byte-for-byte. New direct
rhythm is therefore visible through movement, not an extra brightness pulse.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np
from .articulated_stage import infer_root_rotation, unfolded_targets
from .lighting_map import LightingConfig, compose_world_joints, ema, smoothstep
from .offline_angles import project_angles
from ..beat_conditioned.beat_conditioned_render import _angles
from ...conductor.rig import canonical_rig

@dataclass(frozen=True)
class CompanionConfig:
    style_smoothing_s: float = .65
    confidence_smoothing_s: float = .35
    gesture_gain: float = 1.
    phase_offset_beats: float = 0.

    def __post_init__(self):
        vals=asdict(self)
        if not all(np.isfinite(v) for v in vals.values()):raise ValueError('Finite companion config required')
        if self.style_smoothing_s<=0 or self.confidence_smoothing_s<=0 or not 0<=self.gesture_gain<=2:
            raise ValueError('Positive smoothing and gain in [0,2] required')


def default_lighting_config():
    return LightingConfig(target_space='pose_stage',target_smoothing_s=1e-6,
        angular_controller='offline_projection',tracking_s=.18,
        maximum_speed_deg_s=90.,maximum_acceleration_deg_s2=360.)


def spatial_style(body,fps,smoothing_s):
    """Root-local bounded body properties; no limb velocity sets light speed."""
    body=np.asarray(body,float)
    if body.ndim!=3 or body.shape[1:]!=(22,3) or len(body)<2 or not np.isfinite(body).all():
        raise ValueError('Finite complete (T>=2,22,3) world body required')
    q=infer_root_rotation(body)
    local=np.einsum('tji,tkj->tki',q,body-body[:,0,None])
    hands=local[:,[20,21]]
    openness=np.clip(np.linalg.norm(hands[:,0]-hands[:,1],axis=1)/1.5,0,1.4)
    height=np.tanh((hands[:,:,1].mean(1)-.25)/.55)
    asymmetry=np.tanh((hands[:,0,1]-hands[:,1,1])/.6)
    lean=np.clip(q[:,0,1],-.7,.7)
    facing=q[:,0,2]
    style=np.column_stack((openness,height,asymmetry,lean,facing))
    return ema(style,smoothing_s,fps)


def companion_targets(body,activity,clock,*,fps=40.,config=None,static_style=False):
    """Full surface endpoints plus interpretable, auditable control traces."""
    cfg=config or CompanionConfig();body=np.asarray(body,float)
    n=len(body);times=np.arange(n)/fps
    activity=np.asarray(activity,float)
    if activity.shape!=(n,) or not np.isfinite(activity).all() or np.any((activity<0)|(activity>1)):
        raise ValueError('Activity requires one finite [0,1] value per frame')
    beat,confidence,onset=clock.sample(times,cfg.phase_offset_beats)
    style=spatial_style(body,fps,cfg.style_smoothing_s)
    if static_style:style=np.broadcast_to(style[:1],style.shape).copy()
    opening,height,asymmetry,lean,facing=style.T
    # Quiet stays restrained even if the detector finds a pulse in ambient audio.
    drive=smoothstep((activity-.12)/.55)*ema(confidence,cfg.confidence_smoothing_s,fps)*cfg.gesture_gain
    # Paired light roles, not anatomical endpoints. Different phrase durations
    # provide counterpoint. A continuous slow exchange avoids hard pattern cuts.
    charts=[]
    for rank in range(6):
        side=-1 if rank<3 else 1;pair=min(rank,5-rank)
        shared_x=1.3*lean+.6*facing
        if pair==0:  # Broad four-beat opening + depth arc.
            phase=2*np.pi*beat/4
            x=side*(3.8+.7*opening)+shared_x+drive*side*(2.8*np.cos(phase))
            u=8.0+height+.35*side*asymmetry+drive*(2.5*np.sin(phase))
        elif pair==1:  # Two-beat floor response, offset between sides.
            phase=2*np.pi*beat/2+side*np.pi/2
            x=side*(2.9+.4*opening)+shared_x+drive*(2.7*np.sin(phase))
            u=3.3+.6*height+drive*(2.6*np.cos(phase))
        else:  # Eight-beat shared wall sweep, not a static central pair.
            phase=2*np.pi*beat/8
            x=side*(1.4+.4*opening)+shared_x+drive*(3.5*np.sin(phase))
            u=8.6+.6*height+drive*(2.1*np.cos(phase+side*np.pi/3))
        # Smooth finite-stage saturation; never target a point in mid-air.
        x=7.4*np.tanh(x/7.4)
        u=6.25+6.05*np.tanh((u-6.25)/6.05)
        charts.append(np.column_stack((x,u)))
    chart=np.stack(charts,axis=1)
    targets=unfolded_targets(chart[:,:,0],chart[:,:,1])
    return targets,{'time':times,'beat':beat,'rhythm_confidence':confidence,'rhythm_drive':drive,
        'audio_onset':onset,'spatial_style':style,'chart':chart,'raw_targets':targets}


def compose_companion_world(world,loudness,duration_s,clock,*,lighting_config=None,
                            companion_config=None,rig=None,baseline_bundle=None,static_style=False):
    """Replace only fixture angular channels; keep identical source photometry."""
    cfg=lighting_config or default_lighting_config(); cc=companion_config or CompanionConfig()
    rig=rig or canonical_rig();world=np.asarray(world)
    if cfg.target_space!='pose_stage' or cfg.angular_controller!='offline_projection':
        raise ValueError('Companion requires explicit pose_stage/offline_projection baseline')
    if abs(len(world)/30-duration_s)>1e-6:raise ValueError('Full native timeline mismatch')
    if baseline_bundle is None:
        baseline_bundle=compose_world_joints(world,loudness,duration_s,config=cfg,rig=rig)
    old,old_trace,old_report=baseline_bundle
    frames=old.copy();times=np.arange(round(duration_s*cfg.fps))/cfg.fps
    if old.shape!=(len(times),512) or old.dtype!=np.uint8:raise ValueError('Invalid baseline DMX bundle')
    native_times=np.arange(len(world))/30
    flat=world.reshape(len(world),-1)
    body=np.stack([np.interp(times,native_times,flat[:,i]) for i in range(66)],axis=1).reshape(-1,22,3)
    targets,trace=companion_targets(body,old_trace['activity'],clock,fps=cfg.fps,config=cc,static_style=static_style)
    heads=sorted(rig.by_kind('moving_head'),key=lambda f:(f.position[0],f.name))
    if len(heads)!=6:raise ValueError('Companion requires six moving heads')
    reports=[]
    for rank,f in enumerate(heads):
        pan,tilt=_angles(targets[:,rank]-np.asarray(f.position))
        ticks=project_angles(np.column_stack((pan,tilt)),[f.pan_range_deg,f.tilt_range_deg],old_trace['activity'],cfg)
        a=f.address-1
        frames[:,a:a+4]=np.stack((ticks[:,0]>>8,ticks[:,0]&255,ticks[:,1]>>8,ticks[:,1]&255),axis=1).astype(np.uint8)
        degrees=(ticks/65535-.5)*np.array([f.pan_range_deg,f.tilt_range_deg])
        reports.append({'fixture':f.name,'speed_max_deg_s':float(np.linalg.norm(np.diff(degrees,axis=0)*cfg.fps,axis=1).max()),
            'acceleration_max_deg_s2':float(np.linalg.norm(np.diff(degrees,n=2,axis=0)*cfg.fps**2,axis=1).max())})
    report={'lighting_config':asdict(cfg),'companion_config':asdict(cc),'heads':reports,
        'body_changed':False,'static_style_control':static_style,'direct_audio_rhythm':True,
        'photometry':'all incumbent non-angle bytes unchanged','controller_scope':'offline simulation90/360, not hardware certification',
        'style_channels':['openness','hand_height','asymmetry','lean','facing'],
        'scope':'Procedural companion lighting; not learned motion music conditioning'}
    trace['activity']=old_trace['activity']
    return frames,trace,report
