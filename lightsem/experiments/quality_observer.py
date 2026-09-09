"""Observe emitted motion separately from light level and beat repetition.

Diagnostic measurements, not an aesthetic score. Thresholds belong to a
separately frozen calibration/protocol, and must reject known failure controls.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np


def beam_series(frames, rig):
    d = np.asarray(frames)
    directions, levels = [], []
    for f in rig.fixtures:
        if f.kind != 'moving_head':
            continue
        a = f.address-1
        pan = (d[:, a].astype(float)*256+d[:, a+1])/65535
        tilt = (d[:, a+2].astype(float)*256+d[:, a+3])/65535
        p = np.deg2rad((pan-.5)*f.pan_range_deg)
        t = np.deg2rad((tilt-.5)*f.tilt_range_deg)
        directions.append(np.stack((np.sin(p)*np.cos(t), -np.cos(p)*np.cos(t), np.sin(t)), axis=1))
        # RGB-weighted dimmer: invisible/black commands cannot count as motion.
        rgb = d[:, a+6:a+9].astype(float)/255
        levels.append(d[:, a+4].astype(float)/255*(rgb@np.array([.2126,.7152,.0722])))
    v = np.stack(directions, axis=1)
    w = np.stack(levels, axis=1)
    return v, w


def beam_observation(frames, rig, fps=40.):
    """Angular speed AND two-second spatial span distinguish travel/jitter."""
    v,w = beam_series(frames, rig)
    angular = np.rad2deg(np.arccos(np.clip(np.sum(v[1:]*v[:-1], axis=2), -1, 1)))*fps
    weight = np.minimum(w[1:], w[:-1])
    lit = weight > .03
    angular[angular < 1e-4] = 0.
    total = float(w.sum())
    spans=[]
    for start in range(0,len(v)-int(2*fps)+1,int(fps)):
        chunk=v[start:start+int(2*fps)]
        center=chunk.mean(0);center/=np.maximum(np.linalg.norm(center,axis=1,keepdims=True),1e-9)
        radius=np.rad2deg(np.arccos(np.clip(np.sum(chunk*center[None],axis=2),-1,1)))
        spans.extend((2*np.quantile(radius,.9,axis=0)).tolist())
    power=np.square(np.abs(np.fft.rfft(v-v.mean(0),axis=0)))
    frequency=np.fft.rfftfreq(len(v),d=1/fps)
    hf=float(power[frequency>4].sum()/max(float(power.sum()),1e-12))
    return {'span_2s_median_deg':float(np.median(spans)) if spans else 0.,
            'direction_hf_above4hz_fraction':hf,
            'direction_speed_median_deg_s': float(np.median(angular[lit])) if lit.any() else None,
            'direction_speed_p95_deg_s': float(np.quantile(angular[lit], .95)) if lit.any() else None,
            'lit_motion_fraction': float(np.mean(angular[lit] > 3)) if lit.any() else 0.,
            'lit_fraction': float(lit.mean()),
            'upward_light_fraction': float(np.sum(w*(v[:,:,1] > 0))/total) if total > 0 else None,
            'brightness_weighted_direction_speed': float(np.sum(angular*weight)/weight.sum()) if weight.sum() > 0 else 0.,
            'head_brightness_mean': float(w.mean())}


def phase_explanation(values, beat_sin, beat_cos, bar_phase, fps=40.):
    """Blocked held-out R² of a low-complexity beat/bar Fourier template.

    High R² alone is not collapse: it must be paired with counterfactual
    insensitivity to non-beat audio and/or missing spatial/structural motion.
    """
    y = np.asarray(values, float)
    if y.ndim == 1:
        y = y[:, None]
    beat = np.arctan2(beat_sin, beat_cos)
    bar = 2*np.pi*np.asarray(bar_phase)
    x = np.column_stack([np.ones(len(y))]+[fn(k*p) for p in (beat,bar) for k in range(1,5) for fn in (np.sin,np.cos)])
    if len(x) != len(y) or not np.isfinite(y).all():
        raise ValueError('Aligned finite series required')
    block = max(1, int(round(10*fps)))
    fold = (np.arange(len(y))//block)%2
    residual, variance = 0., 0.
    for i in (0,1):
        train, test = fold != i, fold == i
        if train.sum() < x.shape[1] or test.sum() < 2:
            return {'r2': None, 'reason':'too_short_for_blocked_fit'}
        a, b = x[train], y[train]
        reg = np.eye(x.shape[1])*1e-3;reg[0,0]=0
        coef = np.linalg.solve(a.T@a+reg, a.T@b)
        residual += float(np.square(y[test]-x[test]@coef).sum())
        variance += float(np.square(y[test]-y[test].mean(0)).sum())
    return {'r2': float(1-residual/variance) if variance > 1e-12 else None,
            'constant': variance <= 1e-12, 'harmonics':4,'heldout_block_seconds':10.}


def _photo_normalized(frame):
    x = np.asarray(frame, np.float32)/255
    # Fixed dark floor prevents normalizing codec noise in unlit scenes.
    scale = max(float(np.quantile(x, .99)), .08)
    return np.clip(x/scale, 0, 1)


def image_pair_motion(previous, current):
    """Geometry change after global brightness normalization, with support."""
    a, b = _photo_normalized(previous), _photo_normalized(current)
    support = ((a > .15) | (b > .15)) & ((previous > 6) | (current > 6))
    if not support.any():
        return {'flow_p75_px':0.,'moving_support_fraction':0.,'shape_change':0.,'support_fraction':0.}
    flow = cv2.calcOpticalFlowFarneback((a*255).astype('uint8'), (b*255).astype('uint8'),
        None, .5, 3, 21, 3, 5, 1.2, 0)
    magnitude = np.linalg.norm(flow, axis=2)
    return {'flow_p75_px':float(np.quantile(magnitude[support],.75)),
            'moving_support_fraction':float(np.mean(magnitude[support]>.3)),
            'shape_change':float(np.mean(np.abs(a-b)[support])),
            'support_fraction':float(support.mean())}


def video_observation(path, fps=10., width=320, height=180):
    """Decode existing video only. No model inference or raster rendering."""
    command=['ffmpeg','-v','error','-threads','2','-i',str(Path(path)), '-vf',
             f'fps={fps},scale={width}:{height}:flags=area,format=rgb24',
             '-f','rawvideo','-pix_fmt','rgb24','pipe:1']
    process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    frame_bytes=width*height*3;previous=None;rows=[];levels=[]
    while True:
        raw=process.stdout.read(frame_bytes)
        if not raw:break
        if len(raw)!=frame_bytes:raise RuntimeError('Partial decoded frame')
        rgb=np.frombuffer(raw,np.uint8).reshape(height,width,3)
        # RGB conversion expands limited-range YUV correctly. Reading raw
        # gray Y would leave black at16 and mistake the background for light.
        frame=np.rint(rgb@np.array([.2126,.7152,.0722])).astype(np.uint8)
        levels.append(float(frame.mean()/255))
        if previous is not None:rows.append(image_pair_motion(previous,frame))
        previous=frame.copy()
    error=process.stderr.read().decode();code=process.wait()
    process.stdout.close();process.stderr.close()
    if code or not rows:raise RuntimeError('Video decode failed: '+error)
    summary={k:float(np.median([r[k] for r in rows])) for k in rows[0]}
    summary.update(frames=len(levels),fps=fps,mean_brightness=float(np.mean(levels)),
                   brightness_std=float(np.std(levels)),
                   visible_motion_frame_fraction=float(np.mean([r['flow_p75_px']>.3 and r['shape_change']>.025 for r in rows])))
    return {'summary':summary,'frame_metrics':rows,'brightness':levels}


def failure_flags(beam, screen, phase_r2, nonbeat_response, *, calibration):
    """Independent failure axes; never merge them into a flattering score."""
    s=screen.get('summary',screen)
    return {'motion_collapse':bool(beam['lit_motion_fraction']<calibration['lit_motion_min']),
            'movement_not_visible':bool(s['visible_motion_frame_fraction']<calibration['visible_motion_min']),
            'off_stage_motion':bool(beam['upward_light_fraction'] is not None and beam['upward_light_fraction']>calibration['upward_max']),
            'beat_shortcut':bool(phase_r2 is not None and nonbeat_response is not None and phase_r2>calibration['phase_r2_max'] and nonbeat_response<calibration['nonbeat_response_min']),
            'weak_audio_control':bool(nonbeat_response is not None and nonbeat_response<calibration['nonbeat_response_min'])}
