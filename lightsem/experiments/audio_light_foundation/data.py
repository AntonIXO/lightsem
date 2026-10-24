"""Authenticated MuQ/Base92/body/light rows for the v2 pilot."""

from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path('/mnt/lightsem-data/experiments/lightsem/body_to_dmx_foundation_20261013')
VISUAL = Path('/mnt/lightsem-data/cache/beatsaver_visual')
LIGHT = Path('/mnt/lightsem-data/cache/beatsaver_light')
AUDIO = Path('/mnt/lightsem-data/cache/beatsaver_audio')
IDS = ('1023','10dcc','1181f','11b01','1672d','210e3','21c3a','3b4ab')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()


def _one(root, suffix):
    paths=sorted(Path(root).glob(f'*={suffix}.npz'))
    if len(paths)!=1: raise FileNotFoundError(f'{root}: expected one cache for {suffix}, got {len(paths)}')
    return paths[0]


def _nearest(values, times, target):
    right=np.searchsorted(times,target,side='left').clip(0,len(times)-1)
    left=np.maximum(right-1,0)
    return values[np.where(np.abs(times[left]-target)<=np.abs(times[right]-target),left,right)]


def _motion_proxy(world, n):
    world=np.asarray(world[:n],np.float32)
    root=world[:,0:1]
    rel=world-root
    # A fixed anatomical subset is an auxiliary teacher target only. It is not
    # an authored physical pan/tilt label.
    joints=rel[:,(13,14,16,17,18,19,20,21)]
    horizontal=np.maximum(np.linalg.norm(joints[...,(0,2)],axis=-1),1e-5)
    pan=np.arctan2(joints[...,0],-joints[...,2])/np.pi
    tilt=np.arctan2(joints[...,1],horizontal)/(np.pi/2)
    angle=np.stack((pan,tilt),-1).astype(np.float32)
    # Estimate derivatives on an unwrapped angular trace.  Differentiating
    # atan2 directly turns a harmless -1↔+1 crossing into a 20-unit speed
    # spike and a ~200-unit acceleration spike, which a normalized light head
    # cannot represent.  Keep the angle itself bounded for the aim target,
    # then express speed/acceleration as normalized controls.
    unwrapped=np.unwrap(angle.astype(np.float64),axis=0,period=2.0)
    speed_raw=np.linalg.norm(np.diff(unwrapped,axis=0,prepend=unwrapped[:1]),axis=-1,keepdims=True)*10
    speed=np.clip(speed_raw/4.0,0,1)
    accel_raw=np.diff(speed_raw,axis=0,prepend=speed_raw[:1])*10
    accel=np.clip(accel_raw/20.0,-1,1)
    return rel.reshape(n,-1), angle, speed.astype(np.float32), accel.astype(np.float32)


def load_row(suffix, sample_rate_hz=10, muq_root=None):
    if sample_rate_hz not in (10, 30):
        raise ValueError('sample_rate_hz must be 10 or 30')
    body_path=ROOT/f'body_{suffix}'/'body.npz'; meta_path=ROOT/f'body_{suffix}'/'body.json'
    visual_path=_one(VISUAL,suffix); light_path=_one(LIGHT,suffix); audio_path=_one(AUDIO,suffix)
    meta=json.loads(meta_path.read_text())['metadata']
    timeline_path=Path(meta['audio']['timeline']); base92_path=Path(meta['audio']['base92'])
    if muq_root is None:
        with np.load(timeline_path,allow_pickle=False) as z:
            muq=np.asarray(z['muq'],np.float32); muq_time=np.asarray(z['muq_time'],np.float32)
        with np.load(base92_path,allow_pickle=False) as z:
            base=np.asarray(z['x30'],np.float32); base_time=np.asarray(z['time30'],np.float32)
    else:
        # Foundation-v2 canonical path: use the one-protocol MuQ receipt and
        # rebuild Base92 directly from the authenticated 40-Hz cache. This
        # avoids silently mixing the historical 8-second MuQ timeline with
        # the canonical 30/20/5-second extraction.
        canonical=Path(muq_root)/f'{suffix}.npz'
        with np.load(canonical,allow_pickle=False) as z:
            muq=np.asarray(z['muq'],np.float32); muq_time=np.asarray(z['muq_time'],np.float32)
        with np.load(audio_path,allow_pickle=False) as z:
            mel=np.asarray(z['mel'],np.float32).T
            scalar=('rms','rms_db','flux','flux_lf','flux_lmid','flux_mid','flux_hmid','flux_hf','energy_lf','energy_lmid','energy_mid','energy_hmid','energy_hf','onset','onset_ema','flux_ema','flux_lf_ema','flux_hf_ema','centroid','flatness','rolloff','percussive_ratio','pulse_clarity','beat_phase_sin','beat_phase_cos','beat_accent','bar_phase','beat_period_s')
            names=[x for x in z.files if x.startswith('ch__')]
            base=np.concatenate([mel,np.stack([z['ch__'+x] for x in scalar],1).astype(np.float32)],1)
            base_time=np.arange(len(base),dtype=np.float32)/40.0
    with np.load(body_path,allow_pickle=False) as z:
        world=np.asarray(z['world'],np.float32); body_time=np.asarray(z['time'],np.float32)
    with np.load(visual_path,allow_pickle=False) as z:
        scene_x=np.asarray(z['x'],np.float32); event=np.asarray(z['ev_any'],np.float32)
    with np.load(light_path,allow_pickle=False) as z:
        rgb=np.asarray(z['rgb'],np.float32); inten=np.asarray(z['inten'],np.float32)
    n=min(len(world)//3,len(scene_x),int(np.floor(float(muq_time[-1])*10))+1) if sample_rate_hz==10 else min(len(world),int(np.floor(float(muq_time[-1])*30))+1,len(scene_x)*3)
    target=np.arange(n,dtype=np.float32)/float(sample_rate_hz)
    muq10=_nearest(muq,muq_time,target)
    base10=_nearest(base,base_time,target)
    body10=world[::3][:n] if sample_rate_hz==10 else world[:n]
    rel,angle,speed,accel=_motion_proxy(body10,n)
    light_times=np.asarray(np.load(light_path,allow_pickle=False)['t'],np.float32)
    groups=np.arange(8)%max(1,rgb.shape[0])
    # BeatSaver light caches can contain additive RGB contributions above one
    # when several mapped channels overlap.  The foundation head emits a
    # normalized RGB control in [0, 1], so keep the target in that same
    # physical range instead of asking a sigmoid head to fit impossible values.
    if sample_rate_hz==10:
        blocks=min(n,rgb.shape[1]//4)
        rgb_rate=rgb[:,:blocks*4].reshape(rgb.shape[0],blocks,4,3).mean(2).transpose(1,0,2)
        bright_rate=inten[:,:blocks*4].reshape(inten.shape[0],blocks,4).mean(2).transpose(1,0)[...,None]
        fixture_rgb=np.clip(rgb_rate[:n,groups],0,1); fixture_brightness=np.clip(bright_rate[:n,groups],0,1)
    else:
        rgb_rate=_nearest(rgb.transpose(1,0,2),light_times,target)
        bright_rate=_nearest(inten.transpose(1,0),light_times,target)[...,None]
        fixture_rgb=np.clip(rgb_rate[:,groups],0,1); fixture_brightness=np.clip(bright_rate[:,groups],0,1)
    visual_times=np.arange(len(event),dtype=np.float32)/10.0
    event_rate=_nearest(event,visual_times,target)
    audio=np.concatenate((muq10[:n],base10[:n]),-1)
    if muq_root is not None:
        timeline_path=Path(muq_root)/f'{suffix}.npz'
        base92_path=audio_path
    return {'id':suffix,'audio':audio[:n],'body':rel[:n],'motion':angle[:n],'speed':speed[:n],
            'acceleration':accel[:n],'rgb':fixture_rgb[:n],'brightness':fixture_brightness[:n],
            'event':event_rate[:n,None].astype(np.float32),'paths':{'body':str(body_path),'body_meta':str(meta_path),
            'muq':str(timeline_path),'base92':str(base92_path),'visual':str(visual_path),'light':str(light_path),'audio_cache':str(audio_path)},
            'body_metadata':meta,'raw_world':world,'body_time':body_time}


def load_rows(ids=IDS, sample_rate_hz=10, muq_root=None):
    return [load_row(x, sample_rate_hz=sample_rate_hz, muq_root=muq_root) for x in ids]


def make_fixture_metadata(n=8):
    rows=[]
    for i in range(n):
        theta=2*np.pi*i/n
        rows.append([np.cos(theta),np.sin(theta),.5,1.,0.,0.,1.,1.])
    return np.asarray(rows,np.float32)
