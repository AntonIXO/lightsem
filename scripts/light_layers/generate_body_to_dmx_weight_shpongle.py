#!/usr/bin/env python3
"""Run the old BodyToDMX weight arm on the real full Shpongle body/audio."""
from __future__ import annotations
import argparse, hashlib, json, subprocess
import sys
from pathlib import Path
import librosa
import numpy as np
import torch

from scripts.light_layers.train_body_to_dmx import body_features
from scripts.light_layers.train_body_to_dmx_multisource_roskilde import RoskildeBodyToDMX
from lightsem.model.body_to_dmx import BodyToDMXConfig
from lightsem.conductor.body_to_dmx import scene_to_dmx
from lightsem.dmx import club_rig
from lightsem.dmx.decode import decode_stream, emitted_light
_argv, sys.argv = sys.argv, sys.argv[:1]
from scripts.data.extract_rpmc_causal import features_from_mel, SR, HOP
sys.argv = _argv
from lightsem.audio.online import causal_resample, ema


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audio_features(path: Path) -> np.ndarray:
    proc = subprocess.run(['ffmpeg','-v','error','-i',str(path),'-f','f32le','-ac','1','-ar',str(SR),'pipe:1'], check=True, stdout=subprocess.PIPE)
    y = np.frombuffer(proc.stdout, dtype=np.float32)
    mel_db = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=SR, n_fft=2048, hop_length=HOP, n_mels=128)).astype(np.float32)
    channels, grid = features_from_mel(mel_db)
    fps_in = SR / HOP
    t_out = np.arange(round(len(mel_db[0]) / fps_in * 10.0), dtype=np.float64) / 10.0
    t_in = np.arange(len(mel_db[0]), dtype=np.float64) / fps_in
    cols = []
    for name, value in channels.items():
        for tau in (0.2, 1.0, 4.0):
            col = np.asarray(causal_resample(ema(np.nan_to_num(value), fps_in, tau), t_in, t_out), np.float32).reshape(-1)
            if len(col) != len(t_out):
                col = np.interp(t_out, np.linspace(0., float(t_out[-1]), len(col)), col).astype(np.float32)
            cols.append(col)
    period = np.asarray(grid['period_s'], np.float32)
    if period.ndim == 0:
        period = np.full(len(t_out), float(period), np.float32)
    elif len(period) != len(t_out):
        period = np.interp(t_out, np.linspace(0., float(t_out[-1]), len(period)), period).astype(np.float32)
    period = np.maximum(period, 1e-4)
    phase = np.mod((t_out - float(grid['phase_s'])) / period, 1.0)
    cols.extend([np.sin(2*np.pi*phase), np.cos(2*np.pi*phase), (1-phase)**2.5, period])
    result = np.stack(cols, 1).astype(np.float32)
    if result.shape[1] != 52:
        raise ValueError(f'Expected 52-D old BodyToDMX audio, got {result.shape}')
    return result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--body',type=Path,required=True)
    ap.add_argument('--audio',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--duration',type=float,default=None)
    args=ap.parse_args()
    if args.out.exists(): raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    state=torch.load(args.run/'model.pt',map_location='cpu',weights_only=False)
    cfg=BodyToDMXConfig(**state['config'])
    model=RoskildeBodyToDMX(cfg, roskilde_audio_dim=int(state.get('roskilde_audio_dim',36)), roskilde_scene_dim=int(state.get('roskilde_scene_dim',60)))
    model.load_state_dict(state['model'],strict=True); model.eval()
    stats=state['stats']; bm,bs=torch.tensor(stats['body_mean']),torch.tensor(stats['body_std']); am,ass=torch.tensor(stats['audio_mean']),torch.tensor(stats['audio_std'])
    audio=audio_features(args.audio)
    with np.load(args.body,allow_pickle=False) as z:
        native_time=np.asarray(z['time'],float); world=np.asarray(z['world'],np.float32)
    max_s=min(float(native_time[-1]),float(len(audio)/10.0))
    if args.duration is not None: max_s=min(max_s,float(args.duration))
    n=int(max_s*10)+1
    n=min(n,len(audio))
    times=np.arange(n,dtype=np.float64)/10.0
    body, body_meta=body_features(args.body,times)
    audio=audio[:n]
    world10=np.stack([np.stack([np.interp(times,native_time,world[:,j,k]) for k in range(3)],1) for j in range(22)],1).astype(np.float32)
    with torch.no_grad():
        pred=model(((torch.tensor(body)-bm)/bs)[None],((torch.tensor(audio)-am)/ass)[None])
    scene=pred['scene'][0].numpy(); scene[:,:2]=np.tanh(scene[:,:2]); scene[:,2:]=1/(1+np.exp(-scene[:,2:]))
    event=torch.sigmoid(pred['event_logit'])[0,:,0].numpy(); angles=torch.sigmoid(pred['angle_logit'])[0].numpy().reshape(n,6,2)
    rig=club_rig(); dmx=scene_to_dmx(scene,world10,rig,event,angles=angles); decoded=decode_stream(rig,dmx); light=emitted_light(decoded)
    report={'schema_version':1,'status':'diagnostic_only','reason':'Shpongle input is out-of-domain for the BeatSaver-trained BodyToDMX contract; no physical/authored motion claim','song':'shpongle','audio':str(args.audio.resolve()),'audio_sha256':sha(args.audio),'body':str(args.body.resolve()),'body_sha256':sha(args.body),'model':str((args.run/'model.pt').resolve()),'model_sha256':sha(args.run/'model.pt'),'frames':int(n),'duration_s':float(times[-1]),'fps':10.,'finite_dmx':bool(np.isfinite(dmx).all()),'dmx_sha256':hashlib.sha256(dmx.tobytes()).hexdigest(),'brightness_mean':float(light.mean()),'inactive_fraction':float(np.mean(light.mean(1)<.02)),'event_rate':float(np.mean(event>.5)),'pan_span_deg':((decoded['pan'][:,[i for i,f in enumerate(rig.fixtures) if f.kind=='moving_head']].max(0)-decoded['pan'][:,[i for i,f in enumerate(rig.fixtures) if f.kind=='moving_head']].min(0))*540).tolist(),'tilt_span_deg':((decoded['tilt'][:,[i for i,f in enumerate(rig.fixtures) if f.kind=='moving_head']].max(0)-decoded['tilt'][:,[i for i,f in enumerate(rig.fixtures) if f.kind=='moving_head']].min(0))*270).tolist(),'raw_body':body_meta['raw_metrics']}
    np.savez_compressed(args.out/'demo.npz',time=times.astype(np.float32),world=world10,body_features=body,scene=scene,event=event,angles=angles,dmx=dmx,dimmer=decoded['dimmer'],hue=decoded['hue'],sat=decoded['sat'],value=decoded['val'])
    (args.out/'DEMO.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
