"""Engineering calibration on fixed TRAIN controls, independent of candidate weights."""
from pathlib import Path
from dataclasses import asdict,replace
import json
import subprocess
import wave
import time
import numpy as np
from .night_common import ROOT,BASE,initial_setup
from .night_render import ExpressionConfig,compose_scene,isolate_role,render_video,composition_observation
from .beat_conditioned_render import controls_to_dmx as old_map
from .quality_observer import beam_observation,video_observation
from .stage11_common import authenticated_json,read_authenticated,digest,artifact_manifest,gpu_lock
from lightsem.conductor.rig import canonical_rig

def screen_statistics(path,start_s=0.,duration_s=None):
    command=['ffmpeg','-v','error','-ss',str(start_s),'-i',str(path)]
    if duration_s is not None:command+=['-t',str(duration_s)]
    command+=['-vf','fps=2,scale=214:120','-f','rawvideo','-pix_fmt','rgb24','pipe:1']
    raw=subprocess.check_output(command)
    if not raw:raise ValueError('Empty video decode')
    x=np.frombuffer(raw,np.uint8).reshape(-1,120,214,3).astype(np.float32)/255
    luma=x@np.array([.2126,.7152,.0722]);border=np.concatenate((luma[:,:,:53],luma[:,:,161:]),axis=2)
    return {'frames':len(x),'mean_luma':float(luma.mean()),'border_luma_mean':float(border.mean()),
            'border_luma_p95':float(np.quantile(border.mean((1,2)),.95)),
            'bright_area_mean':float((luma>.4).mean()),'clipped_white_fraction':float((x.min(-1)>.95).mean())}

def silence(path,seconds=20):
    path=Path(path)
    if not path.exists():
        with wave.open(str(path),'wb') as w:w.setnchannels(2);w.setsampwidth(2);w.setframerate(48000);w.writeframes(np.zeros((seconds*48000,2),np.int16).tobytes())

def calibrate(root=ROOT):
    root=Path(root);m,n,_=initial_setup(root);folder=root/'calibration';silence(folder/'silence.wav')
    # Three predeclared canonical TRAIN songs; other TRAIN trajectories only test rig capacity.
    unique={}
    for e in sorted(m['entries'],key=lambda e:(e['canonical_id'],e['key'])):
        if e['partition']=='train' and e['n_frames']>=3200:unique.setdefault(e['canonical_id'],e)
    entries=list(unique.values());selected=[entries[i] for i in (0,len(entries)//3,2*len(entries)//3)]
    cfg=ExpressionConfig();report=[];begun=time.monotonic()
    def clip(e):
        with np.load(e['path']) as z:
            start=(len(z['y'])-800)//4
            return (z['y'][start:start+800]-n['y_center'])/n['y_scale']
    for i,e in enumerate(selected):
        z=clip(e);group=np.stack([z]+[clip(entries[(i*17+j*37)%len(entries)]) for j in range(1,6)])
        modes={'old':old_map(z,n['target_names'],20.)[:2]}
        for mode in ('auto','mirror','independent'):
            d,t,_=compose_scene(group,n['target_names'],20.,config=replace(cfg,composition=mode));modes[mode]=(d,t)
        row={'key':e['key'],'source_sha256':e['sha256'],'modes':{}}
        for mode,(d,t) in modes.items():
            if mode!='old':np.savez_compressed(folder/(e['key']+'_'+mode+'.npz'),dmx=d,**t)
            row['modes'][mode]={}
            for role in ('full','moving_head'):
                frames=d if role=='full' else isolate_role(d,role)
                path=folder/(e['key']+'_'+mode+'_'+role+'.mp4')
                render=render_video(frames,folder/'silence.wav',path,20.)
                row['modes'][mode][role]={'render':render,'screen':screen_statistics(path),'beam':beam_observation(frames,canonical_rig()),
                                        'video':str(path),'sha256':digest(path)}
            if mode!='old':row['modes'][mode]['composition']=composition_observation(d,t)
            print(json.dumps({'stage':'calibration','key':e['key'],'mode':mode,'metrics':row['modes'][mode].get('composition')}),flush=True)
        report.append(row)
    result={'engineering_only':True,'candidate_outputs_used':False,'configuration':asdict(cfg),'rows':report,
            'group_controls':'Distinct real TRAIN motions for geometry/brightness capacity, not proof of music-conditioned groups',
            'elapsed_s':time.monotonic()-begun,'source_sha256':digest(__file__)}
    authenticated_json(folder/'report.json',result)
    authenticated_json(root/'expression_config.json',asdict(cfg))
    return result

if __name__=='__main__':
    with gpu_lock(ROOT):calibrate()
