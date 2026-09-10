"""Frozen diagnostic delivery; failed music qualification cannot be bypassed."""
import argparse
import json
import sys
from pathlib import Path

from .registration import PROBLEM
from ..night_common import BASE,ROOT as NIGHT,STAGE
from ..stage11_common import authenticated_json,read_authenticated,digest,freeze_sources,verify_sources,verify_artifacts,gpu_lock

ROOT=Path('/mnt/lightsem-data/experiments/lightsem/motion_native_audio_20260910')
OUT=ROOT/'delivery/final_diagnostic'


def freeze():
    if (OUT/'protocol.json').exists():return read_authenticated(OUT/'protocol.json')
    from ..stage11_data import load_manifest
    records={r['recording_id']:r for r in load_manifest(STAGE)['recordings']}
    old=read_authenticated(NIGHT/'delivery/pose/protocol.json')
    demo=read_authenticated(ROOT/'data/shpongle/input.json')
    entries=[{'key':'shpongle','path':demo['path'],'sha256':demo['sha256'],'fps':30,
              'duration':demo['source']['source_duration_s'],'audio_path':demo['audio_path'],'audio_offset':0.,'review_start':0.,'review_duration':demo['source']['source_duration_s']}]
    for row in old['selected']:
        e=row['source']['entry'];r=records[e['recording_id']]
        entries.append({'key':e['key'],'path':e['path'],'sha256':e['sha256'],'fps':40,
            'duration':e['duration_s'],'audio_path':r['wav_path'],'audio_offset':e['offset_s'],
            'source_video':r['source_path'],'review_start':row['source']['start_frame']/40,'review_duration':20.})
    files=[ROOT/'music/protocol.json',ROOT/'music/selection_seed_0.json',ROOT/'exposure_roles/report.json',
           ROOT/'data/audio_normalizers.json',ROOT/'data/motion_normalizers.json',ROOT/'data/shpongle/loudness.npz',
           BASE/'run/final.pt',BASE/'normalizers.json',NIGHT/'diagnostics/pose_probes/descriptor256/seed_0/final.pt',
           NIGHT/'delivery/shpongle/full.npz',ROOT/'data/shpongle/input.json',Path(demo['source']['features_path'])]
    files += [ROOT/'music/runs'/a/'seed_0/final.pt' for a in ('regression','flow')]
    for e in entries:
        if digest(e['path'])!=e['sha256']:raise ValueError('Input drift')
        files.extend(Path(e[k]) for k in ('path','audio_path','source_video') if k in e)
    # The model and native geometry are identical to the failed frozen study.
    for name in ('music_model.py','native_codec.py','geometry.py'):
        if digest(Path(__file__).with_name(name))!=digest(ROOT/'music/source/lightsem/experiments/motion_study'/name):raise ValueError('Generator changed')
    snapshot=freeze_sources(OUT)
    p={'entries':entries,'inputs':{str(v):digest(v) for v in files},'source_sha256':snapshot['source_sha256'],
       'native_draw_seeds':[7001+i*104729 for i in range(6)],'incumbent_seed':101,'incumbent_steps':[8,32],
       'segments':{'quiet':[0,20],'active':[70,90],'ending':[250,277]},
       'renderer':{'width':1280,'height':720,'fps':30.,'haze':.25,'cam_orbit':0.,'intensity_gain':2.4,'exposure_profile':'fixed','fixed_exposure_scale':.4,'fixed_role_gains':[1.,.3,.03,.2]},
       'gates':{'quiet_speed_reduction_min':.60,'active_speed_median_min_deg_s':3.,'active_speed_p95_min_deg_s':8.,
                'entry_bulb_ratio_min':2.,'entry_ratio_improvement_over_off_min':1.5,'quiet_bulb_min':.01,
                'whole_frame_clipped_max':.01,'ending_border_luma_max':.20,'speed_max':30.,'acceleration_max':80.},
       'variants':['incumbent','hybrid','brightness_off','independent','hybrid_legacy'],
       'primary_coherence':'quiet emitted pan/tilt vector-speed median, incumbent before versus hybrid after',
       'primary_dynamics':'active/quiet mean bulb-region luminance ratio, matched geometry brightness_off before versus hybrid after',
       'scope':'Diagnostic only: music seed0 failed; no selected native model, seed1, TEST or production promotion. Optical and mapper measurements cannot establish music conditioning.',
       'camera':'native skeleton fixed90pixels/metre, world origin0; no per-frame recentering. Source video in first column when available.',
       'test_access':False,'promotion_authorized':False}
    authenticated_json(OUT/'protocol.json',p)
    return p


def context():
    p=read_authenticated(OUT/'protocol.json');s=verify_sources(OUT)
    if not Path(__file__).resolve().is_relative_to(Path(s['snapshot']).resolve()):raise ValueError('Use frozen delivery launcher')
    for path,sha in p['inputs'].items():
        if digest(path)!=sha:raise ValueError('Delivery input changed: '+path)
    r=read_authenticated(OUT/'experiments.json')
    if r['protocol_sha256']!=digest(OUT/'protocol.json'):raise ValueError('Registration drift')
    return p


def register():
    if (OUT/'experiments.json').exists():return
    p=read_authenticated(OUT/'protocol.json')
    sys.path.insert(0,'/root/vibescience-mcp/src')
    from vibescience_mcp.core import Store
    s=Store('/root/lightsem-vault')
    for stage in ('coherence','dynamics'):
        eid=f'exp-motion-{stage}-20260910'
        if not s.vault.exists('experiments',eid):
            s.start_experiment(id=eid,hypothesis_id=f'h-motion-{stage}-20260910',external_run=str(OUT),config_note=json.dumps(p))
    authenticated_json(OUT/'experiments.json',{'protocol_sha256':digest(OUT/'protocol.json'),'scope':p['scope']})


def interpolate(x,fps,times):
    import numpy as np
    shape=x.shape[1:];x=x.reshape(len(x),-1)
    return np.stack([np.interp(times,np.arange(len(x))/fps,x[:,i]) for i in range(x.shape[1])],axis=1).reshape(len(times),*shape).astype(np.float32)


def infer():
    import numpy as np
    import torch
    from .music_model import MusicModel,MusicConfig
    from .music_evaluation import persist_npz
    from ..beat_conditioned_model import BeatConditionedModel
    from ..night_probes import PoseDecoder
    from ..night_evaluation import generate_groups,normalized_input
    p=context();torch.set_num_threads(4)
    n=read_authenticated(ROOT/'data/audio_normalizers.json');mn=read_authenticated(ROOT/'data/motion_normalizers.json')
    native={}
    for arm in ('regression','flow'):
        state=torch.load(ROOT/'music/runs'/arm/'seed_0/final.pt',map_location='cpu',weights_only=False)
        if state['identity']['protocol_sha256']!=digest(ROOT/'music/protocol.json'):raise ValueError('Model identity drift')
        m=MusicModel(MusicConfig(arm=arm),mn).cuda().eval();m.load_state_dict(state['model']);native[arm]=m
    old=BeatConditionedModel().cuda().eval();old.load_state_dict(torch.load(BASE/'run/final.pt',map_location='cpu',weights_only=False)['model'])
    state=torch.load(NIGHT/'diagnostics/pose_probes/descriptor256/seed_0/final.pt',map_location='cpu',weights_only=False)
    decoder=PoseDecoder(state['input_dim'],state['width']).cuda().eval();decoder.load_state_dict(state['model'])
    mean,std=(np.asarray(state['pose_normalizers'][k],np.float32) for k in ('mean','std'))
    bn=read_authenticated(BASE/'normalizers.json');rows=[]
    for e in p['entries']:
        path=OUT/'inference'/f'{e["key"]}.npz'
        if path.with_suffix('.json').exists():
            r=read_authenticated(path.with_suffix('.json'))
            if digest(path)!=r['sha256']:raise ValueError('Inference drift')
            rows.append(r);continue
        with np.load(e['path']) as z:raw=z['x']
        times=np.arange(round(e['duration']*30))/30
        audio=interpolate(raw,e['fps'],times)
        x=torch.from_numpy(((audio-np.asarray(n['mean']))/np.asarray(n['std'])).astype(np.float32)[None]).cuda()
        with torch.inference_mode():
            values={'regression':native['regression'].generate(x)[0].cpu().numpy()}
            seeds=p['native_draw_seeds'] if e['key']=='shpongle' else p['native_draw_seeds'][:1]
            for i,seed in enumerate(seeds):values['flow'+str(i)]=native['flow'].generate(x,seed=seed)[0].cpu().numpy()
            if e['key']=='shpongle':
                demo=read_authenticated(ROOT/'data/shpongle/input.json')
                with np.load(demo['source']['features_path']) as z:raw40=z['x']
            else:raw40=raw
            legacy_x=normalized_input(raw40,np.zeros((len(raw40),0),np.float32),bn,None)
            for steps in p['incumbent_steps']:
                y,b=generate_groups(old,legacy_x,seeds=(p['incumbent_seed'],),steps=steps)
                pose=(decoder.predict(torch.from_numpy(np.concatenate((y[0],b),axis=1)[None]))[0].numpy()*std+mean).reshape(-1,17,2)
                values['pose'+str(steps)]=interpolate(pose,40,times)
        if any(not np.isfinite(v).all() for v in values.values()):raise ValueError('Nonfinite diagnostic; no discarding draws')
        persist_npz(path,values)
        r={'key':e['key'],'path':str(path),'sha256':digest(path),'native_draw_seeds':seeds,'frames':len(times),'protocol_sha256':digest(OUT/'protocol.json')}
        authenticated_json(path.with_suffix('.json'),r);rows.append(r);print(json.dumps(r),flush=True)
    authenticated_json(OUT/'inference/report.json',{'rows':rows})


def map_lights():
    import numpy as np
    from .lighting_map import compose
    from .music_evaluation import persist_npz
    from ...conductor.rig import canonical_rig
    p=context();e=p['entries'][0]
    r=read_authenticated(OUT/'inference/shpongle.json')
    if digest(r['path'])!=r['sha256']:raise ValueError('Motion drift')
    with np.load(r['path']) as z:features=np.stack([z['flow'+str(i)] for i in range(6)])
    with np.load(ROOT/'data/shpongle/loudness.npz') as z:loud={k:z[k] for k in z.files}
    frames,trace,report=compose(features[0],loud,e['duration'])
    independent,it,ir=compose(features[0],loud,e['duration'],independent_features=features)
    off,ot,orr=compose(features[0],loud,e['duration'],dynamics=False)
    brightness_off=frames.copy()
    for f in canonical_rig().fixtures:
        a=f.address-1+(4 if f.kind=='moving_head' else 0)
        brightness_off[:,a]=off[:,a]
    with np.load(NIGHT/'delivery/shpongle/full.npz') as z:baseline=z['dmx']
    variants={'hybrid':frames,'brightness_off':brightness_off,'independent':independent,'hybrid_legacy':frames,'incumbent':baseline}
    for name,dmx in variants.items():persist_npz(OUT/'lighting'/f'{name}.npz',{'dmx':dmx})
    persist_npz(OUT/'lighting/trace.npz',trace)
    authenticated_json(OUT/'lighting/report.json',{'hybrid':report,'independent':ir,'off':orr,
        'files':{k:{'path':str(OUT/'lighting'/f'{k}.npz'),'sha256':digest(OUT/'lighting'/f'{k}.npz')} for k in variants},
        'matched_brightness_ablation':'Only dimmer bytes differ; pan/tilt, zoom, color and generated motion exactly matched.',
        'motion_qualified':False})


def skeletons():
    import numpy as np
    import torch
    from .music_render import render
    from .geometry import joints_from_features
    p=context()
    for e in p['entries']:
        path=OUT/'skeletons'/f'{e["key"]}.mp4'
        if path.with_suffix('.json').exists():
            r=read_authenticated(path.with_suffix('.json'))
            if digest(path)!=r['sha256']:raise ValueError('Movie drift')
            continue
        start=round(e['review_start']*30);count=round(e['review_duration']*30);sl=slice(start,start+count)
        receipt=read_authenticated(OUT/'inference'/f'{e["key"]}.json')
        if digest(receipt['path'])!=receipt['sha256']:raise ValueError('Motion drift')
        with np.load(receipt['path']) as z:values={k:z[k] for k in z.files}
        models=[]
        # Neutral articulated reference serves only as a fixed scale marker.
        ref=np.zeros((count,139),np.float32);ref[:,2]=.95
        ref[:,3:135]=np.tile([1,0,0,0,1,0],22)
        models.append(('Source recording' if 'source_video' in e else 'Static scale reference (not a model)',joints_from_features(torch.from_numpy(ref)).numpy(),ref[:,135:]))
        for arm in ('regression','flow0'):
            x=values[arm];j=joints_from_features(torch.from_numpy(x)).numpy()
            models.append((arm+' / unqualified',j[sl],x[sl,135:]))
        render(path,models,values['pose8'][sl],values['pose32'][sl],e['audio_path'],e['audio_offset']+start/30,
               e['key']+' / diagnostic',source_video=e.get('source_video'),fixed_camera=True)
        print(str(path),flush=True)


def render_lights():
    import math
    import subprocess
    import numpy as np
    from PIL import Image,ImageDraw
    from .exposure_study import masks,metrics
    from ...sim.renderer import RenderConfig,StageRenderer
    from ...conductor.rig import canonical_rig
    p=context();e=p['entries'][0];reports=read_authenticated(OUT/'lighting/report.json');rows=[]
    for name in p['variants']:
        folder=OUT/'lighting';path=folder/(name+'.mp4');receipt=path.with_suffix('.video.json')
        if receipt.exists():
            r=read_authenticated(receipt)
            if digest(path)!=r['sha256']:raise ValueError('Video drift')
            rows.append(r);continue
        record=reports['files'][name]
        if digest(record['path'])!=record['sha256']:raise ValueError('DMX drift')
        with np.load(record['path']) as z:dmx=z['dmx']
        cfg=RenderConfig(**p['renderer'])
        if name in ('incumbent','hybrid_legacy'):cfg.exposure_profile='legacy'
        rig=canonical_rig();renderer=StageRenderer(rig,cfg)
        regions=masks(rig,cfg,[i for i,f in enumerate(rig.fixtures) if f.kind=='moving_head'])
        cmd=['ffmpeg','-v','error','-nostdin','-n','-f','rawvideo','-pix_fmt','rgb24','-s',f'{cfg.width}x{cfg.height}','-r','30','-i','pipe:0','-i',e['audio_path'],'-map','0:v:0','-map','1:a:0','-c:v','av1_nvenc','-preset','p5','-cq','25','-b:v','0','-pix_fmt','yuv420p','-c:a','aac','-b:a','192k','-t',str(e['duration']),'-shortest',str(path)]
        encoder=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
        observations=[];shots=[];chosen={round(t*30) for t in (0,10,19,50,70,80,90,250,270)}
        try:
            for i in range(math.ceil(e['duration']*30)):
                rgb=renderer.render_frame(dmx[min(round(i*40/30),len(dmx)-1)],0.)
                encoder.stdin.write(rgb.tobytes())
                if i%6==0:
                    m=metrics(rgb,regions);x=rgb.astype(np.float32)/255;luma=x@np.asarray([.2126,.7152,.0722])
                    border=np.ones(luma.shape,bool);border[round(cfg.height*.1):round(cfg.height*.9),round(cfg.width*.1):round(cfg.width*.9)]=False
                    observations.append({'time':i/30,**m,'border_luma':float(luma[border].mean())})
                if i in chosen:
                    shot=Image.fromarray(rgb);shot.thumbnail((640,360));shots.append((i/30,shot))
        finally:
            encoder.stdin.close();error=encoder.stderr.read().decode();code=encoder.wait();renderer.ctx.release()
        if code:raise RuntimeError(error)
        sheet=Image.new('RGB',(1920,1170),'white');pen=ImageDraw.Draw(sheet)
        for k,(t,shot) in enumerate(shots):
            x,y=k%3*640,k//3*390;pen.text((x+10,y+5),f'{name} / {t:.1f}s / diagnostic',fill='black');sheet.paste(shot,(x,y+25))
        sheet.save(path.with_suffix('.png'))
        r={'name':name,'path':str(path),'sha256':digest(path),'observations':observations,'protocol_sha256':digest(OUT/'protocol.json'),'frames':math.ceil(e['duration']*30)}
        authenticated_json(receipt,r);rows.append(r);print(json.dumps({'rendered':name,'frames':r['frames']}),flush=True)
    authenticated_json(OUT/'lighting/videos.json',{'rows':rows})


def observe():
    import numpy as np
    from ...conductor.rig import canonical_rig
    p=context();g=p['gates'];report=read_authenticated(OUT/'lighting/report.json');video=read_authenticated(OUT/'lighting/videos.json')
    videos={r['name']:r for r in video['rows']};rows={}
    for name in p['variants']:
        rec=report['files'][name]
        if digest(rec['path'])!=rec['sha256'] or digest(videos[name]['path'])!=videos[name]['sha256']:raise ValueError('Observed artifact drift')
        with np.load(rec['path']) as z:dmx=z['dmx']
        angles=[];dims=[]
        for f in canonical_rig().by_kind('moving_head'):
            a=f.address-1
            angles.append(np.stack(((dmx[:,a].astype(float)*256+dmx[:,a+1])*f.pan_range_deg/65535,
                                    (dmx[:,a+2].astype(float)*256+dmx[:,a+3])*f.tilt_range_deg/65535),-1))
            dims.append(dmx[:,a+4]/255.)
        angles=np.stack(angles,1);dims=np.stack(dims,1)
        velocity=np.diff(angles,axis=0)*40;speed=np.linalg.norm(velocity,axis=-1)
        acceleration=np.linalg.norm(np.diff(velocity,axis=0)*40,axis=-1)
        segments={}
        for segment,(lo,hi) in p['segments'].items():
            sl=slice(round(lo*40),round(hi*40)-1)
            obs=[r for r in videos[name]['observations'] if lo<=r['time']<hi]
            segments[segment]={'speed_median':float(np.median(speed[sl])),'speed_p95':float(np.quantile(speed[sl],.95)),
                'dimmer_median':float(np.median(dims[sl])),'screen':{k:float(np.mean([r[k] for r in obs])) for k in obs[0] if k!='time'},
                'frame_clipped_max':max(r['frame_clipped'] for r in obs)}
        rows[name]={'segments':segments,'speed_max':float(speed.max()),'acceleration_max':float(acceleration.max()),
            'entry_ratio':segments['active']['screen']['bulb_luma']/max(segments['quiet']['screen']['bulb_luma'],1e-12)}
    before=rows['incumbent']['segments']['quiet']['speed_median'];after=rows['hybrid']['segments']['quiet']['speed_median']
    hybrid=rows['hybrid'];segs=hybrid['segments'];reduction=(before-after)/max(before,1e-12)
    gates={'motion_prerequisite':False,'quiet_pacing':reduction>=g['quiet_speed_reduction_min'],
           'active_movement':segs['active']['speed_median']>=g['active_speed_median_min_deg_s'] and segs['active']['speed_p95']>=g['active_speed_p95_min_deg_s'],
           'entry_contrast':hybrid['entry_ratio']>=g['entry_bulb_ratio_min'] and hybrid['entry_ratio']/rows['brightness_off']['entry_ratio']>=g['entry_ratio_improvement_over_off_min'],
           'quiet_visibility':segs['quiet']['screen']['bulb_luma']>=g['quiet_bulb_min'],
           'ending_glare':segs['ending']['screen']['border_luma']<=g['ending_border_luma_max'] and max(s['frame_clipped_max'] for s in segs.values())<=g['whole_frame_clipped_max'],
           'mechanics':hybrid['speed_max']<=g['speed_max']+1e-8 and hybrid['acceleration_max']<=g['acceleration_max']+1e-8,
           'integrity':True}
    result={'rows':rows,'gates':gates,'quiet_speed_reduction':reduction,'scope':p['scope'],'protocol_sha256':digest(OUT/'protocol.json')}
    authenticated_json(OUT/'lighting/observations.json',result)
    print(json.dumps({'gates':gates,'quiet_speed_reduction':reduction,'entry_ratio':hybrid['entry_ratio'],'entry_ratio_off':rows['brightness_off']['entry_ratio']}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','register','infer','map','skeletons','render','observe']);args=parser.parse_args()
    if args.stage=='freeze':freeze()
    elif args.stage=='register':register()
    elif args.stage=='map':map_lights()
    elif args.stage=='observe':observe()
    else:
        with gpu_lock(ROOT):{'infer':infer,'skeletons':skeletons,'render':render_lights}[args.stage]()
