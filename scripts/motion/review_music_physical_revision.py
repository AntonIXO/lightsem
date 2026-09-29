"""Matched, saved-array visual review. No inference, checkpoints, TEST or vault.
freeze/verify: metadata+hashes only. map: CPU torch. render: explicit shared GPU
lock. review: append-only observer receipt. See implementation report for scope.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import time
import numpy as np

REPO=Path('/root/lightsem')
BASE=Path('/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911')
EXPERIMENT=BASE/'music_physical_revision_01'
NATIVE=Path('/mnt/lightsem-data/experiments/lightsem/motion_native_audio_20260910')
DEFAULT=EXPERIMENT/'visual_review'
ARMS=('reference','old_incumbent','prior_native_flow','new_masked_native','new_physical')
ROWS=('body','original_legacy','closer_zero_angular_flux')
SCOPE='Matched visual diagnostic only; no music-fit proof, selection, hardware safety or default promotion.'

def require(ok,msg):
    if not ok: raise ValueError(msg)

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''): h.update(block)
    return h.hexdigest()

def canonical(x): return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def jhash(x): return hashlib.sha256(canonical(x)).hexdigest()
def rawjson(p): return json.loads(Path(p).read_text())
def loadjson(p):
    e=rawjson(p);require(jhash(e['payload'])==e['sha256'],'Receipt integrity '+str(p));return e['payload']
def writebytes(p,b):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('xb') as f: f.write(b);f.flush();os.fsync(f.fileno())
    p.chmod(0o444)
def savejson(p,x): writebytes(p,canonical({'payload':x,'sha256':jhash(x)})+b'\n')
def savenpz(p,**arrays):
    require(all(np.isfinite(v).all() for v in arrays.values()),'Nonfinite saved array')
    b=io.BytesIO();np.savez_compressed(b,**arrays);writebytes(p,b.getvalue())
def checked(path,sha):
    require(digest(path)==sha,'Changed input '+str(path));return Path(path)

def arrays(path):
    with np.load(path,allow_pickle=False) as z: return {k:z[k] for k in z.files}

@contextmanager
def lock(out):
    import fcntl
    out.mkdir(parents=True,exist_ok=True)
    with (out/'.visual.lock').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);yield

def items(q,entries):
    result=[]
    for c in q['review_clips']:
        e=entries[c['key']]
        result.append(dict(id=f'{c["category"]}_{c["key"]}_{c["start"]}',key=c['key'],
            start=c['start'],frames=c['frames'],full_frames=e['n_frames'],
            audio_path=e['wav_path'],audio_offset_s=e['audio_offset_s'],category=c['category']))
    d=q['shpongle'];result.append(dict(id='full_shpongle_draw7001',key='shpongle',start=0,
        frames=d['frames'],full_frames=d['frames'],audio_path=d['audio_path'],audio_offset_s=0.,category='external diagnostic'))
    require(len(result)==13 and len({r['id'] for r in result})==13,'Expected fixed12+Shpongle')
    require(q['full_seed']==7001 and q['shpongle_seeds']==[7001,111730,216459,321188,425917,530646],'Draw policy changed')
    return result

def freeze(out):
    require(not (out/'freeze.json').exists() and not (out/'source').exists(),'No freeze overwrite')
    # Existence only: do not open numerical results. The recipe must predate them.
    require(not list((EXPERIMENT/'evaluation').glob('*/initial/*/[0-9]*.json')) and
            not list((EXPERIMENT/'evaluation').glob('*/final/*/[0-9]*.json')),
            'Real evaluation already exists; cannot claim a pre-result freeze')
    ep=EXPERIMENT/'evaluation/evaluation_protocol.json';q=loadjson(ep)
    require(q['test_access'] is False and q['selection']=='none','Selection/TEST forbidden')
    manifest=rawjson(NATIVE/'data/audio_manifest.json')
    entries={e['key']:e for e in manifest['entries'] if e['key'] in q['full_keys']}
    require(all(e['partition']=='val' and e['audio_offset_s']==0 for e in entries.values()),'Unexpected cohort/offset')
    plan=items(q,entries)
    stage=rawjson(BASE/'stage_geometry_02/design.json')['config'];stage['stage_lean_gain_m']=0.
    optics=rawjson(BASE/'optics_full_controls_01/design.json')['arms']
    optical={'original_legacy':optics['legacy_native_legacy'],'closer_zero_angular_flux':optics['closer_zero_angular_flux']}
    inputs={str(ep):digest(ep),str(EXPERIMENT/'design.json'):digest(EXPERIMENT/'design.json'),
        str(BASE/'stage_geometry_02/design.json'):digest(BASE/'stage_geometry_02/design.json'),
        str(BASE/'optics_full_controls_01/design.json'):digest(BASE/'optics_full_controls_01/design.json'),
        str(NATIVE/'data/audio_manifest.json'):digest(NATIVE/'data/audio_manifest.json')}
    for asset in q['matched_review_assets'].values():
        checked(asset['report_path'],asset['report_sha256']);inputs[asset['report_path']]=asset['report_sha256']
        for a in asset['arrays']:checked(a['path'],a['sha256']);inputs[a['path']]=a['sha256']
    for item in plan:
        sha=q['inputs'][item['audio_path']];checked(item['audio_path'],sha);inputs[item['audio_path']]=sha
    loud=q['shpongle']['loudness'];checked(loud['path'],loud['sha256']);inputs[loud['path']]=loud['sha256']
    files={}
    for path in sorted((REPO/'lightsem').rglob('*')):
        if path.is_file() and path.suffix in ('.py','.json') and '__pycache__' not in path.parts:
            rel=str(path.relative_to(REPO));writebytes(out/'source'/rel,path.read_bytes());files[rel]=digest(path)
    writebytes(out/'reviewer.py',Path(__file__).read_bytes())
    # Store metadata, not any trained output. Future NPZ identities come from the frozen evaluator envelopes.
    savejson(out/'protocol.json',{'schema':1,'items':plan,'evaluation':q,'entries':entries,
        'mapping':stage,'optical_rows':optical,'arms':list(ARMS),'rows':list(ROWS),
        'direct_audio_dynamics':True,'off_dynamics_diagnostic':'not rendered in this revision',
        'full_recording_mapping':True,'mapping_fps':40,'movie_fps':30,'panel_pixels':[640,360],
        'montage_pixels':[3200,1080],'renderer_time':0.,'body_scale_px_m':60.,
        'body_origin_world':[0.,0.,0.],'old_pose_units':'fixed .45 display metres / torso unit; NOT3D calibration',
        'old_body':'pose8 and pose32 frontal2D displayed separately; lighting unavailable, no invented lift',
        'shpongle_reference':'unavailable: no source motion for external song',
        'prior_native':'historical full native FK ONCE in map; saved full world arrays subsequently sliced',
        'new_native':'ONLY evaluator saved full joints; no reintegration or checkpoint load',
        'sample_seconds':'every integer second of each fixed clip plus exact last frame; no best selection',
        'main_shpongle_seed':7001,'all_six_shpongle_retention_required':True,'scope':SCOPE,
        'human_music_fit_acceptance':None,'test_access':False,'default_promotion':False})
    savejson(out/'freeze.json',{'created_unix':time.time(),'source_files':files,'inputs':inputs,
        'script_sha256':digest(out/'reviewer.py'),'protocol_sha256':digest(out/'protocol.json'),
        'native_renderer_source_sha256':{k:v for k,v in files.items() if k.startswith('lightsem/sim/')},
        'font_path':'/usr/share/fonts/TTF/DejaVuSans.ttf',
        'font_sha256':digest('/usr/share/fonts/TTF/DejaVuSans.ttf'),
        'executables':{str(Path(shutil.which(k)).resolve()):digest(shutil.which(k)) for k in ('ffmpeg','ffprobe')},
        'pre_real_evaluation':True,'renderer_freeze_separate_from_training':True,'scope':SCOPE})
    print('FROZEN',out,flush=True)

def context(out,activate=False):
    f=loadjson(out/'freeze.json');checked(out/'protocol.json',f['protocol_sha256'])
    checked(__file__,f['script_sha256']);checked(out/'reviewer.py',f['script_sha256'])
    for rel,sha in f['source_files'].items():checked(out/'source'/rel,sha)
    for path,sha in f['inputs'].items():checked(path,sha)
    checked(f['font_path'],f['font_sha256'])
    for path,sha in f['executables'].items():checked(path,sha)
    if activate:
        require(not any(k=='lightsem' or k.startswith('lightsem.') for k in sys.modules),'Mixed live/frozen imports')
        sys.path.insert(0,str(out/'source'))
    return loadjson(out/'protocol.json')

def full_slice(world,start,frames):
    require(world.ndim==3 and world.shape[1:]==(22,3) and np.isfinite(world).all(),'Invalid full world')
    require(isinstance(start,int) and isinstance(frames,int) and start>=0 and frames>0 and start+frames<=len(world),'Invalid crop')
    return world[start:start+frames].copy()  # No origin subtraction and no FK.

def verify_evaluator_row(path,q,arm,stage,spec):
    row=loadjson(path);i=row['identity']
    for k,v in {'arm':arm,'checkpoint':'final','stage':stage,'seed':0,'steps':32,'draws':6,
        'precision':'float32','tf32':False,'evaluation_sha256':jhash(q),
        'protocol_sha256':q['protocol_sha256'],'source_sha256':q['source_sha256'],
        'evaluator_sha256':q['evaluator_sha256']}.items():require(i.get(k)==v,'Evaluator identity '+k)
    require(row['spec']==spec and row['finite_all_draws'] and row['draw_seed']==spec['draw_seeds'][0],'Evaluator spec/draw')
    expected=Path(path).with_suffix('.npz');require(Path(row['arrays_path'])==expected,'Unexpected evaluator array path')
    checked(expected,row['arrays_sha256']);return row

def evaluator_inputs(q,key):
    stage='shpongle' if key=='shpongle' else 'full';result={};bindings={}
    for arm in ('masked_native','physical'):
        root=EXPERIMENT/'evaluation'/arm/'final'/stage
        index=0 if key=='shpongle' else q['full_keys'].index(key)
        seeds=q['shpongle_seeds'] if key=='shpongle' else [7001]
        rows=[]
        for draw,seed in enumerate(seeds):
            pos=draw if key=='shpongle' else index
            path=root/f'{pos:03d}.json';row=verify_evaluator_row(path,q,arm,stage,{'key':key,'draw_seeds':[seed]})
            bindings[str(path)]=digest(path);bindings[row['arrays_path']]=row['arrays_sha256'];rows.append(row)
        # All6 retained and verified above; FIRST draw only is opened for main movie.
        result[arm]=arrays(rows[0]['arrays_path'])
    for name in ('times','reference_joints','reference_features'):
        if name in result['masked_native']:
            require(np.array_equal(result['masked_native'][name],result['physical'][name]),'Unpaired full reference/times')
    for arm,z in result.items():
        require(np.isfinite(z['joints']).all() and np.array_equal(z['root'],z['joints'][:,0]),'World/root mismatch')
        require(np.array_equal(z['times'],np.arange(len(z['joints']))/30),'Unexpected native timeline')
    return result,bindings

def historical_asset(q,group,key):
    a=next(a for a in q['matched_review_assets'][group]['arrays'] if a['key']==key)
    checked(a['path'],a['sha256']);return arrays(a['path'])

def validate_worlds(worlds,n):
    for arm,w in worlds.items():require(w.shape==(n,22,3) and np.isfinite(w).all(),'Invalid full world '+arm)

def map_one(out,p,item):
    import torch
    from lightsem.experiments.motion_study.geometry import joints_from_features
    from lightsem.experiments.motion_study.lighting_map import LightingConfig,compose_world_joints
    from lightsem.experiments.motion_study.audio_dynamics import loudness
    torch.set_num_threads(4)
    q=p['evaluation'];key=item['key'];fresh,bindings=evaluator_inputs(q,key);n=item['full_frames']
    target=out/'mapped'/item['id'];target.mkdir(parents=True,exist_ok=False)
    savejson(target/'started.json',{'item':item,'freeze_sha256':digest(out/'freeze.json'),'started_unix':time.time()})
    worlds={'new_masked_native':fresh['masked_native']['joints'],'new_physical':fresh['physical']['joints']}
    if key=='shpongle':
        old=historical_asset(q,'shpongle_previous_delivery',key);features=old['flow0']
        poses={k:old[k] for k in ('pose8','pose32')}
        curve=arrays(q['shpongle']['loudness']['path'])
    else:
        worlds['reference']=fresh['masked_native']['reference_joints']
        prior=historical_asset(q,'previous_native_flow',key);features=prior['features']
        old=historical_asset(q,'incumbent',key);t=np.arange(n)/30
        require(old['times'][0]<=t[0] and t[-1]-old['times'][-1]<=1/30+.000001,'Incumbent support mismatch')
        poses={k:np.stack([np.interp(t,old['times'],old[k].reshape(-1,34)[:,j]) for j in range(34)],1).reshape(n,17,2) for k in ('pose8','pose32')}
        # Full waveform EBU extraction. No clip-wise loudness normalization.
        lr=loudness(item['audio_path'],target/'loudness');curve=arrays(lr['path'])
    require(features.shape==(n,139) and np.isfinite(features).all(),'Historical full native shape')
    with torch.inference_mode():worlds['prior_native_flow']=joints_from_features(torch.from_numpy(features)).numpy()
    validate_worlds(worlds,n)
    require(all(z.shape==(n,17,2) and np.isfinite(z).all() for z in poses.values()),'Invalid historical pose')
    savenpz(target/'full_world.npz',**worlds,**poses,times=np.arange(n)/30)
    cfg=LightingConfig(**p['mapping']);mapped={};reports={}
    for arm,world in worlds.items():
        dmx,trace,report=compose_world_joints(world,curve,n/30,config=cfg,dynamics=True)
        mapped[arm]=dmx;reports[arm]=report
        if 'activity' in mapped:require(np.array_equal(mapped['activity'],trace['activity']),'Unmatched direct audio dynamics')
        mapped['activity']=trace['activity']
    savenpz(target/'full_dmx.npz',**mapped,time=np.arange(round(n/30*40))/40)
    files={str(x.relative_to(target)):digest(x) for x in target.rglob('*') if x.is_file()}
    savejson(target/'map.json',{'item':item,'freeze_sha256':digest(out/'freeze.json'),'files':files,
        'evaluator_bindings':bindings,'mapping_reports':reports,'whole_recording_frames':n,
        'audio_path':item['audio_path'],'audio_sha256':digest(item['audio_path']),
        'audio_offset_s':item['audio_offset_s'],'historical_fk_full_once':True,
        'loudness_alignment':'Full waveform EBU PTS+100ms; mapper interpolates global0-based40Hz timeline; native endpoint <=one frame extension uses endpoint hold',
        'new_reference_worlds':'saved evaluator joints copied exactly, no FK',
        'old_incumbent_lighting':'UNAVAILABLE: archived incumbent has only17-joint frontal2D pose',
        'source_labels':labels(key),'scope':SCOPE})
    print('MAPPED',item['id'],flush=True)

def labels(key):
    return {'reference':'REFERENCE native motion' if key!='shpongle' else 'NO source motion for Shpongle',
        'old_incumbent':'OLD descriptor->2D / 8+32 steps; NOT native3D',
        'prior_native_flow':'PRIOR native flow / saved seed0; first full draw',
        'new_masked_native':'NEW masked_native final30k / draw7001',
        'new_physical':'NEW physical final30k / draw7001'}

def mapped(out,item):
    target=out/'mapped'/item['id'];r=loadjson(target/'map.json')
    require(r['item']==item and r['freeze_sha256']==digest(out/'freeze.json'),'Map identity')
    for rel,sha in r['files'].items():checked(target/rel,sha)
    for path,sha in r['evaluator_bindings'].items():checked(path,sha)
    return target,r

def sample_indices(n):return sorted(set(range(0,n,30))|{n-1})
def dmx_indices(start,frames,n):
    indices=np.rint((np.arange(start,start+frames)/30)*40).astype(int)
    require(indices.min()>=0 and indices.max()<n,'DMX timeline out of bounds');return indices

def optical_dmx(dmx,rig,row):
    result=dmx.copy()
    require(not row['frozen_angles'],'Static optical foil not allowed')
    if row['zoom']=='zero':
        for f in rig.by_kind('moving_head'):result[:,f.address-1+9]=0
    else:require(row['zoom']=='native','Unexpected zoom')
    return result

def body_image(arm,world,poses,index,font):
    from PIL import Image,ImageDraw
    image=Image.new('RGB',(640,360),(248,248,248));d=ImageDraw.Draw(image)
    if world is None and arm!='old_incumbent':
        d.text((12,120),'No reference motion for this external song',fill='black',font=font);return image
    parents=(-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19)
    edges2=((0,1),(0,2),(1,3),(2,4),(0,5),(0,6),(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16))
    outside=0
    for view in range(2):
        ox=view*320
        if arm=='old_incumbent':
            xy=poses['pose8' if view==0 else 'pose32'][index]*.45
            edges=edges2;name='frontal2D '+('8steps' if view==0 else '32steps')
        else:
            w=world[index];xx=w[:,0] if view==0 else (w[:,0]+w[:,2])/np.sqrt(2)
            xy=np.stack((xx,w[:,1]),-1);edges=[(j,parents[j]) for j in range(1,22)]
            name='world front' if view==0 else 'world oblique'
        px=np.stack((160+60*xy[:,0],315-60*xy[:,1]),-1)
        outside+=int(np.count_nonzero((px[:,0]<0)|(px[:,0]>=320)|(px[:,1]<65)|(px[:,1]>=360)))
        # Draw on independent subpanels, so drift cannot spill into another arm.
        tile=Image.new('RGB',(320,295),(248,248,248));pen=ImageDraw.Draw(tile)
        pen.line((0,250,319,250),fill=(180,180,180))
        for a,b in edges:pen.line((float(px[a,0]),float(px[a,1]-65),float(px[b,0]),float(px[b,1]-65)),fill=(20,90,140),width=2)
        image.paste(tile,(ox,65));d.text((ox+8,45),name,fill='black',font=font)
    d.text((8,335),f'fixed60px/m; off-panel anchors {outside}',fill=(150,20,20),font=font)
    if world is not None:d.text((8,20),'root xyz '+np.array2string(world[index,0],precision=2),fill='black',font=font)
    return image

@contextmanager
def encoder(path,n,audio,offset):
    import subprocess
    cmd=['ffmpeg','-v','error','-nostdin','-n','-f','rawvideo','-pix_fmt','rgb24','-s','3200x1080','-r','30','-i','pipe:0',
        '-ss',str(offset),'-i',str(audio),'-map','0:v:0','-map','1:a:0','-frames:v',str(n),
        '-c:v','libx264','-preset','veryfast','-crf','20','-pix_fmt','yuv420p','-c:a','aac','-b:a','192k',
        '-af',f'apad,atrim=duration={n/30}','-t',str(n/30),str(path)]
    # File stderr avoids a blocked pipe. Failed partial video is retained.
    with path.with_suffix('.ffmpeg.log').open('xb') as err:
        proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=err)
        try:yield proc.stdin,cmd
        finally:
            proc.stdin.close();code=proc.wait()
    require(code==0,'ffmpeg encoding failed; partial output retained')

def audit_video(path,n):
    import subprocess
    cmd=['ffprobe','-v','error','-count_frames','-show_streams','-of','json',str(path)]
    info=json.loads(subprocess.check_output(cmd,text=True));v=next(s for s in info['streams'] if s['codec_type']=='video')
    a=next(s for s in info['streams'] if s['codec_type']=='audio')
    require((v['width'],v['height'],int(v['nb_read_frames']))==(3200,1080,n),'Decoded dimensions/count mismatch')
    require(v['r_frame_rate']=='30/1' and abs(float(v['duration'])-n/30)<.04,'Decoded timing mismatch')
    require(abs(float(a['duration'])-n/30)<.08,'Muxed audio duration mismatch')
    return info

def render_one(out,p,item):
    from PIL import Image,ImageDraw,ImageFont
    from lightsem.sim.renderer import RenderConfig,StageRenderer
    from lightsem.conductor.rig import canonical_rig
    from lightsem.experiments.stage11.stage11_common import gpu_lock
    src,m=mapped(out,item);target=out/'renders'/item['id'];target.mkdir(parents=True,exist_ok=False)
    savejson(target/'started.json',{'item':item,'freeze_sha256':digest(out/'freeze.json'),'map_sha256':digest(src/'map.json'),'scope':SCOPE})
    full=arrays(src/'full_world.npz');streams=arrays(src/'full_dmx.npz');start,n=item['start'],item['frames']
    worlds={a:full_slice(full[a],start,n) for a in ARMS if a in full}
    poses={k:full[k][start:start+n] for k in ('pose8','pose32')}
    rig=canonical_rig();font=ImageFont.truetype('/usr/share/fonts/TTF/DejaVuSans.ttf',13)
    indexes=dmx_indices(start,n,len(streams['time']))
    prepared={(row,arm):optical_dmx(streams[arm],rig,p['optical_rows'][row])[indexes]
        for row in ROWS[1:] for arm in worlds}
    selected=set(sample_indices(n));shots=[];raw=hashlib.sha256();renderer=None
    configs={row:RenderConfig(**p['optical_rows'][row]['renderer']) for row in ROWS[1:]}
    with gpu_lock(NATIVE):
        try:
            renderer=StageRenderer(rig,configs['closer_zero_angular_flux'])  # ONE EGL context; all row uniforms reset per frame
            with encoder(target/'matched.mp4',n,item['audio_path'],item['audio_offset_s']+start/30) as (pipe,cmd):
                for i in range(n):
                    image=Image.new('RGB',(3200,1080),(20,20,20))
                    for col,arm in enumerate(ARMS):
                        panel=body_image(arm,worlds.get(arm),poses,i,font);image.paste(panel,(col*640,0))
                        for rid,row in enumerate(ROWS[1:],1):
                            if arm in worlds:
                                renderer.cfg=configs[row]
                                rgb=renderer.render_frame(prepared[(row,arm)][i],0.).copy()
                                require(rgb.shape==(360,640,3),'Stage panel must remain native640x360')
                                panel=Image.fromarray(rgb)
                            else:
                                panel=Image.new('RGB',(640,360),(20,20,20));d=ImageDraw.Draw(panel)
                                d.text((12,150),'UNAVAILABLE: no comparable world3D input',fill='white',font=font)
                            image.paste(panel,(col*640,rid*360))
                    # Native panels are never rescaled. Caption rows cover only top18px.
                    d=ImageDraw.Draw(image)
                    for col,arm in enumerate(ARMS):
                        for rid,row in enumerate(ROWS):
                            x,y=col*640,rid*360;d.rectangle((x,y,x+639,y+17),fill=(12,12,12))
                            d.text((x+4,y+1),f'{row} | {m["source_labels"][arm]}',fill='white',font=font)
                    d.text((4,1060),f'{item["id"]} source t={(start+i)/30:.3f}s | {SCOPE}',fill='white',font=font)
                    raw.update(image.tobytes());pipe.write(image.tobytes())
                    if i in selected:
                        shot=target/f'sample_{i:06d}.jpg';image.save(shot,quality=90,subsampling=0);shots.append({'frame':i,'source_time_s':(start+i)/30,'path':shot.name,'sha256':digest(shot)})
                    if i%300==0:print('RENDER',item['id'],i+1,'/',n,flush=True)
            devices={r:renderer.renderer_name for r in ROWS[1:]}
        finally:
            if renderer is not None:renderer.ctx.release()
    info=audit_video(target/'matched.mp4',n)
    savejson(target/'render.json',{'item':item,'freeze_sha256':digest(out/'freeze.json'),
        'map_sha256':digest(src/'map.json'),'video_sha256':digest(target/'matched.mp4'),
        'frames':n,'fps':30,'panel_pixels':[640,360],'montage_pixels':[3200,1080],
        'renderer_devices':devices,'renderer_time':0.,'decoded_streams':info,'ffmpeg_command':cmd,
        'audio_padding_policy':'At most native final-frame endpoint mismatch; pad audio only to exact native duration, no motion padding',
        'decorated_raw_rgb_sha256':raw.hexdigest(),'samples':shots,'scope':SCOPE,
        'direct_audio_dynamics':True,'off_dynamics_diagnostic':False,'review_status':'pending',
        'human_music_fit_acceptance':None,'default_promotion':False})
    print('RENDERED',item['id'],flush=True)

def verify_render(out,item):
    src,_=mapped(out,item);target=out/'renders'/item['id'];r=loadjson(target/'render.json')
    require(r['item']==item and r['freeze_sha256']==digest(out/'freeze.json') and r['map_sha256']==digest(src/'map.json'),'Render identity')
    checked(target/'matched.mp4',r['video_sha256'])
    require([s['frame'] for s in r['samples']]==sample_indices(item['frames']),'Incomplete sample schedule')
    for s in r['samples']:checked(target/s['path'],s['sha256'])
    return target,r

def review(out,p,observation):
    pending=[i for i in p['items'] if not (out/'reviews'/(i['id']+'.json')).exists()]
    require(pending,'All observer receipts already exist');item=pending[0];target,r=verify_render(out,item)
    o=rawjson(observation)
    require(o.get('item_id')==item['id'],'Review only next missing frozen item')
    require(o.get('reviewer') and o.get('notes'),'Reviewer and notes required')
    require(o.get('coverage') in ('all_frozen_samples','full_playback'),'Declare instrument coverage')
    require(o.get('viewed_sample_frames')==sample_indices(item['frames']),'Inspect every frozen sample, no best selection')
    require(set(o.get('comparisons',{}))==set(ROWS),'Record body and BOTH within-row comparisons')
    require(all(isinstance(v,str) and v.strip() for v in o['comparisons'].values()),'Nonempty within-row notes required')
    savejson(out/'reviews'/(item['id']+'.json'),{'item':item,'render_sha256':digest(target/'render.json'),
        'observation':o,'observation_file_sha256':digest(observation),'created_unix':time.time(),
        'scope':SCOPE,'coverage_is_observer_attestation_not_machine_proof':True,
        'human_music_fit_acceptance':None,'pipeline_qualified':False})
    print('REVIEW RECORDED',item['id'],flush=True)

def report(out,p):
    rows=[]
    for item in p['items']:
        _,r=verify_render(out,item);review_path=out/'reviews'/(item['id']+'.json')
        row={'item':item['id'],'render_sha256':digest(out/'renders'/item['id']/'render.json'),'review_sha256':None}
        if review_path.exists():
            v=loadjson(review_path);require(v['item']==item and v['render_sha256']==row['render_sha256'],'Review identity')
            row['review_sha256']=digest(review_path)
        rows.append(row)
    savejson(out/'review_report.json',{'rows':rows,'all13_rendered':True,'all13_observer_receipts':all(r['review_sha256'] for r in rows),
        'scope':SCOPE,'limitations':['Incumbent only2D; no matched incumbent lighting','Shpongle source motion unavailable',
            'No disabled-direct-audio rendered diagnostic','Fixed world cameras may clip large drift; off-panel anchors disclosed',
            'Lighting carries shared direct audio dynamics; cannot establish music conditioning'],
        'visual_gate':None,'human_music_fit_acceptance':None,'pipeline_qualified':False,'selection':None,
        'seed1_authorized':False,'test_access':False,'default_promotion':False})

def schedule(out,p,stage,budget):
    sub='mapped' if stage=='map' else 'renders';receipt='map.json' if stage=='map' else 'render.json';new=0
    for item in p['items']:
        target=out/sub/item['id']
        if (target/receipt).exists():
            mapped(out,item) if stage=='map' else verify_render(out,item);continue
        if new>=budget:break
        require(not target.exists(),'Incomplete stage retained; do not overwrite '+str(target))
        (map_one if stage=='map' else render_one)(out,p,item);new+=1
    print(stage,'next-missing budget completed',new,flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('freeze','verify','map','render','review','report'))
    parser.add_argument('--output',type=Path,default=DEFAULT)
    parser.add_argument('--budget',type=int,default=1)
    parser.add_argument('--observation',type=Path)
    args=parser.parse_args();require(args.budget>0,'Positive budget required')
    require(args.output.resolve()==DEFAULT.resolve(),'Only declared visual_review output directory')
    with lock(args.output):
        if args.stage=='freeze':freeze(args.output);return
        p=context(args.output,activate=args.stage in ('map','render'))
        if args.stage=='verify':print('Verified frozen visual recipe, inputs, separate native renderer. No model/results read.');return
        if args.stage in ('map','render'):schedule(args.output,p,args.stage,args.budget)
        elif args.stage=='review':require(args.observation is not None,'--observation required');review(args.output,p,args.observation)
        else:report(args.output,p)

if __name__=='__main__':main()
