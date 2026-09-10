"""Read actual encoded movies, independently check mirror rays and save review sheets."""
from pathlib import Path
from dataclasses import replace
import json,subprocess,sys
import numpy as np
ROOT=Path('/mnt/lightsem-data/experiments/lightsem/night_expression_20260909')
sys.path.insert(0,str(ROOT/'source'))
from lightsem.experiments.stage11_common import read_authenticated,authenticated_json,digest,verify_artifacts
from lightsem.experiments.night_common import ARMS,SEEDS,base_context
from lightsem.experiments.night_render import compose_scene,ExpressionConfig
from lightsem.experiments.quality_observer import video_observation
from lightsem.conductor.rig import canonical_rig

def close(a,b,label):
    if not np.allclose(a,b,rtol=1e-6,atol=2e-6):raise AssertionError(label)

def video_info(path,seconds,width,height):
    raw=subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(path)])
    data=json.loads(raw);videos=[s for s in data['streams'] if s['codec_type']=='video'];audio=[s for s in data['streams'] if s['codec_type']=='audio']
    if len(videos)!=1 or len(audio)!=1:raise AssertionError('Missing single video/audio stream')
    v=videos[0]
    if v['codec_name']!='av1' or v['width']!=width or v['height']!=height or v['avg_frame_rate']!='30/1':raise AssertionError('Unexpected video encoding')
    if abs(float(data['format']['duration'])-seconds)>.15:raise AssertionError('Truncated or wrong-duration movie')
    if abs(int(v['nb_frames'])-round(seconds*30))>2:raise AssertionError('Wrong video frame count')
    return {'width':width,'height':height,'frames':int(v['nb_frames']),'duration_s':float(data['format']['duration']),'codec':v['codec_name'],'audio_codec':audio[0]['codec_name']}

def mirror_error(frames,rig):
    heads=sorted(rig.by_kind('moving_head'),key=lambda f:(f.position[0],f.name));rays=[]
    for h in heads:
        ix=h.address-1;d=frames[:,ix:ix+4].astype(float)
        yaw=((d[:,0]*256+d[:,1])/65535-.5)*np.deg2rad(h.pan_range_deg)
        pitch=((d[:,2]*256+d[:,3])/65535-.5)*np.deg2rad(h.tilt_range_deg)
        rays.append(np.stack([np.sin(yaw)*np.cos(pitch),-np.cos(yaw)*np.cos(pitch),np.sin(pitch)],1))
    angles=[]
    for j in range(3):
        right=rays[5-j]*np.array([-1,1,1]);cos=np.clip(np.einsum('ij,ij->i',rays[j],right),-1,1)
        angles.extend(np.degrees(np.arccos(cos)))
    return float(np.max(angles))

def image_at(path,time_s=10):
    raw=subprocess.check_output(['ffmpeg','-v','error','-threads','2','-ss',str(time_s),'-i',str(path),'-frames:v','1','-vf','scale=240:135','-f','rawvideo','-pix_fmt','rgb24','pipe:1'])
    if len(raw)!=240*135*3:raise ValueError('Missing contact-sheet frame')
    from PIL import Image
    return Image.fromarray(np.frombuffer(raw,np.uint8).reshape(135,240,3))

def main():
    from PIL import Image,ImageDraw,ImageFont
    p=read_authenticated(ROOT/'protocol.json');n=base_context()[1];cfg=ExpressionConfig(**read_authenticated(ROOT/'expression_config.json'));rig=canonical_rig()
    out=ROOT/'delivery/review';out.mkdir(exist_ok=True);font=ImageFont.truetype('/usr/share/fonts/TTF/DejaVuSans.ttf',14);rows=[]
    keys=p['evaluation']['render_entries']
    for seed in SEEDS:
        sheet=Image.new('RGB',(1920,6*165),(12,12,12));paint=ImageDraw.Draw(sheet)
        for ai,arm in enumerate(ARMS):
            folder=ROOT/'evaluation'/arm/f'seed_{seed}';report=read_authenticated(folder/'video_report.json');bykey={r['key']:r for r in report['rows']}
            if set(bykey)!=set(keys):raise AssertionError('Video membership')
            for i,key in enumerate(keys):
                r=bykey[key];verify_artifacts(r['artifacts']);path=folder/key/'heads_only.mp4';info=video_info(path,20,854,480)
                render=r['render']
                if 'nvidia' not in render['renderer'].lower() or render['shader_time']!=0 or render['camera_orbit']!=0:raise AssertionError('GPU/static camera rendering contract')
                actual=video_observation(path)['summary']
                for metric,value in r['screen'].items():close(actual[metric],value,'Encoded video observer '+metric)
                with np.load(folder/key/'full.npz') as z:y=z['standardized'][:,:800];events=z['event_scores'][:800]
                d,t,_=compose_scene(y,n['target_names'],20,event_scores=events,config=replace(cfg,composition='mirror'),rig=rig)
                error=mirror_error(d,rig);close(error,r['mirror_emitted_ray_error_deg'],'Emitted mirror geometry')
                sheet.paste(image_at(path),(i*240,ai*165+27));paint.text((i*240+5,ai*165+5),f'{arm} / s{seed} / {key[:11]}',fill=(225,225,225),font=font)
                rows.append({'arm':arm,'seed':seed,'key':key,'video':str(path),'sha256':digest(path),'info':info,'recomputed_screen':actual,'mirror_ray_error_deg':error})
            print(json.dumps({'stage':'independent_video_audit','arm':arm,'seed':seed,'clips':8}),flush=True)
        sheet.save(out/f'all_main_clips_seed_{seed}.jpg',quality=92)
    incumbent=read_authenticated(ROOT/'evaluation/incumbent_adapter/report.json');incumbent_rows=[]
    if len(incumbent['rows'])!=8 or {r['key'] for r in incumbent['rows']}!=set(keys):raise AssertionError('Incumbent adapter membership')
    for r in incumbent['rows']:
        folder=ROOT/'evaluation/incumbent_adapter'/r['key'];verify_artifacts(r['artifacts'])
        for mode,v in r['modes'].items():
            path=folder/(mode+'_heads.mp4');video_info(path,20,854,480);actual=video_observation(path)['summary']
            for metric,value in v['screen'].items():close(actual[metric],value,'Incumbent video observer')
        with np.load(folder/'controls.npz') as z:y=z['standardized'][:,:800];events=z['event_scores'][:800]
        d,t,_=compose_scene(y,n['target_names'],20,event_scores=events,config=replace(cfg,composition='mirror'),rig=rig)
        error=mirror_error(d,rig);target_error=max(float(abs(t['targets'][:,i,0]+t['targets'][:,5-i,0]).max()) for i in range(3))
        incumbent_rows.append({'key':r['key'],'mirror_ray_error_deg':error,'mirror_target_error_m':target_error})
    delivery=ROOT/'delivery/shpongle';inf=read_authenticated(delivery/'inference_report.json');movies=read_authenticated(delivery/'render_report.json');obs=read_authenticated(delivery/'observation_report.json')
    video_rows={};duration=inf['input']['source_duration_s']
    for mode,r in movies['videos'].items():verify_artifacts(r['artifacts']);video_rows[mode]=video_info(r['video'],duration,1920,1080)
    for field in ('config','shader_time','camera_orbit','renderer'):
        if movies['videos']['expression_v4']['render'][field]!=movies['videos']['previous_same_renderer']['render'][field]:raise AssertionError('Shpongle render mismatch')
    heads=video_observation(movies['videos']['heads_only']['video'])['summary']
    for key,value in obs['heads_only']['summary'].items():close(heads[key],value,'Whole-track head visibility')
    from lightsem.experiments.night_calibration import screen_statistics
    ending={mode:screen_statistics(movies['videos'][mode]['video'],duration-60,60) for mode in ('expression_v4','previous_same_renderer')}
    for mode,v in ending.items():
        for key,value in obs['ending'][mode].items():close(v[key],value,'Ending decoded RGB')
    reduction=(ending['previous_same_renderer']['border_luma_mean']-ending['expression_v4']['border_luma_mean'])/ending['previous_same_renderer']['border_luma_mean']
    close(reduction,obs['border_luma_reduction'],'Ending glare reduction')
    if (reduction>=.3)!=obs['ending_glare_gate']:raise AssertionError('Glare gate')
    incumbent_movie=movies['videos'].get('incumbent_new_adapter',movies['videos']['expression_v4'])
    incumbent_ending=screen_statistics(incumbent_movie['video'],duration-60,60)
    incumbent_reduction=(ending['previous_same_renderer']['border_luma_mean']-incumbent_ending['border_luma_mean'])/ending['previous_same_renderer']['border_luma_mean']
    pose_report=read_authenticated(ROOT/'delivery/pose/report.json')
    if len(pose_report['rows'])!=3:raise AssertionError('Pose movie membership')
    pose=[]
    for r in pose_report['rows']:
        if digest(r['path'])!=r['sha256']:raise AssertionError('Pose movie hash')
        pose.append(video_info(r['path'],20,1920,1080))
    comparison=read_authenticated(delivery/'comparison.json');video_info(comparison['path'],duration,1920,540)
    authenticated_json(ROOT/'diagnostics/independent_video_audit.json',{'passed':True,'main_clips':rows,'incumbent_mirror':incumbent_rows,
        'full_movies':video_rows,'pose_movies':pose,'border_luma_reduction':reduction,'ending_glare_gate':reduction>=.3,
        'incumbent_ending_border_luma_reduction':incumbent_reduction,'incumbent_ending_statistics':incumbent_ending,'auditor_sha256':digest(__file__),
        'contact_sheets':[str(out/f'all_main_clips_seed_{s}.jpg') for s in SEEDS]})

if __name__=='__main__':main()
