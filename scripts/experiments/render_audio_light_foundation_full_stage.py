"""Full-width native-style foundation comparison for an authenticated song."""
from __future__ import annotations
import argparse, hashlib, json, math, subprocess
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from lightsem.sim.lit_body_renderer import LitBodyStageRenderer
from lightsem.sim.renderer import RenderConfig
from lightsem.dance.pipeline import accelerator_lock
from scripts.experiments.render_audio_light_foundation_native_style import directions_from_aim, plot_strip


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for c in iter(lambda:f.read(1<<20),b''): h.update(c)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(); p.add_argument('--predictions',type=Path,required=True); p.add_argument('--inference',type=Path,required=True); p.add_argument('--audio',type=Path,required=True); p.add_argument('--patch',type=Path,default=Path('/mnt/lightsem-data/experiments/lightsem/native_first_video_20261015_v1/shpongle/full_run/patch.json')); p.add_argument('--out',type=Path,required=True); p.add_argument('--label',default='Shpongle | foundation 30 Hz | full-stage comparison'); a=p.parse_args()
    if a.out.exists(): raise FileExistsError(a.out)
    with np.load(a.predictions,allow_pickle=False) as z:
        aim=np.asarray(z['aim']); rgb8=np.asarray(z['rgb']); bright8=np.asarray(z['brightness'])[:,:,0]; event=1/(1+np.exp(-np.asarray(z['event'])[:,0])); times=np.asarray(z['time'])
    n=len(times)
    if n<2 or not np.allclose(times,np.arange(n)/30.,atol=1e-6): raise ValueError('predictions must use 30 Hz clock')
    patch=json.loads(a.patch.read_text()); moving_map=np.array([0,1,2,3,4,5,6,7,0,2,4,5,7]); bulb_map=np.arange(50)%8
    dirs=np.zeros((n,63,3),np.float32);dirs[:,:13]=directions_from_aim(aim[:,moving_map]);dirs[:,13:]=[0,-1,0]
    rgb=np.zeros((n,63,3),np.float32);rgb[:,:13]=rgb8[:,moving_map];rgb[:,13:]=rgb8[:,bulb_map]
    dim=np.zeros((n,63),np.float32);dim[:,:13]=bright8[:,moving_map];dim[:,13:]=bright8[:,bulb_map]
    pair=np.linalg.norm(aim[:,:,None,:]-aim[:,None,:,:],axis=-1);ii,jj=np.triu_indices(8,1);spread=pair[:,ii,jj].mean(1);speed=np.linalg.norm(np.diff(aim,axis=0),axis=-1).mean(1)*30
    a.out.mkdir(parents=True);(a.out/'frames').mkdir();np.savez_compressed(a.out/'CONTROLS.npz',time=times,directions=dirs,rgb=rgb,dimmer=dim,aim_spread=spread,aim_speed=speed,brightness=bright8.mean(1),event=event)
    neutral=np.full((22,3),1000.,np.float32);cfg=RenderConfig(width=1280,height=720,fps=30,cam_orbit=0.,haze=.25,intensity_gain=2.4,exposure_profile='fixed',fixed_exposure_scale=.4,fixed_role_gains=(1.,.3,.03,.2),beam_edge_profile='angular',beam_edge_softness_deg=.6,beam_flux_reference_deg=14.,gpu_device=0,require_gpu=True,cam_pos=(0.,6.,11.),cam_target=(0.,2.,-3.),camera_scale=1.02)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',22);small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14);out=a.out/'demo.mp4';command=['ffmpeg','-v','error','-nostdin','-y','-f','rawvideo','-pix_fmt','rgb24','-s','1280x1024','-r','30','-i','pipe:0','-i',str(a.audio),'-map','0:v:0','-map','1:a:0','-frames:v',str(n),'-c:v','libx264','-preset','veryfast','-pix_fmt','yuv420p','-c:a','aac','-b:a','128k','-t',str(n/30),'-movflags','+faststart',str(out)]
    with accelerator_lock():
        renderer=LitBodyStageRenderer(patch,cfg,body_offset=(0.,0.,0.),body_scale=1.,neutral_fill=0.)
        try:
            enc=subprocess.Popen(command,stdin=subprocess.PIPE)
            for i in range(n):
                stage=renderer.render_controls(dirs[i],rgb[i],dim[i],neutral)
                canvas=Image.new('RGB',(1280,1024),(7,10,16));d=ImageDraw.Draw(canvas);d.text((16,8),a.label,font=font,fill='white');d.text((16,42),'30 Hz learned controls | full-width lights | no body input | no native DMX',font=small,fill='#c1d2e2');d.text((16,68),'FOUNDATION LIGHT OUTPUT',font=small,fill='#c1d2e2');canvas.paste(Image.fromarray(stage),(0,84));d.text((16,690),f'Shpongle  t={i/30.:05.2f}s | original audio | diagnostic comparison',font=small,fill='#bacedd');strip=plot_strip({'aim_speed':speed[:i+1],'aim_spread':spread[:i+1],'brightness':bright8[:i+1].mean(1),'event':event[:i+1]},n/30);cursor=round(278+(i/max(n-1,1))*982);ImageDraw.Draw(strip).line((cursor,30,cursor,290),fill='#fff2bb',width=2);canvas.paste(strip,(0,720));enc.stdin.write(canvas.tobytes());
                if i in (0,n//2,n-1):canvas.save(a.out/'frames'/f'frame_{i:04d}.png')
            enc.stdin.close();
            if enc.wait()!=0:raise RuntimeError('ffmpeg failed')
        finally: renderer.close()
    montage=Image.new('RGB',(1920,360));
    for j,i in enumerate((0,n//2,n-1)):montage.paste(Image.open(a.out/'frames'/f'frame_{i:04d}.png').resize((640,360)),(640*j,0))
    montage.save(a.out/'frames/montage.png')
    rec={'schema_version':1,'status':'diagnostic_only','song':'Shpongle','duration_s':n/30,'fps':30,'video':str(out),'video_sha256':sha(out),'audio':str(a.audio),'audio_sha256':sha(a.audio),'predictions':str(a.predictions),'predictions_sha256':sha(a.predictions),'inference':str(a.inference),'patch':str(a.patch),'patch_sha256':sha(a.patch),'full_width_stage':True,'dance_space':False,'learned_beams':8,'presentation_lasers':5,'presentation_bulbs':50,'presentation_duplication':'explicit 8→5/50; no independent extra motion','native_body':False,'native_dmx':False,'state_reset_per_chunk':True,'collapse_gate':{'minimum_per_frame_mean_pairwise_aim':float(spread.min()),'fraction_below_0.2':float(np.mean(spread<.2)),'passed':bool(np.all(spread>=.2))},'frames_inspected':[str(a.out/'frames'/f'frame_{i:04d}.png') for i in (0,n//2,n-1)],'montage':str(a.out/'frames/montage.png'),'promotion':'forbidden'}
    (a.out/'COMPARISON.json').write_text(json.dumps(rec,indent=2,sort_keys=True)+'\n');print(json.dumps(rec,indent=2))

if __name__=='__main__':main()
