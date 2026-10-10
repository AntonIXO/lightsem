"""Render audited saved native body and actual decoded DMX; no generation."""
from pathlib import Path
import argparse,json,subprocess,hashlib
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scripts.motion.evaluate_v5_delivery import OUT,SONGS,verify,read,save,sha,world,bodypath,arrays

def data(song):
    p=verify();t=next(x for x in p['tracks'] if x['id']==song)
    w=world(bodypath('candidate',song));ref=world(bodypath('reference',song))
    folder=OUT/'emitted/candidate'/song
    with np.load(folder/'lights.npz') as z:lights={k:z[k].copy() for k in z.files}
    return p,t,w,ref,lights,read(folder/'patch.json')

def plot(song,w,ref):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['path.simplify']=False
    t=np.arange(len(w))/30;speed=np.linalg.norm(np.diff(w,axis=0)*30,axis=2).mean(1);base=np.linalg.norm(np.diff(ref,axis=0)*30,axis=2).mean(1)
    mask=(t[1:]>=90)&(t[1:]<150);rootmask=(t>=90)&(t<150)
    fig,axs=plt.subplots(3,1,figsize=(12.8,3.6),dpi=100,sharex=True)
    axs[0].plot(t[1:][mask],base[mask],lw=.6,label='selected reference');axs[0].plot(t[1:][mask],speed[mask],lw=.6,label='all-data candidate');axs[0].set_ylabel('Raw m/s');axs[0].legend(fontsize=7,loc='upper right')
    axs[1].plot(t[1:][mask],speed[mask]-base[mask],lw=.6,color='black');axs[1].axhline(0,color='grey',lw=.5);axs[1].set_ylabel('Delta m/s')
    axs[2].plot(t[rootmask],ref[rootmask,0,1],lw=.6);axs[2].plot(t[rootmask],w[rootmask,0,1],lw=.6);axs[2].set_ylabel('Root Y (m)');axs[2].set_xlabel('Original song seconds')
    for ax in axs:ax.grid(alpha=.2);ax.set_xlim(90,150)
    fig.tight_layout(pad=.5);path=OUT/'media'/song/'raw_comparison.png';path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(path)
    fig.savefig(path);plt.close(fig)
    arrays(path.with_suffix('.npz'),time=t,speed_time=t[1:],reference_speed=base,candidate_speed=speed,reference_root_y=ref[:,0,1],candidate_root_y=w[:,0,1])
    return Image.open(path).convert('RGB')

def renderers(patch):
    from lightsem.sim.lit_body_renderer import LitBodyStageRenderer
    from lightsem.sim.renderer import RenderConfig
    cameras=[dict(cam_pos=(2.8,2.15,3.8),cam_target=(0.,1.,-3.5),camera_scale=.20),dict(cam_pos=(0.,6.,11.),cam_target=(0.,2.,-3.),camera_scale=1.02)]
    rs=[]
    for cam,fill in zip(cameras,(.5,.12)):
        cfg=RenderConfig(width=640,height=420,fps=30,cam_orbit=0.,haze=.25,intensity_gain=2.4,exposure_profile='fixed',fixed_exposure_scale=.4,fixed_role_gains=(1.,.3,.03,.2),beam_edge_profile='angular',beam_edge_softness_deg=.6,beam_flux_reference_deg=14.,gpu_device=0,require_gpu=True,**cam)
        rs.append(LitBodyStageRenderer(patch,cfg,body_offset=(0.,0.,-3.5),body_scale=1.,neutral_fill=fill))
    return rs

def frame(rs,w,lights,second,song,strip):
    index=round(second*30);tick=round(second*40)
    pose=w[index].copy();pose-=pose[:1]*np.array([1.,0.,1.])
    values=tuple(lights[k][tick] for k in ('directions','rgb','dimmer'))
    canvas=Image.new('RGB',(1280,900),(8,12,20));draw=ImageDraw.Draw(canvas)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
    draw.text((12,8),f'{song} | RTMPose all-data seed 0 | DIAGNOSTIC: artistic quality unqualified',font=font,fill='white')
    draw.text((12,32),'Own-song raw body + emitted lights | P2 / LUFS | motion caps OFF | no D4/D6',font=font,fill='white')
    for xpos,r in zip((0,640),rs):canvas.paste(Image.fromarray(r.render_controls(*values,pose)),(xpos,65))
    draw.text((12,492),f'Song time {second:.2f}s | root XZ removed for display only; root height retained | neutral fill .50/.12',font=font,fill='white')
    canvas.paste(strip,(0,540));return canvas,pose,values

def smoke(song):
    from lightsem.dance.pipeline import accelerator_lock
    p,t,w,ref,lights,patch=data(song);folder=OUT/'media'/song;folder.mkdir(parents=True,exist_ok=True)
    strip=plot(song,w,ref);rows=[]
    with accelerator_lock():
        rs=renderers(patch)
        try:
            for second in (90.,110.,140.):
                im,pose,values=frame(rs,w,lights,second,song,strip);path=folder/f'smoke_{int(second)}.png';im.save(path)
                diagnostics=[]
                for r in rs:
                    d=r.render_diagnostics(*values,pose)
                    diagnostics.append({'metrics':d['metrics'],'repeat_exact':bool(np.array_equal(d['actual'],r.render_controls(*values,pose)))})
                rows.append({'second':second,'path':str(path),'sha256':sha(path),'views':diagnostics})
        finally:
            for r in rs:r.close()
    save(folder/'SMOKE.json',{'rows':rows,'body_sha256':sha(bodypath('candidate',song)),'lights_sha256':sha(OUT/'emitted/candidate'/song/'lights.npz'),'presentation_only':'root XZ removed, neutral fill .5/.12; original height and motion retained'})

def render(song):
    from lightsem.dance.pipeline import accelerator_lock
    p,t,w,ref,lights,patch=data(song);folder=OUT/'media'/song
    checked=read(folder/'INSPECTED.json');assert checked['smoke_sha256']==sha(folder/'SMOKE.json')
    strip=Image.open(folder/'raw_comparison.png').convert('RGB');audio=folder/'audio_reference.m4a';video=folder/'review.mp4'
    subprocess.run(['ffmpeg','-v','error','-nostdin','-n','-ss','90','-i',t['audio'],'-t','60','-vn','-c:a','aac','-b:a','192k',str(audio)],check=True)
    cmd=['ffmpeg','-v','error','-nostdin','-n','-f','rawvideo','-pix_fmt','rgb24','-s','1280x900','-r','30','-i','pipe:0','-i',str(audio),'-map','0:v:0','-map','1:a:0','-frames:v','1800','-c:v','libx264','-preset','veryfast','-crf','20','-threads','2','-pix_fmt','yuv420p','-c:a','copy','-t','60','-movflags','+faststart',str(video)]
    with accelerator_lock():
        rs=renderers(patch)
        try:
            with (folder/'encode.log').open('xb') as log:
                proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=log)
                try:
                    for i in range(1800):
                        im,_,_=frame(rs,w,lights,90+i/30,song,strip);proc.stdin.write(im.tobytes())
                        if i%300==0:print('RENDER',song,i,flush=True)
                    proc.stdin.close();rc=proc.wait();assert rc==0
                finally:
                    if proc.poll() is None:proc.kill();proc.wait()
        finally:
            for r in rs:r.close()
    save(folder/'RENDER.json',{'video_path':str(video),'video_sha256':sha(video),'body_sha256':sha(bodypath('candidate',song)),'lights_sha256':sha(OUT/'emitted/candidate'/song/'lights.npz'),'audio_sha256':sha(t['audio']),'reference_audio_sha256':sha(audio),'frames':1800,'source_start_s':90,'source_end_s':150,'audio_speed':1.,'diagnostic_only':True})

def audit(song):
    folder=OUT/'media'/song;video=folder/'review.mp4';ref=folder/'audio_reference.m4a'
    subprocess.run(['ffmpeg','-v','error','-xerror','-hwaccel','none','-i',str(video),'-f','null','-'],check=True)
    frames=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_frames','-show_entries','frame=best_effort_timestamp_time','-of','json',str(video)]))['frames']
    pts=np.array([float(x['best_effort_timestamp_time']) for x in frames]);assert len(pts)==1800 and np.allclose(np.diff(pts),1/30,atol=1e-6,rtol=0)
    def pcm(p):return subprocess.check_output(['ffmpeg','-v','error','-i',str(p),'-map','0:a:0','-f','f32le','-acodec','pcm_f32le','-'])
    a,b=pcm(ref),pcm(video);assert a==b,'Decoded audio mismatch'
    save(folder/'AUDIT.json',{'software_decode':True,'frames':len(pts),'uniform_pts_30hz':True,'pcm_equal_to_own_song_reference':True,'pcm_sha256':hashlib.sha256(b).hexdigest(),'video_sha256':sha(video),'render_receipt_sha256':sha(folder/'RENDER.json'),'source_body_visible_in_smoke':True})

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['smoke','render','audit']);a.add_argument('song',choices=SONGS);args=a.parse_args();globals()[args.stage](args.song)
