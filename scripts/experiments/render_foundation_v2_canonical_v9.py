from __future__ import annotations
import hashlib,json,subprocess
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from lightsem.sim.lit_body_renderer import LitBodyStageRenderer
from lightsem.sim.renderer import RenderConfig
from lightsem.dance.pipeline import accelerator_lock
SRC=Path('/mnt/lightsem-data/experiments/lightsem/foundation_audio_light_v2_continuous_20261016_v9_warm_v7/FULL_INFERENCE_3b4ab.npz')
AUDIO=Path('/mnt/lightsem-data/beatsaver/wav/3b4ab.wav')
PATCH=Path('/mnt/lightsem-data/experiments/lightsem/native_first_video_20261015_v1/shpongle/full_run/patch.json')
OUT=Path('/mnt/lightsem-data/experiments/lightsem/foundation_audio_light_v2_continuous_20261016_v9_warm_v7/DEMO_3B4AB_60S_10HZ_FINAL')
N=600; FPS=10.
def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def directions(angle):
 pan=angle[...,0]*np.pi; tilt=angle[...,1]*np.pi/2.; x=np.sin(pan)*np.cos(tilt); y=-np.cos(pan)*np.cos(tilt); z=np.sin(tilt); q=np.stack((x,y,z),-1); return q/np.maximum(np.linalg.norm(q,axis=-1,keepdims=True),1e-8)
def audit(angle,rgb,br):
 pair=np.linalg.norm(angle[:,:,None,:]-angle[:,None,:,:],axis=-1); ii,jj=np.triu_indices(8,1); sp=pair[:,ii,jj].mean(1); vel=np.diff(angle,axis=0,prepend=angle[:1])*FPS; acc=np.diff(vel,axis=0,prepend=vel[:1])*FPS; jerk=np.diff(acc,axis=0,prepend=acc[:1])*FPS; bins=np.floor(np.mod((angle[:,:,0]+1)/2,1)*8).astype(int); sides=np.array([len(np.unique(x)) for x in bins]); h=[hashlib.sha256(angle[:,i].tobytes()).hexdigest() for i in range(8)]; out={'finite':bool(np.isfinite(angle).all() and np.isfinite(rgb).all() and np.isfinite(br).all()),'spread_min':float(sp.min()),'spread_p50':float(np.median(sp)),'spread_p95':float(np.quantile(sp,.95)),'spread_below02_fraction':float(np.mean(sp<.2)),'side_count_min':int(sides.min()),'side_count_p50':float(np.median(sides)),'speed_max':float(np.linalg.norm(vel,axis=-1).max()),'acceleration_max':float(np.linalg.norm(acc,axis=-1).max()),'jerk_max':float(np.linalg.norm(jerk,axis=-1).max()),'rgb_below001_fraction':float(np.mean(rgb<.01)),'rgb_std':float(rgb.std()),'brightness_std':float(br.std()),'exact_duplicate_angle_streams':len(set(h))<8}
 if not out['finite'] or out['spread_below02_fraction']>0 or out['side_count_min']<6 or out['speed_max']>0.30 or out['acceleration_max']>2.0 or out['jerk_max']>20.0 or out['rgb_std']<0.01 or out['exact_duplicate_angle_streams']: raise RuntimeError(json.dumps({'status':'rejected_collapse_or_motion','audit':out}))
 return out
def main():
 with np.load(SRC,allow_pickle=False) as z:
  angle_raw=z['angle'][0,:N]; rgb=z['rgb'][0,:N]; br=z['brightness'][0,:N,:,0]; event_logits=z['event'][0,:N]; ev=1/(1+np.exp(-event_logits[:,0]))
 angle=angle_raw
 a=audit(angle,rgb,br); OUT.mkdir(parents=True,exist_ok=False); np.savez_compressed(OUT/'PREDICTIONS_10HZ.npz',time=np.arange(N)/FPS,aim=angle,rgb=rgb,brightness=br[:,:,None],event=event_logits); np.savez_compressed(OUT/'CONTROLS.npz',time=np.arange(N)/FPS,directions=directions(angle),rgb=np.repeat(rgb,1,axis=1),dimmer=br,aim_spread=np.linalg.norm(angle[:,:,None,:]-angle[:,None,:,:],axis=-1).mean((1,2)),event=ev)
 patch=json.loads(PATCH.read_text()); cfg=RenderConfig(width=640,height=600,fps=FPS,cam_orbit=0.,haze=.25,intensity_gain=2.4,exposure_profile='fixed',fixed_exposure_scale=.4,fixed_role_gains=(1.,.3,.03,.2),beam_edge_profile='angular',beam_edge_softness_deg=.6,beam_flux_reference_deg=14.,gpu_device=0,require_gpu=True,cam_pos=(0.,6.,11.),cam_target=(0.,2.,-3.),camera_scale=1.02); neutral=np.full((22,3),1000.,np.float32); dirs=np.zeros((N,63,3),np.float32); dirs[:,:13]=directions(angle[:,[0,1,2,3,4,5,6,7,0,2,4,5,7]]); dirs[:,13:]=[0.,-1.,0.]; colors=np.zeros((N,63,3),np.float32); colors[:,:13]=rgb[:,[0,1,2,3,4,5,6,7,0,2,4,5,7]]; colors[:,13:]=rgb[:,np.arange(50)%8]; dim=np.zeros((N,63),np.float32); dim[:,:13]=br[:,[0,1,2,3,4,5,6,7,0,2,4,5,7]]; dim[:,13:]=br[:,np.arange(50)%8]
 (OUT/'frames').mkdir(); outvideo=OUT/'demo.mp4'; cmd=['ffmpeg','-v','error','-nostdin','-y','-f','rawvideo','-pix_fmt','rgb24','-s','640x600','-r',str(FPS),'-i','pipe:0','-i',str(AUDIO),'-map','0:v:0','-map','1:a:0','-frames:v',str(N),'-c:v','libx264','-preset','veryfast','-pix_fmt','yuv420p','-c:a','aac','-b:a','128k','-t',str(N/FPS),'-movflags','+faststart',str(outvideo)]
 font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20); small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',12)
 with accelerator_lock():
  renderer=LitBodyStageRenderer(patch,cfg,body_offset=(0.,0.,0.),body_scale=1.,neutral_fill=0.)
  try:
   enc=subprocess.Popen(cmd,stdin=subprocess.PIPE)
   for i in range(N):
    view=renderer.render_controls(dirs[i],colors[i],dim[i],neutral); canvas=Image.new('RGB',(640,600),(7,10,16)); d=ImageDraw.Draw(canvas); d.text((12,8),'Foundation-v2 continuous attention | 10 Hz raw-output diagnostic',font=font,fill='white'); d.text((12,34),'model fixture-metadata anchor + bounded body/audio residual | no native DMX',font=small,fill='#c1d2e2'); canvas.paste(Image.fromarray(view),(0,0)); enc.stdin.write(canvas.tobytes());
    if i in (0,N//2,N-1): canvas.save(OUT/'frames'/f'frame_{i:04d}.png')
   enc.stdin.close();
   if enc.wait()!=0: raise RuntimeError('ffmpeg failed')
  finally: renderer.close()
 montage=Image.new('RGB',(960,360));
 for j,i in enumerate((0,N//2,N-1)): montage.paste(Image.open(OUT/'frames'/f'frame_{i:04d}.png').resize((320,300)),(320*j,0))
 montage.save(OUT/'frames/montage.png'); receipt={'schema_version':1,'status':'diagnostic_only','song':'3b4ab','duration_s':N/FPS,'fps':FPS,'video':str(outvideo),'video_sha256':sha(outvideo),'audio':str(AUDIO),'audio_sha256':sha(AUDIO),'checkpoint':str(SRC.parent/'model.pt'),'checkpoint_sha256':sha(SRC.parent/'model.pt'),'predictions':str(OUT/'PREDICTIONS_10HZ.npz'),'predictions_sha256':sha(OUT/'PREDICTIONS_10HZ.npz'),'patch':str(PATCH),'patch_sha256':sha(PATCH),'learned_beams':8,'presentation_lasers':5,'presentation_bulbs':50,'static_spatial_anchor':False,'model_spatial_anchor':True,'temporal_fallback':False,'body_conditioned':True,'native_body_generated':False,'native_dmx':False,'raw_safety_status':'passed','raw_audit':a,'frames_inspected':[str(OUT/'frames'/f'frame_{i:04d}.png') for i in (0,N//2,N-1)],'montage':str(OUT/'frames/montage.png'),'learned_collapse_audit':'/mnt/lightsem-data/experiments/lightsem/foundation_audio_light_v2_continuous_20261016_v9_warm_v7/COLLAPSE_AUDIT_V9.json','promotion':'diagnostic_only; representation collapse unqualified; no product promotion'}; (OUT/'DEMO.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n'); print(json.dumps(receipt,indent=2))
main()
