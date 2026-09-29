"""Post-hoc diagnostic. Oracle motion is NOT a music-only product candidate."""
import json,subprocess,hashlib
from pathlib import Path
import numpy as np
import torch
from lightsem.experiments.motion_study.lighting_map import compose,LightingConfig
from lightsem.experiments.motion_study.surface_revision import mechanics
from lightsem.experiments.motion_study.spatial_observer import spatial_series,summary
from lightsem.conductor.rig import canonical_rig
from lightsem.sim.renderer import StageRenderer,RenderConfig
from lightsem.experiments.stage11.stage11_common import gpu_lock
ROOT=Path('/mnt/lightsem-data/experiments/lightsem/motion_native_audio_20260910')
OUT=Path('/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/oracle')
OUT.mkdir(exist_ok=True)
torch.set_num_threads(4)
manifest=json.loads((ROOT/'data/audio_manifest.json').read_text())
review=json.loads(Path('/mnt/lightsem-data/experiments/lightsem/motion_representation_20260910/data/manifest.json').read_text())['review_clips']
# First fixed member of each pre-existing review category, no selection from outputs.
clips=[next(c for c in review if c['category']==cat) for cat in ('holds','travel','coordination')]
cfg=LightingConfig(target_space='articulated_surface',surface_gain=1.8,torso_gain=2.,target_smoothing_s=.20,tracking_s=.18)
rconf=json.loads((ROOT/'delivery/surface_revision_02/protocol.json').read_text())['renderer']
rconf.update(width=640,height=360)
rig=canonical_rig();rows=[]
with gpu_lock(ROOT):
 renderer=StageRenderer(rig,RenderConfig(**rconf))
 try:
  for clip in clips:
   key=clip['key'];entry=next(e for e in manifest['entries'] if e['key']==key)
   assert entry['partition']=='val'
   with np.load(entry['path']) as z:real=z['features']
   with np.load(ROOT/f'music/full_sequences/flow/seed_0/{key}.npz') as z:gen=z['features']
   assert real.shape==gen.shape
   loud={'time':np.array([0.,len(real)/30]),'momentary':np.array([-20.,-20.]),'short_term':np.array([-20.,-20.])}
   streams=[];metrics={}
   for arm,features in [('oracle',real),('generated',gen)]:
    dmx,trace,report=compose(features,loud,len(real)/30,config=cfg,dynamics=False)
    lo=clip['start']/30;hi=lo+clip['frames']/30
    # Match every non-angle byte, including zoom/color/dimmer, across bodies.
    if streams:
     mask=np.ones(512,bool)
     for f in rig.by_kind('moving_head'):mask[f.address-1:f.address+3]=False
     dmx[:,mask]=streams[0][:,mask]
    streams.append(dmx)
    s=spatial_series(dmx,rig)
    metrics[arm]={'mechanics':mechanics(dmx,lo,hi),'spatial':summary(s,lo,hi)}
    np.savez_compressed(OUT/f'{key}_{arm}.npz',dmx=dmx)
   name=OUT/f'{clip["category"]}_{key}_oracle_left_generated_right.mp4'
   if name.exists():raise ValueError('No overwriting diagnostic movies')
   cmd=['ffmpeg','-v','error','-nostdin','-n','-f','rawvideo','-pix_fmt','rgb24','-s','1280x360','-r','30','-i','pipe:0','-ss',str(lo),'-i',entry['wav_path'],'-map','0:v:0','-map','1:a:0','-c:v','libx264','-preset','fast','-crf','19','-pix_fmt','yuv420p','-c:a','aac','-t',str(hi-lo),'-shortest',str(name)]
   enc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
   import cv2
   try:
    for i in range(clip['frames']):
     idx=round((lo+i/30)*40)
     imgs=[renderer.render_frame(s[idx],0.).copy() for s in streams]
     for im,label in zip(imgs,['REAL DANCE ORACLE (not playback)','SAVED AUDIO-GENERATED MOTION']):
      cv2.putText(im,label,(12,24),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1,cv2.LINE_AA)
     enc.stdin.write(np.concatenate(imgs,1).tobytes())
     if i in (0,150,300,450):cv2.imwrite(str(OUT/f'{key}_{i}.png'),cv2.cvtColor(np.concatenate(imgs,1),cv2.COLOR_RGB2BGR))
   finally:
    enc.stdin.close();err=enc.stderr.read().decode();code=enc.wait()
   if code:raise RuntimeError(err)
   rows.append({'clip':clip,'metrics':metrics,'movie':str(name)})
   (OUT/'report.json').write_text(json.dumps({'rows':rows,'scope':'Unpreregistered same-mapper capability diagnostic. Source oracle not music-only inference. Fixed .7 activity; all nonangle bytes copied from oracle to generated. Full motion mapped before cropping; fixed preexisting first review clip per category. No TEST or model fitting.','renderer':rconf},indent=2))
   print('COMPLETE',name,flush=True)
 finally:renderer.ctx.release()
