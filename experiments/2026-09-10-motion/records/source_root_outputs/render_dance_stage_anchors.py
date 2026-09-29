import json,math,subprocess
from pathlib import Path
import numpy as np
import cv2
from lightsem.conductor.rig import canonical_rig
from lightsem.sim.renderer import StageRenderer,RenderConfig
from lightsem.experiments.stage11_common import gpu_lock
ROOT=Path('/mnt/lightsem-data/experiments/lightsem/motion_native_audio_20260910')
OUT=Path('/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/stage_geometry_02')
p=json.loads((ROOT/'delivery/surface_revision_02/protocol.json').read_text());cfg=p['renderer'];cfg.update(width=640,height=360)
with np.load(OUT.parent/'stage_geometry_01/candidate.npz') as z:old=z['dmx']
with np.load(OUT/'candidate.npz') as z:new=z['dmx']
path=OUT/'before_after.mp4';assert not path.exists()
with gpu_lock(ROOT):
 renderer=StageRenderer(canonical_rig(),RenderConfig(**cfg))
 cmd=['ffmpeg','-v','error','-nostdin','-n','-f','rawvideo','-pix_fmt','rgb24','-s','1280x360','-r','30','-i','pipe:0','-i','/root/shpongle.flac','-map','0:v:0','-map','1:a:0','-c:v','av1_nvenc','-preset','p5','-cq','23','-b:v','0','-pix_fmt','yuv420p','-c:a','aac','-t',str(p['duration']),'-shortest',str(path)]
 enc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=subprocess.PIPE);samples=[]
 try:
  for i in range(math.ceil(p['duration']*30)):
   idx=min(round(i*40/30),len(old)-1)
   imgs=[renderer.render_frame(d[idx],0.).copy() for d in (old,new)]
   for im,label in zip(imgs,['STAGE CHART ORIGINAL ANCHORS','STAGE CHART CALIBRATED ANCHORS']):cv2.putText(im,label,(12,22),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1,cv2.LINE_AA)
   frame=np.concatenate(imgs,1);enc.stdin.write(frame.tobytes())
   if i%30==0:samples.append(cv2.resize(frame,(640,180)))
   if i%900==0:print('render',i/30,flush=True)
 finally:
  enc.stdin.close();err=enc.stderr.read().decode();code=enc.wait();renderer.ctx.release()
 if code:raise RuntimeError(err)
 for start in range(0,len(samples),30):
  tiles=samples[start:start+30];tiles += [np.zeros_like(tiles[0])]*(30-len(tiles))
  sheet=np.concatenate([np.concatenate(tiles[j:j+3],1) for j in range(0,30,3)],0)
  cv2.imwrite(str(OUT/f'review_{start:03d}.jpg'),cv2.cvtColor(sheet,cv2.COLOR_RGB2BGR))
 (OUT/'render.json').write_text(json.dumps({'path':str(path),'generated_frames':i+1,'duration':p['duration'],'sample_hz':1,'samples':len(samples),'renderer':cfg},indent=2));print('COMPLETE',path,flush=True)
