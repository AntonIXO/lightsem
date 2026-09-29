import json,subprocess
from pathlib import Path
import numpy as np
from run_rhythm_companion import camera_project,render_config,read_exact
from lightsem.conductor.rig import canonical_rig
out=Path('/mnt/lightsem-data/experiments/lightsem/rhythm_companion_20260912T193657Z');video=out/'full_comparison_1080p_av1.mp4';rig=canonical_rig();cfg=render_config(960,540)
y,x=np.mgrid[:540,:960];mask=np.ones((540,960),bool);mask[:48]=False
for bx,by in camera_project([f.position for f in rig.fixtures],cfg):mask&=((x-bx)**2+(y-by)**2>30**2)
cmd=['ffmpeg','-v','error','-nostdin','-threads','4','-i',str(video),'-an','-vf','fps=1,crop=1920:540:0:180','-pix_fmt','rgb24','-threads','1','-f','rawvideo','-']
rows=[]
with (out/'comparison_pixel_decode.log').open('xb') as err:
 proc=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=err)
 while True:
  raw=read_exact(proc.stdout,1920*540*3)
  if not raw:break
  if len(raw)!=1920*540*3:raise RuntimeError('Partial frame')
  image=np.frombuffer(raw,np.uint8).reshape(540,1920,3);values=[]
  for col in range(2):
   rgb=image[:,col*960:(col+1)*960];luma=rgb[:,:,0]*.2126+rgb[:,:,1]*.7152+rgb[:,:,2]*.0722
   values.append(float(luma[mask].mean()))
  rows.append(values)
 if proc.wait():raise RuntimeError('Decode failed')
a=np.asarray(rows);means=a.mean(0)
result={'nominal_1hz_decoded_samples':len(a),'outside_bulbs_and_caption_mean_luma':means.tolist(),
 'candidate_over_baseline':float(means[1]/means[0]),'relative_residual_percent':float(100*(means[1]/means[0]-1)),
 'scope':'Descriptive 1Hz samples from final decoded comparison; different mask/caption exclusion than full raw1080p means; not exact fullpixel mean equality.'}
(out/'comparison_brightness_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
subprocess.run(['ffmpeg','-v','error','-nostdin','-n','-threads','2','-ss','80','-i',str(video),'-frames:v','1','-threads','1',str(out/'comparison_080.png')],check=True)
subprocess.run(['ffmpeg','-v','error','-nostdin','-n','-threads','2','-ss',str(8318/30),'-i',str(out/'full_candidate_1080p_av1.mp4'),'-frames:v','1','-threads','1',str(out/'candidate_final_frame.png')],check=True)
