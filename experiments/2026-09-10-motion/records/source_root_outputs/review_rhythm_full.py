import subprocess
from pathlib import Path
from PIL import Image,ImageDraw
out=Path('/mnt/lightsem-data/experiments/lightsem/rhythm_companion_20260912T193657Z')
for second in (80,196):
 for arm in ('baseline','candidate'):
  path=out/f'full_{arm}_1080p_av1.mp4'
  dest=out/f'full_{arm}_{second:03d}_native.png'
  subprocess.run(['ffmpeg','-v','error','-nostdin','-n','-threads','2','-ss',str(second),'-i',str(path),'-frames:v','1','-threads','1',str(dest)],check=True)
for start in (76,194,264):
 sheet=Image.new('RGB',(1280,6*384),(16,16,16))
 for col,arm in enumerate(('baseline','candidate')):
  path=out/f'full_{arm}_1080p_av1.mp4'
  for i in range(6):
   second=start+i*.4
   raw=subprocess.check_output(['ffmpeg','-v','error','-nostdin','-threads','2','-ss',str(second),'-i',str(path),'-frames:v','1','-vf','scale=640:360','-pix_fmt','rgb24','-threads','1','-f','rawvideo','-'])
   tile=Image.frombytes('RGB',(640,360),raw);sheet.paste(tile,(col*640,i*384+24))
   ImageDraw.Draw(sheet).text((col*640+5,i*384+4),f'{arm} {second:.2f}s',fill='white')
 sheet.save(out/f'full_dense_{start:03d}.jpg',quality=94)
