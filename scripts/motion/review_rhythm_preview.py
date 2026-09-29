from pathlib import Path
import subprocess
from PIL import Image,ImageDraw
out=Path('/mnt/lightsem-data/experiments/lightsem/rhythm_companion_20260912T193657Z')
for start in (0,70,190):
    sheet=Image.new('RGB',(1280,6*384),(16,16,16))
    for col,arm in enumerate(('baseline','companion')):
        movie=out/f'preview_{arm}_{start:03d}.mp4'
        for i,sec in enumerate([0,0.25,0.5,0.75,1,1.25]):
            data=subprocess.check_output(['ffmpeg','-v','error','-threads','2','-ss',str(sec+2),'-i',str(movie),'-frames:v','1','-vf','scale=640:360','-pix_fmt','rgb24','-threads','1','-f','rawvideo','-'])
            im=Image.frombytes('RGB',(640,360),data)
            sheet.paste(im,(col*640,i*384+24))
            ImageDraw.Draw(sheet).text((col*640+8,i*384+4),f'{arm} source {start+sec+2:.2f}s',fill='white')
    sheet.save(out/f'preview_dense_{start:03d}.jpg',quality=94)
