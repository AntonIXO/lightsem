"""Chronological review of rendered evidence, never a silent visual pass.

Requires completed movies. OpenCV decodes all30Hz frames. Contact sheets retain
2Hz across the whole track plus8Hz in declared important intervals. This is
sampled visual evidence, not continuous audiovisual viewing by a human.
"""
import argparse,json,math,subprocess
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw
from lightsem.experiments.stage11.stage11_common import digest,authenticated_json


def review(path,out):
    out.mkdir(parents=True,exist_ok=True)
    info=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_streams','-of','json',str(path)]))['streams'][0]
    a,b=map(int,info['r_frame_rate'].split('/'));fps=a/b;width=info['width'];height=info['height']
    # OpenCV's bundled AV1 decoder cannot decode these files on this host.
    # Use FFmpeg libdav1d explicitly: CPU only, not concurrent GPU work.
    cap=subprocess.Popen(['ffmpeg','-v','error','-threads','2','-c:v','libdav1d','-i',str(path),
        '-an','-f','rawvideo','-pix_fmt','rgb24','pipe:1'],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    rows=[];shots=[];dense=[];previous=None;slow=[];i=0
    intervals=[(16,22),(60,76),(78,86),(108,118),(188,200),(265,277.31)]
    # Fixed crop includes complete truss, wall and stage. Not per-frame zoom.
    crop=(round(width*.23),round(height*.20),round(width*.77),round(height*.95))
    while True:
        raw=cap.stdout.read(width*height*3)
        if not raw:break
        if len(raw)!=width*height*3:raise ValueError('Partial decoded frame')
        t=i/fps;rgb=np.frombuffer(raw,np.uint8).reshape(height,width,3)
        luma=(rgb@np.array([.2126,.7152,.0722],np.float32))/255
        a=cv2.resize(luma,(320,180),interpolation=cv2.INTER_AREA)
        norm=np.clip(a/max(float(np.quantile(a,.99)),.08),0,1)
        row={'time':t,'luma':float(a.mean()),'white_clipped':float((rgb.min(-1)>=250).mean()),
             'support_fraction':float((a>.01).mean()),'normalized_support_fraction':float((norm>.15).mean())}
        if previous is not None:row['normalized_change_1frame']=float(abs(norm-previous).mean())
        slow.append(norm)
        if len(slow)>round(fps*2):
            older=slow.pop(0);row['normalized_change_2s']=float(abs(norm-older).mean())
        previous=norm;rows.append(row)
        take=i%round(fps/2)==0
        # 30Hz supports alternating3/4-frame spacing near8Hz, not exact8Hz.
        dense_take=any(lo<=t<hi for lo,hi in intervals) and round(i*8/fps)!=round((i-1)*8/fps)
        if take or dense_take:
            x0,y0,x1,y1=crop
            image=Image.fromarray(rgb[y0:y1,x0:x1]);image=image.resize((300,234))
            if take:shots.append((t,image))
            if dense_take:dense.append((t,image))
        i+=1
    error=cap.stderr.read().decode();code=cap.wait();cap.stdout.close();cap.stderr.close()
    if code:raise RuntimeError('FFmpeg decode failed: '+error)
    if i<8300:raise ValueError('Incomplete full-track decode')
    sheets=[]
    for label,frames in [('full_2hz',shots),('dense_8hz',dense)]:
        for start in range(0,len(frames),16):
            chunk=frames[start:start+16];sheet=Image.new('RGB',(1200,4*258),(12,12,12));pen=ImageDraw.Draw(sheet)
            for k,(t,image) in enumerate(chunk):
                x,y=k%4*300,k//4*258;pen.text((x+4,y+3),f'{path.stem} | {t:07.3f}s',fill='white');sheet.paste(image,(x,y+24))
            dest=out/f'{label}_{start//16:03d}.jpg';sheet.save(dest,quality=90)
            sheets.append({'path':str(dest),'kind':label,'start':chunk[0][0],'end':chunk[-1][0],'frames':len(chunk)})
    report={'path':str(path),'sha256':digest(path),'decoded_frames':i,'fps':fps,'duration':i/fps,
            'width':width,'height':height,'crop':crop,'sheets':sheets,'rows':rows,
            'visual_pass':None,'reviewer_sha256':digest(__file__),'decoder':'FFmpeg libdav1d CPU','scope':'All-frame automated checks and sampled chronological contact sheets; no music-fit or human acceptance rating.'}
    authenticated_json(out/'report.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('rows','sheets')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('video',type=Path);p.add_argument('out',type=Path);a=p.parse_args();review(a.video,a.out)
