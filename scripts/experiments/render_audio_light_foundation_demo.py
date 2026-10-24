"""Render a short audio-only demo from the privileged-dance light checkpoint."""
from __future__ import annotations
import argparse, hashlib, json, subprocess
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageFilter

from lightsem.experiments.audio_light_foundation.data import load_row, make_fixture_metadata, sha
from lightsem.experiments.audio_light_foundation.model import AudioToLightFoundation, AudioToLightConfig


def main():
    p=argparse.ArgumentParser(); p.add_argument('--song',default='3b4ab'); p.add_argument('--frames',type=int,default=128); p.add_argument('--sample-rate',type=int,choices=(10,30),default=10); p.add_argument('--out',required=True); p.add_argument('--device',default='auto'); p.add_argument('--checkpoint',default='/mnt/lightsem-data/experiments/lightsem/audio_light_foundation_privileged_dance_20261015_v4/target_plus_dance_aux/model.pt'); p.add_argument('--label',default='Audio→Light foundation v2'); args=p.parse_args()
    device=torch.device('cuda' if args.device=='auto' and torch.cuda.is_available() else ('cpu' if args.device=='auto' else args.device))
    checkpoint=Path(args.checkpoint)
    row=load_row(args.song,sample_rate_hz=args.sample_rate); n=min(args.frames,len(row['audio']));
    audio=torch.from_numpy(row['audio'][:n][None]).float().to(device)
    fixtures=torch.from_numpy(make_fixture_metadata())[None].float().to(device)
    ts=torch.arange(n,device=device,dtype=torch.float32)[None]/float(args.sample_rate)
    state=torch.load(checkpoint,map_location=device,weights_only=False)
    model=AudioToLightFoundation(AudioToLightConfig(audio_dim=audio.shape[-1],local_rate_hz=float(args.sample_rate))).to(device).eval()
    model.load_state_dict(state['state_dict'],strict=True)
    # The foundation recipe is trained on fixed local windows.  Carrying the
    # GRU/latent state through an entire multi-minute song was out of
    # distribution and caused a slow state drift into spatial aim collapse.
    # Re-enter the exact training-window contract for each chunk: audio is
    # still the only inference input and no motion is invented at boundaries.
    chunk_frames = 384 if args.sample_rate == 30 else 128
    pieces={k:[] for k in ('aim','speed','acceleration','rgb','brightness','event')}
    with torch.no_grad():
        for start in range(0,n,chunk_frames):
            end=min(n,start+chunk_frames); m=end-start
            chunk_ts=torch.arange(m,device=device,dtype=torch.float32)[None]/float(args.sample_rate)
            chunk_audio=audio[:,start:end]
            chunk_valid=torch.ones((1,m),dtype=torch.bool,device=device)
            out=model(chunk_audio,fixtures,timestamps=chunk_ts,valid=chunk_valid,state=None,duration_seconds=chunk_frames/float(args.sample_rate))
            for k in pieces: pieces[k].append(out[k][0].detach().cpu().numpy())
    pred={k:np.concatenate(v,axis=0) for k,v in pieces.items()}
    for k,v in pred.items():
        if not np.isfinite(v).all(): raise RuntimeError(f'nonfinite prediction: {k}')
    root=Path(args.out); root.mkdir(parents=True,exist_ok=False); (root/'frames').mkdir()
    np.savez_compressed(root/'PREDICTIONS.npz',**pred,time=np.arange(n,dtype=np.float32)/float(args.sample_rate))
    fps=30; count=round(n*fps/float(args.sample_rate)); w,h=1280,720
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',22); small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
    pipe=subprocess.Popen(['ffmpeg','-v','error','-nostdin','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{w}x{h}','-r',str(fps),'-i','pipe:0','-i',str(Path('/mnt/lightsem-data/beatsaver/wav')/f'{args.song}.wav'),'-map','0:v:0','-map','1:a:0','-t',str(count/fps),'-c:v','libx264','-preset','veryfast','-pix_fmt','yuv420p','-c:a','aac','-b:a','128k','-movflags','+faststart',str(root/'demo.mp4')],stdin=subprocess.PIPE)
    for frame in range(count):
        t=frame/fps; x=t*float(args.sample_rate); i=min(n-1,int(x)); j=min(n-1,i+1); a=x-i; aim=pred['aim'][i]*(1-a)+pred['aim'][j]*a; rgb=pred['rgb'][i]*(1-a)+pred['rgb'][j]*a; bright=pred['brightness'][i,:,0]*(1-a)+pred['brightness'][j,:,0]*a; speed=pred['speed'][i,:,0]*(1-a)+pred['speed'][j,:,0]*a; event=1/(1+np.exp(-float(pred['event'][i,0])))
        canvas=Image.new('RGB',(w,h),(5,7,14)); glow=Image.new('RGBA',(w,h),(0,0,0,0)); gd=ImageDraw.Draw(glow)
        origins=[]
        for k in range(8):
            ox=int(90+k*155); oy=120; tx=int(w*(.5+.38*float(aim[k,0]))); ty=int(570-.30*float(aim[k,1])*h/2)
            # Display-only tone mapping makes low learned RGB values visible;
            # saved predictions and control diagnostics remain unchanged.
            display_rgb=np.power(np.clip(rgb[k],0,1),.45)
            alpha=int(70+185*float(bright[k])); col=tuple(np.clip(display_rgb*255,0,255).astype(np.uint8).tolist())+(alpha,); width=int(4+min(16,float(speed[k])*10))
            dx,dy=tx-ox,ty-oy; norm=max((dx*dx+dy*dy)**.5,1); px,py=-dy/norm*width,dx/norm*width
            gd.polygon([(ox-px,oy-py),(ox+px,oy+py),(tx+px*.35,ty+py*.35),(tx-px*.35,ty-py*.35)],fill=col)
            gd.line((ox,oy,tx,ty),fill=tuple(np.clip(display_rgb*255,0,255).astype(np.uint8).tolist())+(min(255,alpha+35),),width=max(2,width//2))
            origins.append((ox,oy,col[:3],int(8+12*bright[k])))
        glow=glow.filter(ImageFilter.GaussianBlur(4)); canvas=Image.alpha_composite(canvas.convert('RGBA'),glow); draw=ImageDraw.Draw(canvas)
        for ox,oy,col,r in origins:
            draw.ellipse((ox-r,oy-r,ox+r,oy+r),fill=col+(255,),outline='white',width=1)
        draw.text((24,20),args.label,font=font,fill='white'); draw.text((24,52),'audio-only inference | dance teacher training-only | no body input',font=small,fill='#b9c9df'); draw.line((40,590,1240,590),fill='#26354b',width=2); draw.text((24,680),f'{args.song}  t={t:05.2f}s  event={event:.2f}  mean brightness={float(bright.mean()):.2f}  display gamma=.45',font=small,fill='#b9c9df')
        pipe.stdin.write(canvas.convert('RGB').tobytes())
        if frame in (0,count//2,count-1): canvas.convert('RGB').save(root/'frames'/f'frame_{frame:04d}.png')
    pipe.stdin.close(); rc=pipe.wait()
    if rc: raise RuntimeError(f'ffmpeg failed: {rc}')
    frame_files=sorted((root/'frames').glob('*.png')); imgs=[Image.open(x).resize((640,360)) for x in frame_files]; montage=Image.new('RGB',(640*len(imgs),360));
    for k,im in enumerate(imgs): montage.paste(im,(k*640,0))
    montage.save(root/'frames'/'montage.png')
    result={'schema_version':1,'status':'diagnostic_only','song':args.song,'audio':str(Path('/mnt/lightsem-data/beatsaver/wav')/f'{args.song}.wav'),'audio_sha256':sha(Path('/mnt/lightsem-data/beatsaver/wav')/f'{args.song}.wav'),'checkpoint':str(checkpoint),'checkpoint_sha256':sha(checkpoint),'prediction_path':str(root/'PREDICTIONS.npz'),'video':str(root/'demo.mp4'),'video_sha256':sha(root/'demo.mp4'),'frames':count,'fps':fps,'duration_s':count/fps,'sample_rate_hz':args.sample_rate,'inference_chunk_frames':chunk_frames,'state_reset_per_chunk':True,'inference_modalities':['MuQ layer 9','Base92','fixture metadata'],'dance_input_at_inference':False,'learned_outputs':['aim','speed','acceleration','rgb','brightness','event'],'no_native_dmx':True,'no_native_body':True,'inspection_montage':str(root/'frames/montage.png'),'promotion':'forbidden'}
    (root/'DEMO.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n'); print(json.dumps(result,indent=2))

if __name__=='__main__': main()
