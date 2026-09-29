#!/usr/bin/env python3
"""Render an edited/baked .blend without regenerating its scene or animation.

Load the .blend before this script on Blender's CLI. Uses its native keyframes
and camera, and muxes the source audio at the embedded LightSem source offset.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import bpy


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--bitrate',default='6000k')
    p.add_argument('--maxrate',default='8000k')
    p.add_argument('--samples',type=int,default=32)
    args=p.parse_args(sys.argv[sys.argv.index('--')+1:])
    if args.output.exists():raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    scene=bpy.context.scene
    patch=json.loads(scene['Fixture patch'])
    binding=Path(scene['Source binding'])
    audio=Path(patch['timeline']['audio']['path'])
    if sha(audio)!=patch['timeline']['audio']['sha256']:
        raise ValueError('Source audio hash mismatch')
    if sha(binding/'timeline.npz')!=patch['timeline']['sha256']:
        raise ValueError('Source timeline hash mismatch')
    fps=scene.render.fps/scene.render.fps_base
    n=scene.frame_end-scene.frame_start+1
    start=float(scene['Source start seconds'])
    if scene.render.engine=='CYCLES':
        sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
        from lightsem.festival.realistic_scene import configure_cycles
        configure_cycles(scene,args.samples)
    else:
        scene.eevee.taa_render_samples=args.samples
    scene.render.image_settings.file_format='PNG'
    scene.render.image_settings.color_mode='RGB'
    scene.render.resolution_percentage=100
    scene.render.use_sequencer=False
    scene.render.threads_mode='FIXED'; scene.render.threads=8
    output=args.output.resolve(); partial=output.with_suffix('.partial.mp4')
    framepath=output.with_name(output.stem+'_working_frame.png')
    cmd=['ffmpeg','-hide_banner','-loglevel','warning','-y',
         '-f','image2pipe','-vcodec','png','-framerate',str(fps),'-i','pipe:0',
         '-ss',str(start),'-i',str(audio),'-map','0:v:0','-map','1:a:0',
         '-c:v','av1_nvenc','-preset','p7','-tune','uhq','-b:v',args.bitrate,
         '-maxrate',args.maxrate,'-bufsize','12000k','-pix_fmt','yuv420p',
         '-c:a','aac','-b:a','128k','-t',str(n/fps),'-movflags','+faststart',str(partial)]
    log=output.with_name(output.stem+'_encode.log').open('w')
    encoder=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=log)
    started=time.monotonic(); blend_hash=sha(bpy.data.filepath)
    try:
        for i,frame in enumerate(range(scene.frame_start,scene.frame_end+1)):
            scene.frame_set(frame)
            scene.render.filepath=str(framepath)
            bpy.ops.render.render(write_still=True)
            data=framepath.read_bytes(); encoder.stdin.write(data)
            if i==min(n-1,round(fps*12)):
                output.with_name(output.stem+'_poster.png').write_bytes(data)
            if i%30==0 or i==n-1:
                status={'status':'rendering','frame':i+1,'frames':n,'elapsed_s':time.monotonic()-started}
                output.with_name(output.stem+'_progress.json').write_text(json.dumps(status,indent=2)+'\n')
                print('PROGRESS',json.dumps(status),flush=True)
        encoder.stdin.close()
        if encoder.wait()!=0:raise RuntimeError('ffmpeg failed; inspect encoder log')
        partial.replace(output)
    except BaseException:
        encoder.terminate(); encoder.wait(); raise
    finally:
        log.close()
    framepath.unlink()
    receipt={'status':'complete','video':str(output),'blend':bpy.data.filepath,
        'blend_sha256_at_render':blend_hash,'sha256':sha(output),'frames':n,'fps':fps,
        'duration':n/fps,'start':start,'resolution':[scene.render.resolution_x,scene.render.resolution_y],
        'samples':args.samples,'render_seconds':time.monotonic()-started,'source_audio':str(audio),
        'source_audio_sha256':sha(audio),'binding_sha256':sha(binding/'timeline.npz'),
        'patch_sha256':sha(binding/'fixture_patch.json'),'binding_directory':str(binding),
        'blender':bpy.app.version_string,'baked':True,'command':cmd,
        'render_script_sha256':sha(__file__),
        'scope':'Native animation render, preserves saved scene edits; source DMX/audio binding authenticated'}
    output.with_name(output.stem+'_render.json').write_text(json.dumps(receipt,indent=2)+'\n')
    output.with_name(output.stem+'_progress.json').write_text(json.dumps({'status':'complete','frames':n})+'\n')
    print('VIDEO_COMPLETE',output,flush=True)


if __name__=='__main__':main()
