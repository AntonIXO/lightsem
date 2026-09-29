#!/usr/bin/env python3
"""Build and render the festival in Blender, with optional native animation bake.

  /opt/blender-lightsem/blender -b -t 8 --python scripts/festival/render_festival.py -- \
    --binding runs/festival_garden/binding --output runs/festival_garden --still 100

Frames are rendered one at a time to a temporary PNG then piped to ffmpeg. No
multi-gigabyte PNG sequence is retained. The original source audio is trimmed
at precisely the same source timestamp used to sample the saved DMX.
"""
from __future__ import annotations

import argparse
import importlib
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

import bpy
import numpy as np
from lightsem.festival.patch import load_timeline, sample_timeline


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path,obj):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj,indent=2)+'\n'); temp.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binding',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--start',type=float,default=80.)
    p.add_argument('--duration',type=float,default=36.)
    p.add_argument('--fps',type=int,default=30)
    p.add_argument('--width',type=int,default=1920)
    p.add_argument('--height',type=int,default=1080)
    p.add_argument('--samples',type=int,default=64)
    p.add_argument('--style',choices=['photographic','legacy'],default='photographic')
    p.add_argument('--haze',type=float,default=.004)
    p.add_argument('--video-bitrate',default=None)
    p.add_argument('--max-video-bitrate',default=None)
    p.add_argument('--still',type=float)
    p.add_argument('--shot',choices=['journey','wide','floor'],default='journey')
    p.add_argument('--blackout',action='store_true')
    p.add_argument('--bake',action='store_true',help='Bake every DMX sample rendered into native Blender keyframes')
    p.add_argument('--build-only',action='store_true')
    p.add_argument('--name',default='lightsem_festival')
    args=p.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
    scene_module=importlib.import_module('lightsem.festival.'+('realistic_scene' if args.style=='photographic' else 'blender_scene'))
    build_scene,apply_dmx,set_camera=scene_module.build_scene,scene_module.apply_dmx,scene_module.set_camera
    args.output.mkdir(parents=True,exist_ok=True)
    # Protect the editable scene as well as the movie before any Blender save.
    targets=[args.output/(args.name+'.blend')]
    if not args.build_only:
        targets.append(args.output/(args.name+('.png' if args.still is not None else '.mp4')))
    for target in targets:
        if target.exists():
            raise FileExistsError(f'Refusing to overwrite {target}; use a new output name/directory')
    source_hashes={str(path.relative_to(ROOT)):sha(path) for path in
        list((ROOT/'lightsem/festival').glob('*.py'))+[Path(__file__)]}
    if any(x<=0 for x in (args.fps,args.width,args.height,args.samples,args.duration)):
        raise ValueError('Dimensions, fps, samples and duration must be positive')
    if args.width%2 or args.height%2:
        raise ValueError('Video dimensions must be even')
    patch=json.loads((args.binding/'fixture_patch.json').read_text())
    timeline=load_timeline(args.binding)
    audio=patch['timeline']['audio']['path']
    if not Path(audio).is_file() or sha(audio)!=patch['timeline']['audio']['sha256']:
        raise ValueError('Source audio is missing or changed since binding export')
    step=float(np.median(np.diff(timeline['time'])))
    source_duration=float(timeline['time'][-1])+step
    if args.start<0 or args.start+args.duration>source_duration+1e-6:
        raise ValueError('Requested interval extends beyond source DMX')
    if args.still is not None and not 0<=args.still<source_duration:
        raise ValueError('Still timestamp outside source DMX')
    print('BUILD',bpy.app.version_string,'fixtures',len(patch['fixtures']),flush=True)
    runtime=build_scene(patch,args.width,args.height,args.samples,args.haze)
    scene=runtime['scene']; scene.render.fps=args.fps
    scene.render.threads_mode='FIXED'; scene.render.threads=8
    n=round(args.duration*args.fps)
    scene.frame_start=1; scene.frame_end=n
    scene['Source start seconds']=args.start
    scene['Source binding']=str(args.binding.resolve())
    scene['Source DMX sha256']=sha(args.binding/'timeline.npz')
    scene['Fixture patch']=json.dumps(patch)
    scene['LightSem scene module']=scene_module.__name__
    # The editable file also has its soundtrack, aligned on the native timeline.
    # The MP4 mux below uses sample-accurate ffmpeg trimming independently.
    se=scene.sequence_editor_create()
    strip=se.strips.new_sound('Shpongle · source audio',str(Path(audio).resolve()),channel=1,
                              frame_start=1-round(args.start*args.fps))
    strip.frame_offset_start=round(args.start*args.fps)
    strip.frame_final_duration=n
    if args.bake:
        strip.sound.pack()
    scene.sync_mode='AUDIO_SYNC'
    scene.render.use_sequencer=False
    text=bpy.data.texts.new('READ ME · LightSem festival')
    text.write('LightSem festival previsualization\n\n'
        'Source DMX and audio are authenticated in the adjacent binding folder.\n'
        'DMX source time = source start seconds + (frame - 1) / FPS.\n'
        'All fixture patch entries retain original source-fixture and slot mapping.\n'
        'This is deterministic fanout, not new learned independent outputs.\n'
        'Virtual laser profiles are graphics controls; not physical scanner output.\n'+
        ('Cycles textured spots project onto passive fabric with physical shadows.\n' if args.style=='photographic'
         else 'Mapped procedural screen media is a projection approximation.\n')+
        'The media shader has its own playback clock; dimmer/color/aim/zoom come from DMX.\n'
        'For live local Art-Net use the Connect Art-Net text block.\n')
    live=bpy.data.texts.new('Connect Art-Net · run manually')
    live.write('import sys, json, bpy\n'
        f'sys.path.insert(0,{str(ROOT)!r})\n'
        'import importlib\n'
        "module=importlib.import_module(bpy.context.scene.get('LightSem scene module','lightsem.festival.blender_scene'))\n"
        'from lightsem.festival.blender_live import install\n'
        "patch=json.loads(bpy.context.scene['Fixture patch'])\n"
        'runtime=module.recover_runtime(patch)\n'
        "receiver=install(runtime,patch,bind='127.0.0.1',port=6454)\n")
    # Set a useful initial frame, and store it even if a later render fails.
    initial=args.start if args.still is None else args.still
    scene['Snapshot source seconds']=initial
    apply_dmx(runtime,sample_timeline(timeline,initial),initial,blackout=args.blackout)
    set_camera(runtime,0,args.duration,args.shot)
    for area in bpy.context.screen.areas:
        if area.type=='VIEW_3D':
            area.spaces.active.region_3d.view_perspective='CAMERA'
            area.spaces.active.shading.type='RENDERED'
    blend=args.output/(args.name+'.blend')
    bpy.ops.wm.save_as_mainfile(filepath=str(blend.resolve()),compress=True)
    if args.build_only:
        return
    if args.still is not None:
        dest=args.output/(args.name+'.png')
        scene.render.filepath=str(dest.resolve())
        before=time.monotonic(); bpy.ops.render.render(write_still=True)
        write_json(args.output/(args.name+'_still.json'),{
            'source_time':initial,'source_index':int(sample_timeline(timeline,initial)['source_index']),
            'render_seconds':time.monotonic()-before,'blackout':args.blackout,
            'resolution':[args.width,args.height],'samples':args.samples,
            'fixtures':len(patch['fixtures']),'image':str(dest.resolve())})
        print('STILL_COMPLETE',dest,flush=True)
        return
    output=args.output/(args.name+'.mp4')
    if output.exists():
        raise FileExistsError(output)
    staging=output.with_suffix('.partial.mp4')
    command=['ffmpeg','-hide_banner','-loglevel','warning','-y',
        '-f','image2pipe','-vcodec','png','-framerate',str(args.fps),'-i','pipe:0',
        '-ss',str(args.start),'-i',str(audio),'-map','0:v:0','-map','1:a:0',
        '-c:v','av1_nvenc','-preset','p7','-tune','uhq',
        '-b:v',args.video_bitrate or ('6000k' if args.style=='photographic' else '1100k'),
        '-maxrate',args.max_video_bitrate or ('8000k' if args.style=='photographic' else '1200k'),'-bufsize','12000k',
        '-pix_fmt','yuv420p','-c:a','aac','-b:a','128k','-t',str(n/args.fps),
        '-movflags','+faststart',str(staging.resolve())]
    framepath=args.output/(args.name+'_working_frame.png')
    log=(args.output/(args.name+'_encode.log')).open('w')
    encoder=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=log)
    started=time.monotonic(); indices=[]
    try:
        for i in range(n):
            elapsed=i/args.fps; t=args.start+elapsed; frame=i+1
            scene.frame_set(frame)
            controls=sample_timeline(timeline,t)
            apply_dmx(runtime,controls,t,frame,args.bake,blackout=args.blackout)
            set_camera(runtime,elapsed,args.duration,args.shot,frame,args.bake)
            scene.render.filepath=str(framepath.resolve())
            bpy.ops.render.render(write_still=True)
            data=framepath.read_bytes(); encoder.stdin.write(data)
            indices.append(int(controls['source_index']))
            if i==0:
                (args.output/(args.name+'_poster.png')).write_bytes(data)
            if i%(10 if args.style=='photographic' else 30)==0 or i==n-1:
                progress={'status':'rendering','frame':frame,'frames':n,
                    'elapsed_s':round(time.monotonic()-started,2),'source_time':t}
                write_json(args.output/(args.name+'_progress.json'),progress)
                print('PROGRESS',json.dumps(progress),flush=True)
        encoder.stdin.close()
        if encoder.wait()!=0:
            raise RuntimeError('ffmpeg failed; inspect encode log')
        staging.replace(output)
    except BaseException:
        encoder.terminate(); encoder.wait()
        raise
    finally:
        log.close()
    if framepath.exists():
        framepath.unlink()
    # Native keyframes survive reopening without allowing Python autoexecution.
    if args.bake:
        for action in bpy.data.actions:
            for fc in action.fcurves:
                for kp in fc.keyframe_points:
                    kp.interpolation='CONSTANT'
        # Camera alone uses smooth interpolation across its per-frame samples.
        action=runtime['camera'].animation_data.action
        for fc in action.fcurves:
            for kp in fc.keyframe_points:
                kp.interpolation='LINEAR'
    scene.frame_set(1)
    bpy.ops.wm.save_as_mainfile(filepath=str(blend.resolve()),compress=True)
    receipt={'status':'complete','video':str(output.resolve()),'blend':str(blend.resolve()),
        'sha256':sha(output),'frames':n,'fps':args.fps,'duration':n/args.fps,'start':args.start,
        'resolution':[args.width,args.height],'samples':args.samples,'render_seconds':time.monotonic()-started,
        'source_audio':str(audio),'source_audio_sha256':sha(audio),
        'binding_sha256':sha(args.binding/'timeline.npz'),'patch_sha256':sha(args.binding/'fixture_patch.json'),
        'source_index_first':indices[0],'source_index_last':indices[-1],
        'blender':bpy.app.version_string,'baked':args.bake,'command':command,'style':args.style,
        'source_files':source_hashes,
        'scope':'Uncalibrated festival previsualization driven by accepted C DMX; deterministic fixture replication'}
    write_json(args.output/(args.name+'_render.json'),receipt)
    write_json(args.output/(args.name+'_progress.json'),{'status':'complete','frames':n})
    print('VIDEO_COMPLETE',output,flush=True)


if __name__=='__main__':
    main()
