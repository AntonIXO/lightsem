#!/usr/bin/env python3
"""Reproducible offline companion-light study and native1080p AV1 delivery.

Run audio with bash run.sh; map/render/audit with bash run_torch.sh.
No training, new motion inference, source retiming, or hardware DMX output.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def save(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False)


def arrays(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def guard(out):
    if shutil.disk_usage(out).free<1024**3+32*1024**2:raise RuntimeError('Storage reserve reached; partial outputs retained')
    size=sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
    if size>1536*1024**2:raise RuntimeError('Study1.5GiB budget reached; partial outputs retained')


def authenticate(out,plan):
    freeze=out/'implementation.json'
    if freeze.exists():
        receipt=json.loads(freeze.read_text())
        revisions=out/'tooling_revisions.json'
        if revisions.exists():
            for revision in json.loads(revisions.read_text()):
                if receipt['files'][revision['path']]!=revision['old_sha256']:
                    raise RuntimeError('Tooling revision chain mismatch')
                receipt['files'][revision['path']]=revision['new_sha256']
        for p,h in receipt['files'].items():
            if digest(p)!=h:raise RuntimeError('Frozen input/code changed: '+p)
        return receipt
    if not (out/'registration.json').exists():raise RuntimeError('Preregister first')
    repo=Path(__file__).resolve().parent
    paths=[repo/'lightsem/audio/rhythm_clock.py',repo/'lightsem/audio/beatgrid.py',
        repo/'lightsem/experiments/motion_study/companion.py',Path(__file__),out/'plan.json']
    paths+=list((repo/'lightsem/experiments/motion_study').glob('*.py'))
    paths+=list((repo/'lightsem/sim').glob('*.py'))
    paths += [Path(plan[k]) for k in ('source_world','source_dmx','loudness','audio')]
    receipt={'files':{str(p):digest(p) for p in paths},'frozen_unix':time.time(),'base_commit':'01a14fe'}
    save(freeze,receipt);return receipt


def audio_stage(out,p):
    import soundfile as sf
    from lightsem.audio.rhythm_clock import clock_from_audio
    y,sr=sf.read(p['audio'],dtype='float32',always_2d=True)
    clock,report=clock_from_audio(y,sr)
    clock.save(out/'rhythm_clock.npz');save(out/'audio_report.json',report)
    print(json.dumps(report),flush=True)


def angles(frames,rig):
    values=[]
    for f in sorted(rig.by_kind('moving_head'),key=lambda f:(f.position[0],f.name)):
        a=f.address-1;d=frames.astype(np.int64)
        ticks=np.stack((d[:,a]*256+d[:,a+1],d[:,a+2]*256+d[:,a+3]),-1)
        values.append((ticks/65535-.5)*np.array([f.pan_range_deg,f.tilt_range_deg]))
    return np.stack(values,axis=1)


def nonangle_mask(rig):
    m=np.ones(512,bool)
    for f in rig.by_kind('moving_head'):m[f.address-1:f.address+3]=False
    return m


def numeric(d,rig,p):
    from lightsem.experiments.quality_observer import beam_series
    from lightsem.experiments.motion_study.spatial_observer import first_hits
    a=angles(d,rig);v=np.linalg.norm(np.diff(a,axis=0)*40,axis=-1)
    acceleration=np.linalg.norm(np.diff(a,n=2,axis=0)*40**2,axis=-1)
    directions,levels=beam_series(d,rig)
    hit,surface,_=first_hits(directions,np.array([f.position for f in rig.by_kind('moving_head')]))
    valid=(surface>=0).all(1)
    if not valid.all():raise RuntimeError('Invalid emitted finite-stage hit; do not drop samples')
    result={'speed_max':float(v.max()),'acceleration_max':float(acceleration.max()),'valid_frame_fraction':float(valid.mean()),'intervals':{}}
    for name,(lo,hi) in p['intervals'].items():
        sl=slice(round(lo*40),round(hi*40)-1);h=hit[round(lo*40):round(hi*40)]
        spread=np.sqrt(np.mean(np.sum((h-h.mean(axis=1,keepdims=True))**2,axis=-1),axis=1))
        result['intervals'][name]={'coordinate_speed_median':float(np.median(v[sl])),
            'hit_x_span_p10_m':float(np.quantile(np.ptp(h[:,:,0],axis=1),.1)),
            'six_head_rms_spread_min_m':float(spread.min()),
            'floor_fraction':float((surface[round(lo*40):round(hi*40)]==0).mean()),
            'wall_fraction':float((surface[round(lo*40):round(hi*40)]==1).mean())}
    return result,a,v


def map_stage(out,p):
    import torch
    torch.set_num_threads(4)
    from lightsem.audio.rhythm_clock import RhythmClock
    from lightsem.experiments.motion_study.companion import (compose_companion_world,default_lighting_config,CompanionConfig)
    from lightsem.experiments.motion_study.lighting_map import compose_world_joints
    from lightsem.conductor.rig import canonical_rig
    rig=canonical_rig();world=arrays(p['source_world'])['world'];copy=world.copy()
    curve=arrays(p['loudness']);clock=RhythmClock.load(out/'rhythm_clock.npz')
    cfg=default_lighting_config();bundle=compose_world_joints(world,curve,p['duration_s'],config=cfg,rig=rig)
    cached=arrays(p['source_dmx'])['candidate']
    if not np.array_equal(cached,bundle[0]):raise RuntimeError('Baseline differs from previous pose-stage DMX')
    streams={'baseline':bundle[0]};reports={};traces={}
    for name,cc,static in [('companion',CompanionConfig(),False),('shifted_clock',CompanionConfig(phase_offset_beats=.5),False),('static_style',CompanionConfig(),True)]:
        print('MAPPING',name,flush=True);guard(out)
        d,tr,r=compose_companion_world(world,curve,p['duration_s'],clock,lighting_config=cfg,
            companion_config=cc,rig=rig,baseline_bundle=bundle,static_style=static)
        if not np.array_equal(d[:,nonangle_mask(rig)],cached[:,nonangle_mask(rig)]):raise RuntimeError('Photometry changed')
        streams[name]=d;traces[name]=tr;reports[name]=r
    if not np.array_equal(world,copy):raise RuntimeError('Saved body changed')
    results={};all_angles={};speeds={}
    for name,d in streams.items():results[name],all_angles[name],speeds[name]=numeric(d,rig,p)
    active=slice(70*40,90*40)
    ratio=results['companion']['intervals']['active']['coordinate_speed_median']/max(results['baseline']['intervals']['active']['coordinate_speed_median'],1e-12)
    phase_rms=float(np.sqrt(np.mean((all_angles['companion'][active]-all_angles['shifted_clock'][active])**2)))
    style_rms=float(np.sqrt(np.mean((traces['companion']['raw_targets'][active]-traces['static_style']['raw_targets'][active])**2)))
    quiet=results['companion']['intervals']['quiet']['coordinate_speed_median'];quiet_base=results['baseline']['intervals']['quiet']['coordinate_speed_median']
    # Descriptive detector association with its own audio onset evidence. A
    # phase-shift null detects phase sensitivity; NOT independent beat ground truth.
    bt=clock.beat_times;bt=bt[(bt>=70)&(bt<90)]
    period=float(np.median(np.diff(bt)))
    at=np.interp(bt,clock.time,clock.onset).mean();off=np.interp(bt+.5*period,clock.time,clock.onset).mean()
    gates={'mechanics':all(r['speed_max']<=90+1e-7 and r['acceleration_max']<=360+1e-7 for r in results.values()),
        'source':True,'photometry':True,'independence':phase_rms>=1 and style_rms>=.1,
        'quiet':quiet<=max(3,1.2*quiet_base),'active_motion_prediction':ratio>=1.25}
    row={'plan':p['id'],'results':results,'active_speed_ratio':ratio,'shifted_clock_angle_rms_deg':phase_rms,
        'static_style_target_rms_m':style_rms,'gates':gates,'baseline_exact':True,'source_unchanged':True,
        'detector_onset_association':{'at_beats':float(at),'half_beat_shift':float(off),'scope':'Same detector evidence; not independent annotations or music-fit'},
        'mapping_reports':reports,'scope':p['scope']}
    np.savez_compressed(out/'streams.npz',**streams,time=bundle[1]['time'])
    np.savez_compressed(out/'trace.npz',**traces['companion'],baseline_speed=speeds['baseline'],candidate_speed=speeds['companion'])
    save(out/'numeric_report.json',row);print(json.dumps(row,indent=2),flush=True)


def render_config(width=1920,height=1080):
    from lightsem.sim.renderer import RenderConfig
    return RenderConfig(width=width,height=height,fps=30.,cam_pos=(0.,6.,11.),cam_target=(0.,2.,-3.),
        cam_orbit=0.,camera_scale=.45,beam_edge_profile='angular',beam_edge_softness_deg=.6,
        beam_flux_reference_deg=14.,exposure_profile='fixed',fixed_exposure_scale=.4,
        fixed_role_gains=(1.,.3,.03,.2),haze=.25,intensity_gain=2.4)


def optical(frames,rig):
    d=frames.copy()
    for f in rig.by_kind('moving_head'):d[:,f.address-1+9]=0
    return d


def encode_command(path,p,width,height,start,n):
    return ['ffmpeg','-hide_banner','-loglevel','warning','-nostdin','-n','-f','rawvideo','-pix_fmt','rgb24',
        '-s',f'{width}x{height}','-r','30','-i','pipe:0','-ss',str(start/30),'-i',p['audio'],
        '-map','0:v:0','-map','1:a:0','-frames:v',str(n),'-c:v','av1_nvenc','-gpu','0','-preset','p6',
        '-tune','hq','-rc','vbr','-cq','20','-b:v','6M','-maxrate','8M','-bufsize','16M',
        '-pix_fmt','yuv420p','-colorspace','bt709','-color_primaries','bt709','-color_trc','bt709',
        '-c:a','aac','-b:a','192k','-af',f'atrim=duration={n/30},asetpts=PTS-STARTPTS',
        '-t',str(n/30),'-movflags','+faststart',str(path)]


def camera_project(points,cfg):
    camera=np.array(cfg.cam_pos);f=np.array(cfg.cam_target)-camera;f=f/np.linalg.norm(f)
    r=np.cross(f,[0,1,0]);r/=np.linalg.norm(r);u=np.cross(r,f)
    delta=np.asarray(points)-camera;depth=delta@f
    return np.column_stack(((.5+(delta@r)/depth/cfg.camera_scale/(cfg.width/cfg.height)/2)*cfg.width,
        (.5-(delta@u)/depth/cfg.camera_scale/2)*cfg.height))


def render_stage(out,p,preview=False):
    from lightsem.conductor.rig import canonical_rig
    from lightsem.sim.renderer import StageRenderer
    from lightsem.experiments.stage11_common import gpu_lock
    from PIL import Image,ImageDraw,ImageFont
    streams=arrays(out/'streams.npz');rig=canonical_rig();cfg=render_config()
    # Full-size native smoke/visual before full-track expansion. No upscaling.
    intervals=[(0,6),(70,82),(190,202)] if preview else [(0,p['duration_s'])]
    receipt={'config':asdict(cfg),'movies':[],'scope':p['scope']}
    pixels=[]; yy,xx=np.mgrid[:cfg.height,:cfg.width]
    mask=np.ones((cfg.height,cfg.width),bool)
    for x,y in camera_project([f.position for f in rig.fixtures],cfg):mask&=((xx-x)**2+(yy-y)**2>60**2)
    with gpu_lock():
        renderer=StageRenderer(rig,cfg);print('GPU',renderer.renderer_name,flush=True)
        for lo,hi in intervals:
            for arm in ('baseline','companion'):
                start=round(lo*30);n=round((hi-lo)*30)
                name=(f'preview_{arm}_{int(lo):03d}' if preview else ('full_candidate_1080p_av1' if arm=='companion' else 'full_baseline_1080p_av1'))
                dest=out/(name+'.mp4');log=out/(name+'_encode.log');d=optical(streams[arm],rig)
                cmd=encode_command(dest,p,cfg.width,cfg.height,start,n)
                metrics=[];rawhash=hashlib.sha256();beg=time.monotonic()
                with log.open('xb') as err:
                    proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=err)
                    try:
                        for i in range(n):
                            if i%30==0:guard(out)
                            src=round((start+i)*40/30)
                            rgb=renderer.render_frame(d[src],0.)
                            rawhash.update(rgb.tobytes())
                            # Pixel measures at native resolution before encode.
                            # Vectorized integer luma avoids large float64 images.
                            luma=(rgb[:,:,0].astype(np.float32)*.2126+rgb[:,:,1]*.7152+rgb[:,:,2]*.0722)
                            metrics.append((float(luma[mask].mean()),float((rgb.max(2)[mask]>=250).mean())))
                            proc.stdin.write(rgb.tobytes())
                            if i%300==0:print('RENDER',name,i,n,'elapsed',round(time.monotonic()-beg,1),flush=True)
                        proc.stdin.close();rc=proc.wait()
                        if rc:raise RuntimeError(log.read_text())
                    except BaseException:
                        try:proc.stdin.close()
                        except BrokenPipeError:pass
                        proc.terminate();proc.wait();raise
                m=np.asarray(metrics)
                row={'arm':arm,'path':str(dest),'start_frame':start,'frames':n,'duration_s':n/30,
                    'raw_sha256':rawhash.hexdigest(),'video_sha256':digest(dest),'encode_command':cmd,
                    'outside_bulb_mean_luma_255':float(m[:,0].mean()),'outside_bulb_clipping_mean':float(m[:,1].mean()),
                    'outside_bulb_clipping_max':float(m[:,1].max()),'seconds':time.monotonic()-beg,
                    'renderer_device':renderer.renderer_name,'native_resolution':[1920,1080]}
                np.savez_compressed(out/(name+'_pixels.npz'),metrics=m)
                save(out/(name+'_render.json'),row);receipt['movies'].append(row)
        renderer.ctx.release()
    save(out/('preview_render.json' if preview else 'full_render.json'),receipt)


def read_exact(pipe,n):
    b=bytearray()
    while len(b)<n:
        x=pipe.read(n-len(b))
        if not x:break
        b.extend(x)
    return bytes(b)


def audit_movie(out,path):
    from PIL import Image,ImageDraw
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(path)],text=True))
    v=next(s for s in probe['streams'] if s['codec_type']=='video');a=next(s for s in probe['streams'] if s['codec_type']=='audio')
    if (v['codec_name'],v['width'],v['height'],v['r_frame_rate'])!=('av1',1920,1080,'30/1'):raise RuntimeError('Wrong delivery format')
    # Full software decode + frame timestamps/count, not a container metadata claim.
    frames=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_frames',
        '-show_entries','frame=best_effort_timestamp_time','-of','json',str(path)],text=True))['frames']
    t=np.array([float(f['best_effort_timestamp_time']) for f in frames]);n=len(t)
    if not np.allclose(t,np.arange(n)/30,atol=1e-5):raise RuntimeError('Video timestamps are discontinuous')
    render_path=path.with_name(path.stem+'_render.json')
    expected=json.loads(render_path.read_text())['frames'] if render_path.exists() else None
    if expected is not None and n!=expected:raise RuntimeError('Wrong decoded frame count')
    if abs(float(a['duration'])-n/30)>.06:raise RuntimeError('Audio duration mismatch')
    result=subprocess.run(['ffmpeg','-v','error','-nostdin','-xerror','-hwaccel','none','-threads','4','-i',str(path),
        '-map','0:v:0','-map','0:a:0','-f','null','-'],capture_output=True,text=True)
    if result.returncode:raise RuntimeError(result.stderr)
    # Audio fingerprint supports identical delivered audio across paired arms.
    pcm=subprocess.check_output(['ffmpeg','-v','error','-nostdin','-i',str(path),'-map','0:a:0','-vn',
        '-ac','2','-ar','44100','-f','f32le','-'])
    # Decoded1Hz sheets: every part of the track; only sheets are retained.
    cmd=['ffmpeg','-v','error','-nostdin','-threads','4','-i',str(path),'-an','-vf','fps=1,scale=480:270',
        '-pix_fmt','rgb24','-threads','1','-f','rawvideo','-']
    sheetdir=out/(path.stem+'_sheets');sheetdir.mkdir(exist_ok=False)
    page=0;count=0;pages=[];sheet=None
    with (sheetdir/'decode.log').open('xb') as err:
        proc=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=err)
        while True:
            raw=read_exact(proc.stdout,480*270*3)
            if not raw:break
            if len(raw)!=480*270*3:raise RuntimeError('Partial decoded review frame')
            slot=count%12
            if slot==0:sheet=Image.new('RGB',(1440,4*294),(16,16,16))
            image=Image.frombytes('RGB',(480,270),raw);x=slot%3*480;y=slot//3*294
            sheet.paste(image,(x,y+24));ImageDraw.Draw(sheet).text((x+5,y+4),f'{path.stem} | {count}s',fill='white')
            count+=1
            if count%12==0:
                dest=sheetdir/f'page_{page:03d}.jpg';sheet.save(dest,quality=92);pages.append(str(dest));page+=1
        if count%12:
            dest=sheetdir/f'page_{page:03d}.jpg';sheet.save(dest,quality=92);pages.append(str(dest))
        if proc.wait()!=0:raise RuntimeError((sheetdir/'decode.log').read_text())
    row={'path':str(path),'video':v,'audio':a,'decoded_frames':n,'expected_frames':expected,'pts_uniform':True,
        'full_software_AV_decode':True,'pcm_sha256':hashlib.sha256(pcm).hexdigest(),'decoded_pcm_samples_per_channel':len(pcm)//8,
        'sheets':pages,'review_sample_frames':count,'scope':'Media integrity + sampled silent pictures, not human audio/music-fit acceptance'}
    save(out/(path.stem+'_audit.json'),row);print('AUDITED',path.name,n,flush=True);return row


def comparison_stage(out,p):
    # Globally match display luma downwards, never framewise. Main candidate stays
    # untouched. Comparison native scenes each downsampled960x540 on1080p canvas.
    rows=json.loads((out/'full_render.json').read_text())['movies']
    means={r['arm']:r['outside_bulb_mean_luma_255'] for r in rows};low=min(means.values())
    gains={k:low/v for k,v in means.items()}
    baseline=out/'full_baseline_1080p_av1.mp4';candidate=out/'full_candidate_1080p_av1.mp4'
    dest=out/'full_comparison_1080p_av1.mp4'
    filters=[]
    for i,arm in enumerate(('baseline','companion')):
        g=gains[arm];label='BEFORE - literal pose mapping' if i==0 else 'AFTER - rhythm-led companion light'
        filters.append(f'[{i}:v]scale=960:540,colorchannelmixer=rr={g}:gg={g}:bb={g},drawtext=text=\'{label}\':fontcolor=white:fontsize=22:x=16:y=12:box=1:boxcolor=black@0.7[v{i}]')
    filters.append('[v0][v1]hstack,pad=1920:1080:0:180:black,drawtext=text=\'Same saved motion and audio - no dance retiming\':fontcolor=white:fontsize=30:x=40:y=800,drawtext=text=\'Same non-angle DMX - one global image-brightness match per arm\':fontcolor=white:fontsize=26:x=40:y=860[v]')
    cmd=['ffmpeg','-v','warning','-nostdin','-n','-threads','4','-i',str(baseline),'-threads','4','-i',str(candidate),
        '-filter_complex_threads','4','-filter_complex',';'.join(filters),'-map','[v]','-map','1:a:0',
        '-frames:v',str(p['frames']),'-c:v','av1_nvenc','-preset','p6','-tune','hq','-rc','vbr','-cq','20',
        '-b:v','5M','-maxrate','7M','-bufsize','14M','-pix_fmt','yuv420p','-c:a','copy','-movflags','+faststart',str(dest)]
    with (out/'comparison_encode.log').open('xb') as log:
        result=subprocess.run(cmd,stderr=log)
    if result.returncode:raise RuntimeError((out/'comparison_encode.log').read_text())
    save(out/'comparison_report.json',{'command':cmd,'gains':gains,'raw_full_record_means':means,
        'scope':'One global precomputed displayRGB gain; decoded/reencoded residual not yet measured; not exact pixel brightness identity'})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['audio','map','preview','render','audit-preview','compare','audit-full'])
    parser.add_argument('--output',type=Path,required=True);a=parser.parse_args();out=a.output
    p=json.loads((out/'plan.json').read_text());authenticate(out,p);guard(out)
    if a.stage=='audio':audio_stage(out,p)
    elif a.stage=='map':map_stage(out,p)
    elif a.stage in ('preview','render'):render_stage(out,p,preview=a.stage=='preview')
    elif a.stage=='compare':comparison_stage(out,p)
    else:
        pattern='preview_*.mp4' if a.stage=='audit-preview' else 'full_*.mp4'
        for path in sorted(out.glob(pattern)):audit_movie(out,path)
if __name__=='__main__':main()
