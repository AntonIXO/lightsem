#!/usr/bin/env python3
"""Fresh baseline-C audio, motion and color on two fixed music excerpts."""
from __future__ import annotations
import argparse
from collections import OrderedDict
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import time

import numpy as np

from lightsem.model.light_layers.visual import sha256, write_json, write_arrays


def checked(plan):
    for path, expected in plan['files'].items():
        if sha256(path) != expected:
            raise RuntimeError('Frozen input changed: '+path)
    root = Path(plan['output'])
    if any(shutil.disk_usage(path).free < 2**30 for path in (root.parent, '/mnt')):
        raise RuntimeError('Free-space floor')
    if root.exists() and sum(p.stat().st_size for p in root.rglob('*') if p.is_file()) > plan['max_new_bytes']:
        raise RuntimeError('New artifact budget')


def prepare(plan):
    import soundfile as sf
    from lightsem.audio.online import extract_bus
    from lightsem.audio.rhythm_clock import clock_from_audio
    from lightsem.experiments.motion_study.audio_dynamics import loudness
    root = Path(plan['output'])
    root.mkdir(exist_ok=False)
    rhythm_inputs, records = [], []
    for track in plan['tracks']:
        out = root/track['id']
        out.mkdir()
        source = sf.info(track['audio'])
        if source.samplerate != 44100:
            raise RuntimeError('These frozen sources must be 44100 Hz')
        y, sr = sf.read(track['audio'], frames=round(plan['prefix_s']*source.samplerate),
                        dtype='float32', always_2d=True)
        if len(y) != round(plan['prefix_s']*sr) or not np.isfinite(y).all():
            raise RuntimeError('Source prefix is incomplete or nonfinite')
        audio = out/'audio_prefix.wav'
        sf.write(audio, y, sr, subtype='FLOAT')
        bus = extract_bus(y=y.mean(1), sr=sr, causal=True)
        expected = round(plan['prefix_s']*40)
        if len(bus.times) != expected:
            raise RuntimeError('Audio bus clock length')
        values = {'times':bus.times.astype(np.float32), 'mel':bus.mel.astype(np.float32),
                  'meta':np.asarray(json.dumps({'fps':bus.fps, 'tempo':bus.tempo,
                      'duration_s':bus.duration_s, 'flags':bus.flags, **bus.meta}))}
        values.update({'ch__'+k:v.astype(np.float32) for k,v in bus.channels.items()})
        write_arrays(out/'audio_bus.npz', values)
        clock, clock_report = clock_from_audio(y, sr)
        clock.save(out/'rhythm_clock.npz')
        write_json(out/'clock_report.json', clock_report)
        measured = loudness(audio, out/'loudness')
        record = {**track, 'source_sha256':sha256(track['audio']), 'prefix_audio':str(audio),
                  'prefix_audio_sha256':sha256(audio), 'channels':y.shape[1],
                  'bus_sha256':sha256(out/'audio_bus.npz'), 'clock_sha256':sha256(out/'rhythm_clock.npz'),
                  'loudness':measured, 'prefix_s':plan['prefix_s'], 'clock_report':clock_report}
        records.append(record)
        rhythm_inputs.append({'recording_id':track['id'], 'canonical_id':track['id'],
                              'partition':'personal_demonstration', 'audio_path':str(audio),
                              'audio_sha256':sha256(audio)})
        checked(plan)
        print('AUDIO_PREPARED', track['id'], clock_report['status'], flush=True)
    manifest = {'checkpoint':plan['beat_checkpoint']['path'],
                'checkpoint_sha256':plan['beat_checkpoint']['sha256'],
                'free_disk_floor_bytes':2**30, 'pilot_rows':2, 'inputs':rhythm_inputs,
                'scope':'Fixed two-song engineering demonstration; no new training or accuracy claim.'}
    write_json(root/'rhythm_inputs.json', manifest)
    write_json(root/'PREPARATION.json', {'status':'two_frozen_audio_prefixes_prepared','records':records,
                                       'rhythm_manifest_sha256':sha256(root/'rhythm_inputs.json')})


def generate(plan):
    import torch
    from lightsem.model.light_layers.training import SourceStore, load_checkpoint, _normalize, generated_history
    from lightsem.model.light_layers.evaluation import candidate_peaks
    from lightsem.model.light_layers.overlay import AccentConfig, AccentEvent, apply_accent_events
    from lightsem.experiments.stage11.stage11_common import gpu_lock
    from lightsem.audio.rhythm_clock import RhythmClock
    from lightsem.experiments.motion_study.companion import compose_companion_world, default_lighting_config
    from lightsem.experiments.motion_study.lighting_map import compose_world_joints
    from lightsem.experiments.visual_next.delivery import color_delivery
    from lightsem.experiments.output_diagnostics import motion_activity, palette_activity, representation_activity, cross_input_motion
    from lightsem.conductor.rig import canonical_rig
    from scripts.motion.run_rhythm_companion import angles
    from scripts.light_layers.prepare_visual_next_demo import FinitePaletteCapture
    root = Path(plan['output'])
    if (root/'GENERATION.json').exists():
        raise RuntimeError('Do not overwrite completed generation')
    preparation = json.loads((root/'PREPARATION.json').read_text())
    rhythm = json.loads((root/'rhythm/complete.json').read_text())
    if rhythm['manifest_sha256'] != sha256(root/'rhythm_inputs.json') or rhythm['rows'] != 2:
        raise RuntimeError('Missing exact learned rhythm extraction')
    beat_rows = {row['recording_id']:row for row in rhythm['records']}
    store = SourceStore.__new__(SourceStore)
    store.max_ram_bytes=128*2**20
    store.audio_cache=OrderedDict()
    store.cache_bytes=0
    store.verified=set()
    store.rows={}
    for row in preparation['records']:
        out = root/row['id']
        assert sha256(out/'audio_bus.npz')==row['bus_sha256']
        assert sha256(out/'rhythm_clock.npz')==row['clock_sha256']
        beat = beat_rows[row['id']]
        assert beat['audio_sha256']==row['prefix_audio_sha256']
        store.rows[row['id']]={'audio':{'path':str(out/'audio_bus.npz'),'sha256':row['bus_sha256']},
                              'rhythm':{'path':beat['path'],'sha256':beat['sha256']}}
    transforms=json.loads(Path(plan['transforms']).read_text())
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    scores={}
    with gpu_lock():
        other=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        if other:raise RuntimeError('Another GPU process is active')
        model, metadata=load_checkpoint(plan['baseline']['checkpoint']['path'],device='cuda')
        assert metadata['config']['recipe']=='I_VR_P0' and metadata['config']['seed']==0
        model.eval()
        for track in plan['tracks']:
            out=root/track['id'];raw=store.audio(track['id']);normalized=_normalize(raw,transforms['audio'])
            assert raw.shape==(round(plan['prefix_s']*40),94) and len(raw)<=model.cfg.max_frames40
            write_arrays(out/'audio94.npz',{'raw':raw,'normalized':normalized})
            with torch.inference_mode(),torch.autocast('cuda',enabled=False):
                value=model(torch.from_numpy(normalized[None]).cuda())
                assert all(torch.isfinite(v).all() for v in value.values())
                capture=FinitePaletteCapture(model)
                rollout=generated_history(capture,value['context10'],value['chromatic_change_logit'],
                                          initial_ids=None,state_ablation=False,cue_threshold=.5,initialize=True)
                result={k:v[0].cpu().numpy() for k,v in {**value,**rollout}.items()}
                result.update(color_score=value['chromatic_change_logit'][0].sigmoid().cpu().numpy(),
                              accent_score=value['accent_logit'][0].sigmoid().cpu().numpy())
                actions=np.unique(np.r_[0,np.flatnonzero(result['color_cues'])]).astype(int)
                previous=result['previous_ids'][actions]
                assert capture.calls==1
                assert np.array_equal(capture.tables.argmax(-1)[np.arange(len(actions)),previous],
                                      result['generated_palette_ids'][actions])
                result.update(action_indices=actions,action_previous_ids=previous,
                              action_argmax_choices=capture.tables.argmax(-1).astype(np.uint8),
                              action_logits=capture.tables[np.arange(len(actions)),previous])
            assert all(np.isfinite(x).all() for x in result.values())
            assert result['previous_ids'][0]==32 and np.array_equal(result['previous_ids'][1:],result['generated_palette_ids'][:-1])
            write_arrays(out/'model_outputs.npz',result);scores[track['id']]=result
            print('COLOR_INFERRED',track['id'],int(result['color_cues'].sum()),flush=True)
        del model
        torch.cuda.empty_cache()
    with np.load(plan['style_world']) as values:world=values['world'][:round(plan['prefix_s']*30)].copy()
    assert world.shape==(round(plan['prefix_s']*30),22,3)
    rig=canonical_rig();config=default_lighting_config();times=np.arange(round(plan['prefix_s']*40))/40
    clip=slice(round(plan['review_start_s']*40),round((plan['review_start_s']+plan['review_duration_s'])*40))
    clip10=slice(clip.start//4,clip.stop//4)
    streams, traces, baselines, clocks, loudnesses, diagnostics = {},{},{},{},{},[]
    for track in plan['tracks']:
        key=track['id'];out=root/key;clock=RhythmClock.load(out/'rhythm_clock.npz')
        with np.load(out/'loudness/loudness.npz') as values:curve={k:values[k] for k in values.files}
        base=compose_world_joints(world,curve,plan['prefix_s'],config=config,rig=rig)
        companion,trace,mapping=compose_companion_world(world,curve,plan['prefix_s'],clock,
                                                      lighting_config=config,rig=rig,baseline_bundle=base)
        predictions=scores[key]
        event_times,event_scores=candidate_peaks(times,predictions['accent_score'],np.array([[0.,plan['prefix_s']]]))
        events=[AccentEvent(float(t),.35,.25) for t in event_times[event_scores>=.5]]
        accent=AccentConfig(max_strength=.35,min_width_s=.25,max_width_s=.25,refractory_s=.2)
        accented=apply_accent_events(companion,rig,times,{f.name:events for f in rig.fixtures if f.kind in ('moving_head','wash')},accent_config=accent)
        dmx,invariants=color_delivery(accented,companion,times,predictions['generated_palette_ids'],
                                     predictions['color_cues'],transforms['palette_prototypes'])
        assert np.array_equal(angles(dmx,rig),angles(companion,rig))
        write_arrays(out/'streams.npz',{'candidate':dmx,'companion':companion,'time':times})
        write_arrays(out/'trace.npz',trace)
        a=angles(dmx,rig);active=(trace['activity']>.25)&(trace['rhythm_confidence']>.25)
        movement=motion_activity(a[clip],audio_active=active[clip])
        if movement['maximum_speed_deg_s']>90+1e-6 or movement['maximum_acceleration_deg_s2']>360+1e-6:
            raise RuntimeError('Fixed simulation mechanical limits failed')
        palettes=palette_activity(predictions['generated_palette_ids'],predictions['color_cues'],predictions['color_score'])
        visible_cues=int(predictions['color_cues'][clip10].sum())
        visible_changes=int(np.count_nonzero(np.diff(predictions['generated_palette_ids'][clip10])))
        diagnostics.append({'track':key,'motion':movement,'palette':palettes,
                            'review_color_cues':visible_cues,'review_palette_id_changes':visible_changes,
                            'context':representation_activity(predictions['context10']),
                            'streams_sha256':sha256(out/'streams.npz'),
                            'model_outputs_sha256':sha256(out/'model_outputs.npz'),
                            'trace_sha256':sha256(out/'trace.npz'),'mapping':mapping,
                            'color_delivery':invariants,'accent_count_full_prefix':len(events)})
        streams[key]=dmx;traces[key]=trace;baselines[key]=base;clocks[key]=clock;loudnesses[key]=curve
        checked(plan);print('FRESH_MOTION_MAPPED',key,flush=True)
    first,second=[v['id'] for v in plan['tracks']]
    assert np.array_equal(traces[first]['spatial_style'],traces[second]['spatial_style'])
    replay,_,_=compose_companion_world(world,loudnesses[first],plan['prefix_s'],clocks[first],
                                      lighting_config=config,rig=rig,baseline_bundle=baselines[first])
    assert np.array_equal(angles(replay,rig),angles(streams[first],rig))
    swap,_,_=compose_companion_world(world,loudnesses[first],plan['prefix_s'],clocks[second],
                                    lighting_config=config,rig=rig,baseline_bundle=baselines[first])
    write_arrays(root/'clock_swap_control.npz',{'candidate':swap,'time':times})
    comparisons={'two_songs':cross_input_motion(angles(streams[first],rig)[clip],angles(streams[second],rig)[clip]),
                 'clock_swap_same_body_and_loudness':cross_input_motion(angles(streams[first],rig)[clip],angles(swap,rig)[clip]),
                 'same_input_motion_replay_exact':True,
                 'exact_same_shared_body_style':True,
                 'normalized_audio_arrays_identical':bool(np.array_equal(store.audio(first),store.audio(second))),
                 'learned_color_score_arrays_identical':bool(np.array_equal(scores[first]['color_score'],scores[second]['color_score']))}
    common_start=max(traces[first]['beat'][clip.start],traces[second]['beat'][clip.start])
    common_end=min(traces[first]['beat'][clip.stop-1],traces[second]['beat'][clip.stop-1])
    if common_end>common_start:
        beat_grid=np.linspace(common_start,common_end,1000);aligned=[]
        for key in (first,second):
            a=angles(streams[key],rig)[clip].reshape(-1,12)
            aligned.append(np.stack([np.interp(beat_grid,traces[key]['beat'][clip],a[:,j]) for j in range(12)],1))
        comparisons['common_beat_index_aligned_angle_rms_deg']=float(np.sqrt(np.mean((aligned[0]-aligned[1])**2)))
    warnings=[]
    for row in diagnostics:
        if row['motion']['constant_motion']:warnings.append(row['track']+': constant emitted motion')
        if row['palette']['inactive_learned_color_head']:warnings.append(row['track']+': learned color head inactive')
        if row['palette']['single_palette_after_activation']:warnings.append(row['track']+': one active palette')
        if row['context']['constant_context']:warnings.append(row['track']+': constant native model context')
    if comparisons['two_songs']['angle_rms_difference_deg']<plan['motion_difference_floor_deg']:
        warnings.append('Different songs produced less than the fixed1degree RMS movement threshold')
    if comparisons['clock_swap_same_body_and_loudness']['angle_rms_difference_deg']<plan['motion_difference_floor_deg']:
        warnings.append('Clock intervention produced less than the fixed1degree RMS movement threshold')
    checked(plan)
    write_json(root/'GENERATION.json',{'status':'two_songs_generated_and_checked_before_render','tracks':diagnostics,
        'comparisons':comparisons,'warnings':warnings,'review_interval_s':[plan['review_start_s'],plan['review_start_s']+plan['review_duration_s']],
        'motion_source':'Same saved unconditioned body, fresh audio-dependent companion mapping. Procedural2/4/8beat movement vocabulary.',
        'scope':'User-requested output-failure/input-sensitivity demonstration. Not new training, fresh-test generalization or a music-conditioned body generator.',
        'render_permission':'Explicit diagnostic request; retain and label failures, never tune them away.'})


def render(plan):
    root=Path(plan['output']);report=json.loads((root/'GENERATION.json').read_text())
    assert report['status']=='two_songs_generated_and_checked_before_render'
    env=dict(os.environ,TORCH_ENV=plan['color_env'],OPENBLAS_NUM_THREADS='2')
    media=[]
    for track in plan['tracks']:
        out=root/track['id'];movie=out/(track['id']+'_baseline_C_1080p_av1.mp4')
        for mode in ('render','audit'):
            command=['bash','run_torch.sh','-u','-m','scripts.rendering.render_light_layers_review',mode,
                '--streams',str(out/'streams.npz'),'--audio',str(out/'audio_prefix.wav'),
                '--start',str(plan['review_start_s']),'--duration',str(plan['review_duration_s']),
                '--output',str(movie),'--label',track['caption'],'--max-bytes','50000000']
            with (out/(mode+'.log')).open('xb') as log:
                subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
        audit=json.loads(movie.with_name(movie.stem+'_audit.json').read_text())
        assert audit['decoded_frames']==round(plan['review_duration_s']*30) and audit['pts_uniform'] and audit['full_software_AV_decode']
        assert movie.stat().st_size<=50_000_000
        media.append({'id':track['id'],'title':track['title'],'path':str(movie),'sha256':sha256(movie),
                      'bytes':movie.stat().st_size,'audit':str(movie.with_name(movie.stem+'_audit.json')),
                      'streams':str(out/'streams.npz'),'source_audio':track['audio'],
                      'source_interval_s':[plan['review_start_s'],plan['review_start_s']+plan['review_duration_s']]})
        checked(plan);print('NEW_SONG_VIDEO_READY',track['id'],flush=True)
    write_json(root/'DELIVERY.json',{'status':'two_new_song_videos_rendered_and_audited','videos':media,
                                    'generation_report_sha256':sha256(root/'GENERATION.json'),
                                    'baseline_promoted':False,'warnings':report['warnings']})


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=('prepare','generate','render'))
    parser.add_argument('--plan',type=Path,required=True)
    args=parser.parse_args();plan=json.loads(args.plan.read_text());checked(plan)
    started=time.monotonic()
    {'prepare':prepare,'generate':generate,'render':render}[args.stage](plan)
    checked(plan)
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>8*2**30:raise RuntimeError('Process RAM budget')
    print('STAGE_COMPLETE',args.stage,round(time.monotonic()-started,3),flush=True)


if __name__=='__main__':main()
