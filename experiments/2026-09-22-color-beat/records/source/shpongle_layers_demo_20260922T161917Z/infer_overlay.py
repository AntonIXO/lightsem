#!/usr/bin/env python3
"""EXPERIMENTAL, fixed seed0 learned-light overlay. No training or rendering."""
from pathlib import Path
from collections import OrderedDict
from dataclasses import asdict
import argparse, colorsys, hashlib, json, random, shutil, time
import numpy as np
import torch
from lightsem.experiments.light_layers.training import (
    SourceStore, AUDIO_SCALARS, _normalize, generated_history, load_checkpoint)
from lightsem.experiments.light_layers.evaluation import validate_transforms, candidate_peaks
from lightsem.experiments.light_layers.overlay import (
    AccentConfig, AccentEvent, apply_overlay, apply_accent_events, allowed_slot_mask,
    sample_accent_events)
from lightsem.conductor.rig import canonical_rig

HERE=Path(__file__).resolve().parent
PLAN_PATH=HERE/'inference_mapping_plan.json'
PLAN=json.loads(PLAN_PATH.read_text())
OUT=Path(PLAN['output_dir'])


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(2**20), b''): h.update(block)
    return h.hexdigest()


def arrdigest(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def ref(path):
    return {'path':str(path), 'sha256':digest(path), 'bytes':Path(path).stat().st_size}


def checked(reference):
    p=Path(reference['path'])
    assert digest(p)==reference['sha256'], 'Changed artifact: '+str(p)
    return p


def save_json(name, data):
    with (OUT/name).open('x') as f:
        json.dump(data,f,indent=2,allow_nan=False)


def guard(reserve=0):
    used=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
    free=shutil.disk_usage(OUT).free
    assert used+reserve <= PLAN['storage']['maximum_new_bytes'], '64MiB demo budget'
    assert free-reserve >= PLAN['storage']['minimum_mnt_free_bytes'], '1GiB /mnt reserve'
    return {'new_artifact_bytes':used, 'mnt_free_bytes':free}


def save_npz(name, **values):
    # Conservative uncompressed size bound before each write.
    guard(sum(v.nbytes for v in values.values())+65536)
    with (OUT/name).open('xb') as f: np.savez_compressed(f, **values)
    guard()


class DemoAudioStore(SourceStore):
    """Only adapt initialization: demo has no TRAIN/DEV split or visual source.

    Inherit SourceStore.audio AND checked unchanged, including every precision,
    clock, endpoint, finite-value, and hash check. Never invoke visual readers.
    """
    def __init__(self, audio, rhythm):
        self.max_ram_bytes=64*1024**2
        self.audio_cache=OrderedDict()
        self.cache_bytes=0
        self.verified=set()
        self.rows={'shpongle_delivery':{'audio':audio, 'rhythm':rhythm}}


def palette_rgb(prototypes):
    p=np.asarray(prototypes,np.float64).reshape(-1,12,6)
    hue=np.arange(12,dtype=np.float64)/12
    saturation=np.arange(6,dtype=np.float64)/5
    z=np.sum(p*saturation[None,None,:]*np.exp(2j*np.pi*hue)[None,:,None],axis=(1,2))
    h=np.where(np.abs(z)==0,0.,np.mod(np.angle(z)/(2*np.pi),1.))
    s=np.abs(z)
    assert np.all(s <= 1+1e-12)
    # Bound only floating-point epsilon, not an artistic transform.
    hsv=np.stack([h,np.minimum(s,1),np.ones(len(s))],1)
    rgb=np.array([colorsys.hsv_to_rgb(*v) for v in hsv],np.float64)
    return rgb,hsv,z


def load_inputs():
    guard(24*1024**2)
    for path, expected in PLAN['source_files'].items():
        assert digest(path)==expected, 'Frozen source changed: '+path
    for key in ['evaluation_binding','transforms','audio','audio_bus','audio_bus_receipt',
                'audio92','audio92_receipt','rhythm','rhythm_receipt','baseline']:
        checked(PLAN[key])
    ck=PLAN['checkpoint']
    checked(ck); checked(ck['training_run_receipt'])
    registration=json.loads(checked(PLAN['evaluation_binding']).read_text())
    assert registration['checkpoints'][PLAN['checkpoint_key']]==ck
    assert (ck['seed'],ck['recipe'],ck['role'])==(0,'S_VR_P0','selected')
    training=json.loads(checked(ck['training_run_receipt']).read_text())
    assert training['status']=='complete' and training['best_weights_sha256']==ck['sha256']
    assert training['bindings']==ck['bindings']
    transforms=json.loads(checked(PLAN['transforms']).read_text())
    validate_transforms(transforms)
    assert transforms==training['frozen_transforms']
    for key in ('norm_sha256','vocab_sha256'):
        assert transforms[key]==ck['bindings'][key]
    assert PLAN['transforms']['sha256']==ck['bindings']['transforms_sha256']
    old_bus_receipt=json.loads(checked(PLAN['audio_bus_receipt']).read_text())
    assert old_bus_receipt['artifacts'][PLAN['audio_bus']['path']]['sha256']==PLAN['audio_bus']['sha256']
    assert old_bus_receipt['identity']['source_audio_sha256']==PLAN['audio']['sha256']
    old_audio92=json.loads(checked(PLAN['audio92_receipt']).read_text())
    assert old_audio92['features_sha256']==PLAN['audio92']['sha256']
    assert old_audio92['source_audio_sha256']==PLAN['audio']['sha256']
    assert old_audio92['source_bus_sha256']==PLAN['audio_bus']['sha256']
    assert old_audio92['feature_names']==[f'mel_{i:02d}' for i in range(64)]+list(AUDIO_SCALARS)
    rhythm_receipt=json.loads(checked(PLAN['rhythm_receipt']).read_text())
    rr=[r for r in rhythm_receipt['records'] if r['recording_id']=='shpongle_delivery']
    assert len(rr)==1 and rr[0]['audio_sha256']==PLAN['audio']['sha256']
    assert rr[0]['sha256']==PLAN['rhythm']['sha256']
    store=DemoAudioStore(PLAN['audio_bus'],PLAN['rhythm'])
    raw=store.audio('shpongle_delivery')
    with np.load(PLAN['audio92']['path'],allow_pickle=False) as z:
        assert np.array_equal(raw[:,:92],z['x']), 'SourceStore first92 != authenticated audio92'
        cache_times=z['times'].copy()
    with np.load(PLAN['audio_bus']['path'],allow_pickle=False) as z:
        meta=json.loads(str(z['meta']))
        audio_times=z['times'].copy()
    assert np.array_equal(cache_times,audio_times)
    with np.load(PLAN['baseline']['path'],allow_pickle=False) as z:
        baseline=z[PLAN['baseline_key']].copy()
        times=z[PLAN['baseline_time_key']].copy()
    assert raw.shape==(11092,94) and baseline.shape==(11092,512)
    assert baseline.dtype==np.uint8
    assert np.array_equal(times,np.arange(11092)/40)
    # Historical bus timestamps are stored float32; accepted DMX is float64.
    # Verify exact float32 quantization, not false bitwise clock equality.
    assert np.array_equal(audio_times,times.astype(np.float32))
    meta['demo_clock_note']='stored bus clock equals exact float32 quantization of canonical40Hz DMX clock'
    meta['demo_bus_clock_max_quantization_error_s']=float(np.max(np.abs(audio_times-times)))
    normalized=_normalize(raw,transforms['audio'])
    assert np.isfinite(normalized).all()
    rgb,hsv,z=palette_rgb(transforms['palette_prototypes'])
    return training,transforms,raw,normalized,baseline,times,rgb,hsv,z,meta,rr[0]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--validate-only',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(4)
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    values=load_inputs()
    training,transforms,raw,normalized,baseline,times,rgb,hsv,z,meta,rhythm_row=values
    # Synthetic software checks do not use a model forward or tune the demo.
    rig=canonical_rig()
    assert np.array_equal(apply_overlay(baseline,rig,times),baseline)
    config=AccentConfig(max_strength=.35,min_width_s=.25,max_width_s=.25,refractory_s=.2)
    assert np.array_equal(sample_accent_events(times,(),config=config),np.zeros(len(times)))
    pr=np.zeros((1,72)); pr[0,5]=1
    primary,_,_=palette_rgb(pr)
    assert np.allclose(primary,[1,0,0])
    preflight={'status':'passed','forward_calls':0,'audio_shape':list(raw.shape),
               'baseline_shape':list(baseline.shape),'first92_equals_authenticated_cache':True,
               'baseline_clock_equals_bus_clock_after_float32_quantization':True, 'immutable_source_checks':True,
               'plan':ref(PLAN_PATH),'helper':ref(__file__),'storage':guard()}
    if args.validate_only:
        save_json('preflight.json',preflight)
        print(json.dumps(preflight),flush=True)
        return
    assert not (OUT/'model_outputs.npz').exists(), 'One full forward already saved; do not repeat'
    save_json('run_started.json',{'plan':ref(PLAN_PATH),'helper':ref(__file__),
              'unix':time.time(),'preflight':preflight,'command':
              'TORCH_ENV=/root/lightsem/.venv/torch-access bash run_torch.sh -u '+str(Path(__file__).resolve())})
    model,metadata=load_checkpoint(PLAN['checkpoint']['path'],device='cuda',
                                  expected_bindings=PLAN['checkpoint']['bindings'])
    assert len(raw)<=model.cfg.max_frames40 and len(raw)%4==0
    assert metadata['config']['seed']==0 and metadata['config']['recipe']=='S_VR_P0'
    save_json('checkpoint_metadata.json',metadata)
    save_npz('audio94.npz',audio94=raw,normalized_audio94=normalized,time40_s=times,source_bus_time40_s=times.astype(np.float32))
    print('Full-song single forward starting: '+str(raw.shape),flush=True)
    torch.cuda.reset_peak_memory_stats()
    started=time.monotonic()
    with torch.no_grad(),torch.autocast(device_type='cuda',enabled=False):
        output=model(torch.from_numpy(normalized[None]).to('cuda'))
        rollout=generated_history(model,output['context10'],output['chromatic_change_logit'],
                                  initial_ids=None,state_ablation=False,cue_threshold=.5,initialize=True)
        palette_logits=model.palette_logits(output['context10'],rollout['previous_ids'])
        for v in [*output.values(),palette_logits]: assert bool(torch.isfinite(v).all())
        to_np=lambda t:t.detach().cpu().numpy()
        scores={k:to_np(v[0]) for k,v in output.items()}
        scores.update({k:to_np(v[0]) for k,v in rollout.items()})
        scores.update(accent_score=to_np(torch.sigmoid(output['accent_logit'])[0]),
                      color_score=to_np(torch.sigmoid(output['chromatic_change_logit'])[0]),
                      palette_logits=to_np(palette_logits[0]),
                      palette_probabilities=to_np(torch.softmax(palette_logits[0],-1)),
                      time40_s=times,time10_s=np.arange(len(raw)//4)/10)
    elapsed=time.monotonic()-started
    peak_memory=torch.cuda.max_memory_allocated()
    save_npz('model_outputs.npz',**scores)
    # No model forward is needed below this point.
    print('Single forward + generated rollout saved in %.3fs'%elapsed,flush=True)
    full_region=np.array([[0.,len(times)/40]])
    candidate_times,candidate_scores=candidate_peaks(times,scores['accent_score'],full_region)
    take=candidate_scores>=.5
    event_times,event_scores=candidate_times[take],candidate_scores[take]
    events=[AccentEvent(float(t),.35,.25) for t in event_times]
    pulse=sample_accent_events(times,events,config=config)
    cues=scores['color_cues']
    cue_times=scores['time10_s'][cues]
    generated=scores['generated_palette_ids']
    assert generated.shape==(2773,) and ((generated>=0)&(generated<32)).all()
    token40=np.repeat(generated,4)
    active=np.zeros(len(times),bool) if not len(cue_times) else times>=cue_times[0]
    tracks={}
    rgb_offsets={'moving_head':(6,7,8),'wash':(2,3,4),'projector':(2,3,4)}
    for f in rig.fixtures:
        if f.kind in rgb_offsets:
            slots=f.address-1+np.array(rgb_offsets[f.kind])
            track=baseline[:,slots].astype(float)/255
            track[active]=rgb[token40[active]]
            tracks[f.name]=track
    colored=apply_overlay(baseline,rig,times,rgb=tracks)
    accented={f.name:events for f in rig.fixtures if f.kind in ('moving_head','wash')}
    overlay=apply_accent_events(colored,rig,times,accented,accent_config=config)
    allowed=allowed_slot_mask(rig)
    protected=~allowed
    assert np.array_equal(overlay[:,protected],baseline[:,protected])
    angle_slots=np.array([f.address-1+j for f in rig.fixtures if f.kind=='moving_head' for j in range(4)])
    assert np.array_equal(overlay[:,angle_slots],baseline[:,angle_slots])
    # Exact baseline before first cue and outside accent support, respecting slots.
    rgb_slots=np.array([f.address-1+j for f in rig.fixtures if f.kind in rgb_offsets for j in rgb_offsets[f.kind]])
    dimmer_slots=np.array([f.address-1+(4 if f.kind=='moving_head' else 0)
                          for f in rig.fixtures if f.kind in ('moving_head','wash')])
    assert np.array_equal(overlay[np.ix_(~active,rgb_slots)],baseline[np.ix_(~active,rgb_slots)])
    assert np.array_equal(overlay[np.ix_(pulse==0,dimmer_slots)],baseline[np.ix_(pulse==0,dimmer_slots)])
    expected=np.floor(np.minimum(baseline[:,dimmer_slots].astype(float)*(1+pulse[:,None]),255)+.5).astype(np.uint8)
    assert np.array_equal(expected,overlay[:,dimmer_slots])
    zero_base=baseline[:,dimmer_slots]==0
    assert (overlay[:,dimmer_slots][zero_base]==0).all()
    save_npz('overlay_dmx.npz',baseline=baseline,overlay=overlay,time=times,
             allowed_slot_mask=allowed,color_active40=active,palette_ids40=token40,
             accent_pulse40=pulse.astype(np.float32),accent_times_s=event_times,
             accent_confidences=event_scores,color_cue_times_s=cue_times,
             palette_rgb=rgb,palette_hsv=hsv,palette_complex_z=z)
    save_json('events.json',{'accent':{'times_s':event_times.tolist(),'confidences':event_scores.tolist(),
                'count':len(events),'amplitude_gain_increment':.35,'full_width_s':.25},
              'color':{'times_s':cue_times.tolist(),'token_ids':generated[cues].tolist(),
                       'count':int(cues.sum()),'first_actual_cue_s':float(cue_times[0]) if len(cue_times) else None},
              'palette_hsv':hsv.tolist(),'palette_rgb':rgb.tolist()})
    equality={'status':'passed','protected_bytes_compared':int(len(times)*protected.sum()),
      'protected_slots_count':int(protected.sum()),'protected_slots_0based':np.flatnonzero(protected).tolist(),
      'changed_protected_bytes':int(np.count_nonzero(overlay[:,protected]!=baseline[:,protected])),
      'baseline_protected_bytes_sha256':arrdigest(baseline[:,protected]),
      'overlay_protected_bytes_sha256':arrdigest(overlay[:,protected]),
      'pan_tilt_bytes_exact':True,'all_strobe_blinder_projector_dimmer_and_other_protected_bytes_exact':True,
      'baseline_full_bytes_sha256':arrdigest(baseline),'overlay_full_bytes_sha256':arrdigest(overlay),
      'changed_allowed_bytes':int(np.count_nonzero(overlay[:,allowed]!=baseline[:,allowed])),
      'rgb_before_first_actual_cue_exact':True,'dimmers_outside_pulse_support_exact':True,
      'baseline_zero_dimmers_stay_zero':True,'same_gain_all_MH_wash_dimmers_before_byte_saturation':True,
      'accent_max_sampled_gain_increment':float(pulse.max()),
      'accent_min_spacing_s':float(np.min(np.diff(event_times))) if len(event_times)>1 else None,
      'baseline_source':PLAN['baseline'],'baseline_source_key':PLAN['baseline_key'],
      'output':ref(OUT/'overlay_dmx.npz')}
    save_json('protected_byte_equality.json',equality)
    for path,expected_hash in PLAN['source_files'].items(): assert digest(path)==expected_hash
    for key in ['audio','audio_bus','audio92','rhythm','baseline','transforms']:
        checked(PLAN[key])
    manifest={'status':'complete','label':PLAN['label'],'claim_scope':PLAN['claim_scope'],
      'plan':ref(PLAN_PATH),'helper':ref(__file__),'inputs':{k:PLAN[k] for k in
      ['audio','audio_bus','audio_bus_receipt','audio92','audio92_receipt','rhythm','rhythm_receipt',
       'baseline','transforms','checkpoint','evaluation_binding']},
      'source_files':PLAN['source_files'],'audio_bus_meta':meta,'rhythm_row':rhythm_row,
      'norm_sha256':transforms['norm_sha256'],'vocab_sha256':transforms['vocab_sha256'],
      'first92_equals_authenticated_cache':True,'runtime':{'torch':torch.__version__,
      'numpy':np.__version__,'device':torch.cuda.get_device_name(),'float32':True,'seed':0,
      'forward_calls':1,'full_song_frames40':len(raw),'context_frames10':len(raw)//4,
      'elapsed_forward_and_rollout_s':elapsed,'peak_allocated_bytes':peak_memory,
      'chunking_used':False,'chunk_equivalence_claim':False,'new_training':False,
      'visual_or_pose_inputs':False,'palette_gold_history':False},
      'mapping':{'accent':PLAN['accent'],'palette':PLAN['palette']},
      'render':{'path':str(OUT/'overlay_dmx.npz'),'baseline_key':'baseline','candidate_key':'overlay',
                'times_key':'time','shape':[11092,512],'dtype':'uint8','fps':40,
                'time_start_s':0.,'time_last_s':float(times[-1]),'support_stop_s':len(times)/40,
                'audio_source':PLAN['audio']['path'],'rig':'lightsem.conductor.rig.canonical_rig'},
      'counts':{'accent_events':len(events),'generated_color_cues':int(cues.sum()),
                'unique_generated_tokens':np.unique(generated).tolist()},
      'artifacts':{p.name:ref(p) for p in OUT.iterdir() if p.is_file()},
      'storage':guard(),'frozen_inputs_unchanged_after':True}
    save_json('provenance_manifest.json',manifest)
    print(json.dumps({'status':'complete','renderer':manifest['render'],
                      'counts':manifest['counts'],'equality':equality,'storage':guard()}),flush=True)


if __name__=='__main__': main()
