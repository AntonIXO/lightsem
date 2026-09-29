"""GPU-only continuous dance-descriptor flow pilot without beat-clock inputs.

CLI: python -m lightsem.experiments.dance_no_clock.dance_no_clock_training --root ROOT
     {train --arm temporal --seed 0 [--resume], evaluate --arm temporal --seed 0}

Only final scheduled weights are evaluated/demonstrated. Intermediate
validations are diagnostics, never checkpoint selection. No legacy inference,
conductor, group-effects engine, event heads or beat-feature extractor is used.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from scipy.stats import wasserstein_distance

from ..stage11.stage11_models import SceneFlowConfig, SceneFlowModel
from ..stage11.stage11_common import (authenticated_json, read_authenticated, digest, json_digest,
    artifact_manifest, verify_artifacts, verify_sources, atomic_torch, atomic_json,
    seed_all, rng_state, restore_rng, gpu_lock, require_space)

ROOT = Path('/mnt/lightsem-data/experiments/lightsem/dance_no_clock_20260908')
LAGS = (1, 5, 10, 20, 40)
ALIGNMENT_NAMES = ('com_speed', 'vert_bounce_amp')
FORBIDDEN = {'pulse_clarity', 'beat_phase_sin', 'beat_phase_cos', 'beat_accent',
             'bar_phase', 'beat_period_s', 'metric_level', 'tempo', 'downbeats'}


def context(root, *, verify_tensors=True):
    root = Path(root)
    protocol = read_authenticated(root/'protocol.json')
    manifest = read_authenticated(root/'data_manifest.json')
    norms = read_authenticated(root/'normalizers.json')
    frozen = verify_sources(root)
    if not protocol.get('frozen'):
        raise RuntimeError('GPU pilot requires frozen protocol/source')
    if manifest['normalizers_sha256'] != digest(root/'normalizers.json'):
        raise RuntimeError('normalizer binding changed')
    pinned = {'data_manifest_sha256': digest(root/'data_manifest.json'),
              'normalizers_sha256': digest(root/'normalizers.json'), 'source_sha256': frozen['source_sha256']}
    if any(protocol.get(key) != value for key,value in pinned.items()):
        raise RuntimeError('manifest/normalizer/source differs from the exact protocol pins')
    entries = manifest['entries']
    keys, song_parts = set(), {}
    for entry in entries:
        if entry['partition'] not in ('train', 'val'):
            raise RuntimeError('beat-free pilot accepts TRAIN/VAL only; test access forbidden')
        if entry['key'] in keys:
            raise RuntimeError('duplicate coach identity')
        keys.add(entry['key'])
        song = entry['canonical_id']
        if song in song_parts and song_parts[song] != entry['partition']:
            raise RuntimeError('canonical TRAIN/VAL overlap')
        song_parts[song] = entry['partition']
        if float(entry['fps']) != 10:
            raise RuntimeError('pilot tensors must already be on the filtered 10 Hz grid')
        if verify_tensors and digest(entry['path']) != entry['sha256']:
            raise RuntimeError('pilot tensor drift: '+entry['path'])
    names = list(norms.get('target_names', manifest.get('target_names', manifest.get('y_names', []))))
    feature_names = list(norms.get('feature_names',manifest.get('feature_names', manifest.get('x_names', []))))
    if len(names) != 20 or FORBIDDEN.intersection(names):
        raise RuntimeError('expected 20 named pose-only targets without metric_level')
    if len(feature_names) != 86 or FORBIDDEN.intersection(feature_names):
        raise RuntimeError('forbidden clock/pulse feature in pilot schema')
    for key, n in (('x_mean', 86), ('x_std', 86), ('y_center', 20), ('y_scale', 20)):
        value = np.asarray(norms[key])
        if value.shape != (n,) or not np.isfinite(value).all() or (key in ('x_std','y_scale') and np.any(value <= 0)):
            raise RuntimeError('invalid training normalization: '+key)
    if any(name not in names for name in ALIGNMENT_NAMES):
        raise RuntimeError('pose alignment targets missing')
    identity = {'protocol_sha256': digest(root/'protocol.json'), 'data_manifest_sha256': digest(root/'data_manifest.json'),
                'normalizers_sha256': digest(root/'normalizers.json'), 'source_sha256': frozen['source_sha256']}
    return protocol, manifest, norms, names, identity


def model_for(config):
    kind = config['flow_kind']
    if kind not in ('temporal', 'pointwise'):
        raise ValueError('flow_kind must be temporal or pointwise')
    return SceneFlowModel(config=SceneFlowConfig(flow_kind=kind, in_dim=86, scene_dim=20,
        d_model=192, n_blocks=3, n_heads=4, attn_window=80, conv_kernel=15, dropout=.1))


class ResidentDance:
    """One GPU tensor; sample song -> recording -> coach -> complete window."""
    def __init__(self, entries, norms, frames=600):
        self.excluded = [{'key': e['key'], 'n_frames': e['n_frames'], 'reason': 'shorter_than_training_window'}
                         for e in entries if e['n_frames'] < frames]
        self.entries = [e for e in entries if e['n_frames'] >= frames]
        if not self.entries:
            raise RuntimeError('no complete training windows; padding is forbidden')
        xs, ys, starts, lengths = [], [], [], []
        offset = 0
        groups = defaultdict(lambda: defaultdict(list))
        for index, entry in enumerate(self.entries):
            with np.load(entry['path'], allow_pickle=False) as z:
                x, y = z['x'].astype(np.float32), z['y'].astype(np.float32)
            if x.shape != (entry['n_frames'],86) or y.shape != (len(x),20) or not np.isfinite(x).all() or not np.isfinite(y).all():
                raise RuntimeError('invalid aligned clock-free tensor: '+entry['key'])
            xs.append((x-np.asarray(norms['x_mean'],np.float32))/np.asarray(norms['x_std'],np.float32))
            ys.append((y-np.asarray(norms['y_center'],np.float32))/np.asarray(norms['y_scale'],np.float32))
            starts.append(offset); lengths.append(len(x)); offset += len(x)
            groups[entry['canonical_id']][entry['recording_id']].append(index)
        self.x = torch.from_numpy(np.concatenate(xs)).cuda()
        self.y = torch.from_numpy(np.concatenate(ys)).cuda()
        self.starts, self.lengths = np.asarray(starts), np.asarray(lengths)
        self.groups = [list(records.values()) for _, records in sorted(groups.items())]

    def batch(self, size, frames, rng):
        selected = []
        for _ in range(size):
            records = self.groups[int(rng.integers(len(self.groups)))]
            coaches = records[int(rng.integers(len(records)))]
            selected.append(coaches[int(rng.integers(len(coaches)))])
        selected = np.asarray(selected)
        offsets = np.array([rng.integers(self.lengths[index]-frames+1) for index in selected])
        indices = self.starts[selected,None]+offsets[:,None]+np.arange(frames)[None]
        index = torch.as_tensor(indices, device='cuda')
        return self.x[index], self.y[index]


def normalized_targets(y, norms):
    return (np.asarray(y,np.float32)-np.asarray(norms['y_center'],np.float32))/np.asarray(norms['y_scale'],np.float32)


def noise_for(length, generation_seed):
    return torch.randn((1,length,20), dtype=torch.float32,
                       generator=torch.Generator(device='cpu').manual_seed(int(generation_seed)))


@torch.no_grad()
def generate_standardized(model, raw_audio, norms, *, noise, steps=8, chunk_size=600):
    """GPU inference over one whole sequence; explicit whole-sequence noise required."""
    if next(model.parameters()).device.type != 'cuda':
        raise RuntimeError('all scientific inference in this pilot must run on GPU')
    x = np.asarray(raw_audio,np.float32)
    if x.ndim != 2 or x.shape[1] != 86 or not np.isfinite(x).all():
        raise ValueError('raw_audio must be finite (T,86) at 10 Hz')
    if tuple(noise.shape) != (1,len(x),20) or noise.dtype != torch.float32:
        raise ValueError('explicit noise must be float32 (1,T,20)')
    x = (x-np.asarray(norms['x_mean'],np.float32))/np.asarray(norms['x_std'],np.float32)
    xt = torch.from_numpy(x)[None].cuda()
    # Keep the ODE state and all arithmetic on GPU as well as the velocity network.
    result = model.sample_chunked(xt, steps=steps, chunk_size=chunk_size,
                                  noise=noise.cuda(), output_device='cuda')[0]
    if not torch.isfinite(result).all():
        raise FloatingPointError('nonfinite generated motion')
    return result.cpu().numpy()


def generate(model, raw_audio, norms, *, noise, steps=8, chunk_size=600):
    """Return native 20 pose-descriptor channels from whole-track raw 86-D audio."""
    value = generate_standardized(model, raw_audio, norms, noise=noise, steps=steps, chunk_size=chunk_size)
    return value*np.asarray(norms['y_scale'],np.float32)+np.asarray(norms['y_center'],np.float32)


def continuous_metrics(reference, prediction, scales):
    real, pred = np.asarray(reference,np.float64), np.asarray(prediction,np.float64)
    if real.shape != pred.shape or real.ndim != 2 or real.shape[1] != 20 or not np.isfinite(real).all() or not np.isfinite(pred).all():
        raise ValueError('expected matching finite (T,20) standardized motion arrays')
    if len(real) <= max(LAGS):
        return {'eligible':False,'reason':'sequence_shorter_than_largest_lag'}
    marginal = np.maximum(np.asarray(scales['marginal_iqr'],np.float64),1e-3)
    if marginal.shape != (20,) or not np.isfinite(marginal).all():
        raise ValueError('invalid 20-channel TRAIN marginal IQR')
    lag_values = {};reference_variations=[];generated_variations=[]
    for lag in LAGS:
        scale = np.maximum(np.asarray(scales['lag_iqr'][str(lag)],np.float64),1e-3)
        if scale.shape != (20,) or not np.isfinite(scale).all():
            raise ValueError('invalid TRAIN lag IQR')
        a, b = real[lag:]-real[:-lag], pred[lag:]-pred[:-lag]
        lag_values[str(lag)] = float(np.mean([wasserstein_distance(a[:,j],b[:,j])/scale[j] for j in range(20)]))
        reference_variations.append(float(np.mean(np.abs(a)/scale)))
        generated_variations.append(float(np.mean(np.abs(b)/scale)))
    reference_variation = float(np.mean(reference_variations))
    generated_variation = float(np.mean(generated_variations))
    def hf(value):
        centered = value-value.mean(0)
        frequency = np.fft.rfftfreq(len(value),d=.1)
        power = np.abs(np.fft.rfft(centered,axis=0))**2/len(value)**2
        return float(power[(frequency>=2)&(frequency<=5)].sum(0).mean())
    rh, ph = hf(real), hf(pred)
    return {'eligible':True,'reason':None,'temporal_w1':float(np.mean(list(lag_values.values()))),
            'lag_w1':lag_values,'marginal_w1':float(np.mean([wasserstein_distance(real[:,j],pred[:,j])/marginal[j] for j in range(20)])),
            'reference_variation':reference_variation,'generated_variation':generated_variation,
            'variation_ratio':generated_variation/reference_variation if reference_variation>1e-12 else None,
            'reference_hf_energy':rh,'generated_hf_energy':ph,'hf_ratio':ph/rh if rh>1e-12 else None}


def alignment_mse(reference, correct, wrong, indices):
    arrays = [np.asarray(value,np.float64)[:,indices] for value in (reference,correct,wrong)]
    if len({a.shape for a in arrays}) != 1 or any(not np.isfinite(a).all() for a in arrays):
        raise ValueError('alignment support must match and be finite')
    if len(arrays[0]) < 10:
        return {'eligible':False,'reason':'shorter_than_one_second','correct_mse':None,'wrong_mse':None,'improvement_fraction':None}
    # Every window is exactly ten frames, centered at its half-frame midpoint;
    # identical valid support is used for truth/correct/wrong with no edge padding.
    arrays = [np.lib.stride_tricks.sliding_window_view(a,10,axis=0).mean(-1) for a in arrays]
    correct_mse, wrong_mse = [float(np.square(arrays[0]-a).mean()) for a in arrays[1:]]
    eligible = wrong_mse>1e-12
    return {'eligible':eligible,'reason':None if eligible else 'zero_wrong_audio_mse',
            'correct_mse':correct_mse,'wrong_mse':wrong_mse,
            'improvement_fraction':(wrong_mse-correct_mse)/wrong_mse if eligible else None,
            'smoothed_frames':len(arrays[0]),'smoothing_frames':10,'smoothing_center_offset_frames':4.5}


def donor_map(entries):
    songs = sorted({e['canonical_id'] for e in entries})
    if len(songs)<2:
        raise RuntimeError('wrong-audio control requires at least two canonical VAL songs')
    first = {song:min((e for e in entries if e['canonical_id']==song),
                      key=lambda e:(e['recording_id'],e['key'])) for song in songs}
    return {song:first[songs[(index+1)%len(songs)]] for index,song in enumerate(songs)}


def summarize(rows):
    """Mean draws/coaches inside recording, then recordings inside canonical song."""
    fields = ('temporal_w1','marginal_w1','reference_variation','generated_variation',
              'reference_hf_energy','generated_hf_energy')
    songs = {}
    for song in sorted({r['canonical_id'] for r in rows}):
        recordings = defaultdict(list)
        for row in rows:
            if row['canonical_id']==song:recordings[row['recording_id']].append(row)
        values = {}
        for field in fields:
            per_rec = []
            for records in recordings.values():
                val = [r['metrics'].get(field) for r in records]
                if all(v is not None and np.isfinite(v) for v in val):per_rec.append(float(np.mean(val)))
            values[field] = float(np.mean(per_rec)) if per_rec else None
        alignment = []
        for records in recordings.values():
            pairs = [r.get('alignment') for r in records]
            if pairs and all(p and p['eligible'] for p in pairs):
                alignment.append({key:float(np.mean([p[key] for p in pairs])) for key in ('correct_mse','wrong_mse')})
        if alignment:
            aligned={key:float(np.mean([p[key] for p in alignment])) for key in ('correct_mse','wrong_mse')}
            aligned['improvement_fraction']=(aligned['wrong_mse']-aligned['correct_mse'])/aligned['wrong_mse']
            values['alignment']=aligned
        else:values['alignment']=None
        for numerator,denominator,key in (('generated_variation','reference_variation','variation_ratio'),
                                         ('generated_hf_energy','reference_hf_energy','hf_ratio')):
            values[key]=values[numerator]/values[denominator] if values[denominator] and values[denominator]>1e-12 else None
        values['recordings'],values['alignment_eligible_recordings']=len(recordings),len(alignment)
        songs[song]=values
    totals = {field:float(np.mean([r[field] for r in songs.values() if r[field] is not None]))
              if any(r[field] is not None for r in songs.values()) else None for field in fields}
    for numerator,denominator,key in (('generated_variation','reference_variation','variation_ratio'),
                                       ('generated_hf_energy','reference_hf_energy','hf_ratio')):
        defined=[row[key] for row in songs.values() if row[key] is not None]
        totals[key] = float(np.mean(defined)) if defined else None
    aligned = [row['alignment'] for row in songs.values() if row['alignment'] is not None]
    totals['alignment']={key:float(np.mean([r[key] for r in aligned])) for key in ('correct_mse','wrong_mse','improvement_fraction')} if aligned else None
    totals['n_canonical_songs'],totals['alignment_eligible_songs']=len(songs),len(aligned)
    return {'overall':totals,'per_canonical':songs}


def evaluate_model(model, entries, norms, target_names, *, generation_seeds=(101,), paired=False, capture_dir=None,
                   donor_entries=None, constant_input=False):
    mapping = donor_map(donor_entries or entries) if paired else None
    indices = [target_names.index(name) for name in ALIGNMENT_NAMES]
    rows=[]
    for entry in entries:
        with np.load(entry['path'],allow_pickle=False) as z:
            x,truth=z['x'].astype(np.float32),normalized_targets(z['y'],norms)
        if constant_input:x=np.broadcast_to(np.asarray(norms['x_mean'],np.float32),x.shape).copy()
        donor=mapping[entry['canonical_id']] if paired else None
        if paired:
            with np.load(donor['path'],allow_pickle=False) as z:wrong_x=z['x'].astype(np.float32)
            if constant_input:wrong_x=np.broadcast_to(np.asarray(norms['x_mean'],np.float32),wrong_x.shape).copy()
        for gs in generation_seeds:
            full_noise=noise_for(len(x),gs)
            prediction=generate_standardized(model,x,norms,noise=full_noise)
            row={key:entry[key] for key in ('key','canonical_id','recording_id','n_frames')}
            row.update(generation_seed=gs,metrics=continuous_metrics(truth,prediction,norms['metric_scales']))
            arrays={'truth_standardized':truth,'prediction_standardized':prediction,'noise':full_noise.numpy()}
            if paired:
                n=min(len(x),len(wrong_x));shared=full_noise[:,:n].clone()
                correct=prediction if n==len(x) else generate_standardized(model,x[:n],norms,noise=shared)
                wrong=generate_standardized(model,wrong_x[:n],norms,noise=shared)
                row['alignment']=alignment_mse(truth[:n],correct,wrong,indices)
                row['alignment'].update(common_frames=n,donor_key=donor['key'],donor_canonical_id=donor['canonical_id'],
                    noise_sha256=hashlib.sha256(shared.numpy().tobytes()).hexdigest())
                arrays.update(paired_truth_standardized=truth[:n],paired_correct_standardized=correct,
                              paired_wrong_standardized=wrong,paired_noise=shared.numpy())
            if capture_dir:
                path=Path(capture_dir)/(entry['key']+f'_g{gs}.npz');path.parent.mkdir(parents=True,exist_ok=True)
                with path.open('xb') as stream:np.savez_compressed(stream,**arrays)
                row.update(capture_path=str(path),capture_sha256=digest(path))
            rows.append(row)
    return rows


def _schedule(step,total):
    warm=max(1,int(.05*total))
    if step<warm:return (step+1)/warm
    position=min(1.,(step-warm)/max(1,total-warm))
    return .1+.9*.5*(1+math.cos(math.pi*position))


def train(root,arm,seed,resume=False):
    if not torch.cuda.is_available():raise RuntimeError('GPU is required; CPU training fallback is forbidden')
    root=Path(root);protocol,manifest,norms,names,binding=context(root)
    cfg=dict(protocol['arms'][arm]);total=int(cfg.get('steps',8000));batch=int(cfg.get('batch',32));frames=int(cfg.get('frames',600))
    if seed not in protocol['training_seeds']:raise ValueError('unplanned training seed')
    run=root/'runs'/arm/f'seed_{seed}'
    if (run/'report.json').exists():raise FileExistsError('completed run cannot be retrained')
    if run.exists() and any(run.iterdir()) and not resume:raise RuntimeError('incomplete run requires explicit --resume')
    require_space(root);run.mkdir(parents=True,exist_ok=True)
    resident=ResidentDance([e for e in manifest['entries'] if e['partition']=='train'],norms,frames)
    validation=[e for e in manifest['entries'] if e['partition']=='val']
    seed_all(seed);model=model_for(cfg).cuda()
    shared={name:hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest() for name,value in model.named_parameters() if not name.startswith('denoiser.')}
    model_config=json.loads(json.dumps(model.config_dict()))
    identity={**binding,'arm':arm,'seed':seed,'config':cfg,'model_config':model_config,
              'initial_shared_parameters_sha256':json_digest(shared),'input_dim':86,'target_dim':20,'target_names':names}
    optimizer=torch.optim.AdamW(model.parameters(),lr=float(cfg.get('lr',3e-4)),weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:_schedule(step,total))
    rng=np.random.default_rng(seed+10000);flow_rng=torch.Generator(device='cuda').manual_seed(seed+30000)
    start_step=0;training_seconds=0.;pending=None
    initial_path=run/'initial.pt'
    if initial_path.exists():
        initial=torch.load(initial_path,map_location='cpu',weights_only=False)
        if initial['identity']!=identity or initial['step']!=0 or any(
            not torch.equal(initial['model'][key],value.detach().cpu()) for key,value in model.state_dict().items()):
            raise RuntimeError('initial checkpoint differs from exact seeded initialization')
    else:
        if resume:raise RuntimeError('resume cannot fabricate a missing initial checkpoint')
        atomic_torch(initial_path,{'schema_version':1,'identity':identity,'step':0,'model':model.state_dict(),
            'model_config':model_config,'normalizers':norms,'target_names':names,'weight_state':'untrained_initialization'})
    if resume:
        blob=torch.load(run/'latest.pt',map_location='cuda',weights_only=False)
        if blob['identity']!=identity:raise RuntimeError('resume identity mismatch')
        model.load_state_dict(blob['model']);optimizer.load_state_dict(blob['optimizer']);scheduler.load_state_dict(blob['scheduler'])
        restore_rng(blob['rng'],rng);flow_rng.set_state(blob['flow_rng'].cpu())
        start_step,training_seconds,pending=blob['step'],blob['training_seconds'],blob['pending_validation']
    started=time.monotonic();model.train()
    milestones=sorted(cfg.get('validation_steps',[2000,4000,8000]))
    def snapshot(step,pending_validation=None):
        return {'schema_version':1,'identity':identity,'model':model.state_dict(),'model_config':model_config,
                'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'rng':rng_state(rng),
                'flow_rng':flow_rng.get_state(),'step':step,'training_seconds':training_seconds,
                'pending_validation':pending_validation,'normalizers':norms,'target_names':names,
                'selection_policy':'final_scheduled_step_only','input_feature_schema':'audio86_no_clock_no_pulse_clarity'}
    def save(step,pending_validation=None):
        blob=snapshot(step,pending_validation);atomic_torch(run/'latest.pt',blob);return blob
    def validate(step):
        path=run/f'validation_step_{step}.json'
        state=rng_state(rng);flow_state=flow_rng.get_state()
        try:
            if path.exists():
                value=read_authenticated(path)
                if value['identity']!=identity or value['step']!=step:raise RuntimeError('validation identity mismatch')
            else:
                begin=time.monotonic()
                rows=evaluate_model(model,validation,norms,names,generation_seeds=(101,),paired=False)
                value={'identity':identity,'step':step,'rows':rows,'summary':summarize(rows),
                       'elapsed_seconds':time.monotonic()-begin,'diagnostic_only':True,'used_for_selection':False}
                authenticated_json(path,value)
        finally:
            restore_rng(state,rng);flow_rng.set_state(flow_state);model.train()
        blob=save(step)
        milestone=run/f'step_{step}.pt'
        if not milestone.exists():atomic_torch(milestone,blob)
    atomic_json(run/'identity.json',identity)
    if not (run/'latest.pt').exists():save(0)
    if pending is not None:validate(int(pending))
    torch.cuda.reset_peak_memory_stats()
    with (run/'training.jsonl').open('a') as log:
        for step in range(start_step,total):
            torch.cuda.synchronize();begin=time.perf_counter()
            x,y=resident.batch(batch,frames,rng)
            optimizer.zero_grad(set_to_none=True)
            loss=model.loss(x,y,generator=flow_rng)
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite flow loss')
            loss.backward();gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(gradient):raise FloatingPointError('nonfinite gradient')
            optimizer.step();scheduler.step();torch.cuda.synchronize()
            training_seconds+=time.perf_counter()-begin
            if step==0 or (step+1)%100==0 or step+1==total:
                value={'step':step+1,'loss':float(loss.detach()),'gradient_norm':float(gradient),
                       'learning_rate':scheduler.get_last_lr()[0],'training_seconds':training_seconds}
                log.write(json.dumps(value)+'\n');log.flush();atomic_json(run/'status.json',{'status':'training',**value})
                print(json.dumps({'arm':arm,'seed':seed,**value}),flush=True)
            if (step+1)%1000==0 or step+1 in milestones:save(step+1,step+1 if step+1 in milestones else None)
            if step+1 in milestones:validate(step+1)
        final_blob=save(total)
    final=run/'final.pt'
    if final.exists():
        existing=torch.load(final,map_location='cpu',weights_only=False)
        if existing['identity']!=identity or existing['step']!=total:raise RuntimeError('existing final checkpoint differs')
        if any(not torch.equal(existing['model'][key],value.detach().cpu()) for key,value in model.state_dict().items()):
            raise RuntimeError('existing final checkpoint weights differ')
    else:atomic_torch(final,final_blob)
    report={'schema_version':1,'status':'completed','identity':identity,'steps':total,'final_step':total,
            'selection_policy':'final_scheduled_step_only','checkpoint_selection_performed':False,
            'training_seconds':training_seconds,'successful_invocation_wall_seconds':time.monotonic()-started,
            'batch':batch,'frames':frames,'excluded_short_training_entries':resident.excluded,
            'gpu':torch.cuda.get_device_name(),'precision':'fp32','nonfinite':False,'oom':False,
            'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
            'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,
            'artifacts':artifact_manifest([final,initial_path,run/'latest.pt',run/'identity.json',run/'training.jsonl']+
                         [run/f'validation_step_{step}.json' for step in milestones]+
                         [run/f'step_{step}.pt' for step in milestones])}
    authenticated_json(run/'report.json',report);atomic_json(run/'status.json',{'status':'completed','step':total})
    del model,resident,optimizer;gc.collect();torch.cuda.empty_cache()
    return report


def load_model(root,arm,seed,device='cuda',*,initial=False):
    """Authenticate final scheduled checkpoint; no best-by-metric selection exists."""
    if device!='cuda':raise RuntimeError('pilot inference must use GPU')
    root=Path(root);protocol,manifest,norms,names,binding=context(root)
    report=read_authenticated(root/'runs'/arm/f'seed_{seed}/report.json')
    identity=report['identity'];cfg=protocol['arms'][arm]
    if (report['status']!='completed' or any(identity.get(key)!=value for key,value in binding.items())
            or identity['arm']!=arm or identity['seed']!=seed or identity['config']!=cfg
            or report['final_step']!=cfg.get('steps',8000)):
        raise RuntimeError('final model report identity differs')
    verify_artifacts(report['artifacts'])
    checkpoint=root/'runs'/arm/f'seed_{seed}'/('initial.pt' if initial else 'final.pt')
    blob=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if blob['identity']!=identity or blob['step']!=(0 if initial else report['final_step']) or json_digest(blob['normalizers'])!=json_digest(norms):
        raise RuntimeError('final checkpoint/calibration differs')
    seed_all(seed)
    model=model_for(cfg);model.load_state_dict(blob['model'])
    if json_digest(model.config_dict())!=json_digest(blob['model_config']):raise RuntimeError('model architecture differs')
    return model.cuda().eval(),blob


def evaluate(root,arm,seed,*,initial=False,constant_input=False):
    root=Path(root);protocol,manifest,norms,names,binding=context(root)
    label='initial_constant' if initial and constant_input else 'initial' if initial else 'constant_input' if constant_input else 'val'
    folder=root/'results'/label/arm/f'seed_{seed}'
    path=folder/'report.json'
    if path.exists():
        report=read_authenticated(path)
        expected={**binding,'arm':arm,'seed':seed,'config':protocol['arms'][arm],
                  'checkpoint_sha256':digest(root/'runs'/arm/f'seed_{seed}'/('initial.pt' if initial else 'final.pt')),
                  'generation_seeds':[101,102],'initial_weights':initial,'constant_input':constant_input}
        if report.get('status')!='completed' or any(report.get('identity',{}).get(key)!=value for key,value in expected.items()):
            raise RuntimeError('existing evaluation identity differs')
        verify_artifacts(report['artifacts']);return report
    if folder.exists() and any(folder.iterdir()):raise RuntimeError('partial evaluation requires explicit recovery')
    model,blob=load_model(root,arm,seed,initial=initial)
    begin=time.monotonic()
    all_entries=[e for e in manifest['entries'] if e['partition']=='val'];entries=all_entries
    if initial or constant_input:
        controls=protocol.get('controls',{})
        ids=controls.get('canonical_ids')
        if ids is None:
            ids=sorted({e['canonical_id'] for e in all_entries},key=lambda key:json_digest([key,'dance-no-clock-controls-v1']))[:int(controls.get('n_songs',4))]
        if not ids or set(ids)-{e['canonical_id'] for e in all_entries}:raise RuntimeError('invalid frozen control subset')
        entries=[e for e in all_entries if e['canonical_id'] in ids]
    rows=evaluate_model(model,entries,norms,names,generation_seeds=(101,102),paired=True,capture_dir=folder/'captures',
                        donor_entries=all_entries,constant_input=constant_input)
    report={'status':'completed','identity':{**binding,'arm':arm,'seed':seed,'config':protocol['arms'][arm],
            'checkpoint_sha256':digest(root/'runs'/arm/f'seed_{seed}'/('initial.pt' if initial else 'final.pt')),
            'generation_seeds':[101,102],'initial_weights':initial,'constant_input':constant_input,
            'canonical_ids':sorted({e['canonical_id'] for e in entries})},
            'rows':rows,'summary':summarize(rows),'target_names':names,'n_rows':len(rows),
            'validation_pilot_only':True,'confirmation':False,'aesthetic_improvement_proven':False,
            'elapsed_seconds':time.monotonic()-begin,
            'artifacts':artifact_manifest([r['capture_path'] for r in rows])}
    authenticated_json(path,report);del model,blob;gc.collect();torch.cuda.empty_cache();return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=ROOT)
    commands=parser.add_subparsers(dest='command',required=True)
    p=commands.add_parser('train');p.add_argument('--arm',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--resume',action='store_true')
    p=commands.add_parser('evaluate');p.add_argument('--arm',required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--initial',action='store_true');p.add_argument('--constant-input',action='store_true')
    args=parser.parse_args(argv)
    with gpu_lock(args.root):
        result=train(args.root,args.arm,args.seed,args.resume) if args.command=='train' else evaluate(args.root,args.arm,args.seed,initial=args.initial,constant_input=args.constant_input)
    label='initial_constant' if getattr(args,'initial',False) and getattr(args,'constant_input',False) else 'initial' if getattr(args,'initial',False) else 'constant_input' if getattr(args,'constant_input',False) else 'val'
    report_path=args.root/'runs'/args.arm/f'seed_{args.seed}/report.json' if args.command=='train' else args.root/'results'/label/args.arm/f'seed_{args.seed}/report.json'
    print(json.dumps({'status':result['status'],'steps':result.get('steps'),'report':str(report_path)},allow_nan=False),flush=True)


if __name__=='__main__':main()
