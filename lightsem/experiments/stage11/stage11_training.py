"""Matched training and leakage-safe evaluation for Stage 11.

No legacy training driver is invoked: their implicit splits, global statistics,
test access and mutable scaler lookup do not belong in this campaign.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import gc
import json
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from .stage11_common import (ROOT, atomic_json, authenticated_json, read_json, digest,
    json_digest, atomic_torch, rng_state, restore_rng, seed_all, require_space,
    artifact_manifest, verify_sources, gpu_lock, read_authenticated)

A_NAMES = ('metric_level', 'vert_period_s', 'vert_bounce_amp', 'com_speed')
C_NAMES = ('qom', 'head_speed', 'foot_speed', 'weight', 'time_eff', 'spread',
           'section_energy', 'hand_speed', 'hand_spread', 'lateral_pos',
           'travel_speed', 'symmetry', 'laterality', 'space_eff', 'flow_eff',
           'jerk', 'vert_bounce')
B_NAMES = ('footfall', 'jump', 'hit')


def _manifest(root):
    from .stage11_data import load_manifest
    return load_manifest(Path(root))


def _audio_features(bus):
    from lightsem.model.light_data import AUDIO_SCALARS
    n = len(bus.times)
    return np.concatenate([bus.mel[:, :n].T,
        np.stack([bus.channels[k][:n] for k in AUDIO_SCALARS], 1)], 1).astype(np.float32)


def prepare_training_data(root=ROOT):
    """Materialize versioned tensors and fit all statistics on permitted training rows."""
    from lightsem.motion.io import load_bus, load_descriptors, slice_bus
    from lightsem.model.light_data import BLOCK_B
    from .stage11_metrics import fit_scales
    root = Path(root)
    target = root / 'training_data.json'
    manifest = _manifest(root)
    manifest_sha = json_digest(manifest)
    if target.exists():
        existing = read_authenticated(target)
        if existing['manifest_sha256'] != manifest_sha:
            raise RuntimeError('Prepared training tensors refer to a different data manifest')
        verify_training_inputs(root, existing)
        return existing
    require_space(root)
    tensor_root = root / 'tensors'
    tensor_root.mkdir(exist_ok=True)
    motion, scene = [], []
    # Statistics use unique old training audio recordings, with target samples
    # balanced across coaches. Index identities remain independent of array order.
    xsum = np.zeros(92, np.float64); xsum2 = xsum.copy(); xcount = 0
    raw_targets, bsum, bcount = [], np.zeros(len(BLOCK_B), np.float64), 0
    scene_xsum = np.zeros(52, np.float64); scene_xsum2 = scene_xsum.copy(); scene_count = 0
    scene_y_train = []
    for r in manifest['recordings']:
        if r['partition'] not in ('train', 'val', 'test') or not r.get('eligible_core',False):
            continue
        if not r.get('visual_path') or not r.get('coaches'):
            raise ValueError('Eligible recording lacks prepared inputs: '+r['recording_id'])
        bus = load_bus(Path(r['audio_path']))
        used_old_audio = False
        for coach in r['coaches']:
            cid = coach['coach_id']
            dst = tensor_root / 'motion' / f'{cid}.npz'
            dst.parent.mkdir(exist_ok=True)
            d = load_descriptors(Path(coach['motion_path']))
            off, dur = float(coach['start_s']), float(coach['duration_s'])
            cut = slice_bus(bus, duration_s=dur, offset_s=off)
            n = min(len(d.times), len(cut.times))
            if n < 160:
                raise ValueError(f'Qualified coach too short: {cid}')
            x = _audio_features(cut)[:n]
            y = np.stack([d.channels[k][:n] for k in A_NAMES+C_NAMES], 1).astype(np.float32)
            b = np.zeros((n, len(BLOCK_B)), np.float32)
            for j, name in enumerate(BLOCK_B):
                ts = np.asarray(d.events.get(name, []), dtype=float)
                idx = np.rint(ts * cut.fps).astype(int)
                idx = idx[(idx >= 0) & (idx < n)]
                b[idx, j] = 1
            if not (np.isfinite(x).all() and np.isfinite(y).all()):
                raise ValueError(f'Nonfinite qualified training example: {cid}')
            np.savez(dst, x=x, y=y, b=b)
            entry = {'key': cid, 'canonical_id': r['canonical_id'], 'recording_id': r['recording_id'],
                     'partition': r['partition'], 'is_old': r['is_old'], 'path': str(dst),
                     'sha256': digest(dst), 'n_frames': n, 'offset_s': off, 'duration_s': n/40}
            motion.append(entry)
            if r['partition'] == 'train' and r['is_old']:
                sample = y[np.linspace(0, n-1, min(n, 2048), dtype=int)]
                raw_targets.append(sample)
                bsum += b.sum(0); bcount += n
                if not used_old_audio:
                    xsum += x.sum(0, dtype=np.float64)
                    xsum2 += np.square(x.astype(np.float64)).sum(0)
                    xcount += n; used_old_audio = True
        with np.load(r['visual_path'], allow_pickle=False) as z:
            x, y = z['x'].astype(np.float32), z['y'].astype(np.float32)
        if len(x) < 600 or x.shape != (len(y), 52) or y.shape[1] != 6:
            raise ValueError(f'Invalid scene training shape: {r["recording_id"]} {x.shape}/{y.shape}')
        dst = tensor_root / 'scene' / (r['recording_id']+'.npz')
        dst.parent.mkdir(exist_ok=True)
        np.savez(dst, x=x, y=y)
        scene.append({'key': r['recording_id'], 'canonical_id': r['canonical_id'],
            'recording_id': r['recording_id'], 'partition': r['partition'], 'is_old': r['is_old'],
            'path': str(dst), 'sha256': digest(dst), 'n_frames': len(x),
            'offset_s': r['active_start_s'], 'duration_s': len(x)/10})
        if r['partition'] == 'train':
            scene_xsum += x.sum(0, dtype=np.float64)
            scene_xsum2 += np.square(x.astype(np.float64)).sum(0)
            scene_count += len(x)
            scene_y_train.append(y)
    if not xcount or not bcount or not scene_count:
        raise RuntimeError('Statistics fit saw zero training rows')
    raw = np.concatenate(raw_targets)
    median = np.median(raw, axis=0)
    iqr = np.quantile(raw, .75, axis=0)-np.quantile(raw, .25, axis=0)
    scale = np.maximum(1.5*iqr, 1e-3)
    p = np.clip(bsum / bcount, 1e-4, 1-1e-4)
    mx = xsum/xcount; sx = np.sqrt(np.maximum(xsum2/xcount-mx*mx, 1e-12))
    smx = scene_xsum/scene_count
    ssx = np.sqrt(np.maximum(scene_xsum2/scene_count-smx*smx, 1e-12))
    # Common feature scales include no held-out statistics. Values/p90 already
    # have recording-relative targets supplied by the versioned data builder.
    sy = np.concatenate([y[::10] for y in scene_y_train])
    sym = sy.mean(0); sys = np.maximum(sy.std(0), 1e-3)
    norms = {'motion': {'x_mean': mx.tolist(), 'x_std': sx.tolist(),
        'y_center': median.tolist(), 'y_scale': scale.tolist(), 'pos_weight': ((1-p)/p).tolist(),
        'a_names': list(A_NAMES), 'c_names': list(C_NAMES), 'b_names': list(BLOCK_B),
        'fit_population': 'old_training_only', 'x_frames': xcount, 'event_frames': bcount},
        'scene': {'x_mean': smx.tolist(), 'x_std': ssx.tolist(), 'y_center': sym.tolist(),
                  'y_scale': sys.tolist(), 'fit_population': 'all_training_only'},
        'scene_metrics': fit_scales(scene_y_train, fps=10)}
    norm_path = root/'normalizers.json'
    authenticated_json(norm_path, norms)
    entries = {'schema_version': 1, 'manifest_sha256': manifest_sha,
        'normalizers_path': str(norm_path), 'normalizers_sha256': digest(norm_path),
        'motion': motion, 'scene': scene, 'prepared_at': time.time(),
        'excluded_recordings': [{'key':r['recording_id'],'reason':r.get('exclusion_reason','ineligible')}
            for r in manifest['recordings'] if not r.get('eligible_core',False)]}
    authenticated_json(target, entries)
    return entries


def verify_training_inputs(root, data=None):
    root=Path(root)
    data=data or read_authenticated(root/'training_data.json')
    manifest=_manifest(root)
    if json_digest(manifest)!=data['manifest_sha256']:
        raise RuntimeError('Data manifest changed')
    read_authenticated(root/'normalizers.json')
    if digest(root/'normalizers.json')!=data['normalizers_sha256']:
        raise RuntimeError('Normalizer drift')
    for family in ('motion','scene'):
        if not data[family]:raise RuntimeError('Empty training family')
        for e in data[family]:
            if digest(e['path'])!=e['sha256']:
                raise RuntimeError('Tensor content drift: '+e['path'])
        sets={part:{e['canonical_id'] for e in data[family] if e['partition']==part} for part in ('train','val','test')}
        if any(sets[a]&sets[b] for a,b in [('train','val'),('train','test'),('val','test')]):
            raise RuntimeError('Canonical split overlap')
        if len(sets['val'])!=40 or len(sets['test'])!=40:
            raise RuntimeError('Incomplete held-out cohort')
    return data


class ResidentDataset:
    """Canonical-song-balanced sampler with deterministic, checkpointed RNG.

    One flat CPU or accelerator tensor avoids per-window disk access. Coach
    identity remains in the hierarchy instead of weighting multi-coach songs.
    """
    def __init__(self, entries, norms, family, *, device='cpu'):
        self.entries = list(entries); self.family = family; self.device = device
        if not self.entries:
            raise ValueError('Empty dataset')
        xs, ys, bs = [], [], []
        self.starts = []; self.lengths = []; off = 0
        for e in self.entries:
            with np.load(e['path'], allow_pickle=False) as z:
                x, y = z['x'].astype(np.float32), z['y'].astype(np.float32)
                if family == 'motion':
                    bs.append(z['b'].astype(np.float32))
            xs.append((x-np.asarray(norms['x_mean'], np.float32))/np.asarray(norms['x_std'], np.float32))
            ys.append((y-np.asarray(norms['y_center'], np.float32))/np.asarray(norms['y_scale'], np.float32))
            self.starts.append(off); self.lengths.append(len(x)); off += len(x)
        self.starts=np.asarray(self.starts); self.lengths=np.asarray(self.lengths)
        self.x=torch.from_numpy(np.concatenate(xs)).to(device)
        self.y=torch.from_numpy(np.concatenate(ys)).to(device)
        self.b=torch.from_numpy(np.concatenate(bs)).to(device) if bs else None
        groups=defaultdict(lambda: defaultdict(list))
        for i,e in enumerate(self.entries):
            groups[e['canonical_id']][e['recording_id']].append(i)
        self.groups=[list(records.values()) for _,records in sorted(groups.items())]
        self.canonical_keys=sorted(groups)

    def _entry(self, group, rng):
        recordings=self.groups[group]
        coaches=recordings[int(rng.integers(len(recordings)))]
        return coaches[int(rng.integers(len(coaches)))]

    def batch(self, batch, frames, rng, device, *, swapped=False, donor_rng=None):
        ng=len(self.groups)
        gidx=rng.integers(0,ng,size=batch)
        entries=[self._entry(int(g),rng) for g in gidx]
        offsets=np.array([rng.integers(0,self.lengths[e]-frames+1) for e in entries])
        ix=self.starts[entries,None]+offsets[:,None]+np.arange(frames)[None]
        it=torch.as_tensor(ix,device=self.x.device)
        x=self.x[it].to(device); y=self.y[it].to(device)
        b=self.b[it].to(device) if self.b is not None else None
        if swapped:
            if ng<2 or donor_rng is None:
                raise ValueError('Swap requires distinct songs and its own RNG')
            dg=(gidx+donor_rng.integers(1,ng,size=batch))%ng
            de=[self._entry(int(g),donor_rng) for g in dg]
            do=np.array([donor_rng.integers(0,self.lengths[e]-frames+1) for e in de])
            di=self.starts[de,None]+do[:,None]+np.arange(frames)[None]
            dx=self.x[torch.as_tensor(di,device=self.x.device)].to(device)
            x=torch.cat([dx[...,:48],x[...,48:]],-1)
        return x,y,b

    def recording(self, i, device='cpu'):
        s,n=int(self.starts[i]),int(self.lengths[i])
        return self.x[s:s+n].to(device),self.y[s:s+n].to(device),None if self.b is None else self.b[s:s+n].to(device)


def _model(family, arm):
    if family == 'motion':
        from lightsem.model.light_trunk import LightTrunk,TrunkConfig
        return LightTrunk(TrunkConfig(in_dim=92,n_a=4,n_b=3,n_c=17,ar_flow=True))
    from .stage11_models import SceneFlowModel
    return SceneFlowModel('pointwise' if arm=='S_pointwise' else 'temporal')


def _motion_predictions(model, x, device, core=640):
    radius=model.cfg.n_blocks*(model.cfg.attn_window-1+model.cfg.conv_kernel-1)
    aa,bb=[],[]
    with torch.no_grad():
        for start in range(0,len(x),core):
            lo=max(0,start-radius);hi=min(len(x),start+core)
            o=model(x[lo:hi].to(device)[None])
            keep=slice(start-lo,hi-lo)
            aa.append(o['a'][0,keep].cpu().numpy())
            bb.append(torch.sigmoid(o['b_logit'][0,keep]).cpu().numpy())
    return np.concatenate(aa),np.concatenate(bb)


def evaluate_motion(model, entries, norms, device, *, capture_dir=None):
    from .stage11_metrics import motion_r2,event_average_precision
    model.eval(); rows=[]
    for e in entries:
        with np.load(e['path'],allow_pickle=False) as z:
            x=(z['x']-np.asarray(norms['x_mean'],np.float32))/np.asarray(norms['x_std'],np.float32)
            real=z['y'][:,:4].astype(np.float32);b=z['b'].astype(np.float32)
        pred,bp=_motion_predictions(model,torch.from_numpy(x),device)
        pred=pred*np.asarray(norms['y_scale'][:4])+np.asarray(norms['y_center'][:4])
        mr=motion_r2(real,pred,channel_names=list(A_NAMES),fps=40,tau=2)
        row={'key':e['key'],'canonical_id':e['canonical_id'],'recording_id':e['recording_id'],
             'n_frames':len(real),'motion':mr,
             'events':{name:event_average_precision(b[:,j],bp[:,j],fps=40,tolerance_s=.2) for j,name in enumerate(norms['b_names'])}}
        if capture_dir:
            path=Path(capture_dir)/(e['key']+'.npz');path.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(path,truth=real,prediction=pred,hazard_truth=b,hazard_prediction=bp)
            row['capture_path']=str(path);row['capture_sha256']=digest(path)
        rows.append(row)
    return rows


def _scene_native_sample(model, x, norms, device, generation_seed, steps=8, noise=None):
    g=torch.Generator(device='cpu').manual_seed(generation_seed)
    with torch.no_grad():
        # Chunk each global vector-field update with its exact receptive halo.
        # A single indexed noise field is shared across every chunk.
        pred=model.sample_chunked(torch.from_numpy(x)[None],steps=steps,chunk_size=600,generator=g,noise=noise)[0].cpu().numpy()
    return pred*np.asarray(norms['y_scale'],np.float32)+np.asarray(norms['y_center'],np.float32)


def evaluate_scene(model, entries, norms, metric_scales, device, *, generation_seeds=(101,),
                   steps=8, controls=False, capture_dir=None, circular_shifts=0):
    from .stage11_metrics import scene_metrics
    model.eval(); rows=[]
    for index,e in enumerate(entries):
        with np.load(e['path'],allow_pickle=False) as z:
            raw_x=z['x'].astype(np.float32);real=z['y'].astype(np.float32)
        x=(raw_x-np.asarray(norms['x_mean'],np.float32))/np.asarray(norms['x_std'],np.float32)
        phase=np.mod(np.arctan2(raw_x[:,48],raw_x[:,49])/(2*np.pi),1)
        beats=np.flatnonzero(np.r_[False,np.diff(phase)<-.5])/10
        for gs in generation_seeds:
            pred=_scene_native_sample(model,x,norms,device,gs,steps)
            metrics=scene_metrics(real,pred,beats,metric_scales,fps=10,circular_shifts=circular_shifts,seed=1100+index)
            row={'key':e['key'],'canonical_id':e['canonical_id'],'generation_seed':gs,
                 'n_frames':len(real),'metrics':metrics}
            if controls:
                donor_candidates=sorted([d for d in entries if d['canonical_id']!=e['canonical_id']],
                    key=lambda d:json_digest([e['key'],d['key'],'donor']))
                donor_rows=[];donor_groups=set()
                for d in donor_candidates:
                    if d['canonical_id'] not in donor_groups:
                        donor_rows.append(d);donor_groups.add(d['canonical_id'])
                    if len(donor_rows)==6:break
                row['donors']=[]
                for d in donor_rows:
                    with np.load(d['path'],allow_pickle=False) as z:dx=z['x'].astype(np.float32)
                    n=min(len(dx),len(raw_x))
                    combined=np.concatenate([dx[:n,:48],raw_x[:n,48:]],1)
                    combined=(combined-np.asarray(norms['x_mean'],np.float32))/np.asarray(norms['x_std'],np.float32)
                    noise=torch.randn((1,n,6),generator=torch.Generator().manual_seed(gs))
                    dp=_scene_native_sample(model,combined,norms,device,gs,steps,noise=noise)
                    matched_prefix=_scene_native_sample(model,x[:n],norms,device,gs,steps,noise=noise)
                    # Both generation contexts AND initial noise match exactly.
                    dm=scene_metrics(real[:n],dp,beats[beats<n/10],metric_scales,fps=10,circular_shifts=0)
                    mm=scene_metrics(real[:n],matched_prefix,beats[beats<n/10],metric_scales,fps=10,circular_shifts=0)
                    row['donors'].append({'key':d['key'],'n_frames':n,'noise_sha256':__import__('hashlib').sha256(noise.numpy().tobytes()).hexdigest(),'matched_metrics':mm,'donor_metrics':dm})
            if capture_dir:
                path=Path(capture_dir)/(e['key']+f'_g{gs}.npz');path.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(path,truth=real,prediction=pred,beats=beats)
                row['capture_path']=str(path);row['capture_sha256']=digest(path)
            rows.append(row)
    return rows


def _selection_score(rows,family):
    groups=defaultdict(list)
    for r in rows:
        if family=='scene':value=r['metrics']['temporal_error']
        else:
            m=r['motion'];value=m.get('primary_r2')
            if value is None:
                channels=m.get('r2_by_channel',m.get('per_channel',{}))
                vals=[channels.get(k) for k in ('com_speed','vert_bounce_amp')]
                value=float(np.mean(vals)) if all(v is not None for v in vals) else None
        if value is not None and np.isfinite(value):groups[r['canonical_id']].append(float(value))
    if not groups:
        raise RuntimeError('No eligible validation songs')
    return float(np.mean([np.mean(v) for v in groups.values()]))


def train_run(root, arm, seed, *, resume=False, engineering_steps=None, engineering_device=None):
    root=Path(root);protocol=read_json(root/'protocol.json');cfg=dict(protocol['arms'][arm]);family=cfg['family']
    if seed not in protocol['training_seeds'] and seed not in protocol.get('extension_seeds',[]):
        raise ValueError('Seed outside frozen protocol')
    data=verify_training_inputs(root) if engineering_steps is None else read_authenticated(root/'training_data.json')
    norms=read_authenticated(root/'normalizers.json')
    run=root/('engineering_runs' if engineering_steps is not None else 'runs')/arm/f'seed_{seed}'
    if (run/'report.json').exists():
        raise RuntimeError(f'Completed output already exists: {run}')
    if run.exists() and any(run.iterdir()) and not resume:
        raise RuntimeError(f'Incomplete output exists; explicit resume required: {run}')
    run.mkdir(parents=True,exist_ok=True)
    if engineering_steps is None:verify_sources(root)
    require_space(root)
    seed_all(seed)
    if engineering_device is not None and engineering_steps is None:
        raise ValueError('Engineering device override cannot enter scientific training')
    device=engineering_device or 'cuda'
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('GPU required; CPU fallback prohibited')
    tr=[e for e in data[family] if e['partition']=='train' and (cfg.get('population')!='old' or e['is_old'])]
    va=[e for e in data[family] if e['partition']=='val']
    if engineering_steps is not None:tr=tr[:8];va=tr[:2]
    # Input caches are useful VRAM occupancy, not anonymous memory padding.
    resident=ResidentDataset(tr,norms[family],family,device=device if cfg.get('resident_gpu',False) else 'cpu')
    model=_model(family,arm).to(device)
    init_sha=json_digest({name:__import__('hashlib').sha256(p.detach().cpu().numpy().tobytes()).hexdigest()
                          for name,p in model.named_parameters() if name.startswith(('inp.','blocks.','encoder.','audio_'))})
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=.01)
    total=int(engineering_steps or cfg['steps'])
    warm=max(1,int(.05*total))
    def multiplier(step):
        if step<warm:return (step+1)/warm
        t=min(1.,(step-warm)/max(1,total-warm))
        return .1+.9*.5*(1+math.cos(math.pi*t))
    sched=(torch.optim.lr_scheduler.OneCycleLR(opt,cfg['lr'],total_steps=total,pct_start=.05)
           if family=='motion' else torch.optim.lr_scheduler.LambdaLR(opt,multiplier))
    rng=np.random.default_rng(seed+10000);donor_rng=np.random.default_rng(seed+20000)
    flow_g=torch.Generator(device=device).manual_seed(seed+30000)
    pw=torch.tensor(norms['motion']['pos_weight'],device=device) if family=='motion' else None
    start_step=0;training_seconds=0.;best_score=None;best_step=None;pending_validation=None;started=time.time()
    identity={'protocol_sha256':digest(root/'protocol.json'),'data_sha256':digest(root/'training_data.json'),
              'normalizers_sha256':digest(root/'normalizers.json'),'arm':arm,'seed':seed,'config':cfg,
              'initial_encoder_sha256':init_sha}
    if engineering_steps is None:identity['source_sha256']=read_json(root/'frozen_sources.json')['source_sha256']
    if resume:
        blob=torch.load(run/'latest.pt',map_location=device,weights_only=False)
        if blob['identity']!=identity:raise RuntimeError('Resume identity mismatch')
        model.load_state_dict(blob['model']);opt.load_state_dict(blob['optimizer']);sched.load_state_dict(blob['scheduler'])
        restore_rng(blob['rng'],rng);donor_rng.bit_generator.state=blob['donor_rng'];flow_g.set_state(blob['flow_rng'].cpu())
        start_step=blob['step'];training_seconds=blob['training_seconds'];best_score=blob['best_score'];best_step=blob['best_step']
        pending_validation=blob.get('pending_validation')
    def snapshot(step, pending=None):
        blob={'schema_version':1,'identity':identity,'model':model.state_dict(),
              'cfg':model.cfg.__dict__ if family=='motion' else model.config_dict(),
              'family':family,'normalizers':norms[family],'metric_scales':norms['scene_metrics'],
              'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),'rng':rng_state(rng),
              'donor_rng':donor_rng.bit_generator.state,'flow_rng':flow_g.get_state(),
              'step':step,'training_seconds':training_seconds,'best_score':best_score,'best_step':best_step,
              'pending_validation':pending}
        return blob
    def save(step,pending=None):
        blob=snapshot(step,pending)
        atomic_torch(run/'latest.pt',blob)
        return blob
    def validate(step):
        nonlocal best_score,best_step
        # Publish the pending state before any outcome is produced. Recovery
        # reuses an authenticated finished capture rather than redrawing it.
        save(step,pending=step)
        path=run/f'validation_step_{step}.json'
        state=rng_state(rng);fg=flow_g.get_state();dg=donor_rng.bit_generator.state
        if path.exists() and Path(str(path)+'.sha256.json').exists():
            result=read_authenticated(path)
            if result['identity']!=identity or result['step']!=step:
                raise RuntimeError('Validation recovery identity mismatch')
            score=result['selection_score']
        else:
            if family=='motion':rows=evaluate_motion(model,va,norms[family],device)
            else:rows=evaluate_scene(model,va,norms[family],norms['scene_metrics'],device,generation_seeds=(101,),controls=False)
            score=_selection_score(rows,family)
            authenticated_json(path,{'identity':identity,'step':step,'rows':rows,'selection_score':score})
        restore_rng(state,rng);flow_g.set_state(fg);donor_rng.bit_generator.state=dg;model.train()
        improved=best_score is None or (score>best_score if family=='motion' else score<best_score)
        if improved:
            best_score=score;best_step=step
            # Commit the weights before latest refers to them as the best.
            atomic_torch(run/'best.pt',snapshot(step))
        save(step)
    atomic_json(run/'identity.json',identity)
    milestones=set(cfg.get('validation_steps',[total])) if engineering_steps is None else {total}
    last_save=time.time();model.train()
    if device=='cuda':torch.cuda.reset_peak_memory_stats()
    if pending_validation is not None:
        validate(int(pending_validation))
    with (run/'training.jsonl').open('a') as log:
        for step in range(start_step,total):
            if device=='cuda':torch.cuda.synchronize()
            ts=time.perf_counter()
            x,y,b=resident.batch(cfg['batch'],cfg['frames'],rng,device,swapped=arm=='S_swapped',donor_rng=donor_rng)
            opt.zero_grad(set_to_none=True)
            if family=='motion':
                out=model(x);la=F.l1_loss(out['a'],y[:,:,:4]);lb=F.binary_cross_entropy_with_logits(out['b_logit'],b,pos_weight=pw)
                lc=model.head_c.loss(out['h'],y[:,:,4:]);loss=la+.3*lb+lc
            else:loss=model.loss(x,y,generator=flow_g)
            if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite loss at {step}')
            loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(grad):raise FloatingPointError(f'Nonfinite gradient at {step}')
            opt.step();sched.step()
            if device=='cuda':torch.cuda.synchronize()
            training_seconds+=time.perf_counter()-ts
            if (step+1)%cfg.get('log_every',100)==0 or step==0 or step+1==total:
                line={'step':step+1,'loss':float(loss.detach()),'grad_norm':float(grad),'lr':sched.get_last_lr()[0],
                      'training_seconds':training_seconds,'wall_seconds':time.time()-started,'at':time.time()}
                log.write(json.dumps(line)+'\n');log.flush();print(json.dumps({'arm':arm,'seed':seed,**line}),flush=True)
                atomic_json(run/'status.json',{'status':'training',**line})
            checkpoint_due=(step+1)%int(cfg.get('checkpoint_every_steps',1000))==0
            if time.time()-last_save>=300 or checkpoint_due or step+1 in milestones:
                save(step+1,pending=step+1 if step+1 in milestones else None);last_save=time.time();require_space(root)
            if step+1 in milestones:
                validate(step+1)
        blob=save(total)
    if engineering_steps is None:
        verify_sources(root)
        verify_training_inputs(root)
    report={'schema_version':1,'identity':identity,'status':'completed','steps':total,
            'windows':total*cfg['batch'],'frames':total*cfg['batch']*cfg['frames'],
            'training_seconds':training_seconds,'wall_seconds':time.time()-started,
            'best_step':best_step,'best_validation_score':best_score,'precision':'fp32',
            'gpu':torch.cuda.get_device_name() if device=='cuda' else 'CPU engineering only','torch':torch.__version__,
            'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30 if device=='cuda' else 0.,
            'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30 if device=='cuda' else 0.,'nonfinite':False,'oom':False,
            'artifacts':artifact_manifest([run/'best.pt',run/'latest.pt',run/'identity.json',run/'training.jsonl'])}
    authenticated_json(run/'report.json',report);atomic_json(run/'status.json',{'status':'completed','at':time.time()})
    del resident,model,opt;gc.collect();torch.cuda.empty_cache()
    return report


def load_run_model(root,arm,seed):
    run=Path(root)/'runs'/arm/f'seed_{seed}'
    blob=torch.load(run/'best.pt',map_location='cpu',weights_only=False)
    model=_model(blob['family'],arm);model.load_state_dict(blob['model'])
    return model,blob
