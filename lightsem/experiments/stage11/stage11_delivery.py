"""Build traceable full-song demonstrations and blind Stage 11 comparisons."""
from __future__ import annotations

from pathlib import Path
import gc
import shutil
import time
import numpy as np
import torch

from .stage11_common import (ROOT, authenticated_json, read_authenticated,
    digest, json_digest, artifact_manifest)
from .stage11_training import load_run_model, _scene_native_sample, verify_training_inputs


def pin_render_assets(root=ROOT):
    root=Path(root);folder=root/'assets';folder.mkdir(exist_ok=True)
    source=Path('/mnt/lightsem-data/ckpt/light_trunk_jd_none_smooth.pt')
    target=folder/'reference_jd_checkpoint.pt'
    if not target.exists():shutil.copy2(source,target);target.chmod(0o444)
    elif digest(source)!=digest(target):raise RuntimeError('Reference checkpoint changed')
    blob=torch.load(target,map_location='cpu',weights_only=False)
    cfg={'reference_checkpoint':str(target),'reference_sha256':digest(target),
         'event_thr':dict(blob.get('event_thr') or {}),'smooth_tau':dict(blob.get('smooth_tau') or {}),
         'postprocessing_policy':'same pinned incumbent thresholds and smoothing across both descriptor arms',
         'scene_mapping_policy':'fixed beat-pulse template plus existing wave preset; no learned RPMC state'}
    authenticated_json(root/'render_assets.json',cfg)
    return cfg


def motion_adapter(model, blob, assets, device):
    """Adapt new checkpoints without a mutable global normalizer lookup."""
    from lightsem.model.infer import LightModel
    lm=LightModel.__new__(LightModel)
    lm.device=device;lm.model=model.to(device).eval();lm.cfg=model.cfg
    norm=blob['normalizers'];lm.mu=torch.tensor(norm['x_mean'],dtype=torch.float32,device=device)
    lm.sd=torch.tensor(norm['x_std'],dtype=torch.float32,device=device)
    names=list(norm['a_names'])+list(norm['c_names'])
    lm.scalers={name:{'median':norm['y_center'][i],'iqr':norm['y_scale'][i]/1.5} for i,name in enumerate(names)}
    lm.block_a=tuple(norm['a_names']);lm.block_c=tuple(norm['c_names'])
    lm.event_thr=dict(assets['event_thr']);lm.smooth_tau=dict(assets['smooth_tau'])
    lm.zero_phase_hz=None;lm.zero_phase_match=False
    return lm


def _render_context(root, split):
    """Use the evaluator's complete frozen bindings before accessing a cohort."""
    from .stage11_evaluation import _context, _test_unlock
    root = Path(root)
    if split not in ('val', 'test'):
        raise ValueError('Rendering supports val or test')
    protocol, data, normalizers, binding = _context(root)
    if protocol.get('frozen') is not True:
        raise RuntimeError('Rendering requires a frozen scientific protocol')
    assets = read_authenticated(root/'render_assets.json')
    if digest(root/'render_assets.json') != protocol.get('render_assets_sha256'):
        raise RuntimeError('Frozen render postprocessing binding changed')
    if digest(assets['reference_checkpoint']) != assets['reference_sha256']:
        raise RuntimeError('Pinned render asset changed')
    unlock = _test_unlock(root, protocol, binding) if split == 'test' else None
    families = list(unlock['eligible_families']) if unlock else ['motion', 'scene']
    return protocol, data, normalizers, binding, assets, unlock, families


def _load_verified_render_model(root, arm, seed, protocol, normalizers, binding, *, unlock=None):
    """Authenticate the selected checkpoint and all transforms before inference."""
    from .stage11_evaluation import _training_report
    root = Path(root)
    if arm not in protocol['arms'] or seed not in protocol['training_seeds'] + protocol.get('extension_seeds', []):
        raise RuntimeError('Render arm/seed lies outside the frozen protocol')
    family = protocol['arms'][arm]['family']
    checkpoint = root/'runs'/arm/f'seed_{seed}'/'best.pt'
    if unlock is not None:
        selected = unlock['arms'].get(arm, {})
        if family not in unlock['eligible_families'] or seed not in selected.get('training_seeds', []):
            raise RuntimeError('Unreviewed family/arm/seed in test render')
        if selected.get('checkpoints', {}).get(str(seed)) != digest(checkpoint):
            raise RuntimeError('Unreviewed checkpoint in test render')
    report = _training_report(root, arm, seed, protocol, binding)
    checkpoint_sha = digest(checkpoint)
    artifact = report['artifacts'].get(str(checkpoint.resolve()))
    if not artifact or artifact['sha256'] != checkpoint_sha:
        raise RuntimeError('Selected render checkpoint is absent from its training artifact manifest')
    model, blob = load_run_model(root, arm, seed)
    if (blob.get('identity') != report['identity'] or blob.get('step') != report.get('best_step')
            or blob.get('family') != family):
        raise RuntimeError('Render checkpoint identity, selected step or family differs from the training report')
    if json_digest(blob.get('normalizers')) != json_digest(normalizers[family]):
        raise RuntimeError('Render checkpoint target/input normalization differs from the frozen calibration')
    if json_digest(blob.get('metric_scales')) != json_digest(normalizers['scene_metrics']):
        raise RuntimeError('Render checkpoint scene-metric calibration differs from the frozen calibration')
    model_cfg = model.config_dict() if hasattr(model, 'config_dict') else model.cfg.__dict__
    if json_digest(blob.get('cfg')) != json_digest(model_cfg):
        raise RuntimeError('Render checkpoint architecture differs from the reconstructed frozen model')
    provenance = {
        'arm': arm, 'family': family, 'training_seed': seed,
        'checkpoint_path': str(checkpoint), 'checkpoint_sha256': checkpoint_sha,
        'training_report_path': str(checkpoint.parent/'report.json'),
        'training_report_sha256': digest(checkpoint.parent/'report.json'),
        'selected_step': int(blob['step']), 'model_config_sha256': json_digest(model_cfg),
        'normalizers_sha256': json_digest(blob['normalizers']),
        'protocol_sha256': binding['protocol_sha256'],
    }
    return model, blob, provenance


def render_campaign(root=ROOT, split='val'):
    from .stage11_data import load_manifest
    from .stage11_render import (frames_scene,prepare_pairs,select_excerpt_start,
        render_frames,extract_audio_excerpt)
    from lightsem.motion.io import load_bus,slice_bus
    from lightsem.conductor.hybrid import ConductorHybrid
    root=Path(root).resolve()
    protocol,data,normalizers,binding,assets,unlock,families=_render_context(root,split)
    verify_training_inputs(root);manifest=load_manifest(root)
    rows=[r for r in manifest['recordings'] if r['partition']==split and r.get('eligible_core')]
    groups={}
    for r in sorted(rows,key=lambda r:json_digest([r['recording_id'],'render-recording'])):
        groups.setdefault(r['canonical_id'],r)
    ids=sorted(groups,key=lambda k:json_digest([k,'stage11-blind-cohort-v1']))[:20]
    if len(ids)!=20:raise RuntimeError('Need 20 complete canonical songs for blind comparisons')
    if set(families)=={'motion','scene'}:
        comparisons=[['D_full','D_old'],['S_temporal','S_pointwise']]
    elif families==['scene'] or set(families)=={'scene'}:
        comparisons=[['S_temporal','S_pointwise'],['S_temporal','S_swapped']]
    else:
        # Same eligible recipe comparison, separately declared sampling seed.
        comparisons=[['D_full','D_old'],['D_full_g102','D_old_g102']]
    needed=sorted({a for pair in comparisons for a in pair})
    candidates=[];models={};device='cuda';conductor=ConductorHybrid()
    frame_dir=root/'results'/'dmx_candidates';frame_dir.mkdir(parents=True,exist_ok=True)
    full_dir=root/'results'/'full_songs';full_dir.mkdir(exist_ok=True)
    full_examples=[]
    for index,canonical in enumerate(ids):
        r=groups[canonical]
        seed=int(json_digest([canonical,'training-seed'])[:8],16)%5
        bus_full=load_bus(Path(r['audio_path']))
        with np.load(r['visual_path'],allow_pickle=False) as z:
            sx=z['x'].astype(np.float32)
        duration=len(sx)/10
        bus=slice_bus(bus_full,duration_s=duration,offset_s=float(r['active_start_s']))
        if abs(len(bus.times)/40-duration)>.05:raise ValueError('Audio/scene duration mismatch')
        offset=select_excerpt_start(canonical,duration,salt='stage11-excerpts-v1',clip_s=30,margin_s=5)
        lo=int(round(offset*40));hi=lo+1200
        frame_arms={};full_arms={};arm_provenance={}
        for named in needed:
            arm=named.split('_g')[0];gs=102 if named.endswith('_g102') else 101
            key=(arm,seed)
            if key not in models:
                # Retain at most one model on the GPU; CPU checkpoint reuse is
                # cheap and never competes with a training process.
                models[key]=_load_verified_render_model(root,arm,seed,protocol,normalizers,binding,unlock=unlock)
            m,blob,model_provenance=models[key];m=m.to(device).eval()
            if blob['family']=='motion':
                lm=motion_adapter(m,blob,assets,device)
                desc=lm.descriptors(bus,seed=gs)
                fr=conductor.render(desc,bus).frames
            else:
                norm=blob['normalizers'];x=(sx-np.asarray(norm['x_mean'],np.float32))/np.asarray(norm['x_std'],np.float32)
                scene=_scene_native_sample(m,x,norm,device,gs,steps=8)
                fr=frames_scene(scene,bus.beats,bus.channels['bar_phase'],duration).frames
            if len(fr)<hi:raise ValueError('Renderer produced a short stream')
            path=frame_dir/f'{canonical}_{named}_seed{seed}.npy'
            if path.exists():raise FileExistsError(path)
            np.save(path,fr[lo:hi]);frame_arms[named]=str(path)
            full_arms[named]=fr
            arm_provenance[named]={**model_provenance,'generation_seed':gs,
                'render_assets_sha256':protocol['render_assets_sha256'],
                'dmx_path':str(path),'dmx_file_sha256':digest(path)}
            m.cpu();gc.collect();torch.cuda.empty_cache()
        candidates.append({'canonical_id':canonical,'music_id':r['recording_id'],
            'audio_path':r['wav_path'],'offset_s':float(r['active_start_s'])+offset,'duration_s':30.,
            'arms':frame_arms,'comparisons':comparisons,'arm_provenance':arm_provenance})
        if index<2:
            wav=extract_audio_excerpt(r['wav_path'],full_dir/f'song{index}.wav',float(r['active_start_s']),duration)
            for named in [comparisons[0][0],comparisons[1][0]]:
                path=full_dir/f'song{index}_{named}.mp4'
                if any(x['path']==str(path) for x in full_examples):continue
                render_frames(full_arms[named],path,audio_path=wav,duration_s=duration)
                full_examples.append({'canonical_id':canonical,'arm':named,'path':str(path),'duration_s':duration,
                                      'model_provenance':{k:v for k,v in arm_provenance[named].items()
                                                          if k not in ('dmx_path','dmx_file_sha256')}})
        print(f'Prepared blind render cohort {index+1}/20',flush=True)
    models.clear();gc.collect();torch.cuda.empty_cache()
    candidate_path=root/'results/render_candidates.json'
    authenticated_json(candidate_path,{'split':split,'candidates':candidates,
        'protocol_sha256':binding['protocol_sha256'],'comparisons':comparisons,'full_examples':full_examples})
    pairs=prepare_pairs(root,protocol,candidates,n_songs=20)
    for candidate in candidates:
        for provenance in candidate['arm_provenance'].values():
            if digest(provenance['dmx_path'])!=provenance['dmx_file_sha256']:
                raise RuntimeError('Source DMX candidate changed while rendering blind materials')
    files=list((root/'results/renders').rglob('*'))+list(full_dir.rglob('*'))
    files += [candidate_path,Path(str(candidate_path)+'.sha256.json')]
    files += [Path(path) for candidate in candidates for path in candidate['arms'].values()]
    files=[p for p in files if p.is_file() and p.name!='ratings.jsonl']
    report={'status':'complete','split':split,'protocol_sha256':binding['protocol_sha256'],
            'n_primary':pairs['n_primary'],'n_repeat':pairs['n_repeat'],'n_sham':pairs['n_sham'],
            'preference_status':'awaiting human ratings','development_only':split!='test',
            'full_examples':full_examples,'pairs_path':str(root/'results/renders/pairs.json'),
            'render_candidates_path':str(candidate_path),'render_candidates_sha256':digest(candidate_path),
            'artifacts':artifact_manifest(files),'completed_at':time.time()}
    authenticated_json(root/f'results/render_{split}.json',report)
    return report
