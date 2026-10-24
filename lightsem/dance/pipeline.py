"""Generate the selected native C, then optionally compose and pace its body."""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
import fcntl
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from .baseline import REPO, checked_asset, load_config, load_model, sha, validate_music
from lightsem.experiments.dynamic_dance.contracts import DanceResult, DancePlan, MusicTimeline


@contextmanager
def accelerator_lock():
    """Share the existing project locks with experimental and production jobs."""
    paths = [Path('/tmp/lightsem-audio-understanding-gpu.lock'),
             Path('/mnt/lightsem-data/experiments/lightsem/stage11_20260908/gpu.lock')]
    handles = []
    try:
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = path.open('a'); handles.append(handle)
            fcntl.flock(handle, fcntl.LOCK_EX)
        yield
    finally:
        for handle in reversed(handles):
            fcntl.flock(handle, fcntl.LOCK_UN); handle.close()


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def cached_track(config, audio_path):
    digest = sha(audio_path)
    return next(((key, item) for key, item in config['cached_tracks'].items()
                 if item['audio_sha256'] == digest), (None, None))


def validate_base92_clock(base, base_time, timeline_time):
    """Require one exact native-30Hz clock across Base92 and observations."""
    timeline_time = np.asarray(timeline_time, dtype=np.float64)
    base_time = np.asarray(base_time, dtype=np.float64)
    expected = np.arange(len(timeline_time), dtype=np.float64) / 30.
    base = np.asarray(base)
    if (base.shape != (len(timeline_time), 92) or base_time.shape != expected.shape
            or not np.array_equal(base_time, expected)
            or not np.array_equal(timeline_time, expected)
            or not np.isfinite(base).all()):
        raise ValueError('Invalid paired native30Hz Base92 features')


def _load_stereo_manifest(path, *, audio_hash, schema, allow_automatic=False):
    """Validate and select one exact-waveform row from a stereo manifest."""
    path = Path(path).expanduser().resolve()
    manifest = json.loads(path.read_text())
    if manifest.get('schema') != schema or manifest.get('feature_dim') != 128:
        raise ValueError('Stereo context manifest schema differs from the active baseline')
    if manifest.get('status') != 'complete' or manifest.get('test_access') is not False:
        raise ValueError('Stereo context manifest is not an authenticated complete artifact')
    if allow_automatic and manifest.get('automatic_for_waveform') is not True:
        raise ValueError('Automatic stereo context is missing its waveform extraction marker')
    if allow_automatic:
        checkpoint_pair = manifest.get('method_checkpoint_sha256')
        if checkpoint_pair is not None:
            if not isinstance(checkpoint_pair, list) or len(checkpoint_pair) != 2:
                raise ValueError('Automatic stereo context teacher hash pair is invalid')
        for key in ('spain_checkpoint_sha256', 'obsrnn_checkpoint_sha256', 'obsrnn_source_commit'):
            if not manifest.get(key):
                raise ValueError(f'Automatic stereo context is missing {key}')
    rows = {row['id']: row for row in manifest.get('rows', [])}
    row = next((value for value in rows.values() if value.get('audio_sha256') == audio_hash), None)
    return manifest, row


def load_stereo_context(audio_path, music, config, *, automatic_manifest=None):
    """Load the hash-bound 128-D external context for the active baseline."""
    spec = config.get('stereo_conditioning')
    if not spec or not spec.get('enabled'):
        return None, None
    manifest_path = Path(spec['context_manifest']).expanduser().resolve()
    if sha(manifest_path) != spec['context_manifest_sha256']:
        raise ValueError('Stereo context manifest hash mismatch')
    audio_hash = sha(audio_path)
    manifest, row = _load_stereo_manifest(manifest_path, audio_hash=audio_hash, schema=spec['schema'])
    if row is None and automatic_manifest is not None:
        manifest_path = Path(automatic_manifest).expanduser().resolve()
        manifest, row = _load_stereo_manifest(manifest_path, audio_hash=audio_hash,
                                               schema=spec['schema'], allow_automatic=True)
        expected_teachers = json.loads(Path(spec['context_manifest']).read_text()).get('method_checkpoint_sha256')
        if (not isinstance(expected_teachers, list) or len(expected_teachers) != 2
                or manifest.get('spain_checkpoint_sha256') != expected_teachers[0]
                or manifest.get('obsrnn_checkpoint_sha256') != expected_teachers[1]):
            raise ValueError('Automatic stereo context teacher checkpoints differ from the active baseline')
    if row is None:
        raise FileNotFoundError(f'No authenticated {spec["fusion"]} context for waveform {audio_hash}')
    context_path = Path(row['context_path']).expanduser().resolve()
    if sha(context_path) != row['context_sha256'] or row.get('audio_sha256') != audio_hash:
        raise ValueError('Stereo context/audio binding mismatch')
    with np.load(context_path, allow_pickle=False) as values:
        source_time = np.asarray(values['time30'], dtype=np.float64)
        source = np.asarray(values['fused30'], dtype=np.float32)
    if source.ndim != 2 or source.shape[1] != 128 or len(source_time) != len(source):
        raise ValueError('Invalid active stereo context shape')
    if not np.isfinite(source).all() or not np.isfinite(source_time).all() or np.any(np.diff(source_time) <= 0):
        raise ValueError('Invalid active stereo context values/clock')
    target = np.asarray(music.time, dtype=np.float64)
    stereo = np.stack([np.interp(target, source_time, source[:, i]) for i in range(128)], axis=1).astype(np.float32)
    return stereo, {'manifest': str(manifest_path), 'manifest_sha256': sha(manifest_path),
                    'context': str(context_path), 'context_sha256': sha(context_path),
                    'audio_sha256': audio_hash, 'schema': manifest['schema'],
                    'fusion': spec['fusion'], 'automatic_for_waveform': bool(manifest.get('automatic_for_waveform'))}


def _manifest_has_audio(path, audio_hash, schema):
    try:
        _, row = _load_stereo_manifest(path, audio_hash=audio_hash, schema=schema)
        return row is not None
    except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError):
        return False


def ensure_automatic_stereo_context(audio_path, feature_root, config, *, audio_python, device):
    """Extract raw SpaIn/oBSRNN context once for an unregistered waveform."""
    spec = config.get('stereo_conditioning')
    if not spec or not spec.get('enabled'):
        return None
    audio_hash = sha(audio_path)
    reference = Path(spec['context_manifest']).expanduser().resolve()
    if _manifest_has_audio(reference, audio_hash, spec['schema']):
        return None
    context_root = Path(feature_root) / 'stereo_context'
    manifest_path = context_root / 'CONTEXT_MANIFEST.json'
    if manifest_path.is_file() and _manifest_has_audio(manifest_path, audio_hash, spec['schema']):
        return manifest_path
    if context_root.exists() and any(context_root.iterdir()):
        if (context_root / 'FAILURE.json').is_file():
            retry = 1
            while True:
                candidate = Path(feature_root) / f'stereo_context_retry{retry}'
                if not candidate.exists():
                    context_root = candidate
                    manifest_path = context_root / 'CONTEXT_MANIFEST.json'
                    break
                retry += 1
        else:
            raise FileExistsError(f'Preserving incomplete automatic stereo context: {context_root}')
    interpreter = audio_python or config['extraction']['python']
    command = [str(interpreter), '-m', 'lightsem.dance.extract_stereo_context',
               '--wav', str(Path(audio_path).resolve()), '--out', str(context_root), '--device', device]
    environment = {**os.environ, 'PYTHONPATH': str(REPO), 'PYTHONDONTWRITEBYTECODE': '1'}
    try:
        subprocess.run(command, cwd=REPO, env=environment, check=True)
    except BaseException as error:
        write_json(context_root / 'FAILURE.json', {'error': f'{type(error).__name__}: {error}',
                                                   'audio_sha256': audio_hash, 'fallback': False})
        raise
    if not manifest_path.is_file():
        raise FileNotFoundError(f'Automatic stereo context did not produce {manifest_path}')
    return manifest_path


def music_input(audio_path, root, config, *, timeline_path=None, base92_path=None,
                audio_python=None, audio_device=None, audio_cache=None):
    key, cached = cached_track(config, audio_path)
    feature_root = Path(audio_cache or '/mnt/lightsem-data/cache/native_c_cli') / sha(audio_path)
    feature_root.mkdir(parents=True, exist_ok=True)
    if timeline_path is not None:
        timeline_path = Path(timeline_path).expanduser().resolve()
    elif cached is not None:
        timeline_path = checked_asset(cached['timeline'])
        base92_path = base92_path or checked_asset(cached['base92'])
    else:
        interpreter = audio_python or config['extraction']['python']
        command = [str(interpreter), '-m', 'lightsem.dance.extract', '--wav', str(audio_path),
                   '--root', str(feature_root), '--device', audio_device or config['extraction']['device']]
        environment = {**os.environ, 'PYTHONPATH': str(REPO), 'PYTHONDONTWRITEBYTECODE': '1'}
        subprocess.run(command, cwd=REPO, env=environment, check=True)
        timeline_path = feature_root / 'features/timelines/input.npz'
    music = MusicTimeline.load(timeline_path)
    validate_music(music, audio_path, config)
    base92_path = Path(base92_path or music.metadata.get('base92', {}).get('path', ''))
    if not base92_path.is_file():
        raise FileNotFoundError('Paired Base92 features are missing; provide --base92 or extract this waveform')
    from lightsem.experiments.dynamic_dance.training import validate_audio_binding
    validate_audio_binding(music, base92_path)
    with np.load(base92_path, allow_pickle=False) as values:
        if 'x30' not in values or 'time30' not in values:
            raise ValueError('Base92 cache must contain native x30 and time30 arrays')
        base = values['x30'].astype(np.float32)
        base_time = np.asarray(values['time30'], dtype=np.float64)
    validate_base92_clock(base, base_time, music.time)
    duration = music.metadata.get('duration_s')
    if duration is not None and abs(float(duration) - len(base) / 30.) > 1 / 30. + 1e-6:
        raise ValueError('Base92/native body coverage differs from analyzed audio duration')
    automatic_manifest = ensure_automatic_stereo_context(
        audio_path, feature_root, config,
        audio_python=audio_python or config['extraction']['python'],
        device=audio_device or config['extraction']['device'])
    stereo, stereo_binding = load_stereo_context(audio_path, music, config,
                                                 automatic_manifest=automatic_manifest)
    return music, base, {'timeline': str(timeline_path), 'timeline_sha256': sha(timeline_path),
                         'base92': str(base92_path.resolve()), 'base92_sha256': sha(base92_path),
                         'audio_sha256': sha(audio_path), 'cached_track': key,
                         'stereo_conditioning': stereo_binding}, stereo


def native_body(music, base, config, *, stereo=None, device='cpu', checkpoint=None, draw_seed=None):
    import torch
    from lightsem.experiments.dynamic_dance.audio import model_features
    from lightsem.experiments.native_dynamics.evaluation import cpu_noise
    from lightsem.experiments.native_dynamics.dataset import CONTEXT_BINS
    from lightsem.experiments.motion_study.geometry import joints_from_features
    model, norms, identity = load_model(config, device, checkpoint)
    if model.config.stereo_dim:
        if stereo is None or len(norms) < 3 or stereo.shape != (len(base), model.config.stereo_dim):
            raise ValueError('Active stereo baseline requires paired native stereo context')
        stereo_mean, stereo_std = norms[2]
    elif stereo is not None:
        raise ValueError('Stereo context supplied to a non-stereo baseline')
    n = len(base); extra, names = model_features(music, np.arange(n) / 30.)
    bins, pad = (n + 14) // 15, (-n) % 15
    # CONTEXT_BINS is the fixed padded width used by the training batches, not
    # an architectural limit at inference.  The native planner has no learned
    # positional table and accepts any finite valid prefix.  Grow the tensors
    # for longer songs so the complete analyzed timeline reaches the planner;
    # never truncate a song to the training width.
    planner_bins = max(CONTEXT_BINS, bins)
    pooled = np.pad(extra, ((0, pad), (0, 0))).reshape(bins, 15, 1061).sum(1)
    pooled /= np.minimum(15, n - np.arange(bins) * 15)[:, None]
    seed = config['draw_seed'] if draw_seed is None else draw_seed
    with torch.inference_mode():
        fine = torch.cat(((torch.tensor(base, device=device) - norms[0][0]) / norms[0][1],
                          (torch.tensor(extra, device=device) - norms[1][0]) / norms[1][1]), -1)[None]
        coarse = torch.zeros(1, planner_bins, 1061, device=device)
        valid = torch.zeros(1, planner_bins, dtype=torch.bool, device=device)
        coarse[0, :bins] = (torch.tensor(pooled, device=device) - norms[1][0]) / norms[1][1]
        valid[0, :bins] = True
        noise = cpu_noise(model.flow, max(config['noise_reference_frames'], n), seed)[:, :n].contiguous().to(device)
        if model.config.stereo_dim:
            stereo_t = (torch.tensor(stereo, device=device) - stereo_mean) / stereo_std
            stereo_pad = np.pad(stereo, ((0, (-n) % 15), (0, 0)))
            stereo_coarse = stereo_pad.reshape(bins, 15, model.config.stereo_dim).sum(1)
            stereo_coarse /= np.minimum(15, n - np.arange(bins) * 15)[:, None]
            # Keep the stereo planner tensor aligned with the full valid
            # planner span; plan_at_frames only reads the valid prefix.
            coarse_stereo = torch.zeros(1, coarse.shape[1], model.config.stereo_dim, device=device)
            coarse_stereo[0, :bins] = (torch.tensor(stereo_coarse, device=device) - stereo_mean) / stereo_std
            native, plan = model.generate(fine, coarse, valid, torch.arange(n, device=device)[None], noise,
                                          stereo_context=stereo_t[None], coarse_stereo=coarse_stereo)
        else:
            native, plan = model.generate(fine, coarse, valid, torch.arange(n, device=device)[None], noise)
        world = joints_from_features(native)[0].cpu().numpy()
        values = native[0].cpu().numpy()
    if not np.isfinite(values).all() or not np.isfinite(world).all():
        raise ValueError('Nonfinite generated native C; no fallback motion is available')
    metadata = {'baseline_id': config['id'], 'checkpoint': identity, 'draw_seed': seed,
                'noise_reference_frames': max(config['noise_reference_frames'], n),
                'noise_sha256': sha_array(noise.cpu().numpy()), 'fps': 30.,
                'motion_origin': 'Actual fixed native C generation', 'newly_generated_motion': True,
                'retiming': False, 'motif_recomposition': False, 'd4max_enabled': False, 'd6_enabled': False,
                'muq': music.metadata['muq'],
                'quiet_gradient_fix_in_checkpoint': identity.get('quiet_gradient_fix_in_checkpoint')}
    body = DanceResult(time=np.arange(n) / 30., world=world, native=values,
                       contacts=values[:, -4:], metadata=metadata)
    return body, extra, {'time': (np.arange(bins) * 15 + 7) / 30,
                         'conditioning': plan['conditioning'][0, :bins].cpu().numpy(),
                         'planner_context_bins': planner_bins,
                         'stereo_attention_window_frames': (
                             int(model.config.frames)
                             if model.config.stereo_fusion == 'dual_cross_attention' else None)}, identity


def sha_array(value):
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def classify_generation_status(config, metrics):
    """Classify a generated body before any mapper or renderer is invoked."""
    if not metrics.get('finite', False) or metrics.get('constant_body', False):
        return 'failure_diagnostic'
    qualification_status = config.get('qualification', {}).get('status')
    if isinstance(qualification_status, str) and 'unqualified' in qualification_status.lower():
        return 'unqualified'
    return 'finite_body_ready'


def d6_prediction(extra, config, *, device='cpu', checkpoint=None):
    import torch
    from lightsem.experiments.dynamic_dance.learning import EffortPlanner, PlannerConfig, plan_condition
    from lightsem.experiments.dynamic_dance.training import planner_predict
    path = checked_asset(config['addons']['d6']['checkpoint'], checkpoint)
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state['planner_config']['input_dim'] != 1061:
        raise ValueError('D6 must use the same MuQ/source feature schema')
    planner = EffortPlanner(PlannerConfig(**state['planner_config']))
    planner.load_state_dict(state['model'], strict=True)
    planner = planner.to(device).eval().requires_grad_(False)
    raw = planner_predict(planner, extra, state['extra_normalizers'], device=device)
    with torch.inference_mode():
        packed = plan_condition(raw['local']).numpy()
    # Planner bins represent half-second intervals; use their centers when
    # mapping learned predictions back to the native clock.
    return {'coarse_time': (np.arange(len(packed)) + .5) * .5,
            'coarse_local_plan': packed}, {'path': str(path), 'sha256': sha(path)}


def modify_body(source, music, config, *, d4max=False, d6=False, telemetry=None):
    """D6 can pace native C directly; D4 is never implicitly enabled."""
    from lightsem.experiments.d4max_d8.motifs import Config, compose, phrase_nodes
    from lightsem.experiments.d4max.pacing import apply_d6
    options = config['addons']['d4max']
    cfg = Config(**{key: value for key, value in options.items() if key in Config.__dataclass_fields__})
    if d4max:
        result, plan, bank = compose(source.world, music, cfg)
        result.metadata.update(source_library_label='fixed_native_C', motion_origin='D4-max recomposition of own-song fixed C')
        # Keep the numeric motif-search inputs in the existing plan artifact so
        # D4 selection is replayable without creating a per-step JSON report.
        plan.diagnostics.update({
            'd4_bank_features': np.asarray(bank.get('features', [])),
            'd4_bank_medoids': np.asarray(bank.get('medoids', [])),
            'd4_bank_labels': np.asarray(bank.get('labels', [])),
            'd4_bank_node_bounds': np.asarray([[n['start'], n['end']] for n in bank.get('nodes', [])], dtype=np.int64),
        })
    else:
        result = source
        nodes = phrase_nodes(music, len(source.world), cfg)
        anchors = [nodes[0]['start']] + [node['end'] for node in nodes]
        plan = DancePlan(time=source.time.copy(), source_time=source.time.copy(),
                         effort=np.ones((len(source.time), 1), np.float32),
                         metadata={'motif_anchor_frames': anchors, 'anchor_scope': 'Audio phrase anchors; no fragment recomposition'})
    if d6:
        if telemetry is None:
            raise ValueError('D6 requires a real MuQ-conditioned planner prediction')
        result, plan, _ = apply_d6(result, plan, telemetry, music,
                                   minimum_rate=config['addons']['d6']['minimum_rate'])
        result.metadata.update(motion_origin='External D6 pacing of ' + ('D4-max composition' if d4max else 'fixed native C'),
                               source_clock_scope='Monotone clock into ' + ('D4 composition' if d4max else 'native C'))
    result.metadata.update(baseline_id=config['id'], d4max_enabled=d4max, d6_enabled=d6,
                           retiming=bool(d6 or d4max), motif_recomposition=d4max,
                           newly_generated_motion=not (d4max or d6),
                           native_speed_learning_claim=False if d6 else None,
                           muq=music.metadata['muq'], generation_provenance=source.metadata.copy())
    if result is source:
        result.metadata.pop('generation_provenance')
    plan.metadata.update(result.metadata)
    result.validate(); plan.validate()
    return result, plan


def generate(audio_path, root, *, baseline_config=None, d4max=False, d6=False, device='cpu',
             checkpoint=None, d6_checkpoint=None, timeline_path=None, base92_path=None,
             draw_seed=None, audio_python=None, audio_device=None, audio_cache=None):
    audio_path = Path(audio_path).expanduser().resolve(); root = Path(root).expanduser().resolve()
    config, config_path = load_config(baseline_config) if baseline_config else load_config()
    checked_asset(config['checkpoint'], checkpoint)
    if d6:
        checked_asset(config['addons']['d6']['checkpoint'], d6_checkpoint)
    if root.exists():
        raise FileExistsError(f'Run directory exists; preserve it and choose another --out: {root}')
    root.mkdir(parents=True)
    try:
        music, base, binding, stereo = music_input(audio_path, root, config, timeline_path=timeline_path,
                                         base92_path=base92_path, audio_python=audio_python, audio_device=audio_device,
                                         audio_cache=audio_cache)
        lock = accelerator_lock() if device.startswith('cuda') else nullcontext()
        with lock:
            source, extra, native_plan, identity = native_body(music, base, config, stereo=stereo, device=device,
                                                             checkpoint=checkpoint, draw_seed=draw_seed)
            telemetry, d6_identity = d6_prediction(extra, config, device=device, checkpoint=d6_checkpoint) if d6 else (None, None)
        source.metadata.update(audio=binding, d6_checkpoint=d6_identity)
        source.save(root / 'source/body.npz')
        result, plan = modify_body(source, music, config, d4max=d4max, d6=d6, telemetry=telemetry)
        if native_plan:
            native_diagnostics = {
                'native_planner_time': np.asarray(native_plan['time']),
                'native_planner_conditioning': np.asarray(native_plan['conditioning']),
                'native_planner_context_bins': int(native_plan['planner_context_bins']),
            }
            if native_plan['stereo_attention_window_frames'] is not None:
                native_diagnostics['native_stereo_attention_window_frames'] = int(
                    native_plan['stereo_attention_window_frames'])
            plan.diagnostics.update(native_diagnostics)
        if telemetry is not None:
            plan.diagnostics.update({
                'd6_coarse_time': np.asarray(telemetry['coarse_time']),
                'd6_coarse_local_plan': np.asarray(telemetry['coarse_local_plan']),
            })
        result.metadata.update(audio=binding, checkpoint=identity, d6_checkpoint=d6_identity)
        result.save(root / 'body.npz'); plan.save(root / 'plan.npz')
        from lightsem.experiments.native_repair.assessment import raw_metrics
        metrics = raw_metrics(result.world, source.world)
        qualification_status = config.get('qualification', {}).get('status')
        status = classify_generation_status(config, metrics)
        report = {'baseline_id': config['id'], 'baseline_config': str(config_path), 'baseline_config_sha256': sha(config_path),
                  'audio': binding, 'checkpoint': identity, 'd4max': d4max, 'd6': d6, 'muq_required': True,
                  'd6_checkpoint': d6_identity, 'raw_body': metrics, 'body_sha256': sha(root / 'body.npz'),
                  'source_body_sha256': sha(root / 'source/body.npz'),
                  'status': status, 'qualification_status': qualification_status,
                  'musical_superiority_claim': False,
                  'quiet_gradient_fix_in_checkpoint': identity.get('quiet_gradient_fix_in_checkpoint')}
        write_json(root / 'GENERATION.json', report)
        return result, source, music, config, report
    except BaseException as error:
        write_json(root / 'FAILURE.json', {'error': f'{type(error).__name__}: {error}', 'fallback': False})
        raise
