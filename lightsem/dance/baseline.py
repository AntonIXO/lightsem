"""Checkpoint and waveform bindings for the user-selected fixed-C baseline."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO / 'configs/dance/fixed_c.json'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def checked_asset(item, override=None):
    path = Path(override or item['path']).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Baseline asset missing: {path}; restore the model/data mount or supply its documented override')
    if sha(path) != item['sha256']:
        raise ValueError(f'Baseline asset SHA256 mismatch: {path}')
    return path


def load_config(path=DEFAULT_CONFIG):
    path = Path(path).expanduser().resolve()
    config = json.loads(path.read_text())
    if config.get('schema_version') != 1 or config.get('status') != 'user_selected_motion_baseline':
        raise ValueError('Expected a selected fixed-C baseline configuration')
    if config.get('muq', {}).get('required') is not True or config['muq'].get('frozen') is not True:
        raise ValueError('Frozen MuQ is mandatory in the baseline and every option combination')
    if config['solver_steps'] != 32 or config['body_fps'] != 30 or config['lighting_fps'] != 40:
        raise ValueError('Fixed C requires its native32-step/30Hz and lighting40Hz contracts')
    if config['lighting']['transfer']['apply_motion_limits'] is not False:
        raise ValueError('The selected simulation baseline has fixture motion limits OFF')
    if config['lighting']['mapper'] != '01' or config['addons']['d6']['maximum_rate'] != 1.5:
        raise ValueError('Unsupported mapper or external clock recipe; select the verified01/positive-D6 definition')
    if any(config['addons'][name]['enabled'] for name in ('d4max', 'd6')):
        raise ValueError('D4-max and D6 must be explicitly enabled at the CLI')
    stereo = config.get('stereo_conditioning')
    if stereo and stereo.get('enabled'):
        if stereo.get('fusion') != 'dual_cross_attention' or stereo.get('feature_dim') != 128:
            raise ValueError('The active stereo baseline must use authenticated dual 128-D conditioning')
        manifest = Path(stereo.get('context_manifest', ''))
        if not manifest.is_file() or sha(manifest) != stereo.get('context_manifest_sha256'):
            raise ValueError('Active stereo context manifest is missing or hash-mismatched')
    return config, path


def validate_music(timeline, audio_path, config):
    """Reject wrong songs, alternate encoders, interventions and fake embeddings."""
    timeline.validate()
    if timeline.metadata.get('input_sha256') != sha(audio_path):
        raise ValueError('MusicTimeline belongs to a different audio input')
    if timeline.metadata.get('intervention') is not None:
        raise ValueError('Default generation requires the original waveform, not an intervention cache')
    recipe = timeline.metadata.get('recipe')
    if isinstance(recipe, dict):
        expected_recipe = config['muq']
        if (recipe.get('muq_layer') != expected_recipe['layer']
                or recipe.get('muq_native_hz') != expected_recipe['native_hz']
                or recipe.get('muq_chunk_seconds') != expected_recipe['chunk_seconds']
                or 'All-In-One' not in str(recipe.get('structure', ''))
                or 'pinned HTDemucs' not in str(recipe.get('separator', ''))):
            raise ValueError('MusicTimeline extractor recipe is not the pinned native contract')
    muq = timeline.metadata.get('muq', {})
    receipt = muq.get('model_receipt', {})
    expected = config['muq']
    weights = receipt.get('artifacts', {}).get('model.safetensors', {}).get('sha256')
    if (muq.get('representation') != 'muq' or muq.get('layer') != expected['layer']
            or receipt.get('model_id') != expected['model_id']
            or receipt.get('revision') != expected['revision']
            or weights != expected['weights_sha256']):
        raise ValueError('MusicTimeline must contain the pinned frozen MuQ layer and checkpoint')
    if timeline.muq.shape[1] != 1024 or not np.isfinite(timeline.muq).all():
        raise ValueError('Invalid MuQ embeddings')


def validate_checkpoint_state(state, *, expected_seed=None, expected_arm='joint'):
    """Validate the self describing provenance carried by a native checkpoint.

    The checkpoint is the authority for its seed and configuration.  Callers may
    constrain those values, but must not manufacture identity from a copied
    product configuration.
    """
    if not isinstance(state, dict):
        raise ValueError('Native checkpoint must contain a mapping state')
    required = ('model', 'config', 'base_config', 'identity', 'normalizers',
                'audio_normalizers', 'extra_normalizers', 'feature_names')
    missing = [key for key in required if key not in state]
    if missing:
        raise ValueError(f'Native checkpoint missing provenance fields: {missing}')
    identity = state['identity']
    if not isinstance(identity, dict) or not isinstance(identity.get('seed'), int):
        raise ValueError('Native checkpoint identity must include an integer seed')
    if identity['seed'] < 0 or (expected_seed is not None and identity['seed'] != expected_seed):
        raise ValueError('Native checkpoint seed does not match the requested run')
    config = state['config']
    if not isinstance(config, dict) or config.get('arm') != identity.get('arm'):
        raise ValueError('Checkpoint identity arm differs from model configuration')
    if expected_arm is not None and config.get('arm') != expected_arm:
        raise ValueError('Checkpoint is not the requested native arm')
    if config.get('extra_dim') != 1061 or config.get('steps') != 32:
        raise ValueError('Checkpoint native configuration violates the fixed contract')
    names = state['feature_names']
    if (not isinstance(names, (list, tuple)) or len(names) != config['extra_dim']
            or any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != len(names)):
        raise ValueError('Checkpoint feature_names are missing, duplicated, or mis-sized')
    for label in ('normalizers', 'audio_normalizers', 'extra_normalizers'):
        norms = state[label]
        if not isinstance(norms, dict) or not {'mean', 'std'} <= set(norms):
            raise ValueError(f'Checkpoint {label} must contain mean and std')
        mean, std = np.asarray(norms['mean']), np.asarray(norms['std'])
        if mean.ndim != 1 or std.shape != mean.shape or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
            raise ValueError(f'Checkpoint {label} contains invalid normalizers')
    return state


def load_model(config, device='cpu', checkpoint=None):
    import torch
    from lightsem.experiments.native_dynamics.model import NativeDynamicsModel, Config
    from lightsem.experiments.native_dynamics.evaluation import setup
    from lightsem.experiments.motion_study.music_model import MusicConfig
    path = checked_asset(config['checkpoint'], checkpoint)
    setup()
    torch.set_num_threads(4)
    state = torch.load(path, map_location='cpu', weights_only=False)
    validate_checkpoint_state(state, expected_seed=config.get('training_seed'), expected_arm='joint')
    from lightsem.experiments.dynamic_dance.contracts import SOURCE_NAMES, SOURCE_FIELDS, MIXTURE_FIELDS
    from lightsem.experiments.dynamic_dance.audio import SECTION_LABELS
    expected_names = ([f'muq_{i}' for i in range(1024)]
                      + [f'{source}_{field}' for source in SOURCE_NAMES for field in SOURCE_FIELDS]
                      + [f'mixture_{k}' for k in MIXTURE_FIELDS]
                      + [f'{event}_{side}_s' for event in ('beat', 'downbeat', 'section')
                         for side in ('previous', 'next')]
                      + [f'section_{label}' for label in SECTION_LABELS])
    if list(state.get('feature_names', ())) != expected_names:
        raise ValueError('Checkpoint feature_names/order differs from the fixed native schema')
    for key, expected_dim in (('audio_normalizers', 92), ('extra_normalizers', 1061)):
        norm = state.get(key)
        if not isinstance(norm, dict) or any(name not in norm for name in ('mean', 'std')):
            raise ValueError(f'Checkpoint is missing {key}')
        mean, std = np.asarray(norm['mean']), np.asarray(norm['std'])
        if mean.shape != (expected_dim,) or std.shape != (expected_dim,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
            raise ValueError(f'Invalid {key} shape or values')
    model_config = Config(**state['config'])
    if model_config.arm != 'joint' or model_config.extra_dim != 1061:
        raise ValueError('Checkpoint is not the MuQ-conditioned joint native C architecture')
    model = NativeDynamicsModel(MusicConfig(**state['base_config']), state['normalizers'], model_config)
    model.load_state_dict(state['model'], strict=True)
    model = model.to(device).eval().requires_grad_(False)
    norms = [tuple(torch.tensor(state[name][key], device=device, dtype=torch.float32)
                   for key in ('mean', 'std')) for name in ('audio_normalizers', 'extra_normalizers')]
    if model_config.stereo_dim:
        for name in ('stereo_mean', 'stereo_std'):
            if name not in state:
                raise ValueError(f'Stereo checkpoint is missing {name}')
        norms.append((torch.tensor(state['stereo_mean'], device=device, dtype=torch.float32),
                      torch.tensor(state['stereo_std'], device=device, dtype=torch.float32)))
    identity = state['identity']
    return model, norms, {'path': str(path), 'sha256': sha(path),
                          'training_seed': identity['seed'],
                          'arm': identity.get('arm'),
                          'stereo_fusion': model_config.stereo_fusion,
                          'stereo_dim': model_config.stereo_dim,
                          'protocol_sha256': identity.get('protocol_sha256'),
                          'quiet_gradient_fix_in_checkpoint': state.get('quiet_gradient_fix_in_checkpoint')}
