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
    return config, path


def validate_music(timeline, audio_path, config):
    """Reject wrong songs, alternate encoders, interventions and fake embeddings."""
    timeline.validate()
    if timeline.metadata.get('input_sha256') != sha(audio_path):
        raise ValueError('MusicTimeline belongs to a different audio input')
    if timeline.metadata.get('intervention') is not None:
        raise ValueError('Default generation requires the original waveform, not an intervention cache')
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


def load_model(config, device='cpu', checkpoint=None):
    import torch
    from lightsem.experiments.native_dynamics.model import NativeDynamicsModel, Config
    from lightsem.experiments.native_dynamics.evaluation import setup
    from lightsem.experiments.motion_study.music_model import MusicConfig
    path = checked_asset(config['checkpoint'], checkpoint)
    setup()
    torch.set_num_threads(4)
    state = torch.load(path, map_location='cpu', weights_only=False)
    model_config = Config(**state['config'])
    if model_config.arm != 'joint' or model_config.extra_dim != 1061:
        raise ValueError('Checkpoint is not the MuQ-conditioned joint native C architecture')
    model = NativeDynamicsModel(MusicConfig(**state['base_config']), state['normalizers'], model_config)
    model.load_state_dict(state['model'], strict=True)
    model = model.to(device).eval().requires_grad_(False)
    norms = [tuple(torch.tensor(state[name][key], device=device, dtype=torch.float32)
                   for key in ('mean', 'std')) for name in ('audio_normalizers', 'extra_normalizers')]
    return model, norms, {'path': str(path), 'sha256': sha(path),
                          'training_seed': config['training_seed'],
                          'quiet_gradient_fix_in_checkpoint': config['trained_recipe']['quiet_gradient_fix_in_checkpoint']}
