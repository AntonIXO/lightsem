from dataclasses import asdict

import torch

from lightsem.model.light_layers.model import LightLayersConfig, build_light_layers
from lightsem.model.light_layers.training import load_checkpoint
from scripts.light_layers.train_visual_next import checkpoint, software_versions


def test_full_report_version_metadata_loads_with_weights_only(tmp_path):
    reference = LightLayersConfig(visual_aux_dim=96, pose_aux_dim=0)
    model = build_light_layers(reference, architecture='independent')
    metadata = {
        'step': 0, 'scope': 'Untrained serialization regression fixture',
        'reference_model_config': asdict(reference), 'model_config': asdict(model.cfg),
        'architecture': 'independent', 'parameter_match': asdict(model.parameter_match),
        'software': software_versions(), 'original_representation_fit_gate_pass': False,
    }
    path = tmp_path / 'untrained.pt'
    checkpoint(path, model, metadata)
    restored, receipt = load_checkpoint(path, device='cpu')
    assert type(receipt['software']['torch']) is str
    assert receipt['original_representation_fit_gate_pass'] is False
    assert all(torch.equal(value, restored.state_dict()[name]) for name, value in model.state_dict().items())
