from pathlib import Path
import json
import numpy as np
import torch
from scripts.experiments.run_joint_open_legacy import JointOpenLegacyModel
from scripts.experiments.evaluate_foundation_v3_own_songs import load_model
from scripts.experiments.run_foundation_v3_audio_only import load_row
from lightsem.experiments.audio_light_foundation.data import make_fixture_metadata

root = Path('/mnt/lightsem-data/experiments/lightsem/foundation_joint_open_legacy_20261022_v1_corrected')
device = torch.device('cuda')
base = load_model(Path('/mnt/lightsem-data/experiments/lightsem/foundation_v3_jerk_safe_20261020_v1/model.pt'), device)
model = JointOpenLegacyModel(base).to(device)
state = torch.load(root / 'model.pt', map_location=device, weights_only=False)
model.load_state_dict(state['state_dict'], strict=True)
model.eval()
with np.load(sorted((root / 'roskilde_cache').glob('*.npz'))[0]) as z:
    features = np.asarray(z['features'][:128], np.float32)
    light = np.asarray(z['light'][:128], np.float32)
fixtures = torch.from_numpy(make_fixture_metadata()[None]).float().to(device)
valid = torch.ones((1, 128), dtype=torch.bool, device=device)
with torch.no_grad():
    output = model.forward_open(torch.from_numpy(features[None]).float().to(device), fixtures, valid=valid, duration_seconds=12.8, auxiliary=True)
aux_std = float(output['open_aux'].std().cpu())
contract = json.loads(Path('/mnt/lightsem-data/experiments/lightsem/audio_light_foundation_v2_contract_20261016_v1/CONTRACT.json').read_text())
row = load_row(contract['rows'][0])
audio = torch.from_numpy(row['audio'][:128][None]).float().to(device)
timestamps = torch.arange(128, device=device, dtype=torch.float32)[None] / 10.0
with torch.no_grad():
    canonical = model.base(audio, fixtures, timestamps=timestamps, valid=valid, duration_seconds=12.8)
print(json.dumps({'open_aux_std': aux_std, 'open_target_std': float(light.std()), 'canonical_path_has_open_encoder': False, 'canonical_output_rgb_std': float(canonical['rgb'].std().cpu())}, sort_keys=True))
