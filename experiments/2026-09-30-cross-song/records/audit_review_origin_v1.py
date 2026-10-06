"""Read-only saved-output audit: color-layout deltas and body provenance.

This is retrospective engineering diagnosis, not model fitting or a new
preregistered experiment. Run with CUDA_VISIBLE_DEVICES='' bash run_torch.sh.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import torch
from lightsem.conductor.rig import canonical_rig
from lightsem.experiments.visual_next.palettes import fixture_groups, group_rgb_mask
from lightsem.experiments.motion_study.geometry import joints_from_features

ROOT = Path('/root/lightsem')
parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, default=Path(__file__).with_name('REVIEW_ORIGIN_AUDIT_V1.json'))
OUT = parser.parse_args().output

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def read(path):
    value = json.loads(Path(path).read_text())
    return value.get('payload', value)

def array(path, key):
    with np.load(path, allow_pickle=False) as value:
        return value[key].copy()

manifest_path = ROOT / 'runs/visual_next_20260929/shpongle_palette_previews_v1/MANIFEST.json'
manifest = read(manifest_path)
paths = {row['letter']: row['streams'] for row in manifest['variants']}
streams = {k: array(p, 'candidate') for k, p in paths.items()}
rgb = group_rgb_mask(fixture_groups(canonical_rig()))
head_rgb = np.zeros(512, dtype=bool)
for fixture in fixture_groups(canonical_rig()):
    if fixture['kind'] == 'moving_head':
        head_rgb[fixture['rgb_slots_zero_based']] = True
pairs = {}
for left, right in [('C', 'P2'), ('C', 'P4'), ('P2', 'P4')]:
    a, b = streams[left], streams[right]
    delta = np.abs(a[:, rgb].astype(float) - b[:, rgb].astype(float))
    assert np.array_equal(a[:, ~rgb], b[:, ~rgb])
    video_indices = np.rint(np.arange(8319) * 40 / 30).astype(int)
    pairs[left + '_vs_' + right] = {
        'all_479_nonRGB_slots_exact': True,
        'all_moving_head_RGB_slots_exact': bool(np.array_equal(a[:, head_rgb], b[:, head_rgb])),
        'video_frames_with_any_RGB_command_difference': int(np.any(delta[video_indices], axis=1).sum()),
        'total_video_frames': len(video_indices),
        'n_40Hz_ticks': len(a),
        'ticks_with_any_RGB_difference': int(np.any(delta, axis=1).sum()),
        'seconds_with_any_RGB_difference': float(np.any(delta, axis=1).sum() / 40),
        'changed_RGB_bytes': int(np.count_nonzero(delta)),
        'RGB_byte_mean_absolute_difference': float(delta.mean()),
        'RGB_byte_p95_absolute_difference': float(np.quantile(delta, .95)),
        'per_tick_max_difference_p50': float(np.median(delta.max(axis=1))),
        'scope': 'DMX RGB command difference, not perceptual image distance',
    }

r1_path = ROOT / 'runs/visual_next_20260929/shpongle_learned_streams_v1/R1_seed0/model_outputs.npz'
ids = array(r1_path, 'generated_palette_ids')
cues = array(r1_path, 'color_cues')
first = int(np.flatnonzero(cues)[0])
active = ids[first:]
unique, counts = np.unique(active, return_counts=True)
r1 = {
    'first_color_cue_s': first / 10,
    'cue_times_s': (np.flatnonzero(cues) / 10).tolist(),
    'subsequent_palette_change_times_s': ((np.flatnonzero(np.diff(active)) + first + 1) / 10).tolist(),
    'active_palette_occupancy': {str(k): float(v / len(active)) for k, v in zip(unique, counts)},
    'model_active_s': len(active) / 10,
    'pre_first_cue_uses_companion_RGB': True,
}

base = Path('/mnt/lightsem-data/experiments/lightsem')
pose = base / 'dance_diagnosis_20260911/pose_stage_01'
source = read(pose / 'freeze.json')['sources']['shpongle_prior_flow0']['world']
saved = pose / 'mapped/shpongle_prior_flow0/full_world.npz'
original = base / 'motion_native_audio_20260910/delivery/final_diagnostic/inference/shpongle.npz'
receipt = read(original.with_suffix('.json'))
assert digest(original) == receipt['sha256']
world = array(saved, 'world')
previous_world = array(source, 'prior_native_flow')
assert np.array_equal(world, previous_world)
torch.set_num_threads(2)
with torch.inference_mode():
    reconstructed = joints_from_features(torch.from_numpy(array(original, 'flow0'))).numpy()
err = float(np.max(np.abs(reconstructed - world)))
assert reconstructed.shape == world.shape and err < 1e-5
protocol_path = original.parent.parent / 'protocol.json'
entry = read(protocol_path)['entries'][0]
assert entry['key'] == 'shpongle' and digest(entry['path']) == entry['sha256']
bindings = {str(p): digest(p) for p in [manifest_path, *paths.values(), r1_path, saved, source, original,
    protocol_path, entry['path'], ROOT / 'lightsem/experiments/motion_study/delivery.py',
    ROOT / 'lightsem/experiments/motion_study/music_model.py',
    ROOT / 'scripts/motion/review_music_physical_support_revision.py',
    ROOT / 'lightsem/experiments/motion_study/companion.py',
    ROOT / 'lightsem/experiments/visual_next/delivery.py', Path(__file__)]}
report = {
    'scope': 'Retrospective read-only audit of existing saved outputs, no training or new generation',
    'color_layout_pairs': pairs,
    'R1': r1,
    'body_origin': {
        'saved_body_exact_to_previous_review': True,
        'original_flow0_FK_max_absolute_difference_m': err,
        'world_shape': list(world.shape),
        'original_model': 'MusicModel conditional native flow, training seed 0',
        'original_draw_seed': receipt['native_draw_seeds'][0],
        'original_audio': entry['audio_path'],
        'original_audio_features': entry['path'],
        'meaning_of_prior_native_flow': 'previous native flow model, NOT an unconditional prior',
        'new_song_usage': 'same saved Shpongle body prefix, no new song-conditioned body inference',
    },
    'input_sha256': bindings,
}
with OUT.open('x') as stream:
    json.dump(report, stream, ensure_ascii=False, indent=2)
print(json.dumps({k:v for k,v in report.items() if k != 'input_sha256'}, indent=2))
