"""Derive three visual ablations from authenticated saved full-song outputs."""
from pathlib import Path
import hashlib
import json
import os
import time
import numpy as np
import torch
from lightsem.conductor.rig import canonical_rig
from lightsem.experiments.light_layers.training import load_checkpoint, generated_history

ROOT = Path(__file__).resolve().parent
DATA = Path('/mnt/lightsem-data/experiments/lightsem/end_to_end_20260928_v1/shpongle_rating_20260929_v1')
SOURCE = Path('/mnt/lightsem-data/experiments/lightsem/shpongle_layers_demo_20260922T161917Z')
PLAN = Path('/root/lightsem/outputs/shpongle_layers_demo_20260922T161917Z/inference_mapping_plan.json')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    plan = json.loads(PLAN.read_text())
    provenance = json.loads((SOURCE / 'provenance_manifest.json').read_text())
    for name in ('model_outputs.npz', 'overlay_dmx.npz'):
        assert sha(SOURCE / name) == provenance['artifacts'][name]['sha256']
    assert sha(plan['checkpoint']['path']) == plan['checkpoint']['sha256']
    with np.load(SOURCE / 'overlay_dmx.npz', allow_pickle=False) as z:
        baseline, full, times = z['baseline'], z['overlay'], z['time']
        active, palette_rgb = z['color_active40'], z['palette_rgb']
        allowed = z['allowed_slot_mask']
    with np.load(SOURCE / 'model_outputs.npz', allow_pickle=False) as z:
        context = z['context10'].copy()
        color_logits = z['chromatic_change_logit'].copy()
        saved_ids, saved_cues = z['generated_palette_ids'].copy(), z['color_cues'].copy()
    assert baseline.shape == full.shape == (11092, 512)
    assert baseline.dtype == full.dtype == np.uint8
    assert np.array_equal(times, np.arange(11092) / 40)
    rig = canonical_rig()
    rgb_slots = np.array([f.address - 1 + off for f in rig.fixtures
                         if f.kind in ('moving_head', 'wash', 'projector')
                         for off in {'moving_head': (6, 7, 8), 'wash': (2, 3, 4), 'projector': (2, 3, 4)}[f.kind]])
    dimmer_slots = np.array([f.address - 1 + (4 if f.kind == 'moving_head' else 0)
                            for f in rig.fixtures if f.kind in ('moving_head', 'wash')])
    mask = np.zeros(512, dtype=bool)
    mask[np.concatenate([rgb_slots, dimmer_slots])] = True
    assert np.array_equal(mask, allowed) and int((~mask).sum()) == 469
    colors, accents = baseline.copy(), baseline.copy()
    colors[:, rgb_slots] = full[:, rgb_slots]
    accents[:, dimmer_slots] = full[:, dimmer_slots]
    recombined = colors.copy()
    recombined[:, dimmer_slots] = accents[:, dimmer_slots]
    assert np.array_equal(recombined, full)
    # Only the small palette head consumes the saved context. No audio encoder,
    # target data, visual encoder, fitting or model forward is permitted here.
    model, metadata = load_checkpoint(plan['checkpoint']['path'], device='cpu',
                                      expected_bindings=plan['checkpoint']['bindings'])
    model.eval()
    def forbidden_forward(*args, **kwargs):
        raise RuntimeError('Full encoder forward forbidden for cached-context rating')
    model.forward = forbidden_forward
    with torch.no_grad():
        c = torch.from_numpy(context[None])
        logits = torch.from_numpy(color_logits[None])
        own = generated_history(model, c, logits, state_ablation=False)
        null = generated_history(model, c, logits, state_ablation=True)
    own_ids = own['generated_palette_ids'][0].numpy()
    null_ids = null['generated_palette_ids'][0].numpy()
    assert np.array_equal(own_ids, saved_ids), 'CPU own-state replay must match the saved demo exactly'
    assert np.array_equal(own['color_cues'][0].numpy(), saved_cues)
    assert np.array_equal(null['color_cues'][0].numpy(), saved_cues)
    no_history = full.copy()
    colors40 = palette_rgb[np.repeat(null_ids, 4)]
    quantized = np.floor(np.clip(colors40, 0, 1) * 255 + .5).astype(np.uint8)
    for f in rig.fixtures:
        if f.kind in ('moving_head', 'wash', 'projector'):
            offsets = {'moving_head': (6, 7, 8), 'wash': (2, 3, 4), 'projector': (2, 3, 4)}[f.kind]
            slots = f.address - 1 + np.array(offsets)
            no_history[np.ix_(active, slots)] = quantized[active]
    rows = []
    DATA.mkdir(exist_ok=False)
    for vid, stream in (('08', colors), ('09', accents), ('10', no_history)):
        assert np.array_equal(stream[:, ~mask], baseline[:, ~mask])
        fs = os.statvfs(DATA)
        assert fs.f_bavail * fs.f_frsize >= 2**30 + 8 * 2**20
        target = DATA / (vid + '_streams.npz')
        with target.open('xb') as f:
            np.savez_compressed(f, candidate=stream, time=times)
        with np.load(target, allow_pickle=False) as z:
            assert np.array_equal(z['candidate'], stream) and np.array_equal(z['time'], times)
        rows.append({'id': vid, 'streams': str(target), 'sha256': sha(target),
                     'dmx_sha256': hashlib.sha256(stream.tobytes()).hexdigest(),
                     'protected_slots_unchanged': 469, 'changed_bytes_vs_companion': int(np.count_nonzero(stream != baseline))})
    assert sum(p.stat().st_size for p in DATA.iterdir()) <= 8 * 2**20
    receipt = {'status': 'complete', 'variants': rows, 'full_encoder_forwards': 0,
               'CPU_palette_head_only': True, 'CPU_own_state_exactly_replays_saved_demo': True,
               'color_cue_times_identical': True, 'source_checkpoint_unchanged': sha(plan['checkpoint']['path']) == plan['checkpoint']['sha256'],
               'color_plus_accent_exactly_reconstructs_combined_demo': True,
               'history_off_differing_frames10': int(np.count_nonzero(null_ids != saved_ids)),
               'checkpoint': plan['checkpoint']['path'], 'elapsed_seconds': time.monotonic() - started,
               'scope': 'Deterministic full-track rating ablations, not the stochastic secondary experiment or a quality result.'}
    with (ROOT / 'PREPARATION.json').open('x') as f:
        json.dump(receipt, f, indent=2)
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
