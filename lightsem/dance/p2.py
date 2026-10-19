"""Own-waveform inference of the accepted frozen Rich96 P2 color model."""
from pathlib import Path
from contextlib import nullcontext
import argparse
import json
import numpy as np

from .baseline import REPO, sha
from .pipeline import accelerator_lock, write_json


def generate(audio, output, device):
    import torch
    from beat_this.inference import Audio2Frames
    from beat_this.model.postprocessor import Postprocessor
    from lightsem.experiments.dynamic_dance_combo import colors as helpers
    from lightsem.model.light_layers.training import SourceStore, _normalize, load_checkpoint, generated_history
    from scripts.light_layers.prepare_visual_next_demo import FinitePaletteCapture
    from collections import OrderedDict
    output.mkdir(parents=True, exist_ok=False)
    baseline = json.loads((REPO / 'configs/light_layers/baseline.json').read_text())
    checkpoint = Path(baseline['checkpoint']['path'])
    if (sha(checkpoint) != baseline['checkpoint']['sha256']
            or sha(helpers.BEAT_CHECKPOINT) != helpers.PINNED[str(helpers.BEAT_CHECKPOINT)]
            or sha(helpers.TRANSFORMS) != helpers.PINNED[str(helpers.TRANSFORMS)]):
        raise ValueError('P2 or BeatThis checkpoint mismatch')
    preparation = helpers.prepare_input_cpu(audio, output / 'input')
    wave, rate, _ = helpers.read_p2_audio(audio)
    with torch.inference_mode():
        predictor = Audio2Frames(checkpoint_path=str(helpers.BEAT_CHECKPOINT), device=device, float16=False)
        beat, downbeat = predictor(wave, rate)
        beats, downbeats = Postprocessor(type='minimal')(beat, downbeat)
    rhythm = output / 'rhythm.npz'
    with rhythm.open('xb') as stream:
        np.savez_compressed(stream, times_s=np.arange(len(beat), dtype=np.float64) / 50,
                            beat_logit=beat.float().cpu().numpy(), downbeat_logit=downbeat.float().cpu().numpy(),
                            beats_s=np.asarray(beats), downbeats_s=np.asarray(downbeats))
    del predictor
    store = SourceStore.__new__(SourceStore)
    store.max_ram_bytes, store.audio_cache, store.cache_bytes, store.verified = 128 * 2**20, OrderedDict(), 0, set()
    store.rows = {'input': {'audio': {'path': preparation['bus'], 'sha256': preparation['bus_sha256']},
                            'rhythm': {'path': str(rhythm), 'sha256': sha(rhythm)}}}
    raw = store.audio('input'); padded, _ = helpers.pad_feature_block(raw)
    transforms = json.loads(helpers.TRANSFORMS.read_text())
    normalized = _normalize(padded, transforms['audio'])
    model, metadata = load_checkpoint(checkpoint, device=device)
    if metadata['config']['recipe'] != 'I_VR_P0':
        raise ValueError('P2 recipe mismatch')
    with torch.inference_mode():
        # The deployed Rich96 model intentionally rejects a single forward
        # beyond 12,000 lighting ticks.  Its offline trunk has a finite
        # receptive field, so use the model's verified halo chunker for longer
        # own-song inputs instead of truncating the song or changing its clock.
        predicted = model.forward_chunked(torch.from_numpy(normalized[None]).to(device))
        capture = FinitePaletteCapture(model)
        rollout = generated_history(capture, predicted['context10'], predicted['chromatic_change_logit'],
                                    initial_ids=None, state_ablation=False, cue_threshold=.5, initialize=True)
        scores = {key: value[0].cpu().numpy() for key, value in rollout.items()}
    if not all(np.isfinite(v).all() for v in scores.values()):
        raise ValueError('Nonfinite P2 output')
    # New-song initial RGB is a fixed shared vocabulary, never a motion source.
    # Known showcase songs reuse their authenticated accepted streams exactly.
    initial = np.zeros((len(padded), 512), np.uint8)
    stream, delivery = helpers.two_group_stream(scores, preparation['lighting_frames'], initial,
                                               transforms['palette_prototypes'])
    slots, _ = helpers.canonical_slots(); left, right = stream[:, slots[0]], stream[:, slots[1]]
    metrics, palette, active = helpers.color_diagnostics(left, right, scores)
    with (output / 'P2.npz').open('xb') as target:
        np.savez_compressed(target, time=np.arange(len(left)) / 40, left=left, right=right,
                            palette_ids40=palette, color_active40=active, color_cues10=scores['color_cues'])
    write_json(output / 'REPORT.json', {'audio': str(audio), 'audio_sha256': sha(audio),
                                        'audio_preparation': {key: preparation[key] for key in (
                                            'audio_duration_s', 'source_sample_rate_hz',
                                            'analysis_sample_rate_hz', 'lighting_frames')},
                                        'checkpoint_sha256': sha(checkpoint), 'recipe': 'P2',
                                        'initial_rgb': 'Fixed black before first generated color cue; no historical soundtrack/controller substituted',
                                        'diagnostics': metrics, 'delivery': delivery,
                                        'P2_sha256': sha(output / 'P2.npz')})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wav', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args()
    with accelerator_lock() if args.device == 'cuda' else nullcontext():
        generate(args.wav.resolve(), args.out.resolve(), args.device)


if __name__ == '__main__':
    main()
