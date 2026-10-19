"""Whole-song accepted P2 colors, independently of body motion and dimmers.

Shpongle reuses the accepted full-record stream exactly. Cure uses the same
frozen Rich96/history/300 ms/two-group recipe on its complete waveform. Its old
180-second inference remains a comparator: the offline model can legitimately
change the prefix when its context extends beyond the former crop.
"""
from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import subprocess
import time

import numpy as np


REPO = Path('/root/lightsem')
DEFAULT_ROOT = Path('/mnt/lightsem-data/experiments/lightsem/dynamic_dance_combo_20261005_v1')
BASELINE = REPO/'configs/light_layers/baseline.json'
P2_SOURCE = REPO/'runs/visual_next_20260929/shpongle_palette_previews_v1/P2.npz'
OLD_CURE = REPO/'runs/cross_song_20260930_v1/02'
CUE_ROOT = Path('/mnt/lightsem-data/experiments/lightsem/lighting-cue-model-20260922t094347z')
CHECKPOINT = CUE_ROOT/'observer_fits_v1/04_seed0_I_VR_P0/best.pt'
TRANSFORMS = CUE_ROOT/'observer_train_transforms_v1.json'
BEAT_CHECKPOINT = CUE_ROOT/'weights/beat_this_small0.ckpt'
SH_PREDICTIONS = Path('/mnt/lightsem-data/experiments/lightsem/end_to_end_20260928_v1/comparison_demos_v1/03_I_VR_P0/model_outputs.npz')
NIGHT_PYTHON = Path('/root/lightsem/.venv/night-access/bin/python')
PINNED = {
    str(P2_SOURCE): 'd704d75fb9be47483f3ea55ec3d2cf259e42aedfa6bc670645d2c9fa3c2d62e5',
    str(CHECKPOINT): '8de865906cfabb49290971eccf6016ab38edebe885eec7e6369f020fb5544b8a',
    str(TRANSFORMS): 'a4672acf0ff0b95fa3fad2b8a1f481810029f8bde4b30e10bfccfacc94740b39',
    str(BEAT_CHECKPOINT): '6074be2c4d490c5f6101fcc374a1ec72ae93456e23bb6019783b849f5dc7d47b',
    str(SH_PREDICTIONS): '135a02cef71b695cf0faa0370ab6bc602a741e303dee47f4f9ea822776f9fea6',
    str(OLD_CURE/'streams.npz'): 'ea3d5bcb5de11f3c21e557eb1906171aec0dae1dc236a96eb28d5feb41e9f875',
    str(OLD_CURE/'model_outputs.npz'): 'fac26ceb2b3d05787c8904e2407140c18c1add1ad4ba3cf29b32fed9b5258553',
}
TRACKS = {
    'shpongle': {'audio': '/root/shpongle.flac', 'sha256': '95e4388424ef62950bdeae78e47b8e71839ed28c6e01189fe15f9ae6c902b6c0'},
    'cure_for_me': {'audio': '/root/lightsem/experiments/legacy/stages/outputs/stage8/demo_songs/Cure_For_Me.wav', 'sha256': '79fbcfbc5f99e200075801a4ef157055dfe2593a98ac2e545747f89b28f47a9f'},
}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def _npz(path, **values):
    with Path(path).open('xb') as stream:
        np.savez_compressed(stream, **values)


def checked_sources():
    for path, expected in PINNED.items():
        if sha(path) != expected:
            raise ValueError('Frozen P2 input changed: '+path)
    baseline = json.loads(BASELINE.read_text())
    expected = {'recipe': 'I_VR_P0', 'seed': 0, 'palette_history': True,
                'color_transition_s': .3, 'color_layout': 'two_groups',
                'color_transition_curve': 'causal_smoothstep_channel_rgb'}
    if any(baseline.get(key) != value for key, value in expected.items()):
        raise ValueError('Accepted P2 recipe changed')
    if baseline['checkpoint'] != {'path': str(CHECKPOINT), 'sha256': PINNED[str(CHECKPOINT)]}:
        raise ValueError('Accepted P2 checkpoint changed')
    if baseline['reference_streams'] != str(P2_SOURCE) or baseline['reference_streams_sha256'] != PINNED[str(P2_SOURCE)]:
        raise ValueError('Accepted P2 full stream changed')
    return {**PINNED, str(BASELINE): sha(BASELINE)}


def frame_count(duration_s, fps=40):
    if not np.isfinite(duration_s) or duration_s <= 0:
        raise ValueError('Positive finite whole-song duration required')
    return int(math.ceil(float(duration_s)*fps-1e-9))


def read_p2_audio(audio):
    """Read a source and normalize the P2 analysis rate to 44.1 kHz.

    The native MuQ path accepts arbitrary source rates, but the historical P2
    bus and BeatThis input are defined at 44.1 kHz.  Keep the original path,
    hash, and duration in the receipt while resampling only the analysis copy.
    """
    import soundfile as sf
    from scipy.signal import resample_poly

    wave, source_rate = sf.read(audio, dtype='float32', always_2d=False)
    wave = np.asarray(wave, dtype=np.float32)
    if wave.ndim not in (1, 2) or not len(wave) or not np.isfinite(wave).all():
        raise ValueError('Expected finite nonempty mono or multichannel source')
    source_rate = int(source_rate)
    if source_rate <= 0:
        raise ValueError('Source sample rate must be positive')
    duration_s = len(wave) / source_rate
    if source_rate != 44100:
        wave = resample_poly(wave, 44100, source_rate, axis=0).astype(np.float32)
    return wave, 44100, {'source_rate_hz': source_rate,
                         'analysis_rate_hz': 44100,
                         'source_duration_s': float(duration_s)}


def extend_endpoint(values, frames):
    """At most one extra native tick, only for a final fractional audio frame."""
    values = np.asarray(values)
    if frames < len(values) or frames > len(values)+1 or not len(values):
        raise ValueError('Only zero or one final endpoint-hold tick is admitted')
    return np.concatenate((values, values[-1:]), axis=0) if frames > len(values) else values.copy()


def pad_feature_block(raw94):
    """Complete the last four-frame inference block, with explicit edge padding."""
    raw94 = np.asarray(raw94)
    if raw94.ndim != 2 or raw94.shape[1] != 94 or not len(raw94) or not np.isfinite(raw94).all():
        raise ValueError('Finite full-record audio94 required')
    pad = (-len(raw94)) % 4
    return np.pad(raw94, ((0, pad), (0, 0)), mode='edge'), pad


def canonical_slots():
    from lightsem.conductor.rig import canonical_rig
    from lightsem.experiments.visual_next.palettes import fixture_groups, group_rgb_mask
    rig = canonical_rig()
    heads = sorted(rig.by_kind('moving_head'), key=lambda fixture: fixture.position[0])
    slots = [fixture.address-1+np.array([6, 7, 8]) for fixture in (heads[0], heads[-1])]
    return slots, group_rgb_mask(fixture_groups(rig))


def prepare_input_cpu(audio, outputdir):
    """Extract historical float32 P2 bus (not the dancer's float16 Base92)."""
    from lightsem.audio.online import extract_bus
    from lightsem.model.light_layers.training import AUDIO_SCALARS
    outputdir = Path(outputdir)
    outputdir.mkdir(parents=True, exist_ok=True)
    report_path, bus_path = outputdir/'INPUT.json', outputdir/'audio_bus.npz'
    if report_path.exists():
        report = json.loads(report_path.read_text())
        if report['audio_sha256'] != sha(audio) or report['bus_sha256'] != sha(bus_path):
            raise ValueError('Prepared P2 input changed')
        return report
    if bus_path.exists():
        raise FileExistsError('Preserve unreceipted P2 bus')
    wave, sample_rate, audio_info = read_p2_audio(audio)
    if wave.ndim == 1:
        wave = wave[:, None]
    bus = extract_bus(y=wave.mean(1), sr=sample_rate, causal=True)
    values = {'times': bus.times.astype(np.float32), 'mel': bus.mel.astype(np.float32),
              'meta': np.asarray(json.dumps({'fps': bus.fps, 'tempo': bus.tempo,
                  'duration_s': bus.duration_s, 'flags': bus.flags, **bus.meta}))}
    values.update({'ch__'+key: bus.channels[key].astype(np.float32) for key in AUDIO_SCALARS})
    _npz(bus_path, **values)
    report = {'audio': str(audio), 'audio_sha256': sha(audio), 'bus': str(bus_path),
              'bus_sha256': sha(bus_path), 'audio_duration_s': audio_info['source_duration_s'],
              'source_sample_rate_hz': audio_info['source_rate_hz'],
              'analysis_sample_rate_hz': audio_info['analysis_rate_hz'],
              'bus_frames': len(bus.times), 'lighting_frames': frame_count(audio_info['source_duration_s']),
              'mel_dtype': 'float32; no float16 round trip', 'causal_spectral': True,
              'oracle_beat_grid': True, 'source_sha256': sha(__file__)}
    _json(report_path, report)
    return report


def _extract_beat_subprocess(audio, outputdir):
    """Executed only inside the parent's single GPU lock; never a standalone job."""
    import soundfile as sf
    import torch
    from beat_this.inference import Audio2Frames
    from beat_this.model.postprocessor import Postprocessor
    outputdir = Path(outputdir)
    if os.environ.get('LIGHTSEM_COMBO_COLOR_LOCK_OWNER') != str(os.getppid()):
        raise RuntimeError('BeatThis child requires the guarded parent process')
    if sha(BEAT_CHECKPOINT) != PINNED[str(BEAT_CHECKPOINT)]:
        raise ValueError('Frozen BeatThis checkpoint changed')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    wave, sample_rate = sf.read(audio, dtype='float32', always_2d=False)
    predictor = Audio2Frames(checkpoint_path=str(BEAT_CHECKPOINT), device='cuda:0', float16=False)
    with torch.inference_mode():
        beat, down = predictor(wave, sample_rate)
        if not bool(torch.isfinite(beat).all() & torch.isfinite(down).all()):
            raise ValueError('Nonfinite full-record BeatThis output')
        beats, downbeats = Postprocessor(type='minimal')(beat, down)
    b, d = beat.float().cpu().numpy(), down.float().cpu().numpy()
    if len(b) != len(d) or abs(len(b)/50-len(wave)/sample_rate) > .1:
        raise ValueError('Full-record BeatThis duration mismatch')
    path = outputdir/'rhythm.npz'
    _npz(path, times_s=np.arange(len(b), dtype=np.float64)/50, beat_logit=b,
         downbeat_logit=d, beats_s=np.asarray(beats), downbeats_s=np.asarray(downbeats))
    _json(outputdir/'RHYTHM.json', {'path': str(path), 'sha256': sha(path),
        'audio_sha256': sha(audio), 'checkpoint_sha256': PINNED[str(BEAT_CHECKPOINT)],
        'frames': len(b), 'torch': torch.__version__, 'source_sha256': sha(__file__)})


def _infer_cure(outputdir, preparation):
    import torch
    from collections import OrderedDict
    from lightsem.model.light_layers.training import SourceStore, _normalize, load_checkpoint, generated_history
    from scripts.light_layers.prepare_visual_next_demo import FinitePaletteCapture
    rhythm_path = outputdir/'RHYTHM.json'
    if not rhythm_path.exists():
        environment = dict(os.environ, LIGHTSEM_COMBO_COLOR_LOCK_OWNER=str(os.getpid()),
                           OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
        subprocess.run([str(NIGHT_PYTHON), '-m', 'lightsem.experiments.dynamic_dance_combo.colors', '_beat', '--audio', preparation['audio'],
                        '--output', str(outputdir)], env=environment, check=True)
    rhythm = json.loads(rhythm_path.read_text())
    if rhythm['audio_sha256'] != preparation['audio_sha256'] or sha(rhythm['path']) != rhythm['sha256'] or rhythm['checkpoint_sha256'] != PINNED[str(BEAT_CHECKPOINT)]:
        raise ValueError('Full-song learned rhythm identity changed')
    store = SourceStore.__new__(SourceStore)
    store.max_ram_bytes, store.audio_cache = 128*2**20, OrderedDict()
    store.cache_bytes, store.verified = 0, set()
    store.rows = {'cure_for_me': {'audio': {'path': preparation['bus'], 'sha256': preparation['bus_sha256']},
                                'rhythm': {'path': rhythm['path'], 'sha256': rhythm['sha256']}}}
    raw94 = store.audio('cure_for_me')
    padded, padding = pad_feature_block(raw94)
    transforms = json.loads(TRANSFORMS.read_text())
    normalized = _normalize(padded, transforms['audio'])
    _npz(outputdir/'audio94.npz', raw=raw94, padded_raw=padded, normalized=normalized)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, metadata = load_checkpoint(CHECKPOINT, device='cuda')
    if metadata['config']['recipe'] != 'I_VR_P0' or metadata['config']['seed'] != 0:
        raise ValueError('P2 checkpoint recipe or seed drift')
    with torch.inference_mode(), torch.autocast('cuda', enabled=False):
        output = model(torch.from_numpy(normalized[None]).cuda())
        if not all(bool(torch.isfinite(value).all()) for value in output.values()):
            raise ValueError('Nonfinite P2 encoder output')
        capture = FinitePaletteCapture(model)
        rollout = generated_history(capture, output['context10'], output['chromatic_change_logit'],
                                    initial_ids=None, state_ablation=False, cue_threshold=.5, initialize=True)
        result = {key: value[0].cpu().numpy() for key, value in {**output, **rollout}.items()}
        actions = np.unique(np.r_[0, np.flatnonzero(result['color_cues'])]).astype(int)
        previous = result['previous_ids'][actions]
        if capture.calls != 1 or not np.array_equal(capture.tables.argmax(-1)[np.arange(len(actions)), previous], result['generated_palette_ids'][actions]):
            raise ValueError('Own-history palette action audit failed')
        if result['previous_ids'][0] != 32 or not np.array_equal(result['previous_ids'][1:], result['generated_palette_ids'][:-1]):
            raise ValueError('P2 palette history drift')
        result.update(action_indices=actions, action_previous_ids=previous,
                      action_argmax_choices=capture.tables.argmax(-1).astype(np.uint8),
                      action_logits=capture.tables[np.arange(len(actions)), previous])
    if not all(np.isfinite(value).all() for value in result.values()):
        raise ValueError('Nonfinite P2 output')
    _npz(outputdir/'model_outputs.npz', **result)
    del model
    torch.cuda.empty_cache()
    return result, {'native_bus_frames': len(raw94), 'inference_frames': len(padded),
        'terminal_feature_padding_ticks': padding, 'terminal_padding_policy': 'repeat final feature row only to complete four-frame block',
        'discarded_inference_ticks': len(padded)-preparation['lighting_frames'],
        'torch': torch.__version__, 'checkpoint_metadata': metadata}


def two_group_stream(scores, frames, companion_rgb_source, prototypes):
    """Use historical pre-cue RGB; all non-RGB bytes are new zero placeholders."""
    from lightsem.experiments.visual_next.delivery import color_delivery
    companion = np.asarray(companion_rgb_source)
    if companion.dtype != np.uint8 or companion.ndim != 2 or companion.shape[1] != 512:
        raise ValueError('Expected authenticated canonical RGB source')
    ids, cues = scores['generated_palette_ids'], scores['color_cues']
    count = len(ids)*4
    if frames > count + 3 or frames < count - 3:
        raise ValueError('Only the final four-frame padded block may differ from inference length')
    first = np.flatnonzero(cues)
    prefix_needed = count if not len(first) else min(count, int(first[0])*4+1)
    if prefix_needed > len(companion):
        raise ValueError('Historical pre-first-cue RGB does not cover new inactive prefix')
    _, mask = canonical_slots()
    zero = np.zeros((count, 512), np.uint8)
    before = zero.copy()
    common = min(count, len(companion))
    before[:common, mask] = companion[:common, mask]
    stream, delivery = color_delivery(zero, before, np.arange(count)/40., ids, cues, prototypes, layout='two_groups')
    if np.any(stream[:, ~mask]):
        raise ValueError('Color-only overlay changed non-RGB placeholders')
    if frames > count:
        # The decoded whole-song clock can contain up to three real terminal
        # ticks beyond the last complete 4-frame model block.  Hold the last
        # emitted color state over those ticks; no new cue is invented.
        stream = np.concatenate((stream, np.repeat(stream[-1:], frames-count, axis=0)), axis=0)
    return stream[:frames].copy(), {**delivery, 'canonical_nonRGB_bytes_all_zero': True,
                                   'historical_RGB_used_only_before_first_new_cue': True}


def color_diagnostics(left, right, scores):
    left, right = np.asarray(left), np.asarray(right)
    if left.dtype != np.uint8 or right.dtype != np.uint8 or left.shape != right.shape or left.ndim != 2 or left.shape[1] != 3:
        raise ValueError('Expected two matching uint8 RGB streams')
    ids10, cues10 = np.asarray(scores['generated_palette_ids']), np.asarray(scores['color_cues'])
    ids40 = np.repeat(ids10, 4)
    if len(ids40) < len(left):
        ids40 = extend_endpoint(ids40, len(left))
    ids40 = ids40[:len(left)]
    cue_ticks = np.flatnonzero(cues10)*4
    cue_ticks = cue_ticks[cue_ticks < len(left)]
    active = np.arange(len(left)) >= cue_ticks[0] if len(cue_ticks) else np.zeros(len(left), bool)
    visible = ids40[active]
    palette, counts = np.unique(visible, return_counts=True)
    report = {'frames40': len(left), 'color_cue_count': len(cue_ticks),
              'palette_changes_while_active': int(np.count_nonzero(np.diff(visible))),
              'palette_occupancy_seconds': {str(key): float(count/40) for key, count in zip(palette, counts)},
              'palette_occupancy_fraction_active': {str(key): float(count/max(len(visible), 1)) for key, count in zip(palette, counts)},
              'left_rgb_changing_ticks': int(np.count_nonzero(np.any(np.diff(left.astype(int), axis=0), axis=1))),
              'right_rgb_changing_ticks': int(np.count_nonzero(np.any(np.diff(right.astype(int), axis=0), axis=1))),
              'two_color_groups_identical': bool(np.array_equal(left, right)),
              'color_cue_times_s': (cue_ticks/40).tolist(), 'finite': True}
    return report, ids40, active


def generate(root=DEFAULT_ROOT, track='shpongle', *, acquire_gpu_lock=True):
    """Return (left RGB, right RGB, report), preserving all original artifacts.

    Invoke with torch-access Python. Only Cure consumes GPU, sequentially:
    night-access BeatThis child, then the frozen Rich96 FP32 model. A caller
    already holding the same authoritative lock may set acquire_gpu_lock=False.
    """
    import soundfile as sf
    identity = track if isinstance(track, str) else track['id']
    if identity not in TRACKS:
        raise ValueError('Only the two authorized showcase songs are registered')
    source = TRACKS[identity]
    audio = source['audio'] if isinstance(track, str) else track.get('audio', source['audio'])
    if sha(audio) != source['sha256']:
        raise ValueError('Showcase waveform changed')
    bindings = checked_sources()
    audio_info = sf.info(audio)
    duration = audio_info.frames/audio_info.samplerate
    frames = frame_count(duration)
    outputdir = Path(root)/'colors'/identity
    outputdir.mkdir(parents=True, exist_ok=True)
    report_path, output_path = outputdir/'REPORT.json', outputdir/'P2.npz'
    if report_path.exists():
        report = json.loads(report_path.read_text())
        if report['source_bindings'] != bindings or report['audio_sha256'] != source['sha256']:
            raise ValueError('Saved P2 report inputs differ')
        for path, expected in report['artifacts'].items():
            if sha(path) != expected:
                raise ValueError('Saved P2 output changed')
        with np.load(output_path, allow_pickle=False) as values:
            return values['left'].copy(), values['right'].copy(), report
    if output_path.exists():
        raise FileExistsError('Preserve unreceipted P2 output')
    started = time.monotonic()
    slots, _ = canonical_slots()
    details = {}
    if identity == 'shpongle':
        with np.load(P2_SOURCE, allow_pickle=False) as values:
            stream = values['candidate'].copy()
            if not np.array_equal(values['time'], np.arange(len(stream))/40):
                raise ValueError('Accepted P2 source clock changed')
        left = extend_endpoint(stream[:, slots[0]], frames)
        right = extend_endpoint(stream[:, slots[1]], frames)
        with np.load(SH_PREDICTIONS, allow_pickle=False) as values:
            scores = {key: values[key].copy() for key in ('generated_palette_ids', 'color_cues')}
        details = {'route': 'reuse exact accepted full-song Shpongle P2 RGB',
                   'original_ticks_exact': bool(np.array_equal(left[:len(stream)], stream[:, slots[0]]) and np.array_equal(right[:len(stream)], stream[:, slots[1]])),
                   'terminal_RGB_endpoint_hold_ticks': frames-len(stream),
                   'terminal_RGB_endpoint_hold_start_s': len(stream)/40,
                   'terminal_partial_audio_interval_s': max(0., duration-len(stream)/40),
                   'model_forwards': 0}
    else:
        preparation = prepare_input_cpu(audio, outputdir)
        from lightsem.experiments.stage11.stage11_common import gpu_lock
        with gpu_lock() if acquire_gpu_lock else nullcontext():
            scores, inference = _infer_cure(outputdir, preparation)
        transforms = json.loads(TRANSFORMS.read_text())
        with np.load(OLD_CURE/'streams.npz', allow_pickle=False) as values:
            companion, old_candidate, old_time = values['companion'].copy(), values['candidate'].copy(), values['time'].copy()
        stream, delivery = two_group_stream(scores, frames, companion, transforms['palette_prototypes'])
        left, right = stream[:, slots[0]].copy(), stream[:, slots[1]].copy()
        with np.load(OLD_CURE/'model_outputs.npz', allow_pickle=False) as values:
            old_scores = {key: values[key].copy() for key in ('generated_palette_ids', 'color_cues')}
        from lightsem.experiments.visual_next.delivery import color_delivery
        old, _ = color_delivery(old_candidate, companion, old_time, old_scores['generated_palette_ids'],
                                old_scores['color_cues'], transforms['palette_prototypes'], layout='two_groups')
        old_lr = np.stack((old[:, slots[0]], old[:, slots[1]]), 1)
        new_lr = np.stack((left[:len(old)], right[:len(old)]), 1)
        changed = np.any(old_lr != new_lr, axis=(1, 2))
        details = {'route': 'fresh full-song frozen Rich96/P2 inference; no prefix splice',
                   'inference': inference, 'delivery': delivery,
                   'old_180s_prefix_comparison': {'ticks': len(old), 'changed_ticks': int(changed.sum()),
                       'changed_fraction': float(changed.mean()), 'exact': bool(not changed.any()),
                       'first_changed_s': float(np.flatnonzero(changed)[0]/40) if changed.any() else None,
                       'max_abs_RGB_byte_difference': int(np.abs(old_lr.astype(int)-new_lr.astype(int)).max()),
                       'interpretation': 'Same frozen recipe on longer offline context; prefix changes are reported, not hidden by splicing.'},
                   'model_forwards': 1}
    diagnostics, ids40, active = color_diagnostics(left, right, scores)
    _npz(output_path, time=np.arange(frames)/40., left=left, right=right,
         palette_ids40=ids40, color_active40=active, generated_palette_ids=scores['generated_palette_ids'],
         color_cues10=scores['color_cues'])
    if checked_sources() != bindings:
        raise ValueError('P2 inputs changed during generation')
    artifacts = {str(path): sha(path) for path in sorted(outputdir.iterdir()) if path.is_file()}
    report = {'schema_version': 1, 'track': identity, 'audio': str(audio), 'audio_sha256': source['sha256'],
              'audio_duration_s': duration, 'source_bindings': bindings, 'source_sha256': sha(__file__),
              'recipe': 'I_VR_P0 seed0, own palette history, threshold0.5, argmax, two groups, causal300msRGB fade',
              'new_baseline': False, 'motion_or_brightness_reused': False, 'diagnostics': diagnostics,
              'details': details, 'artifacts': artifacts, 'elapsed_s': time.monotonic()-started,
              'scope': 'Color only. No activity floor, forced cue, palette repair or autonomous motion.'}
    _json(report_path, report)
    return left, right, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('prepare', 'generate', '_beat'))
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--track', choices=tuple(TRACKS))
    parser.add_argument('--audio')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.stage == '_beat':
        _extract_beat_subprocess(args.audio, args.output)
    elif args.stage == 'prepare':
        print(json.dumps(prepare_input_cpu(args.audio, args.output), sort_keys=True))
    else:
        if args.track is None:
            parser.error('--track is required for generate')
        _, _, report = generate(args.root, args.track)
        print(json.dumps({'track': args.track, 'frames40': report['diagnostics']['frames40'],
                          'report': str(args.root/'colors'/args.track/'REPORT.json')}, sort_keys=True))


if __name__ == '__main__':
    main()
