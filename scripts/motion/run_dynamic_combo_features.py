"""Duration-matched donor waveforms and fully recomputed musical features.

Preparation is CPU-only and freezes the exact analyzed stereo float32 samples
before any learned extraction. Gain is the original donor intervention, applied
once by AudioAnalyzer after decoding the lossless duration-matched source.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from lightsem.experiments.dynamic_dance.audio import (
    guard_disk, load_waveform, read_receipt, waveform_counterfactual, write_receipt,
)
from lightsem.experiments.dynamic_dance.contracts import sha256


OLD_ROOT = Path('/mnt/lightsem-data/experiments/lightsem/dynamic_dance_20261005_v1')
DEFAULT_ROOT = Path('/mnt/lightsem-data/experiments/lightsem/dynamic_dance_combo_20261005_v1')
TRACKS = ('shpongle', 'cure_for_me')
MAX_WAVE_BYTES = 100_000_000


def waveform_sha(wave):
    return hashlib.sha256(np.ascontiguousarray(wave, dtype=np.float32).tobytes()).hexdigest()


def match_duration(wave, target_samples):
    wave = np.asarray(wave)
    if wave.dtype != np.float32 or wave.ndim != 2 or wave.shape[0] != 2 or not np.isfinite(wave).all():
        raise ValueError('Expected finite stereo float32 waveform in channels-first order')
    if type(target_samples) is not int or target_samples <= 0:
        raise ValueError('Positive integer target sample count required')
    out = np.zeros((2, target_samples), np.float32)
    common = min(wave.shape[1], target_samples)
    out[:, :common] = wave[:, :common]
    return out, {'original_samples': wave.shape[1], 'target_samples': target_samples,
                 'retained_source_samples': common,
                 'cropped_tail_samples': max(0, wave.shape[1]-target_samples),
                 'zero_padded_tail_samples': max(0, target_samples-wave.shape[1]),
                 'crop_start_sample': 0, 'sample_rate': 44100,
                 'policy': 'retain chronological prefix; crop excess tail or append exact zeros; no loop or repeated samples'}


def _binding_path(root):
    return Path(root)/'records/DONOR_DURATION_BINDINGS_v1.json'


def verify_binding(root):
    binding = read_receipt(_binding_path(root))
    for path, expected in binding['files'].items():
        if sha256(path) != expected:
            raise ValueError('Frozen donor preparation input changed: '+path)
    if sum(Path(row['entry']['wav_path']).stat().st_size for row in binding['tracks'].values()) >= MAX_WAVE_BYTES:
        raise ValueError('Lossless donor waveform byte budget exceeded')
    return binding


def prepare(root=DEFAULT_ROOT):
    import soundfile as sf
    root = Path(root)
    if _binding_path(root).exists():
        return verify_binding(root)
    plan_path = root/'PLAN.json'
    plan = json.loads(plan_path.read_text())
    tracks = {row['id']: row for row in plan['tracks']}
    if set(tracks) != set(TRACKS):
        raise ValueError('Expected the fixed two showcase songs')
    out = root/'features/donor_audio'
    out.mkdir(parents=True, exist_ok=True)
    rows, files = {}, {str(plan_path): sha256(plan_path)}
    for song in TRACKS:
        old_path = OLD_ROOT/f'features/timelines/{song}__donor.json'
        old = json.loads(old_path.read_text())['metadata']
        donor_path = Path(old['input_path'])
        if sha256(donor_path) != old['input_sha256']:
            raise ValueError('Original donor waveform changed')
        intervention = old['intervention']
        if intervention['kind'] != 'gain' or intervention['original_song'] != song or not np.isfinite(intervention['db']):
            raise ValueError('Expected the original fixed donor gain intervention')
        own_audio = Path(tracks[song]['audio'])
        target = sf.info(own_audio)
        if target.samplerate != 44100:
            raise ValueError('Showcase clock must remain 44.1 kHz')
        wave = load_waveform(donor_path)
        old_gained, repeated_intervention = waveform_counterfactual(wave, 44100, intervention)
        if waveform_sha(old_gained) != old['analyzed_waveform_sha256'] or repeated_intervention != intervention:
            raise ValueError('Original donor gain replay differs from historical analyzed waveform')
        matched, duration = match_duration(wave, int(target.frames))
        analyzed, repeated_intervention = waveform_counterfactual(matched, 44100, intervention)
        if not np.array_equal(analyzed[:, :duration['retained_source_samples']], old_gained[:, :duration['retained_source_samples']]):
            raise ValueError('Retained donor samples changed beyond the original gain')
        extension = '.flac' if song == 'shpongle' else '.wav'
        path = out/(song+'_donor_matched'+extension)
        if path.exists():
            raise FileExistsError('Preserve partially prepared donor waveform: '+str(path))
        guard_disk(root, required_bytes=analyzed.nbytes+2**20)
        if song == 'shpongle':
            if not np.array_equal(matched, np.round(matched*32768)/32768):
                raise ValueError('Shpongle donor is not losslessly representable as original PCM16')
            sf.write(path, matched.T, 44100, format='FLAC', subtype='PCM_16')
            encoding = 'FLAC PCM16; bit-exact original PCM16 samples plus zeros'
        else:
            sf.write(path, matched.T, 44100, format='WAV', subtype='FLOAT')
            encoding = 'WAV FLOAT32; exact original floating samples, no quantization or clipping'
        reloaded = load_waveform(path)
        if not np.array_equal(reloaded, matched):
            raise ValueError('Lossless donor file round trip failed')
        reanalyzed, _ = waveform_counterfactual(reloaded, 44100, intervention)
        if not np.array_equal(reanalyzed, analyzed):
            raise ValueError('Prepared donor does not reproduce frozen analyzed samples')
        key = song+'__donor_matched'
        timeline = root/f'features/timelines/{key}.npz'
        base92 = root/f'features/base92/{key}.npz'
        entry = {'key': key, 'partition': 'counterfactual', 'canonical_id': song+'__donor',
                 'wav_path': str(path), 'wav_sha256': sha256(path), 'timeline_path': str(timeline)}
        row = {'track': song, 'condition': 'donor', 'music_timeline': str(timeline),
               'base92': str(base92), 'entry': entry, 'intervention': intervention,
               'analyzed_waveform_sha256': waveform_sha(analyzed),
               'raw_matched_waveform_sha256': waveform_sha(matched),
               'duration': {**duration, 'target_seconds': target.frames/44100,
                            'zero_padded_tail_s': duration['zero_padded_tail_samples']/44100,
                            'cropped_tail_s': duration['cropped_tail_samples']/44100},
               'encoding': encoding, 'bytes': path.stat().st_size,
               'source': {'historical_metadata': str(old_path), 'donor_path': str(donor_path),
                          'donor_sha256': old['input_sha256'], 'historical_analyzed_waveform_sha256': old['analyzed_waveform_sha256'],
                          'recipient_audio': str(own_audio), 'recipient_sha256': sha256(own_audio)},
               'gain_policy': 'Preserve historical scalar gain exactly, applied once after decode; no new loudness fit after duration matching',
               'finite': bool(np.isfinite(analyzed).all()), 'peak_abs_after_gain': float(np.abs(analyzed).max()),
               'padding_waveform_exactly_zero': bool(not np.any(analyzed[:, duration['retained_source_samples']:]))}
        rows[song] = row
        for source in (old_path, donor_path, own_audio, path):
            files[str(source)] = sha256(source)
    total = sum(row['bytes'] for row in rows.values())
    if total >= MAX_WAVE_BYTES:
        raise ValueError(f'Preserve prepared waveforms but do not extract: {total} bytes exceeds budget')
    binding = {'schema_version': 1, 'scope': 'Full-song donor duration correction only; original PLAN and historical features remain immutable',
               'override_conditions': ['donor'], 'tracks': rows, 'files': files,
               'waveform_total_bytes': total, 'waveform_byte_limit': MAX_WAVE_BYTES,
               'source_sha256': sha256(__file__), 'prepared_before_gpu_extraction': True,
               'all_audio_paths_recomputed': ['Base92', 'MuQ', 'HTDemucs', 'BeatThis', 'AllInOne', 'physical_observations'],
               'silence_note': 'Shpongle donor ends early; trailing exact silence is a declared intervention consequence, not extrapolated donor features.'}
    write_receipt(_binding_path(root), binding)
    return verify_binding(root)


def extract(root, track):
    import torch
    from lightsem.experiments.dynamic_dance.audio import AudioAnalyzer
    from lightsem.experiments.dynamic_dance.contracts import MusicTimeline
    root = Path(root)
    binding = verify_binding(root)
    if track not in binding['tracks']:
        raise ValueError('Unregistered donor track')
    row = binding['tracks'][track]
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    analyzer = AudioAnalyzer(device='cuda', structure_device='cpu')
    receipt = analyzer.extract(row['entry'], root, intervention=row['intervention'])
    timeline = MusicTimeline.load(row['music_timeline'])
    if timeline.metadata['analyzed_waveform_sha256'] != row['analyzed_waveform_sha256']:
        raise ValueError('Extracted MusicTimeline did not use the prepared donor samples')
    if receipt['base92']['analyzed_waveform_sha256'] != row['analyzed_waveform_sha256']:
        raise ValueError('Base92 and MusicTimeline analyzed different donor samples')
    if abs(timeline.metadata['duration_s']-row['duration']['target_seconds']) > 1e-10:
        raise ValueError('Extracted donor duration drift')
    verify_binding(root)
    return {'track': track, 'timeline': row['music_timeline'], 'timeline_sha256': sha256(row['music_timeline']),
            'base92': row['base92'], 'base92_sha256': sha256(row['base92']),
            'analyzed_waveform_sha256': row['analyzed_waveform_sha256'], 'seconds': receipt['duration_s']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('prepare', 'extract'))
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--track', choices=TRACKS)
    parser.add_argument('--guarded', action='store_true')
    args = parser.parse_args()
    if args.stage == 'prepare':
        value = prepare(args.root)
        print(json.dumps({'binding': str(_binding_path(args.root)), 'bytes': value['waveform_total_bytes'],
                          'durations': {song: row['duration'] for song, row in value['tracks'].items()}}, sort_keys=True))
    else:
        if not args.guarded or args.track is None:
            parser.error('extract requires --guarded and --track')
        from scripts.motion.dynamic_combo_guard import authenticate_guarded_job
        authenticate_guarded_job(__file__)
        print(json.dumps(extract(args.root, args.track), sort_keys=True))


if __name__ == '__main__':
    main()
