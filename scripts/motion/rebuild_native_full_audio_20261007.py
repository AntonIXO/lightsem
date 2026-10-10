"""Rebuild exact decoded Just Dance waveforms for a new native-fit run.

The output is isolated from the legacy cache.  Each WAV is decoded directly
from the source video, measured from the resulting samples, and authenticated
with source and waveform hashes.  No mean-dB admission filter is applied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from multiprocessing import Pool
from pathlib import Path
import subprocess

import numpy as np
from lightsem.experiments.native_dynamics.full_fit_admission import verify_normalization


SOURCE = Path('/mnt/lightsem-data/justdance')
MANIFEST = Path('/mnt/lightsem-data/cache/justdance_manifest.json')


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def decode_one(job):
    rid, filename, output = job
    output = Path(output)
    wav = output / f'{rid}.wav'
    receipt = output / f'{rid}.json'
    source = SOURCE / filename
    if receipt.exists() and wav.exists():
        prior = json.loads(receipt.read_text())
        if prior['source_sha256'] != sha256(source) or prior['wav_sha256'] != sha256(wav):
            raise ValueError(f'Stale audio cache for {rid}; reprobe in a new artifact root')
        return prior
    if wav.exists() or receipt.exists():
        raise RuntimeError(f'partial audio output for {rid}')
    output.mkdir(parents=True, exist_ok=True)
    partial = output / f'{rid}.partial.wav'
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-y',
               '-i', str(source), '-map', '0:a:0', '-vn', '-ac', '2', '-ar', '44100',
               '-c:a', 'pcm_f32le', str(partial)]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f'ffmpeg failed for {rid}: {result.stderr[-500:]}')
    partial.replace(wav)
    import soundfile as sf
    x, sr = sf.read(wav, dtype='float32', always_2d=True)
    if sr != 44100 or x.shape[1] != 2 or len(x) < 8 * sr or not np.isfinite(x).all():
        raise ValueError(f'invalid decoded waveform for {rid}: sr={sr} shape={x.shape}')
    samples = x.reshape(-1).astype(np.float64)
    rms = float(np.sqrt(np.mean(samples * samples)))
    peak = float(np.max(np.abs(samples)))
    mean_db = float(20 * np.log10(max(rms, 1e-12)))
    peak_db = float(20 * np.log10(max(peak, 1e-12)))
    nonzero = float(np.mean(np.abs(samples) > 1e-7))
    classification = ('silent' if peak == 0 else
                      'low_level_usable' if mean_db < -40 else 'normal')
    value = {'schema_version': 1, 'recording_id': rid, 'source_path': str(source),
             'source_sha256': sha256(source), 'wav_path': str(wav),
             'wav_sha256': sha256(wav), 'sample_rate': sr, 'channels': 2,
             'frames': int(len(x)), 'duration_s': float(len(x) / sr),
             'rms_db': mean_db, 'peak_db': peak_db, 'nonzero_fraction': nonzero,
             'classification': classification, 'command': command}
    receipt.write_text(json.dumps(value, indent=2) + '\n')
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ids-file', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    ids = {line.strip() for line in args.ids_file.read_text().splitlines()
           if line.strip() and not line.lstrip().startswith('#')}
    rows = {row['yt_id']: row for row in json.loads(MANIFEST.read_text())
            if row.get('yt_id')}
    missing = sorted(ids - rows.keys())
    if missing:
        raise KeyError(f'IDs missing from fresh source manifest: {missing[:10]}')
    jobs = [(rid, rows[rid]['file'], str(args.output)) for rid in sorted(ids)]
    with Pool(max(1, min(args.workers, len(jobs)))) as pool:
        results = []
        for i, value in enumerate(pool.imap_unordered(decode_one, jobs), 1):
            results.append(value)
            if i == 1 or i % 25 == 0 or i == len(jobs):
                print(f'AUDIO {i}/{len(jobs)} {value["recording_id"]} {value["classification"]}', flush=True)
    results.sort(key=lambda row: row['recording_id'])
    report_path = args.output / 'MANIFEST.json'
    prior = {}
    if report_path.exists():
        previous = json.loads(report_path.read_text())
        prior = {row['recording_id']: row for row in previous.get('rows', [])}
    prior.update({row['recording_id']: row for row in results})
    # A low-level track may have an explicit, authenticated gain receipt. Keep
    # the normalized waveform binding when a later reprobe rebuilds the raw
    # manifest; never silently revert training to the unnormalized file.
    for rid, row in list(prior.items()):
        receipt = args.output / f'{rid}_NORMALIZATION.json'
        if receipt.exists():
            norm = json.loads(receipt.read_text())
            row['wav_path'] = norm['output_wav']
            row['wav_sha256'] = norm['output_wav_sha256']
            row['normalization_receipt'] = str(receipt)
            row['classification'] = 'low_level_usable'
            row['normalization'] = 'explicit_authenticated_gain'
            verify_normalization(row)
    results = [prior[key] for key in sorted(prior)]
    report = {'schema_version': 1, 'status': 'complete', 'source_manifest': str(MANIFEST),
              'source_manifest_sha256': sha256(MANIFEST), 'ids_file': str(args.ids_file),
              'ids_file_sha256': sha256(args.ids_file), 'rows': results,
              'classification_counts': {k: sum(r['classification'] == k for r in results)
                                        for k in ('normal', 'low_level_usable', 'silent')},
              'audio_code_sha256': sha256(Path(__file__))}
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'rows': len(results),
                      'classification_counts': report['classification_counts']}, sort_keys=True))


if __name__ == '__main__':
    main()
