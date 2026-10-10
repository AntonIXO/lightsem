"""Storage-gated extraction of missing max-manifest MuQ streams.

This command is intentionally separate from admission. It writes only into a
new package feature directory, refuses TEST recordings, and refuses to start
unless the estimated output fits while preserving the 20 GiB free-space floor.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--package', type=Path, required=True)
    p.add_argument('--model-receipt', type=Path, default=Path('/mnt/lightsem-data/experiments/lightsem/night_expression_20260909/features/muq_model.json'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--minimum-free-gib', type=float, default=20.0)
    args = p.parse_args()
    package, output = args.package, args.output
    report = json.loads((package/'MUQ_ADMISSION_REPORT.json').read_text())
    ids = report['max_missing_recordings']
    if not ids:
        print(json.dumps({'status': 'complete', 'missing': 0})); return
    stage = json.loads(Path('/mnt/lightsem-data/experiments/lightsem/stage11_20260908/data/manifest.json').read_text())
    records = {r['recording_id']: r for r in stage['recordings']}
    forbidden = [rid for rid in ids if records[rid]['partition'] == 'test']
    if forbidden:
        raise RuntimeError('Refusing TEST MuQ extraction: ' + ','.join(forbidden))
    model = json.loads(args.model_receipt.read_text())
    if model.get('model_id') != 'OpenMuQ/MuQ-large-msd-iter' or model.get('revision') != '0562a57814f6f8bbd9fdea0a25921a2fce1a841a':
        raise RuntimeError('Pinned MuQ receipt mismatch')
    estimated = int(report.get('max_missing_estimated_bytes', 0))
    free = shutil.disk_usage(output).free
    if free - estimated < args.minimum_free_gib * 2**30:
        raise RuntimeError(f'storage gate: free={free/2**30:.2f}GiB estimated={estimated/2**30:.2f}GiB floor={args.minimum_free_gib:.2f}GiB')
    output.mkdir(parents=True, exist_ok=True)
    import soundfile as sf
    import soxr
    import torch
    from muq import MuQ

    def extract_waveform(model_obj, waveform):
        sr, rate, core, context, size = 24000, 25, 20*24000, 5*24000, 30*24000
        waveform = np.asarray(waveform, np.float32)
        padded = np.pad(waveform, (context, size), mode='reflect')
        n = int(np.ceil(len(waveform)/sr*rate)); out = {layer: [] for layer in (3, 6, 9, 12)}
        with torch.inference_mode():
            for start in range(0, len(waveform), core):
                x = torch.from_numpy(padded[start:start+size])[None].cuda()
                result = model_obj(x, output_hidden_states=True)
                take = min(20*rate, n-start//(sr//rate))
                for layer in out:
                    z = result.hidden_states[layer][0, 5*rate:5*rate+take]
                    if z.shape != (take, 1024) or not torch.isfinite(z).all(): raise FloatingPointError('invalid MuQ output')
                    out[layer].append(z.cpu().numpy().astype(np.float16))
        return {layer: np.concatenate(chunks) for layer, chunks in out.items()}

    output.mkdir(parents=True, exist_ok=True); model_obj = MuQ.from_pretrained(model['path']).cuda().float().eval(); rows = []
    for rid in ids:
        row = records[rid]; wav = Path(row['wav_path']); x, sr = sf.read(wav, dtype='float32', always_2d=True); x = x.mean(1)
        if sr != 24000: x = soxr.resample(x, sr, 24000)
        features = extract_waveform(model_obj, x); artifacts = {}
        for layer, array in features.items():
            path = output/f'{rid}_layer{layer}.npy'; partial = output/f'{rid}_layer{layer}.npy.partial';
            if path.exists(): raise RuntimeError(f'output already exists: {path}')
            with partial.open('wb') as stream: np.save(stream, array)
            partial.replace(path); artifacts[str(path)] = {'bytes': path.stat().st_size, 'sha256': digest(path)}
        rows.append({'recording_id': rid, 'canonical_id': row['canonical_id'], 'partition': row['partition'], 'source_waveform': str(wav),
                     'waveform_sha256': digest(wav), 'frames': int(len(next(iter(features.values())))), 'native_hz': 25., 'time_origin_s': 0.,
                     'context_s': 30, 'core_s': 20, 'model_revision': model['revision'], 'artifacts': artifacts})
    receipt = {'status': 'complete', 'recordings': ids, 'rows': rows, 'model_id': model['model_id'], 'revision': model['revision'],
               'minimum_free_gib': args.minimum_free_gib, 'estimated_bytes': estimated, 'source_manifest_sha256': digest(package/'max_masked_window_train_manifest.json')}
    (output/'COMPLETE.json').write_text(json.dumps(receipt, indent=2) + '\n'); print(json.dumps(receipt, indent=2))


if __name__ == '__main__': main()
