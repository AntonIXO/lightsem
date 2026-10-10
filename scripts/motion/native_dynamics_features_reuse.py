"""Build an authenticated all-fit feature index with exact-cache reuse.

Reuse is allowed only for the same audio implementation, waveform hash,
MuQ model/revision/layer/weights, timestamps, and Base92 recipe. New streams
are extracted into the new campaign root; nothing in an old cache is changed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from lightsem.experiments.native_dynamics.data import read, write, sha
from lightsem.experiments.native_dynamics.feature_cache import verify_cached_features
from lightsem.experiments.native_dynamics.full_fit_admission import verify_muq, verify_normalization
from lightsem.experiments.dynamic_dance.audio import (
    AudioAnalyzer, load_waveform, extract_base92, guard_disk, model_features,
    read_receipt,
)
from lightsem.experiments.dynamic_dance.contracts import MusicTimeline


def _waveform_id_from_receipt(cache_root: Path, key: str) -> str:
    receipt = read_receipt(cache_root/'features/receipts'/f'{key}.json')
    return receipt['base92']['analyzed_waveform_sha256']


def _unique(entries):
    result = {}
    for row in entries:
        key = row['feature_key']
        binding = (row['wav_sha256'], str(Path(row['wav_path']).resolve()),
                   row['frames'], row['timeline_path'])
        if key in result and result[key][0] != binding:
            raise ValueError(f'Conflicting waveform or clock for feature key {key}')
        result.setdefault(key, (binding, row))
    return [row for _, row in result.values()]


def prepare(root: Path, reuse_root: Path, guard_job: str, partial_root: Path | None = None):
    root, reuse_root = Path(root), Path(reuse_root)
    partial_root = Path(partial_root) if partial_root is not None else None
    manifest = read(root/'data/MANIFEST.json')
    if manifest.get('status') == 'partial_diagnostic':
        raise ValueError('Partial pose admission cannot authorize features')
    if (root/'features/COMPLETE.json').exists():
        raise FileExistsError('Completed feature report exists')
    protocol = read(root/'TRAINING_PROTOCOL.json')
    expected = protocol['muq']
    fd_path = root/'data/FD_MANIFEST.json'
    if not fd_path.exists():
        raise FileNotFoundError('Full-fit FD_MANIFEST.json is required')
    fd = read(fd_path)
    rows = manifest['entries'] + fd['entries']
    unique = _unique(rows)
    analyzer = None
    products, audits, bindings = {}, [], []
    reused, extracted = [], []
    started = time.monotonic()
    for index, row in enumerate(unique, 1):
        key = row['feature_key']
        guard_disk(root, required_bytes=256*2**20)
        if row.get('normalization_receipt'):
            verify_normalization(row)
        if sha(row['wav_path']) != row['wav_sha256']:
            raise ValueError(f'Waveform changed: {key}')
        # Existing products remain external, immutable, and fully receipt-bound.
        try:
            waveform_id = _waveform_id_from_receipt(reuse_root, key)
            product, audit, binding = verify_cached_features(
                row, expected, reuse_root, waveform_id)
            products[key], audits_row, bindings_row = product, audit, binding
            audits.append(audits_row); bindings.append(bindings_row); reused.append(key)
            print('REUSE_FEATURE', index, len(unique), key, flush=True)
            continue
        except (FileNotFoundError, KeyError):
            pass
        # Recover products from an interrupted run only after revalidating the
        # complete receipt and every referenced artifact in its partial root.
        if partial_root is not None:
            try:
                waveform_id = _waveform_id_from_receipt(partial_root, key)
                product, audit, binding = verify_cached_features(
                    row, expected, partial_root, waveform_id)
                products[key], audits_row, bindings_row = product, audit, binding
                audits.append(audits_row); bindings.append(bindings_row); reused.append(key)
                print('RESUME_FEATURE', index, len(unique), key, flush=True)
                continue
            except (FileNotFoundError, KeyError):
                pass
        if analyzer is None:
            analyzer = AudioAnalyzer('cuda', 'cpu')
        timeline_path = root/'features/timelines'/f'{key}.npz'
        entry = dict(row, key=key, timeline_path=str(timeline_path))
        wave = load_waveform(row['wav_path'])
        wave_id = hashlib.sha256(np.ascontiguousarray(wave).tobytes()).hexdigest()
        base = extract_base92(wave, np.arange(row['frames'])/30, entry,
                              root, wave_id, None)
        del wave
        entry.update(base92_path=base['path'], base92_sha256=base['npz_sha256'],
                     base92_receipt=base,
                     base92_provenance='Exact local waveform; scalar gain only when separately authenticated')
        receipt = analyzer.extract(entry, root)
        if sha(receipt['timeline_path']) != receipt['timeline_sha256']:
            raise ValueError(f'Changed timeline payload: {key}')
        sealed = read_receipt(root/'features/receipts'/f'{key}.json')
        if sealed != receipt:
            raise ValueError(f'Feature receipt differs: {key}')
        timeline = MusicTimeline.load(receipt['timeline_path'])
        audit = verify_muq(timeline, expected, row['wav_sha256'])
        if timeline.metadata['analyzed_waveform_sha256'] != base['analyzed_waveform_sha256']:
            raise ValueError(f'Base92/MuQ waveform mismatch: {key}')
        products[key] = {
            'timeline_path': receipt['timeline_path'], 'timeline_sha256': receipt['timeline_sha256'],
            'base92_path': base['path'], 'base92_sha256': base['npz_sha256'],
            'seconds': receipt['elapsed_s'], 'feature_key': key,
            'feature_receipt_sha256': sha(root/'features/receipts'/f'{key}.json'),
            'reused': False,
        }
        bindings.append({'key':key,'waveform_sha256':base['analyzed_waveform_sha256'],
                         'wav_file_sha256':row['wav_sha256'],
                         'base92_receipt_sha256':sha(base['path'].replace('.npz','.json')),
                         'timeline_receipt_sha256':sha(root/'features/receipts'/f'{key}.json'),
                         'muq_model':timeline.metadata['muq'],
                         'timestamps':timeline.metadata['timestamps']})
        audits.append(dict(key=key, **audit)); extracted.append(key)
        print('EXTRACT_FEATURE', index, len(unique), key, receipt['elapsed_s'], flush=True)
    reports = []
    for row in manifest['entries']:
        product = products[row['feature_key']]
        reports.append(dict(key=row['key'], partition=row['partition'], **product))
    fd_rows = []
    for row in fd['entries']:
        product = products[row['feature_key']]
        if 'fd_aligned' in product:
            aligned = Path(product['fd_aligned'])
        else:
            # Bind freshly extracted FD audio to the paired motion clock.
            # A prior alignment cannot stand in for the newly measured stream.
            timeline = MusicTimeline.load(product['timeline_path'])
            target = np.arange(row['frames'], dtype=np.float64) / 30
            extra, _ = model_features(timeline, target)
            with np.load(product['base92_path'], allow_pickle=False) as arrays:
                if not np.array_equal(arrays['time30'], target):
                    raise ValueError('Fresh FineDance Base92 clock mismatch')
                x = arrays['x30'].copy()
            aligned = root/'features/fd_aligned'/f"{row['key']}.npz"
            aligned.parent.mkdir(parents=True, exist_ok=True)
            with aligned.open('xb') as stream:
                np.savez_compressed(stream, x=x, extra=extra)
        digest = sha(aligned)
        fd_rows.append(dict(row, audio_path=str(aligned), audio_sha256=digest,
                            cache_path=str(aligned), cache_sha256=digest,
                            feature_receipt=product,
                            inputs={product['timeline_path']:product['timeline_sha256'],
                                    product['base92_path']:product['base92_sha256']}))
    # The input manifest is frozen by the guard and supervision receipt.
    # Publish the feature-bound view as a new output, never overwrite it.
    write(root/'features/FD_MANIFEST.json', dict(fd, entries=fd_rows,
         feature_source='exact own-track native MuQ; authenticated reuse or fresh extraction'))
    write(root/'features/MUQ_IDENTITY_AUDIT.json', {
        'expected':expected,'recordings':audits,'verified_streams':len(audits),
        'reused_streams':len(reused),'fresh_streams':len(extracted),
        'reuse_root':str(reuse_root),
        'policy':'Reuse only after model, waveform, timestamps, context, code and artifact hash verification.'})
    write(root/'features/BASE92_MUQ_BINDING.json', {
        'schema_version':1,'status':'verified','recipe_version':2,'rows':bindings,
        'count':len(bindings),'reused_streams':len(reused),'fresh_streams':len(extracted),
        'policy':'Base92 and MuQ share exact analyzed waveform bytes and native timestamps; no aggregate matrix.'})
    write(root/'features/COMPLETE.json', {
        'entries':reports,'data_manifest_sha256':sha(root/'data/MANIFEST.json'),
        'fd_manifest_sha256':sha(root/'features/FD_MANIFEST.json'),
        'fd_manifest_path':str(root/'features/FD_MANIFEST.json'),
        'fd_input_manifest_sha256':sha(root/'data/FD_MANIFEST.json'),
        'base92_muq_binding_sha256':sha(root/'features/BASE92_MUQ_BINDING.json'),
        'elapsed_s':time.monotonic()-started,'guard_job':guard_job,
        'source':sha(__file__),'unique_audio_streams':len(unique),
        'jd_coach_references':len(reports),'fd_records':len(fd['entries']),
        'original_source_partitions':dict(Counter(r.get('source_partition',r['partition']) for r in unique)),
        'no_training':True,'scope':'explicit all-partition fit',
        'models':'Frozen MuQ layer9, HTDemucs, BeatThis, AllInOne; exact own-song waveform for Base92 and 1061 features',
        'reuse_root':str(reuse_root),'reused_streams':len(reused),'fresh_streams':len(extracted)})
    return {'unique':len(unique),'reused':len(reused),'fresh':len(extracted)}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True); p.add_argument('--reuse-root',type=Path,required=True)
    p.add_argument('--partial-root',type=Path)
    p.add_argument('--guarded',action='store_true'); p.add_argument('--guard-script',type=Path)
    a=p.parse_args()
    if not a.guarded: raise ValueError('Frozen GPU guard required')
    guard_path=a.guard_script or Path(__file__).with_name('native_full_fit_guard_rtmpose_v5_20261009.py')
    spec=importlib.util.spec_from_file_location('_native_feature_guard',guard_path)
    guard=importlib.util.module_from_spec(spec); spec.loader.exec_module(guard)
    receipt=guard.authenticate_guarded_job(__file__)
    print(json.dumps(prepare(a.root,a.reuse_root,receipt['identity']['job_id'],a.partial_root)))


if __name__=='__main__': main()
