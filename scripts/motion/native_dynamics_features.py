"""Exact per-recording MuQ/Base92; coach streams share only their own audio."""
from pathlib import Path
import argparse
from collections import Counter
import hashlib
import importlib.util
import time
import numpy as np
from lightsem.experiments.native_dynamics.data import read,write,sha,ROOT
from lightsem.experiments.native_dynamics.full_fit_admission import verify_muq, verify_normalization
from lightsem.experiments.dynamic_dance.audio import (
    AudioAnalyzer,load_waveform,extract_base92,guard_disk,model_features,read_receipt,
)
from lightsem.experiments.dynamic_dance.contracts import MusicTimeline


def audio_work_items(entries):
    """Require a single exact waveform/timeline/clock for each feature key."""
    unique={}
    for row in entries:
        key=row['feature_key']
        binding=(row['wav_sha256'],str(Path(row['wav_path']).resolve()),row['frames'],row['timeline_path'])
        if key in unique and unique[key][0]!=binding:
            raise ValueError(f'Conflicting waveform or clock for feature key {key}')
        unique.setdefault(key,(binding,row))
    return [row for _,row in unique.values()]


def prepare(root, guard_job):
    root=Path(root); manifest=read(root/'data/MANIFEST.json')
    if manifest.get('status')=='partial_diagnostic': raise ValueError('Partial cache cannot authorize training')
    if (root/'features/COMPLETE.json').exists(): raise FileExistsError('Completed feature report exists')
    protocol=read(root/'TRAINING_PROTOCOL.json')
    expected=protocol.get('muq') or read('/root/lightsem/configs/dance/fixed_c.json')['muq']
    fd_path=root/'data/FD_MOTION_OVERLAP_MANIFEST.json'
    if not fd_path.exists():fd_path=root/'data/FD_MOTION_MANIFEST.json'
    fd=read(fd_path) if fd_path.exists() else {'entries':[]}
    rows=manifest['entries']+fd['entries']
    unique=audio_work_items(rows)
    analyzer=AudioAnalyzer('cuda','cpu'); products={}; audits=[]; bindings=[]; started=time.monotonic()
    for index,row in enumerate(unique,1):
        guard_disk(root,required_bytes=256*2**20)
        if row.get('normalization_receipt'): verify_normalization(row)
        if sha(row['wav_path'])!=row['wav_sha256']:raise ValueError('Waveform changed')
        entry={k:row[k] for k in ('partition','wav_path','wav_sha256','timeline_path')}
        entry['key']=row['feature_key']
        wave=load_waveform(row['wav_path'])
        waveid=hashlib.sha256(np.ascontiguousarray(wave).tobytes()).hexdigest()
        base=extract_base92(wave,np.arange(row['frames'])/30,entry,root,waveid,None)
        del wave
        entry.update(base92_path=base['path'],base92_sha256=base['npz_sha256'],
                     base92_receipt=base,
                     base92_provenance='Exact local waveform; scalar gain only when separately authenticated')
        receipt=analyzer.extract(entry,root)
        if sha(receipt['timeline_path'])!=receipt['timeline_sha256']:
            raise ValueError('Changed timeline payload')
        # Authenticate the receipt sidecar even for an in-process new result.
        sealed=read_receipt(root/'features/receipts'/f"{entry['key']}.json")
        if sealed!=receipt:raise ValueError('Feature receipt differs')
        timeline=MusicTimeline.load(receipt['timeline_path'])
        audit=verify_muq(timeline,expected,row['wav_sha256'])
        if timeline.metadata['analyzed_waveform_sha256']!=base['analyzed_waveform_sha256']:
            raise ValueError('Base92 and MuQ analyze different waveforms')
        products[entry['key']]={'timeline_path':receipt['timeline_path'],'timeline_sha256':receipt['timeline_sha256'],
            'base92_path':base['path'],'base92_sha256':base['npz_sha256'],'seconds':receipt['elapsed_s'],
            'feature_key':entry['key'],'feature_receipt_sha256':sha(root/'features/receipts'/f"{entry['key']}.json")}
        bindings.append({'key':entry['key'],'waveform_sha256':base['analyzed_waveform_sha256'],
                         'wav_file_sha256':row['wav_sha256'],'base92_receipt_sha256':sha(base['path'].replace('.npz','.json')),
                         'timeline_receipt_sha256':sha(root/'features/receipts'/f"{entry['key']}.json"),
                         'muq_model':timeline.metadata['muq'],'timestamps':timeline.metadata['timestamps']})
        audits.append(dict(key=entry['key'],**audit))
        if row.get('domain')=='fd':
            extra,_=model_features(timeline,np.arange(row['frames'])/30)
            with np.load(base['path'],allow_pickle=False) as z: x=z['x30']
            aligned=root/'features/fd_aligned'/f"{row['key']}.npz"
            aligned.parent.mkdir(parents=True,exist_ok=True)
            if aligned.exists():raise FileExistsError(aligned)
            with aligned.open('xb') as stream:np.savez_compressed(stream,x=x,extra=extra.astype(np.float16))
            products[entry['key']]['fd_aligned']=str(aligned)
        print('RECORDING_FEATURE',index,len(unique),entry['key'],receipt['elapsed_s'],flush=True)
    reports=[dict(key=row['key'],partition=row['partition'],**products[row['feature_key']]) for row in manifest['entries']]
    if fd['entries']:
        fd_rows=[]
        for row in fd['entries']:
            product=products[row['feature_key']];path=product['fd_aligned'];digest=sha(path)
            fd_rows.append(dict(row,audio_path=path,audio_sha256=digest,cache_path=path,cache_sha256=digest,
                feature_receipt=product,inputs={product['timeline_path']:product['timeline_sha256'],
                                              product['base92_path']:product['base92_sha256']}))
        write(root/'data/FD_MANIFEST.json',dict(fd,entries=fd_rows,feature_source='own-track native MuQ; no aggregate matrix'))
    write(root/'features/MUQ_IDENTITY_AUDIT.json',{'expected':expected,'recordings':audits,'verified_streams':len(audits)})
    write(root/'features/BASE92_MUQ_BINDING.json',{'schema_version':1,'status':'verified',
        'recipe_version':2,'rows':bindings,'count':len(bindings),
        'policy':'Base92 and MuQ share the exact analyzed waveform bytes and explicit native timestamps; no aggregate audio matrix.'})
    write(root/'features/COMPLETE.json',{'entries':reports,'data_manifest_sha256':sha(root/'data/MANIFEST.json'),
        'fd_manifest_sha256':sha(root/'data/FD_MANIFEST.json') if fd['entries'] else None,
        'base92_muq_binding_sha256':sha(root/'features/BASE92_MUQ_BINDING.json'),
        'elapsed_s':time.monotonic()-started,'guard_job':guard_job,'source':sha(__file__),
        'unique_audio_streams':len(unique),'jd_coach_references':len(reports),'fd_records':len(fd['entries']),
        'original_source_partitions':dict(Counter(r.get('source_partition',r['partition']) for r in unique)),
        'no_training':True,'scope':'explicit all-partition fit' if protocol.get('no_validation_split') else 'manifest partitions',
        'models':'Frozen MuQ layer9, HTDemucs, BeatThis, AllInOne; exact own-song waveform for Base92 and 1061 features'})


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--guarded',action='store_true');p.add_argument('--guard-script',type=Path);a=p.parse_args()
    if not a.guarded:raise ValueError('Frozen GPU guard required')
    guard_path=a.guard_script or Path(__file__).with_name('native_dynamics_guard.py')
    spec=importlib.util.spec_from_file_location('_native_features_guard',guard_path)
    guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
    receipt=guard.authenticate_guarded_job(__file__)
    prepare(a.root,receipt['identity']['job_id'])

if __name__=='__main__':main()
