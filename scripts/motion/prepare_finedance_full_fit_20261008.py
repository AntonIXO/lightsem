"""Prepare every locally paired FineDance source, retaining original split metadata."""
from pathlib import Path
import argparse
from collections import Counter
import json
import numpy as np
import soundfile as sf
import torch
from lightsem.experiments.motion_study.geometry import prepare_native, FPS
from lightsem.experiments.native_dynamics.data import sha, write

SPLIT=Path('/mnt/lightsem-data/experiments/lightsem/motion_representation_20260910/data/split.json')
PRIOR=Path('/mnt/lightsem-data/experiments/lightsem/motion_representation_20260910/data/manifest.json')


def prepare(root):
    torch.set_num_threads(2)
    root=Path(root)
    target=root/'data/FD_MOTION_OVERLAP_MANIFEST.json'
    if target.exists(): raise FileExistsError(target)
    sources=json.loads(SPLIT.read_text())['entries']
    prior={r['key']:r for r in json.loads(PRIOR.read_text())['entries']}
    entries=[]; exclusions=[]
    for source in sources:
        key=source['key']; motion=Path(source['motion_path']); wav=Path(source['wav_path'])
        if not motion.exists() or not wav.exists():
            exclusions.append({'key':key,'reason':'missing_local_pair'}); continue
        if sha(motion)!=source['motion_sha256'] or sha(wav)!=source['wav_sha256']:
            raise ValueError(f'Changed FineDance source {key}')
        raw=np.load(motion,allow_pickle=False)
        info=sf.info(wav)
        motion_duration=len(raw)/FPS
        difference=motion_duration-info.duration
        # Bind only the measured overlap when released audio and motion tails
        # differ. Do not invent an offset or synthesize either tail.
        bound_frames=min(len(raw),int(np.floor(info.duration*FPS+1e-8)))
        if bound_frames < 480:
            exclusions.append({'key':key,'reason':'overlap_shorter_than_training_window',
                               'difference_s':difference,'source_partition':source['partition']});continue
        if not np.isfinite(raw).all():
            exclusions.append({'key':key,'reason':'nonfinite_source_motion'});continue
        old=prior.get(key); path=None
        candidate_path=root/'data/fd_motion'/f'{key}.npz'
        if candidate_path.exists():
            try:
                with np.load(candidate_path,allow_pickle=False) as prior_value:
                    if len(prior_value['features']) != bound_frames:
                        candidate_path=root/'data/fd_motion_overlap'/f'{key}.npz'
            except Exception:
                candidate_path=root/'data/fd_motion_overlap'/f'{key}.npz'
        if candidate_path.exists():
            try:
                with np.load(candidate_path,allow_pickle=False) as prior_value:
                    if all(len(prior_value[name])==bound_frames for name in ('features','joints','rotations_6d','contacts')):
                        path=candidate_path
            except Exception:
                pass
        if path is None and old and Path(old['path']).exists() and sha(old['path'])==old['sha256']:
            try:
                with np.load(old['path'],allow_pickle=False) as prior_value:
                    if all(len(prior_value[name])==bound_frames for name in ('features','joints','rotations_6d','contacts')):
                        path=Path(old['path'])
            except Exception:
                pass
        if path is None:
            value=prepare_native(raw[:bound_frames])
            if not all(np.isfinite(x).all() for x in value.values()):
                exclusions.append({'key':key,'reason':'nonfinite_native_conversion'});continue
            path=candidate_path;path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('xb') as stream: np.savez_compressed(stream,**value)
        row=dict(source,domain='fd',partition='train',source_partition=source['partition'],
            historical_excluded=source.get('excluded',False),excluded=False,exclusion_reason=None,
            path=str(path),sha256=sha(path),duration_s=bound_frames/FPS,audio_duration_s=info.duration,
            n_frames=bound_frames,frames=bound_frames,
            audio_offset_s=0.,duration_difference_s=difference,fps=FPS,
            released_motion_duration_s=motion_duration,
            motion_tail_not_paired_s=max(0.,motion_duration-bound_frames/FPS),
            audio_tail_not_paired_s=max(0.,info.duration-bound_frames/FPS),
            alignment_status='assumed released common t0; tail difference disclosed, not a verified offset estimate',
            feature_key=f'fd_{key}',timeline_path=str(root/'features/timelines'/f'fd_{key}.npz'))
        entries.append(row)
    report={'schema_version':1,'status':'full_fit_motion_paired_no_split','entries':entries,'exclusions':exclusions,
            'source_split_sha256':sha(SPLIT),'source_split_preserved':True,'source_partitions':dict(Counter(r['partition'] for r in sources)),
            'requested_recordings':len(sources),'admitted_recordings':len(entries),'admitted_groups':len({r['canonical_id'] for r in entries}),
            'test_payload_opened':True,'legacy_author_ignore_policy':'Metadata retained; finite paired data not discarded solely for author exclusion label',
            'audio_binding':'assumed released frame0 to audio t0; real overlapping samples only; no fitted offset or extrapolated tail',
            'geometry_sha256':sha(Path('/root/lightsem/lightsem/experiments/motion_study/geometry.py'))}
    write(target,report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('entries','exclusions')},indent=2))
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();prepare(a.root)
