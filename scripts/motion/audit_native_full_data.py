"""Inventory all local motion assets and expanded TRAIN admission before fitting.

No held-out pose payload is opened. Candidate coach tracks retain the original
confidence, visibility, timestamp, jump and 80%-coverage window checks. Low
whole-record coverage is reported separately from usable masked windows.
"""
from pathlib import Path
from collections import Counter
import argparse
import json
import shutil

import numpy as np
import soundfile as sf
import torch

from lightsem.experiments.native_dynamics.data import Quality,pose_observations,read,sha,write

CACHE=Path('/mnt/lightsem-data/cache')
PARENT=Path('/mnt/lightsem-data/experiments/lightsem/native_dynamics_jd_20261005_v1')
FD=Path('/mnt/lightsem-data/experiments/lightsem/dynamic_dance_20261005_v1/learning/cache/manifest.json')


def audit(output):
    torch.set_num_threads(2)
    old=read(PARENT/'data/MANIFEST.json');mapping=read(CACHE/'justdance_songs.json')
    catalog=read(CACHE/'justdance_manifest.json');known={r['yt_id']:r for r in mapping['rows']}
    protected={r['key'] for r in old['entries'] if r['partition']!='train'}
    protected_groups={r['group'] for r in old['entries'] if r['partition']!='train'}
    cfg=Quality();rows=[];rejected=[];inventory=Counter()
    for item in sorted(catalog,key=lambda r:r['yt_id']):
        key=item['yt_id'];group=known.get(key,{}).get('song_key');pose=CACHE/'justdance_pose'/f'{key}.npz'
        wav=CACHE/'justdance_wav'/f'{key}.wav';source=Path('/mnt/lightsem-data/justdance')/item['file']
        inventory['catalog_recordings']+=1
        inventory['pose_present']+=pose.exists();inventory['wav_present']+=wav.exists()
        inventory['video_present']+=source.exists()
        if key in protected or group in protected_groups:
            rejected.append({'key':key,'reason':'protected development group; pose payload unopened'});continue
        if known.get(key,{}).get('excluded'):
            rejected.append({'key':key,'reason':'existing exclusion retained'});continue
        title=known.get(key,{}).get('title_norm',item['file']).lower()
        if 'cure for me' in title or 'shpongle' in title:
            rejected.append({'key':key,'reason':'showcase quarantine'});continue
        if not pose.exists() or not (wav.exists() or source.exists()):
            rejected.append({'key':key,'reason':'missing paired source'});continue
        duration=sf.info(wav).duration if wav.exists() else item['duration_s']
        if not cfg.frames/cfg.fps<=duration<=512:
            rejected.append({'key':key,'reason':'outside 16s minimum/512s planner capacity'});continue
        with np.load(pose,allow_pickle=False) as z:coaches=z['K'].shape[1]
        for coach in range(coaches):
            try:
                arrays,quality=pose_observations(pose,duration,cfg,coach_index=coach)
                if abs(quality['native_duration_s']-duration)>cfg.maximum_av_duration_difference_s:
                    raise ValueError('Pose/audio duration binding mismatch')
                if not quality['eligible_windows']:
                    raise ValueError('No original 80%-coverage 16s window')
                row={'key':key,'coach_index':coach,'known_group':group,'duration_s':duration,
                     'pose_path':str(pose),'source_path':str(source),'wav_path':str(wav),
                     'wav_present':wav.exists(),'frames':len(arrays['time']),
                     'quality':{k:v for k,v in quality.items() if k!='pose_meta'},
                     'strict_original_record_QA':cfg.minimum_duration_s<=duration<=cfg.maximum_duration_s and quality['coverage']>=cfg.minimum_coverage,
                     'group_identity_status':'known_legacy_group' if group else 'requires_duplicate_resolution',
                     'title':title}
                rows.append(row)
            except (ValueError,KeyError) as error:
                rejected.append({'key':key,'coach_index':coach,'reason':str(error)})
        if inventory['catalog_recordings']%50==0:
            print(json.dumps({'catalog_scanned':inventory['catalog_recordings'],'candidate_tracks':len(rows)}),flush=True)
    fd=read(FD)
    counts={p:{'records':len(r:=[r for r in fd['entries'] if r['partition']==p]),
               'groups':len({x['canonical_id'] for x in r})} for p in ('train','val')}
    report={'schema_version':1,'status':'admission_audit_not_training_manifest','inventory':dict(inventory),
            'inputs':{str(p):sha(p) for p in (PARENT/'data/MANIFEST.json',CACHE/'justdance_songs.json',CACHE/'justdance_manifest.json',FD)},
            'FineDance':counts,'candidates':rows,'rejected':rejected,
            'candidate_recordings':len({r['key'] for r in rows}),
            'candidate_tracks':len(rows),'strict_original_record_QA_tracks':sum(r['strict_original_record_QA'] for r in rows),
            'known_candidate_groups':len({r['known_group'] for r in rows if r['known_group']}),
            'missing_wav_candidate_recordings':len({r['key'] for r in rows if not r['wav_present']}),
            'reasons':dict(Counter(r['reason'] for r in rejected)),
            'protected_development_keys':sorted(protected),'protected_development_groups':sorted(protected_groups),
            'scope':'TRAIN candidates only. Unknown duplicate identities are not admitted. Window coverage gate remains80%; global-record coverage is descriptive for expanded masked-window admission. Coach hours are not independent song hours.',
            'storage_free_GiB':{str(p):shutil.disk_usage(p).free/2**30 for p in (Path('/root/lightsem'),CACHE)},
            'source_sha256':sha(__file__)}
    write(output,report)
    print(json.dumps({k:report[k] for k in ('candidate_recordings','candidate_tracks','strict_original_record_QA_tracks','known_candidate_groups','missing_wav_candidate_recordings','reasons')},indent=2))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    audit(p.parse_args().output)
