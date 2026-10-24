from __future__ import annotations
import json, hashlib, os, glob
from pathlib import Path
import numpy as np
contract_path=Path('/mnt/lightsem-data/experiments/lightsem/audio_light_foundation_v2_contract_20261016_v1/CONTRACT.json')
contract=json.loads(contract_path.read_text())
root=Path('/mnt/lightsem-data/experiments/lightsem/audio_light_foundation_v2_muq_20261016_v3')
receipt=json.loads((root/'COMPLETE_V2.json').read_text())
rows={x['id']:x for x in contract['rows']}
ids=set(rows); rid={x['id'] for x in receipt['records']}; wav_ids={Path(x).stem for x in glob.glob('/mnt/lightsem-data/beatsaver/wav/*.wav')}
shape_bad=[]; nonfinite=[]; time_bad=[]
for rec in receipt['records']:
    with np.load(rec['path'], allow_pickle=False) as z:
        shape=tuple(z['muq'].shape); tm=np.asarray(z['muq_time']);
        if shape[1:]!=(1024,): shape_bad.append({'id':rec['id'],'shape':shape})
        if not np.isfinite(z['muq']).all(): nonfinite.append(rec['id'])
        if len(tm)>1 and not np.all(np.diff(tm)>0): time_bad.append(rec['id'])
old=[]
for id_,row in rows.items():
    if row.get('muq_existing') is None: continue
    old_path=row['muq_existing']['path']; new_path=str(root/f'{id_}.npz')
    with np.load(old_path,allow_pickle=False) as a, np.load(new_path,allow_pickle=False) as b:
        x=np.asarray(a['muq'],np.float32); y=np.asarray(b['muq'],np.float32)
        old.append({'id':id_,'old_shape':list(x.shape),'new_shape':list(y.shape),
                    'mean_abs_error':float(np.abs(x-y).mean()),
                    'max_abs_error':float(np.abs(x-y).max()),
                    'flattened_corr':float(np.corrcoef(x.reshape(-1),y.reshape(-1))[0,1]),
                    'old_protocol':{'path':old_path,'chunk_seconds':8,'overlap_stitch':'timestamp-aligned average'},
                    'new_protocol':{'path':new_path,'chunk_seconds':30,'core_seconds':20,'context_seconds':5,'overlap_stitch':'5s context discard'}})
out={'schema_version':1,'protocol':'audio-light-foundation-v2-muq-channel-audit-20261016-v2',
     'canonical_extraction':{'receipt':str(root/'COMPLETE_V2.json'),'receipt_sha256':hashlib.sha256((root/'COMPLETE_V2.json').read_bytes()).hexdigest(),'records':len(receipt['records']),'wav_records':len(wav_ids),'missing_wav_ids':sorted(wav_ids-rid),'extra_ids':sorted(rid-wav_ids),'unique_wav_sha256':len({r['wav_sha256'] for r in receipt['records']}),'model_id':receipt['model_id'],'revision':receipt['revision'],'layer':receipt['layer'],'sample_rate_hz':receipt['sample_rate'],'native_hz':receipt['native_hz'],'saved_dtype':receipt['saved_dtype'],'all_muqs_1024d':not shape_bad,'all_finite':not nonfinite,'all_timestamps_strict':not time_bad,'shape_bad':shape_bad,'nonfinite':nonfinite,'timestamp_bad':time_bad},
     'channel_contract':{'base92_dim':92,'muq_dim':1024,'total_dim':1116,'base92_layout':'mel64+scalar28 at 40Hz, timestamp-aligned to 10Hz','muq_layout':'layer9 at 25Hz, timestamp-aligned to 10Hz','zero_padding_forbidden':True,'legacy_visual_52d_forbidden':True},
     'old_experiment_reconciliation':{'old_native_cache_rows':len(old),'old_cache_is_not_reused':True,'reason':'old native cache used 8s overlap-aware crops; canonical v2 uses 30s/20s/5s windows. Same model/revision and shape do not imply byte-equivalent context-boundary values; all 300 rows were re-extracted under one canonical protocol.','rows':old}}
for p in [Path('/root/lightsem/docs/research/2026-10-16-audio-light-foundation-v2/MUQ_EXTRACTION_CHANNEL_AUDIT.json'),Path('/mnt/lightsem-data/experiments/lightsem/audio_light_foundation_v2_muq_20261016_v3/MUQ_EXTRACTION_CHANNEL_AUDIT.json')]:
 p.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
 print(p)
