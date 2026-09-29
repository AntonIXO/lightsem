import json,numpy as np
from pathlib import Path
OUT=Path('/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/music_physical_revision_01/visual_review')
p=json.loads((OUT/'protocol.json').read_text())['payload'];refs={r['key']:r for r in p['evaluation']['matched_review_assets']['incumbent']['arrays']};rows=[]
for item in p['items']:
 if item['key']=='shpongle':continue
 with np.load(refs[item['key']]['path']) as z:
  t=z['times'];oldn=len(t);lo=float(t[0]);hi=float(t[-1]);dt=float(np.median(np.diff(t)))
 row={'item':item,'old_frames':oldn,'old_start':lo,'old_last_time':hi,'median_dt':dt,'native_last_time':(item['full_frames']-1)/30}
 rows.append(row)
path=OUT.parent/'legacy_support_audit.json';assert not path.exists();path.write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
