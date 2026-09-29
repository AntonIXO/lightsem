import importlib.util,sys,json
from pathlib import Path
OUT=Path('/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/music_physical_revision_01/visual_review_support_revision_01')
AUDIT=OUT.parent.parent/'physical_visual_audits_01'
s=importlib.util.spec_from_file_location('frozen_review',OUT/'reviewer.py');m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m)
observations={}
for path in AUDIT.rglob('*observation*.json'):
 o=json.loads(path.read_text())
 if isinstance(o,dict) and 'item_id' in o and 'viewed_sample_frames' in o and 'comparisons' in o:
  if o['item_id'] in observations:raise ValueError('Ambiguous observer file '+o['item_id'])
  observations[o['item_id']]=path
with m.lock(OUT):
 p=m.context(OUT)
 for item in p['items']:
  if (OUT/'reviews'/(item['id']+'.json')).exists():continue
  if item['id'] not in observations:print('WAITING observer',item['id']);break
  m.review(OUT,p,observations[item['id']])
