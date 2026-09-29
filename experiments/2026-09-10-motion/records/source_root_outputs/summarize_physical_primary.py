import importlib.util,sys,json
from pathlib import Path
OUT=Path('/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/music_physical_revision_01/evaluation')
s=importlib.util.spec_from_file_location('frozen_evaluator',OUT/'evaluator.py');m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m)
p,q=m.context(OUT);rows={}
for arm in m.ARMS:
 r=m.load_json(OUT/arm/'final/primary/report.json');assert r['complete'] and r['expected_items']==119
 rows[arm]={'quality':m.quality(r['summary']),'conditioning':{c:m.comparison(r['summary']['per_song'],c) for c in ('donor','shift')}}
a,b=rows['physical']['quality'],rows['masked_native']['quality']
report={'scope':'Complete119window final-checkpoint primary cohorts only; supplementalshort/fullcontinuity/visual still pending; not final pipeline report.','rows':rows,'sliding_excess_reduction':1-a['sliding_excess']/b['sliding_excess'],'primary_sliding_absolute_pass':a['sliding_excess']<=.2,'primary_conditioning_pass':all(c['relative_improvement']>=.05 and c['relative_ci95'][0]>0 for c in rows['physical']['conditioning'].values()),'pipeline_qualified':False}
m.save_json(OUT/'primary_analysis.json',report);print(json.dumps(report,indent=2))
