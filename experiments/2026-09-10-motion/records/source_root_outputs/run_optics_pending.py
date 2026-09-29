"""Schedule fixed original arms in order, using the immutable native runner."""
import argparse,importlib.util,sys
from pathlib import Path
OUT=Path('/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/optics_full_controls_01')
p=argparse.ArgumentParser();p.add_argument('--budget',type=int,default=1);a=p.parse_args();assert a.budget>0
path=OUT/'frozen/run_dance_optics_full.py';s=importlib.util.spec_from_file_location('frozen_optics_runner',path);m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m)
loaded=m.load_frozen(OUT);done=0
for name in loaded[1]['arms']:
 target=OUT/'renders'/name
 if (target/'render.json').exists():
  r=m.authenticated(target/'render.json')
  for f,sha in r['artifacts'].items():
   if m.digest(target/f)!=sha:raise RuntimeError('Completed render artifact changed: '+str(target/f))
  continue
 if target.exists():raise RuntimeError('Retained incomplete arm; inspect, do not overwrite: '+name)
 if done>=a.budget:break
 m.render_one(OUT,loaded,name);done+=1
print('BATCH COMPLETE',done,flush=True)
