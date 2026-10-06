"""Commands for the frozen experimental full-song combination campaign."""
import argparse,json
from pathlib import Path
from lightsem.experiments.dynamic_dance_combo import campaign as c

def main():
 p=argparse.ArgumentParser();p.add_argument('stage',choices=('colors','generate','postprocess','map','audit','freeze-audio','source-checks','paired-checks'))
 p.add_argument('--root',type=Path,required=True);p.add_argument('--track',required=True);p.add_argument('--condition',default='correct');p.add_argument('--method');p.add_argument('--guarded',action='store_true');a=p.parse_args()
 if a.stage in ('colors','generate'):
  if not a.guarded:raise ValueError('GPU-capable commands require frozen sequential guard')
  from scripts.motion.dynamic_combo_guard import authenticate_guarded_job
  authenticate_guarded_job(__file__)
 if a.stage=='colors':
  from lightsem.experiments.dynamic_dance_combo.colors import generate
  _,_,r=generate(a.root,c.track(a.root,a.track),acquire_gpu_lock=False);value={'path':str(a.root/'colors'/a.track/'REPORT.json')}
 elif a.stage=='generate':value={'method':c.generate(a.root,a.track,a.condition)}
 elif a.stage=='postprocess':value={'methods':c.postprocess(a.root,a.track,a.condition)}
 elif a.stage=='map':r=c.map_one(a.root,a.track,a.method);value={'method':a.method,'frames':r['frames'],'status':r['status']}
 elif a.stage=='source-checks':
  from lightsem.experiments.dynamic_dance_combo.preflight import source_checks
  value=source_checks(a.root,a.track)
 elif a.stage=='paired-checks':
  from lightsem.experiments.dynamic_dance_combo.preflight import paired_checks
  value=paired_checks(a.root)
 elif a.stage=='freeze-audio':
  from lightsem.experiments.dynamic_dance_combo.diagnostics import freeze_release_intervals
  value=freeze_release_intervals(a.root,c.track(a.root,a.track))
 else:
  from lightsem.experiments.dynamic_dance_combo.diagnostics import compare
  r=compare(a.root,a.method,c.track(a.root,a.track));value={'method':a.method,'primary':r.get('primary'),'physical':r.get('physical_engineering_status')}
 print(json.dumps(value,default=str,allow_nan=False))
if __name__=='__main__':main()
