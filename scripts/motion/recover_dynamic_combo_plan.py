"""Complete an interrupted plan serialization from exact preserved body arrays."""
from pathlib import Path
import argparse,json,shutil
import numpy as np
from lightsem.experiments.dynamic_dance_combo import campaign as c
from lightsem.experiments.dynamic_dance_combo.clock_metadata import nonnegative_effort
from lightsem.experiments.dynamic_dance.contracts import DanceResult,DancePlan

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root
 auth=c.read(r/'records/D6_TELEMETRY_RECOVERY_AUTHORIZATION_v1.json')
 for path,digest in auth['archived_files'].items():
  if c.sha(path)!=digest:raise ValueError('Preserved partial artifact changed')
 for method in ['C3__mute_drums','C4__mute_drums']:
  source=r/'failed_runs/d6_plan_telemetry/partial_bodies'/method/'shpongle';target=r/'bodies'/method/'shpongle'
  target.parent.mkdir(parents=True,exist_ok=True);shutil.copytree(source,target)
  for f in source.iterdir():
   if c.sha(f)!=c.sha(target/f.name):raise ValueError('Recovered original file differs')
 c3=c.body_path(r,'C3__mute_drums','shpongle');c4=c.body_path(r,'C4__mute_drums','shpongle');body=DanceResult.load(c4);source=DanceResult.load(c3)
 effort,audit=nonnegative_effort(body.source_time,body.time,fps=body.metadata['fps'])
 plan=DancePlan(time=body.time,source_time=body.source_time,effort=effort.astype(np.float32),metadata={**body.metadata,'derived_effort_roundoff':audit,'serialization_recovery':str(r/'records/D6_TELEMETRY_RECOVERY_AUTHORIZATION_v1.json')})
 plan.save(c4.parent/'plan.npz')
 with np.load(r/'generation/shpongle/mute_drums/telemetry.npz') as z:telemetry={k:z[k] for k in z.files}
 rates=c.learned_proposals(source.world,source.time,telemetry)
 c.arrays(c4.parent/'rate_proposals.npz',time=source.time,desired_rates=rates,**{k:telemetry[k] for k in ['coarse_time','coarse_local_plan']})
 gp=c.body_path(r,'G78__mute_drums','shpongle')
 c.save(r/'generation/shpongle/mute_drums/POSTPROCESS.json',{'source':str(gp),'source_sha256':c.sha(gp),'C3':str(c3),'C3_sha256':c.sha(c3),'C4':str(c4),'C4_sha256':c.sha(c4),'condition':'mute_drums','recovery':'Plan-only local-clock roundoff; every previously written body/source-clock/contact byte unchanged.'})
 c.save(r/'records/D6_TELEMETRY_RECOVERY_COMPLETE.json',{'body_sha256':c.sha(c4),'expected_body_sha256':auth['body_sha256'],'body_bytes_unchanged':c.sha(c4)==auth['body_sha256'],'audit':audit})
 print(json.dumps({'recovered':str(c4),'body_bytes_unchanged':c.sha(c4)==auth['body_sha256'],'negative_intervals':audit['negative_interval_count']}))
if __name__=='__main__':main()
