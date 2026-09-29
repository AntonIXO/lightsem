"""Unpreregistered analysis of saved VAL summaries only; no new inference."""
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
ROOT=Path('/mnt/lightsem-data/experiments/lightsem/motion_native_audio_20260910')
OUT=Path('/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911')
names=json.loads((ROOT/'data/motion_normalizers.json').read_text())['summary']['names']
def es(draws,target):
 first=np.sqrt(np.mean((draws-target)**2,axis=1)).mean(0)
 if len(draws)==1:return first
 pair=np.sqrt(np.mean((draws[:,None]-draws[None,:])**2,axis=2))
 return first-.5*pair.sum((0,1))/(len(draws)*(len(draws)-1))
result={}
for arm in ('regression','flow'):
 report=json.loads((ROOT/f'music/evaluation/{arm}/seed_0/final/report.json').read_text())
 records=defaultdict(list)
 for row in report['rows']:
  with np.load(row['summary_path']) as z:
   ref=z['reference']; pred=z['correct']; scores={c:es(z[c],ref) for c in ('correct','donor','shift')}
   # Actual matched summary controls. Time reversal preserves each generated
   # draw's marginal summary values but destroys its original ordering.
   scores['reverse']=es(pred[:,::-1],ref)
   scores['constant_draw_mean']=es(np.broadcast_to(pred.mean(1,keepdims=True),pred.shape),ref)
   scores['correct_temporal_rms']=np.std(pred,axis=1).mean(0)
   scores['reference_temporal_rms']=np.std(ref,axis=0)
   records[(row['canonical_id'],row['key'])].append(scores)
 groups=defaultdict(list)
 for (group,key),rows in records.items():groups[group].append({k:np.mean([r[k] for r in rows],0) for k in rows[0]})
 per={g:{k:np.mean([r[k] for r in rows],0) for k in rows[0]} for g,rows in groups.items()}
 means={k:np.mean([r[k] for r in per.values()],0) for k in next(iter(per.values()))}
 result[arm]={'songs':len(per),'channels':{name:{k:float(v[i]) for k,v in means.items()} for i,name in enumerate(names)}}
 for name,values in result[arm]['channels'].items():
  c=values['correct']; print(arm,name,'correct %.3f donor %.3f shift %.3f reverse %.3f constant %.3f pred/ref temporal rms %.2f'%(c,values['donor'],values['shift'],values['reverse'],values['constant_draw_mean'],values['correct_temporal_rms']/max(values['reference_temporal_rms'],1e-12)),flush=True)
result['scope']='Post-hoc univariate energy score by channel; equal recording then song means. Not the frozen multivariate primary, not causal music fit or a new experiment. Reversal and constant controls reuse generated draw values; no TEST access.'
(OUT/'summary_channel_diagnosis.json').write_text(json.dumps(result,indent=2))
