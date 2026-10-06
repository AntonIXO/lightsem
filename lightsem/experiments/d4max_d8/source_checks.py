"""Numerical source, count and paired-output admission; no motion generation."""
from pathlib import Path
import numpy as np
from lightsem.experiments.dynamic_dance_combo.campaign import read,sha,save,track,body_path,compile_world
from lightsem.experiments.dynamic_dance_combo.brightness import load_verified_loudness
from lightsem.experiments.dynamic_dance.contracts import DanceResult
from lightsem.conductor.body_transfer import decode
from lightsem.conductor.body_transfer_v2 import make_outputs

def source_checks(root,song):
 root=Path(root);plan=read(root/'PLAN.json');t=track(root,song)
 body=DanceResult.load(body_path(root,'D7_reference',song));pose=body.world[min(18*30,len(body.world)-1)].astype(float)
 n=240;times=np.arange(n)/40;frozen=np.broadcast_to(pose,(n,22,3)).copy()
 with np.load(root/'colors'/song/'P2.npz') as z:left=z['left'][:n];right=z['right'][:n]
 loud,_=load_verified_loudness(t['loudness_receipt'],t['audio'])
 def compile(w,counts):
  outputs=make_outputs(**counts);u,p,intent,trace,hard=compile_world(w,times,outputs,left,right,loud,plan['transfer'])
  return {'universes':u,'patch':p,'decoded':decode(u,p),'hard':hard}
 base=compile(frozen,plan['counts']);angles=base['decoded']['angles_deg'];moving=np.array([x['kind']!='bulb' for x in base['patch']])
 stable=bool(np.array_equal(angles[80:,moving],np.broadcast_to(angles[80:81,moving],angles[80:,moving].shape)))
 if not stable:raise ValueError('Frozen body emitted autonomous angular movement')
 rows=[]
 for name,joint in [('left_wrist',20),('right_wrist',21),('left_ankle',7),('right_ankle',8),('left_elbow',18),('right_elbow',19)]:
  if not any(f.get('role_index')==joint for f in base['patch']):raise ValueError('Invalid probe: requested joint has no assigned output')
  w=frozen.copy();w[80:160,joint,0]+=.12
  case=compile(w,plan['counts']);delta=np.max(np.abs(case['decoded']['angles_deg']-angles),axis=(0,2));assigned=[i for i,x in enumerate(case['patch']) if x.get('role_index')==joint]
  changes=[case['patch'][i]['name'] for i in np.flatnonzero(delta>1e-8)]
  passed=bool(assigned and any(delta[i]>.0083 for i in assigned))
  if not passed:raise ValueError('Assigned limb did not affect emitted output: '+name)
  rows.append({'limb':name,'assigned':[case['patch'][i]['name'] for i in assigned],'changed_outputs':changes,'passed':passed,'max_angle_change_deg':float(delta.max())})
 # Deliberately imposed pose yaw is a source intervention, never a demo driver.
 w=frozen.copy();q=w[80:160]-w[80:160,:1];a=.25;rotation=np.array([[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]])
 w[80:160]=q@rotation.T+w[80:160,:1]
 common=compile(w,plan['counts']);expanded=compile(w,{'beams':20,'lasers':17,'bulbs':80});single=compile(w,{'beams':1,'lasers':0,'bulbs':0})
 references={x['name']:i for i,x in enumerate(common['patch'])}
 comparisons=[]
 for label,result in [('single',single),('expanded',expanded)]:
  for i,f in enumerate(result['patch']):
   if f['name'] in references:
    j=references[f['name']]
    for k in ['angles_deg','rgb','dimmer']:
     if not np.array_equal(result['decoded'][k][:,i],common['decoded'][k][:,j]):raise ValueError('Arbitrary-count common emitted channel changed')
  comparisons.append({'case':label,'outputs':len(result['patch']),'common_streams_exact':True})
 expanded_frozen=compile(frozen,{'beams':20,'lasers':17,'bulbs':80})
 extra=[]
 for i,f in enumerate(expanded['patch']):
  if f['name'] in references:continue
  key='dimmer' if f['kind']=='bulb' else 'angles_deg'
  delta=np.abs(expanded['decoded'][key][:,i]-expanded_frozen['decoded'][key][:,i])
  changed=delta>1e-8 if delta.ndim==1 else np.any(delta>1e-8,axis=-1)
  indices=np.flatnonzero(changed)
  extra.append({'name':f['name'],'role':f.get('body_region'),'capability':f['kind'],
    'source_step_frame':80,'first_response_frame':int(indices[0]) if len(indices) else None,
    'last_response_frame':int(indices[-1]) if len(indices) else None,'response_observed':bool(len(indices))})
 p=root/'preflight'/f'{song}_source_v2.json';value={'status':'passed','source':str(body_path(root,'D7_reference',song)),'source_sha256':sha(body_path(root,'D7_reference',song)),
 'frozen_body_angular_stability':stable,'frozen_body_brightness_scope':'Brightness is deliberately allowed to follow audio even when body frozen; motion remains stationary.',
 'limb_controls':rows,'count_controls':comparisons,'extra_fixture_yaw_response':extra,'fixture_counts':[1,63,117],'time_dependence_scope':'Only explicit diagnostic body perturbations; never rendered asgenerateddance. Extra streams share source information, not independent gestures.'}
 save(p,value);return value

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--track',required=True);a=p.parse_args();source_checks(a.root,a.track)
