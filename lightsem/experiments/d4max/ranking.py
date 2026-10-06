"""Observed saved-output ranking: common window, fixed proxies and physical gates."""
from pathlib import Path
import json,csv
import numpy as np
from scipy.stats import rankdata
from ..dynamic_dance.contracts import DanceResult,MusicTimeline
from ..dynamic_dance import diagnostics as d,transfer_audit as ta
from ..dynamic_dance_combo import diagnostics as cd
from ..dynamic_dance_combo.campaign import read,save,sha
from lightsem.conductor.body_transfer import decode


def load_pair(root,method,song,start=60.,end=180.):
 root=Path(root);bp=root/'bodies'/method/song/'body.npz';folder=root/'timelines'/method/song
 body=DanceResult.load(bp);bm=(body.time>=start-1e-8)&(body.time<end-1e-8)
 if bm.sum()!=round((end-start)*30):raise ValueError('Common native interval missing')
 pins={str(p):sha(p) for p in [bp,bp.with_suffix('.json'),folder/'timeline.npz',folder/'patch.json',folder/'report.json']}
 report=read(folder/'report.json')
 for key,path in [('body_sha256',bp),('timeline_sha256',folder/'timeline.npz'),('patch_sha256',folder/'patch.json')]:
  if report.get(key)!=sha(path):raise ValueError('Saved mapping provenance mismatch: '+str(path))
 with np.load(folder/'timeline.npz') as z:
  mask=(z['time']>=start-1e-8)&(z['time']<end-1e-8);data={k:z[k][mask] for k in ['time','body','targets','universes','dimmer']}
 if mask.sum()!=round((end-start)*40):raise ValueError('Common light interval missing')
 patch=read(folder/'patch.json');moving=np.array([p['kind']!='bulb' for p in patch]);controls=decode(data['universes'],patch)
 vectors=data['targets'][:,moving]-np.array([p['position'] for p in patch])[moving];length=np.linalg.norm(vectors,axis=-1)
 requested=np.stack([np.rad2deg(np.arctan2(vectors[...,0],-vectors[...,1])),np.rad2deg(np.arcsin(np.clip(vectors[...,2]/length,-1,1)))],-1)
 ranges=np.array([[p['pan_range_deg'],p['tilt_range_deg']] for p in patch if p['kind']!='bulb'])
 return {'world':body.world[bm],'time':body.time[bm],'contacts':body.contacts[bm] if body.contacts is not None else None,'ltime':data['time'],'requested':requested,'emitted':controls['angles_deg'][:,moving],'controls':controls,'ranges':ranges,'patch':patch,'pins':pins,'metadata':body.metadata}


def score(pair,reference,music,release,plan):
 w,t=pair['world'],pair['time'];time=pair['ltime'];req,em=pair['requested'],pair['emitted'];start,end=plan['ranking']['window_s']
 floor=float(np.quantile(reference['world'][:,d.FEET,1],.02))
 physical=d.body_dynamics(w,t,contacts=pair['contacts'],reference_body=reference['world'],floor_y=floor)
 emission=d.emission_diagnostics(req,em,time,tick_deg=float(pair['ranges'].max()/65535))
 def clipped(xs):return [[max(start,a),min(end,b)] for a,b in xs if min(end,b)>max(start,a)]
 attacks=[v for v in release['attack_times'] if start<=v<end]
 series=d.dynamics_series(w,t)
 musical={}
 for stage,values,clock in [('requested',req,time),('emitted',em,time)]:
  event=ta.motor_event_alignment(values,clock,attacks,start_s=start,end_s=end)
  recurrence=ta.feature_recurrence(values/pair['ranges'],clock,music.muq,music.muq_time)
  rel=d.release_contrast(ta.motor_speed(values,clock).mean(1),clock,clipped(release['release_intervals']),clipped(release['reference_intervals']),denominator_floor=1.)
  musical[stage]={'event':event,'recurrence':recurrence,'release':rel}
 musical['body']={'recurrence':d.recurrence_alignment(w,t,music.muq,music.muq_time),'event':d.event_alignment(d.gesture_events(series['relative_speed_m_s'],t),attacks,start_s=start,end_s=end),'release':d.release_contrast(series['relative_speed_m_s'],t,clipped(release['release_intervals']),clipped(release['reference_intervals']))}
 m=musical['emitted'];values=[m['event']['pooled']['f1_minus_null_p95'],m['recurrence']['lift_over_null_p95'],m['release']['lift_over_null_p95']]
 return {'body':physical,'emission':emission,'music':musical,'metrics':values,'brightness':{'mean_emitted_dimmer':float(pair['controls']['dimmer'].mean()),'scope':'Excluded from motion ranking'},'inputs':pair['pins']}


def run(root):
 root=Path(root);plan=read(root/'PLAN.json');methods=plan['ranking']['methods'];scores={};details={};patch_reference=None
 for track in plan['tracks']:
  song=track['id'];music=MusicTimeline.load(track['music_timeline']);release=read(root/'records/audio_intervals'/f'{song}.release.json')
  refrow=next(x for x in methods if x['id']=='D7_full');reference=load_pair(refrow['root'],refrow['artifact'],song)
  refscore=score(reference,reference,music,release,plan);details[song]={}
  for row in methods:
   pair=reference if row['id']=='D7_full' else load_pair(row['root'],row['artifact'],song)
   if pair['patch']!=reference['patch']:raise ValueError('Common saved-motion patch mismatch')
   value=refscore if row['id']=='D7_full' else score(pair,reference,music,release,plan)
   gates=d.comparison_gates(value['body'],refscore['body'],candidate_emission=value['emission'],baseline_emission=refscore['emission'])
   failed=[key for key in cd.PHYSICAL_GATES if gates['gates'][key]['status']!='pass']
   for key,maximum in [('emitted_speed_deg_s',90.),('emitted_acceleration_deg_s2',360.)]:
    if value['emission'][key]['max']>maximum+1e-6:failed.append(key+'_hard_limit')
   value.update(physical_eligible=not failed,physical_failures=failed,engineering_gates=gates)
   details[song][row['id']]=value
   print('rank measured',song,row['id'],flush=True)
  matrix=np.array([[np.nan if v is None else v for v in details[song][row['id']]['metrics']] for row in methods])
  ranked=np.full(matrix.shape,np.nan)
  for column in range(3):
   good=np.isfinite(matrix[:,column]);ranked[good,column]=rankdata(-matrix[good,column],method='average')
  for i,row in enumerate(methods):
   value=details[song][row['id']];value['metric_ranks']=[float(x) if np.isfinite(x) else None for x in ranked[i]];value['mean_rank']=float(ranked[i].mean()) if np.isfinite(ranked[i]).all() else None
 records=[]
 for row in methods:
  per={song:details[song][row['id']] for song in details};good=all(v['physical_eligible'] and v['mean_rank'] is not None for v in per.values());ranks=[v['mean_rank'] for v in per.values()]
  records.append({**row,'eligible':good,'worst_song_mean_rank':max(ranks) if all(v is not None for v in ranks) else None,'overall_mean_rank':float(np.mean(ranks)) if all(v is not None for v in ranks) else None,'songs':{s:{k:v[k] for k in ['metrics','metric_ranks','mean_rank','physical_eligible','physical_failures']} for s,v in per.items()}})
 eligible=sorted([v for v in records if v['eligible']],key=lambda x:(x['worst_song_mean_rank'],x['overall_mean_rank']))
 raw=sorted([v for v in records if v['overall_mean_rank'] is not None],key=lambda x:(x['worst_song_mean_rank'],x['overall_mean_rank']))
 result={'protocol':plan['ranking'],'plan_sha256':sha(root/'PLAN.json'),'winner':eligible[0]['id'] if eligible else None,'raw_metric_leader':raw[0]['id'] if raw else None,'rows':records,'engineering_order':[v['id'] for v in eligible],'all_metric_order':[v['id'] for v in raw],'details':details,'baseline_promoted':False}
 save(root/'reports/RANKING.json',result)
 with (root/'reports/RANKING.csv').open('x',newline='') as f:
  writer=csv.writer(f);writer.writerow(['method','song','event_lift','recurrence_lift','release_lift','event_rank','recurrence_rank','release_rank','mean_rank','physical_eligible','physical_failures'])
  for row in records:
   for s,v in row['songs'].items():writer.writerow([row['id'],s,*v['metrics'],*v['metric_ranks'],v['mean_rank'],v['physical_eligible'],';'.join(v['physical_failures'])])
 return result
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();run(a.root)
