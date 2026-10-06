"""D4-max experiment: saved own-song D7 library, immutable controls, no GPU generation."""
from pathlib import Path
from dataclasses import fields
import shutil
import numpy as np
from ..dynamic_dance_combo.campaign import read,sha,save,arrays,track,body_path,_save_composed,map_one,learned_proposals
from ..dynamic_dance_combo.diagnostics import freeze_release_intervals
from ..dynamic_dance.contracts import DanceResult,DancePlan,MusicTimeline
from ..dynamic_dance.retiming import d4_motif_recomposition,retime_with_effort
from .motifs import Config,build_bank,compose
from ..d4max.pacing import apply_d6
ROOT=Path('/mnt/lightsem-data/experiments/lightsem/d4max_d8_20261005_v2')


def initialize(root):
 root=Path(root);plan=read(root/'PLAN.json')
 for t in plan['tracks']:freeze_release_intervals(root,t)
 for t in plan['tracks']:
  source=Path(t['source_body']);dest=body_path(root,'D7_reference',t['id'])
  for src in source.parent.glob('*'):
   if src.is_file():
    dst=dest.parent/src.name;dst.parent.mkdir(parents=True,exist_ok=True)
    with dst.open('xb') as f:f.write(src.read_bytes())
  folder=root/'colors'/t['id'];folder.mkdir(parents=True,exist_ok=False)
  for src,name in [(Path(t['colors']),'P2.npz'),(Path(t['color_report']),'REPORT.json')]:
   with (folder/name).open('xb') as f:f.write(src.read_bytes())
 save(root/'preflight/INITIALIZED.json',{'reference':'Existing complete own-song G78 bodies copied byte-for-byte; no generation/retraining.','plan_sha256':sha(root/'PLAN.json')})


def generate(root,song):
 root=Path(root);t=track(root,song);plan=read(root/'PLAN.json')
 cfg=Config(**{k:v for k,v in plan['d4max'].items() if k in {f.name for f in fields(Config)}})
 source_path=body_path(root,'D7_reference',song);source=DanceResult.load(source_path);music=MusicTimeline.load(t['music_timeline'])
 bank=build_bank(source.world,music,cfg)
 folder=root/'banks'/song;folder.mkdir(parents=True,exist_ok=False)
 arrays(folder/'features.npz',features=bank['features'],labels=bank['labels'],medoids=bank['medoids'],**bank['feature_reference'])
 save(folder/'BANK.json',{k:v for k,v in bank.items() if k not in ('features','labels','medoids','feature_reference')})
 meta={'components':['G78_library','D4_max'],'source_role':'Saved D7+D8 source; D8 active in G78 generation.','d6_enabled':False,'generation_provenance':source.metadata,'generated_source':str(source_path),'generated_source_sha256':sha(source_path),'motion_origin':'Recomposed recurring motifs from this song own saved G78 body','newly_generated_motion':False,'source_library_label':'G78 (D7+D8)','bank_sha256':sha(folder/'features.npz')}
 rr,pp=d4_motif_recomposition(source.world,music,30.)
 _save_composed(root,'S4',song,rr,pp,source_path,{**meta,'components':['G78_library','standard_D4'],'motion_origin':'Standard D4 control on identical full G78 source'})
 all_methods=['D7_reference','S4']
 for condition in ['correct',*t['waveform_controls']]:
  binding=t if condition=='correct' else t['waveform_controls'][condition]
  target=MusicTimeline.load(binding['music_timeline'])
  with np.load(binding['local_plan_telemetry']) as z:telemetry={k:z[k].copy() for k in z.files}
  suffix='' if condition=='correct' else '__'+condition
  result,planned,_=compose(source.world,music,cfg,bank=bank,target_music=target)
  extra={**meta,'condition':condition,'music_timeline':binding['music_timeline'],'music_timeline_sha256':sha(binding['music_timeline']),'library_audio':t['music_timeline'],'library_fixed':True,'sensitivity_scope':'Composition/pacing query path only; G78 library and its original audio tags fixed. Both songs have separate own-song libraries.'}
  p=_save_composed(root,'C3'+suffix,song,result,planned,source_path,extra)
  body=DanceResult.load(p);dp=DancePlan.load(p.parent/'plan.npz')
  r6,p6,proposals=apply_d6(body,dp,telemetry,target,minimum_rate=.25)
  extra6={**extra,'components':['G78_library','D4_max','D6_anchored'],'d6_enabled':True,'motion_origin':'D4-max motif composition with learned within-motif pacing; exact musical motif anchors','planner_telemetry':binding['local_plan_telemetry'],'planner_telemetry_sha256':sha(binding['local_plan_telemetry'])}
  p6path=_save_composed(root,'C4'+suffix,song,r6,p6,p,extra6)
  arrays(p6path.parent/'rate_proposals.npz',time=body.time,desired_rates=proposals)
  all_methods+=['C3'+suffix,'C4'+suffix]
  if condition=='correct':
   again,againp,_=compose(source.world,music,cfg,bank=bank,target_music=target)
   assert np.array_equal(again.world,result.world) and np.array_equal(again.source_time,result.source_time)
   repeat,_,_=apply_d6(body,dp,telemetry,target,minimum_rate=.25)
   assert np.array_equal(repeat.world,r6.world) and np.array_equal(repeat.source_time,r6.source_time)
   save(root/'preflight'/f'{song}_repeat.json',{'C3_body_and_clock_exact':True,'C4_body_and_clock_exact':True,'source_sha256':sha(source_path),'C4_anchor_max_error_s':p6.metadata['pacing']['anchor_max_error_s']})
   rz,pz,_=apply_d6(body,dp,telemetry,target,minimum_rate=0.)
   _save_composed(root,'C4_zero_allowed',song,rz,pz,p,{**extra6,'control':'Same anchored D6 allowing rate0; isolates positive-rate safeguard'})
   all_methods.append('C4_zero_allowed')
   rs,ps,_=compose(source.world,music,cfg,bank=bank,target_music=target,control='shuffle')
   _save_composed(root,'C3_shuffled',song,rs,ps,source_path,{**extra,'control':'same-family-histogram circularly shifted assignment'})
   all_methods.append('C3_shuffled')
   for method,control in [('C4_shifted','shift'),('C4_loudness','rms')]:
    rc,pc,_=apply_d6(body,dp,telemetry,target,control=control,minimum_rate=.25)
    _save_composed(root,method,song,rc,pc,p,{**extra6,'control':control});all_methods.append(method)
   ru,pu=retime_with_effort(body.world,body.time,proposals,target,metadata={'source_order_preserved':True})
   _save_composed(root,'C4_unanchored',song,ru,pu,p,{**extra6,'control':'Former regional D6 integration; motif boundaries not constrained','source_order_preserved':True});all_methods.append('C4_unanchored')
  print(song,condition,'body saved',flush=True)
 save(root/'bodies'/(song+'_COMPLETE.json'),{'methods':all_methods,'source_sha256':sha(source_path)})
 return all_methods


def map_all(root,song):
 for method in read(Path(root)/'bodies'/f'{song}_COMPLETE.json')['methods']:
  map_one(root,song,method);print(song,method,'emitted',flush=True)


def main():
 import argparse
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['init','generate','map','compare']);p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--track');a=p.parse_args()
 if a.stage=='init':initialize(a.root)
 elif a.stage=='generate':generate(a.root,a.track)
 elif a.stage=='map':map_all(a.root,a.track)
 else:
  from .diagnostics import compare
  for method in ['C3','C4']:compare(a.root,method,a.track)
if __name__=='__main__':main()
