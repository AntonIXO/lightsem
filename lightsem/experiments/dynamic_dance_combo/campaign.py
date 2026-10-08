"""Frozen full-track generation, paired composition and actual emitted mapping."""
from pathlib import Path
import hashlib,json,math
import numpy as np
from .brightness import load_verified_loudness,apply_brightness
from .clock_metadata import nonnegative_effort
from ..dynamic_dance.contracts import DanceResult,DancePlan,MusicTimeline
from ..dynamic_dance.retiming import d4_motif_recomposition,retime_with_effort,resample_body
from ..dynamic_dance.training import observed_regional_speed
from ...dance.lighting import (_exact_clock, _load_p2_colors, _normalise_counts,
                               _resample_body_full_interval, _stream_duplicate_groups,
                               _validate_body_for_lighting)

ROOT=Path('/mnt/lightsem-data/experiments/lightsem/dynamic_dance_combo_20261005_v1')
OLD=Path('/mnt/lightsem-data/experiments/lightsem/dynamic_dance_20261005_v1')

def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,value):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
 with Path(str(p)+'.sha256.json').open('x') as f:json.dump({'sha256':sha(p)},f);f.write('\n')
def arrays(p,**data):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 if not all(np.isfinite(x).all() for x in data.values()):raise ValueError('Nonfinite output arrays')
 with p.open('xb') as f:np.savez_compressed(f,**data)
def track(root,song):return next(t for t in read(Path(root)/'PLAN.json')['tracks'] if t['id']==song)
def body_path(root,method,song):return Path(root)/'bodies'/method/song/'body.npz'
def suffix(condition):return '' if condition=='correct' else '__'+condition

def generate(root,song,condition='correct'):
 import torch
 from .generation import generate_full
 from ..dynamic_dance.training import load_learned
 torch.use_deterministic_algorithms(True)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 root=Path(root);plan=read(root/'PLAN.json');t=track(root,song)
 timeline=t['music_timeline'] if condition in ('correct','plan_shift','repeat') else t['waveform_controls'][condition]['music_timeline']
 base=t['base92'] if condition in ('correct','plan_shift','repeat') else t['waveform_controls'][condition]['base92']
 if condition=='donor':
  binding_path=root/'records/DONOR_DURATION_BINDINGS_v1.json'
  if sha(binding_path)!=read(str(binding_path)+'.sha256.json')['sha256']:raise ValueError('Donor duration binding changed')
  bound=read(binding_path)['tracks'][song]
  timeline,base=bound['music_timeline'],bound['base92']
 # One maximum-song draw, cropped in TIME identically across inputs. The model
 # noise API internally draws channels first; hashing the final tensor matters.
 m,_,_=load_learned(OLD,'D7',0,'cpu')
 maxframes=max(x['body_frames'] for x in plan['tracks'])
 noise=m.noise(1,maxframes,generator=torch.Generator(device='cpu').manual_seed(7001),device='cpu')[:,:t['body_frames']].contiguous()
 del m
 result=generate_full(OLD,timeline,base,t['duration_s'],'cuda',7001,initial_noise=noise,
                      plan_intervention='shift_global_half' if condition=='plan_shift' else None,
                      output_common_reference=condition=='correct')
 folder=root/'generation'/song/condition
 folder.mkdir(parents=True,exist_ok=False)
 arrays(folder/'telemetry.npz',**result['telemetry'])
 save(folder/'metadata.json',result['metadata'])
 generated='G78_shift' if condition=='plan_shift' else 'G78'+suffix(condition)
 for key,method in [('reference','D7_reference'),('combined',generated)]:
  if key not in result or result[key] is None:continue
  output=result[key]
  metadata={**result['metadata'],'method':method,'song':song,'condition':condition,
            'generation_component':'reference_D7_only' if key=='reference' else 'combined_D7_D8',
            'planner_active':key=='combined','effort_is_generated_plan':key=='combined',
            'fixed_effort_marker_for_D7':key=='reference',
            'motion_origin':'new full-song D7 flow' if key=='reference' else 'D7 flow plus bounded D8 global-minus-local plan residual',
            'independent_motion_source':False,'newly_generated_motion':True,'product_baseline':False,
            'body_generation':'full30Hz trajectory, identical explicit noise for D7 reference and combination',
            'telemetry':str(folder/'telemetry.npz'),'telemetry_sha256':sha(folder/'telemetry.npz')}
  p=body_path(root,method,song)
  DanceResult(time=result['time'],world=output['world'],native=output['native'],contacts=output['contacts'],metadata=metadata).save(p)
  coarse=result['telemetry']['global_plan'] if key=='combined' else np.ones((len(result['time']),5))
  DancePlan(time=result['time'],source_time=result['time'],effort=coarse[:,:5],metadata=metadata).save(p.parent/'plan.npz')
 save(folder/'COMPLETE.json',{'method':generated,'song':song,'condition':condition,'outputs':{str(p):sha(p) for p in folder.glob('*') if p.is_file()},'body_sha256':sha(body_path(root,generated,song))})
 return generated

def learned_proposals(world,time,telemetry):
 from scipy.ndimage import gaussian_filter1d
 p=telemetry['coarse_local_plan'];coarse=telemetry['coarse_time']
 speed=np.expm1(p[:,:5]);hold=p[:,9:14];prob=p[:,14:].reshape(len(p),5,4)
 target=speed*(1-hold)+.025*hold
 desired=np.stack([np.interp(time,coarse,target[:,i]) for i in range(5)],-1)
 ratio=desired/np.maximum(observed_regional_speed(world,30.),.05)
 rates=np.empty_like(ratio)
 for i in range(5):
  levels=np.stack([gaussian_filter1d(ratio[:,i],scale*30/4,mode='nearest') for scale in (.5,1.,2.,4.)],-1)
  probs=np.stack([np.interp(time,coarse,prob[:,i,j]) for j in range(4)],-1)
  rates[:,i]=(levels*probs).sum(-1)
 return rates

def _save_composed(root,method,song,result,plan,source_path,extra):
 result.time=np.arange(len(result.world),dtype=float)/30
 plan.time=result.time.copy()
 meta={**result.metadata,**extra,'method':method,'song':song,'product_baseline':False,
       'source_body':str(source_path),'source_body_sha256':sha(source_path)}
 result.metadata=meta;plan.metadata={**plan.metadata,**meta}
 if meta.get('source_order_preserved') and np.any(plan.effort<0):
  effort,roundoff=nonnegative_effort(result.source_time,result.time,fps=meta['fps'])
  plan.effort=effort.astype(np.float32)
  plan.metadata['derived_effort_roundoff']=roundoff
 p=body_path(root,method,song);result.save(p);plan.save(p.parent/'plan.npz')
 return p

def postprocess(root,song,condition='correct'):
 if condition in ('plan_shift','repeat'):
  raise ValueError('Unsupported postprocess for generation-only control: '+condition)
 root=Path(root);t=track(root,song);timeline_path=t['music_timeline'] if condition in ('correct','plan_shift') else t['waveform_controls'][condition]['music_timeline']
 if condition=='donor':timeline_path=read(root/'records/DONOR_DURATION_BINDINGS_v1.json')['tracks'][song]['music_timeline']
 music=MusicTimeline.load(timeline_path);s=suffix(condition)
 source_path=body_path(root,'G78'+s,song);source=DanceResult.load(source_path)
 with np.load(root/'generation'/song/condition/'telemetry.npz') as z:telemetry={k:z[k] for k in z.files}
 result,plan=d4_motif_recomposition(source.world,music,30.)
 metadata={'components':['D7','D8','D4'],'motion_origin':'same-song fragments from D7 + D8 generated body','music_timeline':timeline_path,'condition':condition,
           'generation_provenance':source.metadata,
           'generated_source':str(source_path),'generated_source_sha256':sha(source_path),'d6_enabled':False}
 c3=_save_composed(root,'C3'+s,song,result,plan,source_path,metadata)
 # D6 is a distinct OPTIONAL pacing operation on exactly the saved C3 source.
 c3body=DanceResult.load(c3);time=np.arange(len(c3body.world))/30
 rates=learned_proposals(c3body.world,time,telemetry)
 result6,plan6=retime_with_effort(c3body.world,time,rates,music,metadata={'components':['D7','D8','D4','D6'],'source_order_preserved':True})
 meta6={**metadata,'components':['D7','D8','D4','D6'],'d6_enabled':True,'motion_origin':'learned contact-aware pacing of the same C3 motif composition',
        'source_clock_scope':'monotone clocks into C3; C3 separately has nonmonotone original-fragment provenance',
        'fragment_provenance':c3body.metadata.get('fragment_provenance',[]),'fragment_clock_domain':'C3 source time, not final output time',
        'd4_plan':str(c3.parent/'plan.npz'),'d4_plan_sha256':sha(c3.parent/'plan.npz'),'learned_duration_proposal':'trained local planner, no retraining or target-motion labels at inference'}
 c4=_save_composed(root,'C4'+s,song,result6,plan6,c3,meta6)
 arrays(c4.parent/'rate_proposals.npz',time=time,desired_rates=rates,**{k:telemetry[k] for k in ('coarse_time','coarse_local_plan')})
 if condition=='correct':
  rs,ps=d4_motif_recomposition(source.world,music,30.,control='shuffle')
  _save_composed(root,'C3_shuffled',song,rs,ps,source_path,{**metadata,'control':'shuffled_recurrence'})
  rms=np.interp(time,music.time,music.mixture[:,0]);q=max(float(np.quantile(rms,.9)),1e-8)
  loud=.2+1.3*np.clip(rms/q,0,1)
  for method,proposal in [('C4_loudness',loud),('C4_shifted',np.roll(rates,len(rates)//2,axis=0))]:
   rr,pp=retime_with_effort(c3body.world,time,proposal,music,metadata={'source_order_preserved':True})
   _save_composed(root,method,song,rr,pp,c3,{**meta6,'control':method,'duration_control_scope':'circularly shifted five-region learned proposals before fixed feasibility projection' if method=='C4_shifted' else 'RMS-only global proposals using existing .2+1.3*normalized-RMS schedule'})
 save(root/'generation'/song/condition/'POSTPROCESS.json',{'source':str(source_path),'source_sha256':sha(source_path),'C3':str(c3),'C3_sha256':sha(c3),'C4':str(c4),'C4_sha256':sha(c4),'condition':condition})
 return ['C3'+s,'C4'+s]+(['C3_shuffled','C4_loudness','C4_shifted'] if condition=='correct' else [])

def compile_world(world40,time,outputs,left,right,loudness,transfer,mapper_id=None):
 from lightsem.conductor.body_transfer_v2 import map_body
 from lightsem.conductor.body_transfer import TransferConfig,compile_intent
 from scripts.motion.run_body_portfolio import hard_checks
 # The mapper is part of the production lighting configuration.  Older PLAN
 # files carried it beside the transfer settings, so accept that spelling too
 # without allowing it to leak into TransferConfig.
 transfer=dict(transfer)
 if mapper_id is None: mapper_id=transfer.pop('mapper', '01')
 else: transfer.pop('mapper', None)
 mapper_id=str(mapper_id).zfill(2)
 from lightsem.conductor.body_transfer_v2 import CATALOG
 if mapper_id not in CATALOG:
  raise ValueError(f'Unknown body mapper {mapper_id!r}')
 cfg=TransferConfig(**transfer)
 intent=map_body(world40,outputs,mapper_id,40.)
 restored,trace=apply_brightness(world40,time,outputs,intent,loudness)
 dmx,patch,final=compile_intent(restored,outputs,left,right,cfg)
 hard=hard_checks(dmx,patch,final,40.,enforce_motion_limits=cfg.apply_motion_limits)
 # Keep mapper identity and attribution attached to the intent that actually
 # reached DMX; reports must never infer it from a requested config alone.
 final['algorithm_id']=mapper_id
 return dmx,patch,final,trace,hard

def map_one(root,song,method):
 from lightsem.conductor.body_transfer_v2 import make_outputs
 from lightsem.conductor.body_transfer import decode
 root=Path(root);plan=read(root/'PLAN.json');t=track(root,song);p=body_path(root,method,song);body=DanceResult.load(p)
 if len(body.world)!=t['body_frames']:raise ValueError('Full native song required')
 frames=int(t['lighting_frames'])
 if frames < 2: raise ValueError('Full song must provide at least two lighting frames')
 time=np.arange(frames,dtype=np.float64)/40.
 world_native, body_time = _validate_body_for_lighting(body, float(time[-1]))
 world=_resample_body_full_interval(world_native,body_time,time)
 colors=root/'colors'/song/'P2.npz'
 left,right,palette,color_active,color_identity=_load_p2_colors(colors,t['audio'],frames,
                                                                allow_fixture_report=True)
 loudness,loudrep=load_verified_loudness(t['loudness_receipt'],t['audio'])
 counts=_normalise_counts(plan.get('counts'))
 outputs=make_outputs(**counts)
 mapper_id=plan.get('mapper', plan.get('lighting',{}).get('mapper', plan.get('transfer',{}).get('mapper','01')))
 dmx,patch,intent,bright,hard=compile_world(world,time,outputs,left,right,loudness,plan['transfer'],mapper_id=mapper_id)
 decoded=decode(dmx,patch)
 for i,f in enumerate(patch):
  expected=left if f['position'][0]<0 else right
  if not np.array_equal(np.rint(decoded['rgb'][:,i]*255).astype(np.uint8),expected):raise ValueError('P2 RGB changed')
 if not np.array_equal(np.rint(intent['dimmer']*255).astype(np.uint8),np.rint(decoded['dimmer']*255).astype(np.uint8)):raise ValueError('Emitted dimmer differs from intended restoration')
 attribution=np.zeros((len(outputs),22),dtype=np.uint8)
 for index,joints in enumerate(intent.get('attribution', ())):
  attribution[index,np.asarray(joints,dtype=int)]=1
 duplicate_groups=_stream_duplicate_groups(decoded)
 folder=root/'timelines'/method/song;folder.mkdir(parents=True,exist_ok=False)
 data={'time':time,'body':world,'universes':dmx,'targets':intent['targets'],'dimmer':intent['dimmer'],'source_points':intent['source_points'],'field_points':intent['field_points'],'attribution_mask':attribution,'brightness_envelope':bright['level'] if 'level' in bright else bright['envelope']['level']}
 if palette is not None:data['palette_ids40']=palette
 if color_active is not None:data['color_active40']=color_active
 rate_roundoff=None
 if body.source_time is not None:
  source=np.asarray(body.source_time);source=source[:,None] if source.ndim==1 else source
  data['source_time']=np.stack([np.interp(time,body.time,source[:,i]) for i in range(source.shape[1])],-1)
  if body.metadata.get('d6_enabled') or method.startswith('C4'):
   rates,rate_roundoff=nonnegative_effort(source,body.time)
  else:
   dp=DancePlan.load(p.parent/'plan.npz');rates=dp.effort
  data['playback_rate']=np.stack([np.interp(time,body.time,rates[:,i]) for i in range(rates.shape[1])],-1)
 cuts=np.array(sorted(set(float(x['target_seconds'][0]) for x in body.metadata.get('fragment_provenance',[]))))
 if body.metadata.get('d6_enabled') and body.source_time is not None:
  clock=np.asarray(body.source_time);clock=clock[:,0] if clock.ndim==2 else clock
  cuts=body.time[np.clip(np.searchsorted(clock,cuts,side='left'),0,len(clock)-1)]
 data['intervention_time']=cuts
 arrays(folder/'timeline.npz',**data);save(folder/'patch.json',patch)
 arrays(folder/'brightness.npz',**{k:v for k,v in bright.items() if isinstance(v,np.ndarray)})
 report={'status':'engineering_valid' if hard.get('pass',hard.get('passed',True)) else 'diagnostic','method':method,'track':song,'frames':len(time),'hard_checks':hard,'P2_RGB_exact':True,
         'mapper':intent.get('algorithm_id',str(mapper_id).zfill(2)),'counts':counts,
         'mapper_attribution':{'fixture_count':len(outputs),
                               'covered_joint_count':int(np.count_nonzero(attribution.any(axis=0))),
                               'uncovered_joints':np.flatnonzero(~attribution.any(axis=0)).tolist(),
                               'exact_duplicate_fixture_groups':duplicate_groups},
         'color':color_identity,
         'body_sha256':sha(p),'timeline_sha256':sha(folder/'timeline.npz'),'patch_sha256':sha(folder/'patch.json'),
         'inputs':{str(root/'PLAN.json'):sha(root/'PLAN.json'),str(p):sha(p),str(p.with_suffix('.json')):sha(p.with_suffix('.json')),str(colors):sha(colors),str(colors.parent/'REPORT.json'):sha(colors.parent/'REPORT.json'),'body_metadata_sha256':sha(p.with_suffix('.json'))},
         'brightness':{k:v for k,v in bright.items() if not isinstance(v,np.ndarray)},'loudness':loudrep,'derived_rate_roundoff':rate_roundoff,
         'brightness_scope':'same original song global audio envelope for all candidates and intervention diagnostics; regional/body brightness may differ','product_baseline':False}
 save(folder/'report.json',report)
 return report
