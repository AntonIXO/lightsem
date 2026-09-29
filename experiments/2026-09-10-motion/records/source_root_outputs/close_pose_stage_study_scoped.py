import json,hashlib
from pathlib import Path
from log_to_vault import Client
OUT=Path('/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/pose_stage_01');DEST=OUT/'vibescience_final_result.json';assert not DEST.exists()
def auth(path):
 x=json.loads(Path(path).read_text());assert hashlib.sha256(json.dumps(x['payload'],sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()==x['sha256'];return x['payload']
def unrelated():return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('/root/vibescience-vault').rglob('*') if p.is_file()}
def call(c,n,a):
 r=c.call(n,a)
 if not isinstance(r,dict) or r.get('ok') is False:raise RuntimeError(repr(r))
 return r
numeric=auth(OUT/'numeric_report.json');visual=auth(OUT/'render_report.json');assert numeric['all_numeric_gates_pass']
assert all(visual[k] for k in ['all_approved_entries_rendered','all_candidate_pixel_gates_pass','all_independent_sampled_observations_present','all_software_media_gates_pass'])
assert visual['coverage_counts']['completed']==14 and visual['human_music_fit_acceptance'] is None and visual['pipeline_qualified'] is False
match=auth(OUT/'renders/shpongle_prior_flow0_global_matched/render.json');assert match['brightness_reference']['all_repeated_frame_rawhashes_equal']
assert (OUT/'numeric_vault_receipt.json').exists() and (OUT/'observer_audits/media/final_manifest.json').exists()
summary=Path('/root/lightsem/outputs/dance_pose_stage_final_outcome.md');media=Path('/root/lightsem/outputs/dance_pose_stage_media_audit.md');assert summary.exists() and media.exists()
note='Supports ONLY registered fixed-saved-body pose-chart bundle and scoped emitted-geometry/control claim: NEW exact-edge0.5-4Hz mean-per-head motion/radius .6359372158->1.4567158668 (2.29066x vs same-body stage02) inShpongle70-90;8/8travel/coord improve vs required6/8, all12reference item gates pass. Quietmedian unchanged2.816059829, fullsimulator90/360 limits/finite hits/nonangle parity/selected all-head-cloud thresholds pass.14movies and14independent sampled observations complete;106sheets/1090unique entry-frame samples and300nativefast inspections, NOTcontinuousAVorhumanacceptance. Independent all-frame/audioPTS anddecode checks pass. GlobalRGB controlscalars[1,.9351254485] retain rawframe/streamidentity; means2.9368334504 vs2.9620804781 include+.8597percent roundingresidual. Greaterouter-shaft pose-linkedmotion persistsunderdimming, butfan/steadycentralpair/tangles andunclearlowpose cues remain. No literalbodyreadability or universalgesturepreservation: prototypeprojectionnullspace persists. Pair overlapnotwholegroupcollapseor3Dintersection. This is NOTnewlearnedmusicconditioning, hardware, preference, visual/productqualification ordefaultpromotion. Fixedpriorflow0 and12mocapwindows only;noinference/TEST/seed1. Physicalconditioning andabsolute-sliding failures unchanged. Latermetadataaudit scopesoldES toordered0.5s summaries, notfinebeatlighting; directloudness changescurrenttarget/speed/photometry andpreventslearned-attribution fromappearance alone. See outputs/dance_pose_stage_final_outcome.md andalllinkedimmutableevidence. Goalactive; pipelineNOTqualified.'
before=unrelated();c=Client()
try:
 assert c.initialize()['name']=='vibescience-lightsem'
 eid='exp-dance-pose-stage-20260911';leaves=call(c,'leaves',{'unevaluated_only':True});assert any(x['experiment_id']==eid and not x['closed'] for x in leaves['leaves'])
 gates=call(c,'record_gate_results',{'experiment_id':eid,'results':[{'gate_id':'pixels','passed':True,'evidence':str(OUT/'render_report.json')+'; raw fixed-mask clipping gate only'},{'gate_id':'coverage','passed':True,'evidence':str(OUT/'render_report.json')+'; all14sampled observer attestations; NOT continuousAV/humanacceptance; '+str(media)},{'gate_id':'brightness','passed':True,'evidence':str(OUT/'renders/shpongle_prior_flow0_global_matched/render.json')+'; globalprocedure withreported.8597percent residual, not exactmeanmatching'}]})
 result=call(c,'close_experiment',{'experiment_id':eid,'notes':note})
 update=call(c,'update_problem',{'id':'dancing-light-generative-20260903','add_update':note})
 assert unrelated()==before
 DEST.write_text(json.dumps({'result':result,'gate_results':gates,'problem_update':update,'scope':note,'unrelated_vault_unchanged':True,'summary_sha256':hashlib.sha256(summary.read_bytes()).hexdigest(),'media_report_sha256':hashlib.sha256(media.read_bytes()).hexdigest()},indent=2));print(eid,result['verdict'])
finally:c.close()
