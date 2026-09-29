import json,hashlib
from pathlib import Path
from scripts.vault.log_to_vault import Client
PLAN=Path('/root/lightsem/experiments/2026-09-10-motion/outputs/loose/pose_stage_study_plan.json');p=json.loads(PLAN.read_text());OUT=Path(p['artifact_directory']);DEST=OUT/'registration.json';assert not DEST.exists();assert p['parent_decisions_final'] is True
assert not (OUT/'mapped').exists() and not (OUT/'measurements.json').exists(),'Candidate execution must not precede registration'
def hashes():return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('/root/vibescience-vault').rglob('*') if p.is_file()}
def call(c,n,a):
 r=c.call(n,a)
 if not isinstance(r,dict) or r.get('ok') is False:raise RuntimeError(repr(r))
 return r
before=hashes();c=Client()
try:
 assert c.initialize()['name']=='vibescience-lightsem'
 diagnostic=call(c,'register_diagnostic',{'id':'motion-light-broadband-motion-radius','name':'Emitted midshaft broadband motion over projected cone radius','unit':'ratio','direction':'higher_better','description':'NEW fixed.5-4Hz exact-edge Welch PSD integration, mean of six per-head midshaft RMS/mean equivalent half-peak cone radius ratios; fixed camera and same savedbody. Not the older half-open1-2Hz diagnostic, decoded opticalflow, musicfit or preference.'})
 hypothesis=call(c,'propose_hypothesis',{'id':p['hypothesis_id'],'problem_id':'dancing-light-generative-20260903','statement':'A fixed anatomical root-local pose chart, rather than fixture-locked lateral lanes, increases emitted broadband motion relative to cone size on the saved Shpongle body while respecting registered group-spread, mechanics and control limits.','predicted_effects':[{'diagnostic_id':diagnostic['id'],'direction':'up','magnitude_note':'Candidate>=2*same-body stage02 on the NEW.5-4Hz emitted geometric instrument in70-90s; all independent gates required.'}],'rationale':p['prior_evidence']+' Current source and physical studies still fail musicconditioning; this is a mapper mechanism test, not a music-model fix. It builds on static stage02 and completed optics/lean evidence. Pair convergence is not equivalent to global collapse.','plan':json.dumps(p,sort_keys=True),'gates':[{'id':k,'description':v,'blocking':True} for k,v in p['gates'].items()],'ambition_class':'mechanism','gap_closure_estimate':.25})
 experiment=call(c,'start_experiment',{'id':p['experiment_id'],'hypothesis_id':hypothesis['id'],'git_ref':'working-tree; exact source snapshot required before execution','external_run':str(OUT),'config_note':json.dumps(p,sort_keys=True),'parent_experiment_id':'exp-dance-stage-anchors-20260911'})
 note='20260911-labelled studies completed: physical directional_only (conditioning and absolute sliding fail), static stage anchors supports scoped surface/trajectory controls, optics supports scoped geometry/brightness controls, signedlean directional_only (torso.4326->.4801px fails2x/1px). No pipeline orhumanAVqualification. Unpreregistered syntheticgestureaudit nowproves rawtargetxordering locked and strong lateral saturation; prototype permits crossings but millimetre pairmerges and49.1cm proper-FK projectionnullspace remain. New pose-stage emitted-motion study preregistered BEFORE any realcandidate mapping, with explicit earlystops and unchangedsource/music. See experiments/2026-09-10-motion/outputs/loose/dance_physical_final_outcome.md, dance_optics_final_outcome.md, dance_gesture_transfer_design.md and pose_stage_study_plan.json. Caption-loss allegation in lean audit NOTcorroborated by parent45sample glyphcheck; preserveparentaddendum.'
 call(c,'update_problem',{'id':'dancing-light-generative-20260903','add_update':note})
 assert hashes()==before
 OUT.mkdir(parents=True,exist_ok=True)
 DEST.write_text(json.dumps({'diagnostic':diagnostic,'hypothesis':hypothesis,'experiment':experiment,'registered_by':'parent','registered_before_candidate_execution':True,'plan_sha256':hashlib.sha256(PLAN.read_bytes()).hexdigest(),'unrelated_vault_unchanged':True,'unrelated_hashes':before},indent=2));print('REGISTERED',experiment['id'])
finally:c.close()
