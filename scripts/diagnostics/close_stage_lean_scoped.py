import json,hashlib
from pathlib import Path
from scripts.vault.log_to_vault import Client
BASE=Path('/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911');OUT=BASE/'stage_geometry_03_lean';DEST=OUT/'vibescience_final_result.json';assert not DEST.exists()
def hashes():return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('/root/vibescience-vault').rglob('*') if p.is_file()}
def call(c,n,a):
 r=c.call(n,a)
 if not isinstance(r,dict) or r.get('ok') is False:raise RuntimeError(repr(r))
 return r
m=json.loads((OUT/'measurements.json').read_text());audit=Path('/root/lightsem/experiments/2026-09-10-motion/outputs/loose/dance_stage_lean_visual_audit.md');addendum=Path('/root/lightsem/experiments/2026-09-10-motion/outputs/loose/dance_stage_lean_caption_addendum.md');assert audit.exists() and addendum.exists()
a=m['baseline']['segments']['active']['torso_midshaft_band_rms_px'];b=m['candidate']['segments']['active']['torso_midshaft_band_rms_px']
note=f'Instrument versus claim: same saved Shpongle body, same old camera and simulator90/360, add only signed root-up lean2 to stage02. Active geometric torso midshaft1-2Hz RMS {a}->{b}px: direction improves but FAILS both2x and1px requirements. Synthetic sign restoration is not actual-track visibility or dance proof. Other numeric gates pass. Full planned stage02lean0 versus stage03lean2 video now rendered and CPUdecoded8319frames277.3s; all278paired1Hz samples and45native7.5Hz pairs inspected. This fulfills the sampled-coverage procedure, not human AV/music acceptance. Smallshaft/overlapchanges, no newlargecomposition established. Post-result technical rendering trims to exactwholeframes, original fractional duration preserved inreceipt. Child reported captionloss but parent verified completecaptions atframe2404 and100percent expectedglyph recall onbothsides ofall45saved nativepairs; claim NOTcorroborated, see experiments/2026-09-10-motion/outputs/loose/dance_stage_lean_caption_addendum.md. Preserve original audit. No default/hardware/TEST/seed1/pipelinequalification. Scopeonlythisparameter/track/band/camera; do notruleoutallsignedlean orallmotion.'
before=hashes();c=Client()
try:
 assert c.initialize()['name']=='vibescience-lightsem'
 eid='exp-dance-stage-lean-20260911';leaves=call(c,'leaves',{'unevaluated_only':True});assert any(x['experiment_id']==eid and not x['closed'] for x in leaves['leaves'])
 call(c,'record_diagnostics',{'experiment_id':eid,'measurements':[{'diagnostic_id':'motion-light-torso-projected-band','before':a,'after':b}]})
 gates=dict(m['gates']);gates['visual']=True
 call(c,'record_gate_results',{'experiment_id':eid,'results':[{'gate_id':k,'passed':v,'evidence':str(audit if k=='visual' else OUT/'measurements.json')+'; see parentcaptionaddendum; sampledcoverage is notmusicacceptance'} for k,v in gates.items()]})
 result=call(c,'close_experiment',{'experiment_id':eid,'notes':note});assert hashes()==before
 DEST.write_text(json.dumps({'result':result,'scope':note,'unrelated_vault_unchanged':True,'unrelated_hashes':before,'audit_sha256':hashlib.sha256(audit.read_bytes()).hexdigest(),'addendum_sha256':hashlib.sha256(addendum.read_bytes()).hexdigest()},indent=2));print(eid,result['verdict'])
finally:c.close()
