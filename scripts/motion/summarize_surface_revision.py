import json,hashlib
from pathlib import Path
from vibescience_mcp.core import Store
out=Path('/mnt/lightsem-data/experiments/lightsem/motion_native_audio_20260910/delivery/surface_revision_02')
def hashes():return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('/root/vibescience-vault').rglob('*') if p.is_file()}
before=hashes();s=Store('/root/lightsem-vault')
s.update_problem('dancing-light-generative-20260903',add_update='2026-09-10 spatial correction complete as diagnostic, NOT product acceptance. Reused saved Shpongle flow0, no training/TEST. Rev01 fixes persistent convergence but fails movement. Rev02 gain/bandwidth bundle improves first-surface span1.8166->6.4905m, soft-cone overlap70.507%->6.614%, active translation-removed4s extent.4542(old rejected)/.3721(rev01)->1.1301m. Active pan/tilt median9.742deg/s vs incumbent51.028; quiet2.971 vs40.872. Every non-angle DMX byte and optical settings held fixed. All8319 encoded video frames analysed; all555 chronological2Hz samples plus160 dense samples inspected, not continuous human audiovisual viewing. Local visible reconvergence194-201s and limited floor-facing fan vocabulary remain; visual gate FAILED and computed rev02 verdict directional_only. Do NOT claim fully fixed dancing lights or generator qualification. Full before/after retained at '+str(out/'shpongle_before_after.mp4'))
result={'recall':s.recall(problem_id='dancing-light-generative-20260903'),'calibration':s.calibration(),'doctor':s.doctor()}
after=hashes();assert before==after
result['unrelated_vault_unchanged']=True;result['hashes']=before
(out/'final_vault_audit.json').write_text(json.dumps(result,default=str,indent=2))
print('Pinned LightSem record updated; unrelated vault unchanged')
