import json,hashlib
from pathlib import Path
from scripts.vault.log_to_vault import Client
out=Path('/mnt/lightsem-data/experiments/lightsem/rhythm_companion_20260912T193657Z');eid=json.loads((out/'registration.json').read_text())['experiment']['id']
def hashes():return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('/root/vibescience-vault').rglob('*') if p.is_file()}
def call(c,n,a):
 r=c.call(n,a)
 if not isinstance(r,dict) or r.get('ok') is False:raise RuntimeError(str(r))
 return r
before=hashes();c=Client()
try:
 assert c.initialize()['name']=='vibescience-lightsem'
 media=call(c,'record_gate_results',{'experiment_id':eid,'results':[{'gate_id':'media','passed':True,'evidence':'Threefull277.3s deliveries,each8319decoded1920x1080AV1frames at30fps. Candidate/baseline rendered natively1080p; comparison has2x960x540scenes on1080p canvas. FullsoftwareAVdecode,allvideoPTS,allaudiopacketPTS/durations pass. Each44100HzstereoPCM has12228930samples/channel,identical hashes acrossall3files. AACpriming1024/finaldiscard702 honored; sourceFLAC6.667mswhole-frame trim. Fourfixedsourcewaveformwindows bestlag0. Fullrawoutsidebulbs anychannel>=250clip0botharms. Comparison277nominal1Hzluma2.275195/2.230331 afterglobalmatch,-1.97%candidate residual. No dropped/paddedvideo. Silentvisualcoverage85nominal1Hzcandidate samples+18densepaired samples+stills; nohumanAVmusicfit. Full source/artifacts and hashes:'+str(out)}]})
 update=call(c,'update_problem',{'id':'dancing-light-generative-20260903','add_update':'2026-09-12 rhythm-companion engineering implementation and1080pAV1delivery COMPLETE.85focusedtests pass. Products: '+str(out/'full_candidate_1080p_av1.mp4')+' and '+str(out/'full_comparison_1080p_av1.mp4')+'. All3fullvideos media/audiotimingverified; final_outcome.json contains scopedreview. Savedbody neverchanged; motion controls slowstyle and audio clocks lightgestures, notdancerretiming. Numeric25%speedprediction remains FAILED (14.31%increase), despite implementationmechanismpasses. Repetitivepairedmotifs and unannotatedrealbeataccuracy remain limits. Nooverallmusicfit/humanacceptance/defaultpromotion/hardwarequalification/newcommit. Leave experiment unclosed rather than turn a direction-only diagnostic record into support for the missedpreregisteredmagnitude.'})
 assert hashes()==before
 (out/'final_vault_receipt.json').write_text(json.dumps({'media':media,'update':update,'unrelated_vault_unchanged':True,'closed':False,'reason':'Originalmagnitude>=1.25failed; do notautoclosebroadsupports'},indent=2));print('FINAL MEDIA RECORDED')
finally:c.close()
