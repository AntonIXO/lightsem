"""Close mapper claims from frozen measurements and explicit sampled-frame review."""
import json
import sys
from pathlib import Path

from .delivery import ROOT,OUT
from .registration import PROBLEM
from ..stage11_common import authenticated_json,read_authenticated,digest,verify_sources


def close():
    verify_sources(OUT)
    p=read_authenticated(OUT/'protocol.json')
    r=read_authenticated(OUT/'lighting/observations.json')
    review=read_authenticated(OUT/'review.json')
    if r['protocol_sha256']!=digest(OUT/'protocol.json') or review['observations_sha256']!=digest(OUT/'lighting/observations.json'):
        raise ValueError('Review or observation belongs to another delivery')
    for path,sha in p['inputs'].items():
        if digest(path)!=sha:raise ValueError('Frozen delivery input changed')
    for row in read_authenticated(OUT/'lighting/videos.json')['rows']:
        if digest(row['path'])!=row['sha256']:raise ValueError('Reviewed video changed')
    other={str(p):digest(p) for p in Path('/root/vibescience-vault').rglob('*.md')}
    sys.path.insert(0,'/root/vibescience-mcp/src')
    from vibescience_mcp.core import Store
    store=Store('/root/lightsem-vault');rows=[]
    for stage in ('coherence','dynamics'):
        eid=f'exp-motion-{stage}-20260910';exp=store.vault.read_experiment(eid)
        gates={g.id:r['gates'].get(g.id,False) for g in store.get_hypothesis(exp.hypothesis_id).gates}
        gates['visual_review']=review[stage]['passed']
        if stage=='coherence':
            did='motion-light-quiet-angular-speed'
            before=r['rows']['incumbent']['segments']['quiet']['speed_median'];after=r['rows']['hybrid']['segments']['quiet']['speed_median']
            gates['regional_expression']=review['regional_expression']['passed']
        else:
            did='motion-light-entry-luminance-ratio'
            before=r['rows']['brightness_off']['entry_ratio'];after=r['rows']['hybrid']['entry_ratio']
        if not exp.closed:
            store.record_diagnostics(eid,[{'diagnostic_id':did,'before':before,'after':after}])
            store.record_gate_results(eid,[{'gate_id':k,'passed':v,'evidence':json.dumps({'observations':str(OUT/'lighting/observations.json'),'review':review,'gates':r['gates']})} for k,v in gates.items()])
            store.close_experiment(eid,notes=p['scope']+' '+json.dumps({'gates':gates,'quiet_reduction':r['quiet_speed_reduction'],'review_type':review['type']})+
                ' Directional brightness or quiet-pacing improvement is not approval of the generated choreography. All five complete lighting variants are retained; no ending-position fade or additional random procedural motion was used.',
                artifacts=[str(OUT/'lighting/observations.json'),str(OUT/'review.json'),str(OUT/'lighting/videos.json')],
                key_numbers={'before':before,'after':after,'all_gates':int(all(gates.values())),'motion_prerequisite':0},
                do_not_retry=['Do not promote the native generator from direct-audio lighting metrics. Do not relax the failed active-movement minimum or substitute a different time interval.'])
            if not all(gates.values()):
                store.interpret_experiment(eid,lesson='Shared articulated targets, quantified mechanical limits and direct measured loudness separate useful lighting behavior from generator quality. A downstream improvement cannot reverse weak conditionality or physical-motion failure upstream.',failure_mode='recipe_conditional')
            exp=store.vault.read_experiment(eid)
        rows.append({'stage':stage,'experiment':eid,'verdict':exp.verdict.value,'before':before,'after':after,'gates':gates})
    if other!={str(p):digest(p) for p in Path('/root/vibescience-vault').rglob('*.md')}:raise ValueError('Unrelated vault changed')
    result={'rows':rows,'native_model_qualified':False,'confirmation_authorized':False,'test_access':False,
            'optical_ramps_qualified':read_authenticated(ROOT/'exposure_roles/vibescience_results.json')['qualified'],
            'unrelated_vault_unchanged':True,'calibration':store.calibration(),
            'recall':store.recall(problem_id=PROBLEM),'causal_map':store.causal_map(problem_id=PROBLEM)}
    authenticated_json(OUT/'vibescience_results.json',result)
    print(json.dumps(rows))


if __name__=='__main__':close()
