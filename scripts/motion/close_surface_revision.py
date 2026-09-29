"""Pinned Store closure; independent gates, original evidence never rewritten."""
import argparse,json,hashlib
from pathlib import Path
from vibescience_mcp.core import Store


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def hashes():return {str(p):digest(p) for p in Path('/root/vibescience-vault').rglob('*') if p.is_file()}


def run(out):
    plan=json.loads((out/'design_preregistered.json').read_text())
    report=json.loads((out/'measurements.json').read_text());row=report['rows']['articulated_surface'];segs=row['segments']
    review=json.loads((out/'review.json').read_text())
    for p,sha in plan['inputs'].items():
        if digest(p)!=sha:raise ValueError('input drift')
    frozen=json.loads((out/'frozen_sources.json').read_text())
    for p,sha in frozen['files'].items():
        if digest(out/'source'/p)!=sha:raise ValueError('source drift')
    before=hashes();s=Store('/root/lightsem-vault')
    reg=json.loads((out/'registration.json').read_text());eid=reg['experiment']
    passed={g:all(v['gates'].get(g,True) for v in segs.values()) for g in ['spatial_spread','surface_coverage','not_concentrated','surface_valid','cone_separation','trajectory','not_just_translation','changing_shape']}
    active=segs['active']['mechanics'];quiet=segs['quiet']['mechanics']
    passed.update(mechanics=active['max']<=30+1e-8 and active['acceleration_max']<=80+1e-8,
                  quiet_pacing=quiet['median']<=40.872*.4,
                  active_movement=active['median']>=3 and active['p95']>=8,
                  photometric_identity=row['photometric_bytes_identical'] and report['legacy_dmx_identical'],
                  visual_review=review['visual_review'])
    if 'bandwidth' in eid:
        old=json.loads((Path(plan['parent'])/'measurements.json').read_text())['rows']['articulated_surface']['segments']['active']['summary']
        did='motion-light-relative-surface-extent';b=old['relative_trajectory_4s_median_m'];a=segs['active']['summary']['relative_trajectory_4s_median_m']
    else:
        old=json.loads((out/'old_spatial.json').read_text())['hybrid']['full']['summary']
        did='motion-light-surface-horizontal-span';b=old['hit_x_span']['median'];a=segs['full']['summary']['hit_x_span']['median']
    exp=s.vault.read_experiment(eid)
    if exp.closed:raise ValueError('Never overwrite a closure')
    s.record_diagnostics(eid,[{'diagnostic_id':did,'before':b,'after':a}])
    s.record_gate_results(eid,[{'gate_id':g,'passed':passed[g],'evidence':json.dumps({'measurements':str(out/'measurements.json'),'review':str(out/'review.json'),'gate':g,'segments':segs,'scope':'One saved Shpongle mapper diagnostic; no generator qualification.'})} for g in plan['gates']])
    videos=[out/(arm+'.mp4') for arm in plan['arms']]
    wall=0.
    for p in videos:
        receipt=json.loads(p.with_suffix('.video.json').read_text())
        if digest(p)!=receipt['sha256']:raise ValueError('Video drift')
        wall+=receipt['wall_seconds']
    verdict=s.close_experiment(eid,notes='Instrument: saved-track surface geometry, actual emitted DMX and completed renders; claim limited to this mapper recipe and song. Visual review is sampled and cannot establish continuous audiovisual human acceptance or music conditioning. '+json.dumps(review),
        artifacts=[str(out/'measurements.json'),str(out/'review.json'),str(out/'protocol.json')]+[str(v) for v in videos],
        key_numbers={'before':b,'after':a,'active_speed_median':active['median'],'quiet_speed_median':quiet['median'],'all_gates':int(all(passed.values())),'generator_qualified':0,'human_product_acceptance':0},
        do_not_retry=['Do not mistake widened static surface anchors for choreography. Do not qualify the music generator from direct-audio brightness or downstream mapper geometry. Do not erase the old user-rejected collapse or change frozen spatial thresholds.'],
        gpu_hours=wall/3600,wall_hours=wall/3600)
    s.interpret_experiment(eid,lesson='Surface-space targets remove the persistent central focus caused by elevated inward targets. Articulation bandwidth and a complete changing composition are separate gates; spatial spread alone is not dancing to music. '+review['choreography']['notes'],failure_mode='recipe_conditional')
    after=hashes();assert before==after,'UNRELATED VAULT CHANGED'
    result={'experiment':eid,'verdict':verdict,'gates':passed,'before':b,'after':a,'unrelated_vault_unchanged':True,'unrelated_vault_hashes':before,'doctor':s.doctor()}
    (out/'vibescience_results.json').write_text(json.dumps(result,default=str,indent=2))
    print(json.dumps({'experiment':eid,'verdict':verdict,'gates':passed,'before':b,'after':a},default=str),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);a=p.parse_args();run(a.out)
