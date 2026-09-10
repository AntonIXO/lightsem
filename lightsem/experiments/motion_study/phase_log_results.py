"""Close the precommitted phase128 comparison, retaining every capacity outcome."""
import json
import sys
from pathlib import Path

from .phase_registration import ROOT
from .registration import PROBLEM
from .log_results import bootstrap
from ..stage11_common import read_authenticated, authenticated_json, digest, verify_sources


def main():
    protocol = read_authenticated(ROOT/'codec/protocol.json')
    selection = read_authenticated(ROOT/'codec/final_selection.json')
    verify_sources(ROOT/'codec')
    for path,sha in protocol['inputs'].items():
        if digest(path)!=sha:
            raise ValueError('Frozen input changed')
    reports = {r['arm']:read_authenticated(ROOT/'codec/evaluation'/r['arm']/'final/report.json') for r in selection['rows']}
    initial = {arm:read_authenticated(ROOT/'codec/evaluation'/arm/'initial/report.json') for arm in reports}
    baseline = read_authenticated(protocol['primary_baseline'])
    arm = protocol['primary_arm']
    row = next(r for r in selection['rows'] if r['arm']==arm)
    intervals = {scope:bootstrap(baseline[scope]['per_song'],reports[arm][scope]['per_song'],protocol)
                 for scope in ('window_summary','full_summary')}
    authenticated_json(ROOT/'codec/statistics_phase128.json',intervals)
    sys.path.insert(0,'/root/vibescience-mcp/src')
    from vibescience_mcp.core import Store
    unrelated = {str(p):digest(p) for p in Path('/root/vibescience-vault').rglob('*.md')}
    store = Store('/root/lightsem-vault')
    eid = 'exp-motion-native-phase-codec-20260910'
    experiment = store.vault.read_experiment(eid)
    if not experiment.closed:
        measurements = [{'diagnostic_id':'motion-codec-velocity-nrmse',
            'before':baseline['full_summary']['metrics']['velocity_nrmse'],
            'after':reports[arm]['full_summary']['metrics']['velocity_nrmse']}]
        for name in reports:
            did = 'motion-phase-'+name+'-fit-velocity-nrmse'
            store.register_diagnostic(id=did,name=did,direction='lower_better',unit='ratio',
                description='Supplementary observed random orthogonal initialization to analytic TRAIN-fit full-VAL velocity NRMSE; not a substituted primary.')
            measurements.append({'diagnostic_id':did,'before':initial[name]['full_summary']['metrics']['velocity_nrmse'],
                'after':reports[name]['full_summary']['metrics']['velocity_nrmse']})
        store.record_diagnostics(eid,measurements)
        hypothesis = store.get_hypothesis(experiment.hypothesis_id)
        gates = {**row['gates'],'integrity':True}
        store.record_gate_results(eid,[{'gate_id':g.id,'passed':gates[g.id],
            'evidence':json.dumps({'selection':str(ROOT/'codec/final_selection.json'),'primary':arm,
                'window':row['window'],'full':row['full'],'intervals':intervals})} for g in hypothesis.gates])
        store.close_experiment(eid,notes='Primary phase128 versus historical original temporal128, not compute matched. '+json.dumps(intervals)+
            ' All four capacities are reported; none substituted for the fixed primary. Exact root/contact latent channels do not repair limb temporal errors. No TEST or confirmation.',
            artifacts=[str(ROOT/'codec/final_selection.json'),str(ROOT/'codec/statistics_phase128.json')],
            key_numbers={'primary_before':measurements[0]['before'],'primary_after':measurements[0]['after'],
                         'eligible_arms':len(selection['eligible_arms'])},
            do_not_retry=['Do not repeat these truncated independent Haar-band projections expecting native velocity fidelity; any revision needs a new identity.'])
        if not row['eligible']:
            store.interpret_experiment(eid,lesson='Pose accuracy alone hides severe temporal distortion from bandwise truncation. Even the largest candidate fails velocity, kinetic energy, planted feet and short-hold gates. Retain full native detail as an explicit uncompressed engineering reference before music generation; do not call it learned compression success.',failure_mode='recipe_conditional')
        experiment = store.vault.read_experiment(eid)
    if unrelated!={str(p):digest(p) for p in Path('/root/vibescience-vault').rglob('*.md')}:
        raise ValueError('Unrelated vault changed')
    result = {'experiment_id':eid,'verdict':experiment.verdict.value,'bootstrap':intervals,
        'selection_sha256':digest(ROOT/'codec/final_selection.json'),'all_arms':selection['rows'],
        'calibration':store.calibration(diagnostic_id='motion-codec-velocity-nrmse'),
        'causal_map':store.causal_map(problem_id=PROBLEM),'recall':store.recall(problem_id=PROBLEM),
        'unrelated_vault_unchanged':True}
    authenticated_json(ROOT/'codec/vibescience_results.json',result)
    print(json.dumps({'verdict':result['verdict'],'intervals':intervals}))


if __name__=='__main__':
    main()
