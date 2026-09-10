"""Record measured codec outcomes and computed verdicts in the pinned project vault."""
import json
import random
import statistics
import sys
from pathlib import Path

from . import ROOT
from .registration import PROBLEM
from ..stage11_common import authenticated_json, read_authenticated, digest, verify_sources, verify_artifacts


def bootstrap(before, after, protocol):
    groups = sorted(before)
    assert groups == sorted(after)
    deltas = [after[g]['velocity_nrmse']-before[g]['velocity_nrmse'] for g in groups]
    rng = random.Random(protocol['evaluation']['bootstrap_seed'])
    values = sorted(statistics.mean(rng.choices(deltas,k=len(deltas)))
                    for _ in range(protocol['evaluation']['bootstrap_draws']))
    return {'paired_canonical_groups':len(groups),'after_minus_before':statistics.mean(deltas),
            'ci95':[values[int(.025*len(values))],values[int(.975*len(values))]],
            'method':'paired canonical-song percentile bootstrap; 10000 draws, frozen seed'}


def main():
    root = ROOT
    selection = read_authenticated(root/'codec/final_selection.json')
    protocol = read_authenticated(root/'codec/protocol.json')
    verify_sources(root/'codec')
    for path,sha in protocol['inputs'].items():
        if digest(path) != sha:
            raise ValueError('Frozen input changed: '+path)
    reports, initial = {}, {}
    for row in selection['rows']:
        arm = row['arm']
        run = read_authenticated(root/'codec/runs'/arm/'seed_0/report.json')
        verify_artifacts(run['artifacts'])
        reports[arm] = read_authenticated(root/'codec/evaluation'/arm/'seed_0/final/report.json')
        initial[arm] = read_authenticated(root/'codec/evaluation'/arm/'seed_0/initial/report.json')
    base = reports['frame24']
    sys.path.insert(0,'/root/vibescience-mcp/src')
    from vibescience_mcp.core import Store
    unrelated = Path('/root/vibescience-vault')
    before = {str(p):digest(p) for p in unrelated.rglob('*.md')}
    store = Store('/root/lightsem-vault')
    output = []
    for arm,eid in [('temporal64','exp-motion-codec-20260910'),
                    ('temporal128','exp-motion-codec-temporal128-20260910')]:
        report = reports[arm]
        selected_row = next(r for r in selection['rows'] if r['arm']==arm)
        intervals = {scope:bootstrap(base[scope]['per_song'],report[scope]['per_song'],protocol)
                     for scope in ('window_summary','full_summary')}
        evidence = root/'codec'/f'statistics_{arm}.json'
        authenticated_json(evidence,intervals)
        experiment = store.vault.read_experiment(eid)
        if not experiment.closed:
            measurements = [{'diagnostic_id':'motion-codec-velocity-nrmse',
                             'before':base['full_summary']['metrics']['velocity_nrmse'],
                             'after':report['full_summary']['metrics']['velocity_nrmse']}]
            for name in ('frame24',arm):
                did = f'motion-codec-{name}-learning-velocity-nrmse'
                store.register_diagnostic(id=did,name=did,direction='lower_better',unit='ratio',
                    description='Supplementary observed initialization-to-final full-VAL velocity NRMSE; no new prediction or selection rule.')
                measurements.append({'diagnostic_id':did,
                    'before':initial[name]['full_summary']['metrics']['velocity_nrmse'],
                    'after':reports[name]['full_summary']['metrics']['velocity_nrmse']})
            store.record_diagnostics(eid,measurements)
            gates = {**selected_row['gates'],'integrity':True}
            hypothesis = store.get_hypothesis(experiment.hypothesis_id)
            store.record_gate_results(eid,[{'gate_id':g.id,'passed':gates[g.id],
                'evidence':json.dumps({'arm':arm,'selection':str(root/'codec/final_selection.json'),
                    'window':selected_row['window'],'full':selected_row['full'],
                    'bootstrap':intervals,'integrity':'frozen sources, inputs, checkpoint artifacts verified'})}
                for g in hypothesis.gates])
            result = store.close_experiment(eid,
                notes='Full-sequence velocity NRMSE is primary; paired song-group intervals and fixed-window outcomes are separate. '
                      'All trained/initial states are retained. Absolute/source/physical/visual gates are independently reported. '
                      'The frozen quiet16s gate has zero eligible quiet/stationary windows and is vacuous; '
                      'post-result two-second diagnostics expose nonzero excess motion. No confirmation or TEST access. '
                      'An accurate contact classifier does not establish planted-foot geometry. '+json.dumps(intervals),
                artifacts=[str(root/'codec/final_selection.json'),str(evidence),str(root/'delivery/codec_frame24_temporal64/report.json')],
                key_numbers={'primary_before':measurements[0]['before'],'primary_after':measurements[0]['after'],
                             'eligible':int(selected_row['eligible'])},
                do_not_retry=['Do not repeat this 20k BF16 codec recipe or extend the same run identity; any revision must be newly frozen.'])
            experiment = store.vault.read_experiment(eid)
            if not selected_row['eligible']:
                store.interpret_experiment(eid,
                    lesson='This codec recipe retains recognizable poses but fails dynamics, physical contact and/or full-track fidelity. '
                           'Root-relative diagnostics are needed because fixing accumulated root drift alone may not repair limb velocity. '
                           'Do not start music training from an ineligible representation; freeze a distinct revision with direct short-hold coverage.',
                    failure_mode='recipe_conditional')
        output.append({'arm':arm,'experiment_id':eid,'verdict':experiment.verdict.value,
                       'prediction_match':experiment.prediction_match.model_dump(),
                       'gates':[g.model_dump() for g in experiment.gate_results], 'bootstrap':intervals})
    after = {str(p):digest(p) for p in unrelated.rglob('*.md')}
    if before != after:
        raise RuntimeError('Unrelated vault changed')
    summary = {'experiments':output,'unrelated_vault_unchanged':True,
               'calibration':store.calibration(diagnostic_id='motion-codec-velocity-nrmse'),
               'causal_map':store.causal_map(problem_id=PROBLEM),
               'recall':store.recall(problem_id=PROBLEM),'selection_sha256':digest(root/'codec/final_selection.json')}
    authenticated_json(root/'codec/vibescience_results.json',summary)
    print(json.dumps([{'arm':r['arm'],'verdict':r['verdict']} for r in output]))


if __name__ == '__main__':
    main()
