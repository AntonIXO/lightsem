"""Record complete native-music, integration and first exposure outcomes."""
import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from .registration import PROBLEM
from ..stage11_common import read_authenticated,authenticated_json,digest,verify_sources,verify_artifacts

ROOT = Path('/mnt/lightsem-data/experiments/lightsem/motion_native_audio_20260910')


def log_music(store):
    p = read_authenticated(ROOT/'music/protocol.json')
    selection = read_authenticated(ROOT/'music/selection_seed_0.json')
    verify_sources(ROOT/'music')
    for path,sha in p['inputs'].items():
        if digest(path)!=sha:raise ValueError('Frozen music input changed')
    rows = {r['arm']:r for r in selection['rows']}
    for arm in rows:
        verify_artifacts(read_authenticated(ROOT/'music/runs'/arm/'seed_0/report.json')['artifacts'])
    flow,reg = rows['flow'],rows['regression']
    eid = 'exp-motion-native-audio-20260910'
    exp = store.vault.read_experiment(eid)
    if not exp.closed:
        measurements = [
            {'diagnostic_id':'motion-correct-audio-energy-score','before':flow['controls']['donor']['before'],'after':flow['controls']['donor']['after']},
            {'diagnostic_id':'motion-native-flow-vs-regression-energy-score','before':reg['final']['correct_es'],'after':flow['final']['correct_es']}]
        for arm,row in rows.items():
            for control in ('donor','shift'):
                did = f'motion-native-{arm}-{control}-energy-score'
                store.register_diagnostic(id=did,name=did,direction='lower_better',unit='standardized_rms',
                    description='Supplementary complete within-model control: wrong-audio ES before, correct-audio ES after; primary remains the preregistered flow donor comparison.')
                c = row['controls'][control]
                measurements.append({'diagnostic_id':did,'before':c['before'],'after':c['after']})
            did = f'motion-native-{arm}-learning-energy-score'
            store.register_diagnostic(id=did,name=did,direction='lower_better',unit='standardized_rms',
                description='Supplementary initialization-to-final correct-audio ES, same frozen119windows and score space.')
            measurements.append({'diagnostic_id':did,'before':row['initial']['correct_es'],'after':row['final']['correct_es']})
        store.record_diagnostics(eid,measurements)
        h = store.get_hypothesis(exp.hypothesis_id)
        store.record_gate_results(eid,[{'gate_id':g.id,'passed':flow['gates'][g.id],
            'evidence':json.dumps({'selection':str(ROOT/'music/selection_seed_0.json'),'controls':flow['controls'],
                                 'quality':flow['quality'],'all_gates':flow['gates']})} for g in h.gates])
        store.close_experiment(eid,
            notes='Direction and qualification differ. Flow improves unconditional distribution score substantially over initialization and trained regression, '
                  'but neither donor nor within-track shift reaches5%; both flow CIs include zero. Own-label foot sliding fails. '
                  'Full-chunk agreement passes, yet sampled full-track poses/travel fail visual acceptance. '
                  'Paired corpus128TRAIN/36VAL records109/31groups; actual32s sampler125TRAIN107groups and119VALwindows33records28groups. '
                  'Short TRAIN006/177/189 and VAL171/190/192 excluded only by fixed crop length, transparently recorded; no claim all short holds were trained. '
                  'No confirmation or TEST. '+json.dumps(flow['controls']),
            artifacts=[str(ROOT/'music/selection_seed_0.json'),str(ROOT/'music/review/render_report_seed_0.json')],
            key_numbers={'donor_relative_improvement':flow['controls']['donor']['relative_improvement'],
                         'shift_relative_improvement':flow['controls']['shift']['relative_improvement'],
                         'sliding_excess_m_s':flow['quality']['sliding_excess'],'qualified':0},
            do_not_retry=['Do not promote or extend this native13930k recipe under the same identity. A revision must address physical contact, weak conditionality and short-record handling explicitly.'])
        store.interpret_experiment(eid,
            lesson='Learning the marginal motion distribution and reducing energy score versus a regressor do not establish music timing. Physical foot constraints are missing from the native quadratic objective; contact labels remain populated while feet slide. Global chunk consistency does not bound generated root travel. Preserve this failed recipe and freeze a distinct targeted revision.',
            failure_mode='recipe_conditional')
        exp = store.vault.read_experiment(eid)
    result = {'experiment_id':eid,'verdict':exp.verdict.value,'selection_sha256':digest(ROOT/'music/selection_seed_0.json'),
              'prediction_match':exp.prediction_match.model_dump(),'gates':[g.model_dump() for g in exp.gate_results]}
    authenticated_json(ROOT/'music/vibescience_results.json',result)
    return result


def log_integration(store):
    out = ROOT/'music/review'
    verify_sources(out)
    p = read_authenticated(out/'protocol.json')
    cfg = read_authenticated(out/'integration_experiment.json')
    report = read_authenticated(out/'incumbent/report.json')
    groups = defaultdict(list)
    for row in report['rows']:
        if digest(row['path'])!=row['sha256']:raise ValueError('Incumbent output changed')
        groups[row['canonical_id']].append(row['metrics'])
    grouped = []
    for group,rows in sorted(groups.items()):
        grouped.append({step:{name:statistics.mean(r[step][name] for r in rows) for name in rows[0][step]} for step in ('8','32')})
    before = statistics.mean(r['8']['high_frequency_fraction'] for r in grouped)
    after = statistics.mean(r['32']['high_frequency_fraction'] for r in grouped)
    energy_ratio = statistics.mean(r['32']['velocity_energy_tl2_s2'] for r in grouped)/statistics.mean(r['8']['velocity_energy_tl2_s2'] for r in grouped)
    rng = random.Random(cfg['bootstrap_seed']);bootstrap=[]
    for _ in range(cfg['bootstrap_draws']):
        sample = rng.choices(grouped,k=len(grouped))
        a,b = (statistics.mean(r[s]['high_frequency_fraction'] for r in sample) for s in ('8','32'))
        bootstrap.append((a-b)/max(a,1e-12))
    bootstrap.sort()
    stats = {'before':before,'after':after,'relative_reduction':(before-after)/max(before,1e-12),
             'ci95':[bootstrap[250],bootstrap[9750]],'velocity_energy_ratio':energy_ratio,'canonical_groups':len(grouped)}
    authenticated_json(out/'integration_statistics.json',stats)
    gates = {'paired_inputs':True,'magnitude':stats['relative_reduction']>=cfg['minimum_relative_reduction'],
             'bootstrap':stats['ci95'][0]>0,'active_motion_retained':cfg['velocity_energy_ratio'][0]<=energy_ratio<=cfg['velocity_energy_ratio'][1],
             'integrity':all(digest(path)==sha for path,sha in p['inputs'].items())}
    eid=cfg['experiment_id'];exp=store.vault.read_experiment(eid)
    if not exp.closed:
        store.record_diagnostics(eid,[{'diagnostic_id':cfg['diagnostic_id'],'before':before,'after':after}])
        store.record_gate_results(eid,[{'gate_id':g,'passed':v,'evidence':json.dumps(stats)} for g,v in gates.items()])
        store.close_experiment(eid,notes='Same incumbent checkpoint, descriptor256 pose decoder, full audio and global noise seed101; only Euler step count changes. '+json.dumps(stats)+
            'Sparse2D snapshots look similar. This is not a comparison against fixed-bone3D reconstruction or a broad guarantee of perceptual improvement.',
            artifacts=[str(out/'incumbent/report.json'),str(out/'integration_statistics.json')],
            key_numbers={'before':before,'after':after,'relative_reduction':stats['relative_reduction'],'all_gates':int(all(gates.values()))})
        if not all(gates.values()):
            store.interpret_experiment(eid,lesson='Increasing integration steps alone did not meet the preregistered jitter-improvement and preservation gates; the learned motion/decoder remains a separate issue.',failure_mode='recipe_conditional')
        exp=store.vault.read_experiment(eid)
    result={'experiment_id':eid,'verdict':exp.verdict.value,'statistics':stats,'gates':gates}
    authenticated_json(out/'integration_vibescience.json',result)
    return result


def log_exposure(store):
    out=ROOT/'exposure';verify_sources(out)
    p=read_authenticated(out/'protocol.json');report=read_authenticated(out/'report.json')
    eid=read_authenticated(out/'experiment.json')['experiment_id'];exp=store.vault.read_experiment(eid)
    if not exp.closed:
        measurements=[]
        for candidate in report['candidates']:
            scale=candidate['scale'];did='motion-render-response-fixed-'+str(round(scale*100)).zfill(3)
            store.register_diagnostic(id=did,name=did,direction='higher_better',unit='ratio',
                description='Observed whole-rig bulb-region80/20-percent dimmer response for a prespecified fixed exposure scale; legacy before, candidate after. Not a substituted selected primary endpoint.')
            measurements.append({'diagnostic_id':did,'before':report['legacy_response'],'after':candidate['scenes']['whole_rig']['response']})
        # The selection returned none. Keep the selected-primary measurement
        # undefined rather than substituting a favorable scale after outcomes.
        if report['selected_scale'] is not None:
            candidate=next(c for c in report['candidates'] if c['scale']==report['selected_scale'])
            measurements.append({'diagnostic_id':'motion-render-dimmer-response','before':report['legacy_response'],'after':candidate['scenes']['whole_rig']['response']})
        store.record_diagnostics(eid,measurements)
        candidates=report['candidates']
        gates={name:any(all(s['gates'][name] for s in c['scenes'].values()) for c in candidates)
               for name in ('ramp_monotonicity','dynamic_range','clipping','beam_visibility')}
        gates['integrity']=report['default_pixel_max_abs']==0
        store.record_gate_results(eid,[{'gate_id':k,'passed':v,'evidence':str(out/'report.json')} for k,v in gates.items()])
        store.close_experiment(eid,notes='All five scales were actually measured and retained. Selection found no candidate passing all scenes. '
            'The selected-primary endpoint is therefore undefined; no favorable scale is substituted. The computed direction verdict may be inconclusive for that reason. '
            'Every scale improves raw whole-rig response, but single-beam visibility fails; high scales compress whole-rig response through tonemapping. '
            'The actual historical renderer and the new default legacy profile are pixel-identical across all63ramps. '
            'A role-balanced fixed-radiance revision needs a new identity and a fixed primary parameter set.',
            artifacts=[str(out/'report.json'),str(out/'protocol.json')],key_numbers={'legacy_response':report['legacy_response'],'eligible_scales':0,'legacy_pixel_difference':report['default_pixel_max_abs']},
            do_not_retry=['Do not silently adopt a failed fixed scale or relax the global single-beam visibility threshold in this protocol.'])
        store.interpret_experiment(eid,lesson='One shared fixed exposure does not simultaneously retain single-beam visibility and whole-rig response under identical per-kind radiance gains. Fixed role calibration is a distinct intervention; a future primary must be fixed before outcomes even if a selection gate fails.',failure_mode='recipe_conditional')
        exp=store.vault.read_experiment(eid)
    result={'experiment_id':eid,'verdict':exp.verdict.value,'selected_scale':report['selected_scale'],'report_sha256':digest(out/'report.json')}
    authenticated_json(out/'vibescience_results.json',result)
    return result


def main():
    sys.path.insert(0,'/root/vibescience-mcp/src')
    from vibescience_mcp.core import Store
    unrelated={str(p):digest(p) for p in Path('/root/vibescience-vault').rglob('*.md')}
    store=Store('/root/lightsem-vault')
    results=[log_music(store),log_integration(store),log_exposure(store)]
    if unrelated!={str(p):digest(p) for p in Path('/root/vibescience-vault').rglob('*.md')}:raise ValueError('Unrelated vault changed')
    authenticated_json(ROOT/'completed_stage_records.json',{'results':results,'unrelated_vault_unchanged':True,
        'calibration':store.calibration(),'causal_map':store.causal_map(problem_id=PROBLEM),'recall':store.recall(problem_id=PROBLEM)})
    print(json.dumps([{'experiment':r['experiment_id'],'verdict':r['verdict']} for r in results]))


if __name__=='__main__':main()
