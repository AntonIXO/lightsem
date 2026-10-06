"""Register and close the fixed LightSem body-transfer engineering comparison.

All writes use /root/lightsem-vault. Directional verdicts are computed by Store;
two familiar songs and geometric proxies do not establish artistic quality.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

# The vault environment deliberately has no NumPy/Torch dependency.
_catalog_source = Path(__file__).resolve().parents[2] / 'lightsem/conductor/body_transfer_v2/__init__.py'
CATALOG = next(ast.literal_eval(node.value) for node in ast.parse(_catalog_source.read_text()).body
               if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'CATALOG' for t in node.targets))

PROBLEM = 'body-transfer-portfolio-20261004'
CROSS = 'bt2-geometric-crossing-fraction'
COVERAGE = 'bt2-emitted-six-region-coverage'
DUPLICATES = 'bt2-exact-desired-target-pairs'
TITLES = {
    '1804.05653': 'Neural Kinematic Networks for Unsupervised Motion Retargetting',
    '2005.05732': 'Skeleton-Aware Networks for Deep Motion Retargeting',
    '2306.08006': 'Pose-aware Attention Network for Flexible Motion Retargeting by Body Part',
    '2109.07431': 'Contact-Aware Retargeting of Skinned Motion',
    '2303.08658': 'Skinned Motion Retargeting with Residual Perception of Motion Semantics & Geometry',
    '2410.20986': 'Skinned Motion Retargeting with Dense Geometric Interaction Perception',
    '2509.26633': 'OmniRetarget: Interaction-Preserving Data Generation for Humanoid Whole-Body Loco-Manipulation and Scene Interaction',
    '0905.2635': 'Point-Set Registration: Coherent Point Drift',
    '1306.0895': 'Sinkhorn Distances: Lightspeed Computation of Optimal Transportation Distances',
    '1211.0053': 'The Emerging Field of Signal Processing on Graphs: Extending High-Dimensional Data Analysis to Networks and Other Irregular Domains',
    '1408.6552': 'Bearing Rigidity and Almost Global Bearing-Only Formation Stabilization',
    '2006.06071': 'Affective Movement Generation using Laban Effort and Shape and Hidden Markov Models',
    '2509.24469': 'LaMoGen: Laban Movement-Guided Diffusion for Text-to-Motion Generation',
}


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def store():
    sys.path.insert(0, '/root/vibescience-mcp/src')
    from vibescience_mcp.core import Store
    return Store('/root/lightsem-vault')


def register(root, revision=2):
    problem = PROBLEM if revision == 2 else PROBLEM + '-v' + str(revision)
    tag = 'bt' + str(revision)
    receipt = root / 'records/VIBESCIENCE_REGISTRATION.json'
    if receipt.exists():
        raise FileExistsError('Registration already exists; preserve it')
    s = store()
    for identifier, name, direction, description in (
        (CROSS, 'Body transfer fixed geometric projected ray crossing fraction', 'lower_better',
         'Strict segment crossings, sampled up to480 times and4096 deterministic pairs in seconds60–180; excludes collinear overlap. Descriptive proxy, not exact photometry or choreography.'),
        (COVERAGE, 'Body transfer emitted six-region sensitivity coverage', 'higher_better',
         'Fraction of head,torso,left/right arm,left/right leg controlled12degree perturbations that change actual16bit moving angles by at least1tick. Brightness and colors excluded.'),
        (DUPLICATES, 'Body transfer exact desired-target trajectory duplicate pairs', 'lower_better',
         'Exact target-coordinate trajectory pairs for default13moving outputs; different fixture origins do not count as distinct source information.'),
    ):
        s.register_diagnostic(name=name, unit='fraction' if identifier != DUPLICATES else 'pairs',
                              direction=direction, description=description, id=identifier)
    paper_ids = {}
    for arxiv, title in TITLES.items():
        identifier = 'bt2-paper-' + arxiv.replace('.', '-')
        if not s.vault.exists('papers', identifier):
            s.add_paper(title=title, arxiv_id_or_url='https://arxiv.org/abs/'+arxiv,
                key_claims=['Verified source mechanism; LightSem mapping is an original deterministic adaptation, not a reproduction or paper-validated lighting result.'], id=identifier)
        paper_ids[arxiv] = identifier
    if not s.vault.exists('problems', problem):
        s.create_problem('Preserve accepted body gestures in arbitrary-N emitted lighting',
            description=('Rework the body-to-light adapter on accepted separate Shpongle/Cure bodies and fixed P2. '
                'Prior V1 exact cross-type target duplicates, missing head response, projection nullspace, persistent brightness and mechanical attenuation motivate this portfolio. '
                'Read prior motion-surface, surface-bandwidth, pose-stage, lean, projection and optics records without changing their verdicts. '
                'The separate audio-understanding campaign informs timing/absence controls, not new inputs to these comparisons. '
                '15 algorithms ×2 familiar songs; all outcome claims remain engineering-scoped.'),
            paper_refs=list(paper_ids.values()), id=problem)
    rows = []
    for algorithm, (name, papers) in CATALOG.items():
        intervention = tag+'-mapping-'+algorithm+'-20261004'
        s.register_intervention(name=name+' with stable anatomical capability readout',
            description=f'Body-only field plus V{revision} capability readout and unchanged limits/P2. Comparison against V1 estimates the combined change, not isolated causality of the paper mechanism. V3 repairs absolute structure and proximity readout for03/08; other13methods repeat the same inputs and are not independent replication.',
            id=intervention)
        for track in ('shpongle', 'cure_for_me'):
            hypothesis = f'h-{tag}-{algorithm}-{track}-20261004'
            if not s.vault.exists('hypotheses', hypothesis):
                s.propose_hypothesis(problem,
                    statement=f'{name} and V2 capability mapping reduce projected ray crossings for {track} while preserving six-region source sensitivity.',
                    rationale='Anatomical field representation and separated static anchors should avoid the shared central aim region; source sensitivity prevents static spread from qualifying.',
                    interventions=[intervention], papers=[paper_ids[p] for p in papers],
                    predicted_effects=[{'diagnostic_id': CROSS, 'direction': 'down',
                        'magnitude_note': 'At least0.10 absolute reduction; additionally source-region coverage>=V1 and hard invariants. No statistical generalization claim.'},
                        {'diagnostic_id': DUPLICATES, 'direction': 'down', 'magnitude_note': 'Report independently; distinct targets alone do not establish choreography.'}],
                    gates=[{'id':'crossing_magnitude','description':'Crossing fraction decreases by at least0.10 absolute.'},
                           {'id':'source_sensitivity','description':'Six-region emitted angular coverage is at least V1.'},
                           {'id':'hard_invariants','description':'Finite values, mechanical bounds, constant-body nonRGB output, P2 equality, source identity.'}],
                    plan='Freeze source/config/data. Run counterexample controls before full trajectories. Report intent, actualDMX, optical proxies, brightness and color separately. Preserve all failures. User review decides choreography.',
                    ambition_class='mechanism', id=hypothesis)
            experiment = f'exp-{tag}-{algorithm}-{track}-20261004'
            if not s.vault.exists('experiments', experiment):
                s.start_experiment(hypothesis, external_run=str(root/'timelines'/algorithm/track),
                    config_note='Preregistered before new emitted-output comparisons. Pipeline implementation tests are engineering checks. Scientific input/source manifest will be frozen in PLAN.json.', id=experiment)
            rows.append({'algorithm':algorithm,'track':track,'hypothesis':hypothesis,'experiment':experiment})
    save(receipt, {'problem':problem,'revision':revision,'vault':'/root/lightsem-vault','rows':rows,
        'primary_diagnostic':CROSS,'magnitude_absolute':.10,
        'scope':'Combined mapper/readout engineering comparison; no artistic or music-conditioning qualification.'})
    print('REGISTERED',len(rows),flush=True)


def close(root):
    s = store()
    registration = read(root/'records/VIBESCIENCE_REGISTRATION.json')
    results = []
    for row in registration['rows']:
        algorithm, track = row['algorithm'], row['track']
        before = read(root/'baseline'/track/'report.json')
        after = read(root/'timelines'/algorithm/track/'report.json')
        a, b = before.get('diagnostics',before), after['diagnostics']
        bp = read(root/'preflight'/'v1'/f'{track}.json')
        ap = read(root/'preflight'/algorithm/f'{track}.json')
        measurements = [
            {'diagnostic_id':CROSS,'before':a['spatial']['crossing_fraction'],'after':b['spatial']['crossing_fraction']},
            {'diagnostic_id':COVERAGE,'before':bp['emitted_source_region_coverage'],'after':ap['emitted_source_region_coverage']},
            {'diagnostic_id':DUPLICATES,'before':len(a['duplicates']['desired_target_pairs']),'after':len(b['duplicates']['desired_target_pairs'])}]
        gates = [
            {'gate_id':'crossing_magnitude','passed':measurements[0]['before']-measurements[0]['after']>=.10,
             'evidence':json.dumps(measurements[0])},
            {'gate_id':'source_sensitivity','passed':measurements[1]['after']>=measurements[1]['before'],
             'evidence':json.dumps(measurements[1])},
            {'gate_id':'hard_invariants','passed':bool(after.get('hard_invariants_passed',False)),
             'evidence':str(root/'timelines'/algorithm/track/'report.json')}]
        experiment = s.vault.read_experiment(row['experiment'])
        if not experiment.closed:
            s.record_diagnostics(row['experiment'], measurements)
            s.record_gate_results(row['experiment'], gates)
            verdict = s.close_experiment(row['experiment'],
                notes='Computed verdict concerns a geometric clutter proxy and engineering gates only. Motion/brightness/color remain separate. Two familiar examples are not generalization evidence. Choreography requires review.',
                artifacts=[str(root/'timelines'/algorithm/track/'report.json')],
                key_numbers={m['diagnostic_id']+'_'+phase:m[phase]
                             for m in measurements for phase in ('before','after')})
        else:
            verdict = {'experiment':experiment}
        results.append({**row,'measurements':measurements,'gates':gates,
                        'computed':json.loads(json.dumps(verdict,default=lambda o:o.model_dump(mode='json') if hasattr(o,'model_dump') else str(o)))})
    save(root/'records/VIBESCIENCE_VERDICTS.json',{'results':results,'scope':'Engineering verdicts only; no automatic product promotion.'})
    print('CLOSED',len(results),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('register','close'))
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--revision',type=int,choices=(2,3),default=2)
    args = parser.parse_args()
    register(args.root,args.revision) if args.stage == 'register' else close(args.root)
