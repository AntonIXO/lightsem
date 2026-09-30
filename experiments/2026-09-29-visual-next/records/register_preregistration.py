"""Register the reviewed literature and prospective studies through pinned MCP."""
from pathlib import Path
import gzip
import hashlib
import json
import re
import time

from scripts.vault.log_to_vault import Client, VAULT

ROOT = Path(__file__).resolve().parent
PAYLOAD = json.loads((ROOT / 'NATIVE_PAYLOADS_V1.json').read_text())
PROBLEM = 'dancing-light-generative-20260903'
VAULTS = ('/root/lightsem-vault', '/root/vibescience-vault')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot():
    return {root: {str(p): sha(p) for p in Path(root).rglob('*.md')} for root in VAULTS}


def save(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, ensure_ascii=False)
        f.write('\n')


def main():
    assert VAULT == VAULTS[0]
    assert sha(ROOT / 'PREREGISTRATION_V1.md') == PAYLOAD['protocol_sha256']
    before = snapshot()
    with gzip.open(ROOT / 'VAULT_BEFORE_REGISTRATION.json.gz', 'wt') as f:
        json.dump(before, f)
    protected = [Path(VAULT) / 'problems' / (PROBLEM + '.md'),
                 Path(VAULT) / 'runbooks/rb-baseline06-image-supervision-20260929.md']
    for p in protected:
        with gzip.open(ROOT / (p.stem + '_before.md.gz'), 'wb') as f:
            f.write(p.read_bytes())
    receipts = ROOT / 'native_receipts'
    receipts.mkdir(exist_ok=False)
    created, calls = [], []
    client = Client()
    counter = 0

    def call(tool, args, expected_id=None):
        nonlocal counter
        result = client.call(tool, args)
        counter += 1
        save(receipts / f'{counter:02d}_{tool}.json', {'tool': tool, 'arguments': args, 'result': result})
        if isinstance(result, str) or (isinstance(result, dict) and result.get('ok') is False):
            raise RuntimeError(f'{tool} returned an error; inspect receipt{counter}')
        if expected_id is not None:
            assert isinstance(result, dict) and result.get('id') == expected_id, (tool, result)
        calls.append({'tool': tool, 'id': expected_id})
        print(tool, expected_id or '', 'verified', flush=True)
        return result

    try:
        assert client.initialize()['name'] == 'vibescience-lightsem'
        old_problem = call('get_problem', {'id': PROBLEM}, PROBLEM)
        call('recall', {'problem_id': PROBLEM, 'query':
            'cut-aware independent content events temporal mask propagation Cutie vs SAM2 richer color temporal visual auxiliary and equal-information fixture oracle'})
        tags = call('list_tags', {})
        assert {'audio-to-light','cross-modal','evaluation-metric','matched-baseline','metric-gaming'} <= {x['id'] for x in tags}
        for category, tool in [('papers', 'add_paper'), ('diagnostics', 'register_diagnostic'),
                               ('interventions', 'register_intervention'), ('hypotheses', 'propose_hypothesis')]:
            for args in PAYLOAD[category]:
                path = Path(VAULT) / category / (args['id'] + '.md')
                assert not path.exists(), 'Refuse overwriting existing record ' + str(path)
                result = call(tool, args, args['id'])
                if category == 'hypotheses':
                    assert result['status'] == 'proposed'
                    assert result['predicted_effects'] == args['predicted_effects']
                created.append(str(path))

        prior_description = old_problem.get('description', '')
        prior_synthesis = old_problem.get('synthesis', '')
        summary = ('Current product-development baseline C is user-selected: independent Rich96 I_VR_P0 selectedseed0, '
            'rhythm companion02, generated palette history ON,300ms causal smoothstep color transitions. Music is the '
            'playback input; internal saved/generated poses still shape beam geometry while the audio clock supplies timing. '
            'The visual light model has no pose auxiliary; YOLO boxes/SAM2 are offline target-construction tools. '
            'The next questions are independently valid content-color labels, temporal masks, color-faithful spatial/temporal '
            'supervision, and constrained fixture-wise color output. One-track informed preference selected C; historical '
            'observer29/30null failure,31/40coverage failure and the negative shared-model palette-history study remain unchanged. '
            'No new experiments or scientific verdicts were produced in the2026-09-29 research/preregistration task.')
        new_ids = [p['id'] for p in PAYLOAD['papers']]
        update = ('Perplexity Deep Research completed after retrying the same full-context query with900s timeout; '
            '17primary papers fetched/read with exact identities/hashes. Four hypotheses and blocking gates are preregistered: '
            + ', '.join(h['id'] for h in PAYLOAD['hypotheses']) + '. '
            'Protocol: ' + str(ROOT / 'PREREGISTRATION_V1.md') + ' SHA256 ' + PAYLOAD['protocol_sha256'] + '. '
            'Independent annotation and executable model/source bindings remain pending; no invented observations/noise/power. '
            'Original description preserved verbatim below; original Markdown snapshot is also retained in the research records.\n\n'
            + prior_description)
        call('update_problem', {'id': PROBLEM, 'description': summary,
            'paper_refs': sorted(set(old_problem.get('paper_refs', [])) | set(new_ids)), 'add_update': update}, PROBLEM)
        current = ('CURRENT STATE —2026-09-29, literature-grounded preregistration, no new experiments.\n\n' + summary +
            '\n\nFIRST NEXT WORK: independently annotate the frozen12TRAIN calibration/31historicalDEV source windows '
            'and validate the instrument; then isolated cut/mask contrasts. Richer-feature student fitting is conditional '
            'on instrument/representation gates. The fixture oracle must beat an equally privileged global oracle; '
            'it establishes capacity only. Preserve C until a separately qualified product decision. '
            'Full research synthesis: ' + str(ROOT / 'RESEARCH_SYNTHESIS.md') +
            '\n\nHISTORICAL SYNTHESIS PRESERVED — older defaults/framing are superseded by the current state above:\n\n' + prior_synthesis)
        assert not re.search(r'^##\s', current, flags=re.M)
        call('synthesize_problem', {'problem_id': PROBLEM, 'synthesis': current})
        actual = call('get_problem', {'id': PROBLEM}, PROBLEM)
        assert actual['description'] == summary and prior_synthesis in actual['synthesis']
        assert actual['status'] == old_problem['status']
        assert actual['topic_tags'] == old_problem['topic_tags'] and actual['problem_tags'] == old_problem['problem_tags']

        priority = {'cut-targets': 1, 'temporal-masks': 2, 'fixture-oracle': 3, 'color-temporal': 4}
        runbooks = []
        for h in PAYLOAD['hypotheses']:
            short = h['id'].removeprefix('h-visual-next-').removesuffix('-20260929')
            rid = h['id'].replace('h-', 'rb-', 1)
            args = {'id': rid, 'problem_id': PROBLEM, 'priority': priority[short],
                'prompt': 'Execute ONLY the prospective study ' + h['id'] + ' after its prerequisite gates. '
                    'Its prediction is already native/proposed; do not create a duplicate hypothesis. '
                    'Start a new experiment with lineage to the relevant preserved parent only after the annotation, '
                    'instrument and implementation/source/weight bindings are valid. No new experiment was run during preregistration.\n\n' + h['plan'],
                'preconditions': 'Independent annotations, eligible support, instrument validation and exact executable/weight bindings are PENDING. '
                    'Baseline C/source artifacts are fixed; do not infer readiness or confirmation from this queued record.',
                'gates': [g['id'] for g in h['gates']], 'topic_tags': ['audio-to-light','evaluation-metric']}
            assert not (Path(VAULT) / 'runbooks' / (rid + '.md')).exists()
            call('register_runbook', args, rid)
            call('update_runbook', {'runbook_id': rid, 'hypothesis_id': h['id'],
                'note': 'Literature and prospective protocol registered before computation; prerequisites pending.'}, rid)
            runbooks.append(rid)
            created.append(str(Path(VAULT) / 'runbooks' / (rid + '.md')))
        call('update_runbook', {'runbook_id': 'rb-baseline06-image-supervision-20260929',
            'status': 'done', 'hypothesis_id': PAYLOAD['hypotheses'][0]['id'],
            'preconditions': 'Research/preregistration completed; four specific queued runbooks now carry execution. Independent annotations and experiments remain pending.',
            'note': 'Completed the requested full-context Perplexity research,17primary-paper verification/reading, '
                'and four native prospective hypotheses: ' + ', '.join(h['id'] for h in PAYLOAD['hypotheses']) +
                '. This closes only the research-planning intent, not the experiments. Baseline C remains accepted.'},
            'rb-baseline06-image-supervision-20260929')
        recalled = call('recall', {'problem_id': PROBLEM, 'query': 'visual-next temporal masks fixture oracle content-color'})
    finally:
        client.close()

    after = snapshot()
    changes = {root: sorted(p for p in set(before[root]) | set(after[root])
                            if before[root].get(p) != after[root].get(p)) for root in VAULTS}
    assert not changes[VAULTS[1]], 'Foreign vault changed during writes'
    allowed = set(created) | {str(p) for p in protected}
    assert set(changes[VAULT]) == allowed, changes
    # Check actual Markdown after reindex/readback, guarding against serialization loss.
    for h in PAYLOAD['hypotheses']:
        body = (Path(VAULT) / 'hypotheses' / (h['id'] + '.md')).read_text()
        assert h['plan'] in body, 'Native plan was truncated'
        assert 'status: proposed' in body
    before_experiments = {p: h for p, h in before[VAULT].items() if '/experiments/' in p}
    after_experiments = {p: h for p, h in after[VAULT].items() if '/experiments/' in p}
    assert before_experiments == after_experiments
    receipt = {'status': 'PREREGISTERED_AND_READ_BACK',
        'hypotheses': [h['id'] for h in PAYLOAD['hypotheses']], 'new_papers': len(PAYLOAD['papers']),
        'new_diagnostics': len(PAYLOAD['diagnostics']), 'new_interventions': len(PAYLOAD['interventions']),
        'runbooks': runbooks, 'calls': calls, 'protocol_sha256': PAYLOAD['protocol_sha256'],
        'created_records': created, 'changed_markdown': changes,
        'all_historical_experiment_bytes_preserved': True, 'foreign_vault_unchanged': True,
        'native_plan_readback_complete': True, 'new_experiments': 0,
        'scientific_results_or_noise_estimates_recorded': False}
    save(ROOT / 'NATIVE_REGISTRATION.json', receipt)
    print('PREREGISTERED_AND_READ_BACK', receipt['hypotheses'], flush=True)


if __name__ == '__main__':
    main()
