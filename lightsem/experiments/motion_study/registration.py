"""Register separate scientific claims in the explicitly pinned LightSem vault."""
import json
import sys
from pathlib import Path

from . import ROOT
from ..stage11.stage11_common import authenticated_json, read_authenticated, digest

PROBLEM = 'dancing-light-generative-20260903'
STUDIES = (
    ('codec', 'motion-codec-velocity-nrmse', 'lower_better',
     'A temporal continuous motion codec preserves clean dance dynamics better than a framewise 24-channel encoder.',
     'Compare same-data seed0 final-step codecs. Absolute joint error <=.02 body height; velocity NRMSE <=.15 using TRAIN RMS; contact F1 >=.90; active travel and kinetic ratios .9..1.1. Lowest bitrate eligible codec wins; separate source/visual gates.',
     ['source_quality', 'joint_fidelity', 'velocity_fidelity', 'contacts', 'travel_energy', 'continuity', 'visual_review', 'integrity']),
    ('audio', 'motion-correct-audio-energy-score', 'lower_better',
     'Conditional temporal flow generates coherent motion that responds correctly to music beyond mismatched and shifted audio controls.',
     'Correct-audio energy score >=5% lower than donor and time-shift controls; song-group95%CI excludes0; compare matched deterministic regression; direct lighting audio dynamics disabled for this test.',
     ['representation_prerequisite', 'conditioning_magnitude', 'conditioning_ci', 'continuity', 'visual_review', 'integrity']),
    ('coherence', 'motion-light-quiet-angular-speed', 'lower_better',
     'Shared body choreography and bounded regional expression reduce excessive rig movement while retaining active choreography.',
     'Shpongle quiet-intro median emitted pan/tilt speed >=60% below frozen incumbent; active movement and spatial development retained; shared versus independent sampled sequences compared with the same generator.',
     ['motion_prerequisite', 'quiet_pacing', 'active_movement', 'regional_expression', 'visual_review', 'integrity']),
    ('dynamics', 'motion-light-entry-luminance-ratio', 'higher_better',
     'Direct audio loudness controls improve visible quiet-to-entry brightness contrast without losing quiet-stage visibility.',
     'Matched motion, geometry and renderer; envelope on/off. Visible quiet introduction, clearly stronger loud entry, no added ending glare. Measure bulb-region luminance and clipping separately from DMX.',
     ['motion_prerequisite', 'entry_contrast', 'quiet_visibility', 'ending_glare', 'visual_review', 'integrity']),
    ('exposure', 'motion-render-dimmer-response', 'higher_better',
     'Fixed calibrated exposure restores visible dimmer response compressed by time-varying total-intensity normalization.',
     'Same DMX single-fixture and whole-rig ramps under legacy/fixed exposure; monotonic bulb luminance and meaningful dynamic range while bounding clipping and preserving beam detail.',
     ['ramp_monotonicity', 'dynamic_range', 'clipping', 'beam_visibility', 'integrity']),
)


def register(root=ROOT):
    root = Path(root)
    report = root / 'vibescience_registration.json'
    if report.exists():
        return read_authenticated(report)
    sys.path.insert(0, '/root/vibescience-mcp/src')
    from vibescience_mcp.core import Store
    other = Path('/root/vibescience-vault')
    before = {str(p): digest(p) for p in other.rglob('*.md')}
    store = Store('/root/lightsem-vault')  # Serial use: index rebuilding is not concurrency safe.
    old = store.get_problem(PROBLEM)
    marker = 'Accepted representation-first implementation 2026-09-10'
    if marker not in old.description:
        store.update_problem(id=PROBLEM, description=(
            marker + '. Build convincing music-driven stage lighting. Music is the only playback input; '
            'an internal generated skeleton/motion sequence is explicitly allowed. Hybrid measured audio '
            'dynamics may control brightness and movement intensity. Quiet passages must remain visible '
            'but restrained. Establish clean motion-source and temporal-codec quality before audio training. '
            'Prior human review rejects flashing/juggling reconstructions and excessive Shpongle movement; '
            'only the incumbent audio-to-descriptions-to-pose path is currently visually convincing. '
            'Glare improvement is useful but does not establish visible head-brightness dynamics. '
            'Exploration seed0; confirm only qualifying finalist and matched baseline at seed1. Accelerated '
            'BF16/compile/prefetch in matched arms, no earlier 2–3h ceiling. Historical nulls and gates remain valid.'),
            add_update='2026-09-10 explicit user authorization: implement the accepted representation-first plan. '
                       'This supersedes prior no-internal-motion restrictions. Previous description preserved:\n' + old.description)
    rows = []
    for stage, diagnostic, direction, statement, bar, gates in STUDIES:
        intervention = f'motion-{stage}-20260910'
        store.register_diagnostic(id=diagnostic, name=diagnostic, direction=direction,
                                  unit='ratio' if stage != 'coherence' else 'degrees_per_second',
                                  description=statement + ' ' + bar)
        store.register_intervention(id=intervention, name=intervention, description=statement)
        hid = f'h-motion-{stage}-20260910'
        if store.vault.exists('hypotheses', hid):
            h = store.get_hypothesis(hid)
        else:
            h = store.propose_hypothesis(id=hid, problem_id=PROBLEM, statement=statement,
                rationale='Accepted plan; source quality, codec fidelity, music conditioning, rig coherence, '
                          'audio dynamics and renderer response are separate causal questions. No historical gate is relaxed.',
                interventions=[intervention], predicted_effects=[{'diagnostic_id': diagnostic,
                    'direction': 'down' if direction == 'lower_better' else 'up', 'magnitude_note': bar}],
                gates=[{'id': g, 'description': bar + ' Component: ' + g} for g in gates],
                plan=str(root) + '. Source/runtime/protocol frozen before each scientific stage. '
                     'One GPU, seed0 exploration, six inference draws, no TEST access before qualification.',
                ambition_class='mechanism')
        rows.append({'stage': stage, 'hypothesis_id': h.id, 'diagnostic_id': diagnostic, 'gates': gates})
    after = {str(p): digest(p) for p in other.rglob('*.md')}
    if before != after:
        raise RuntimeError('Unrelated vault changed during registration; investigate')
    result = {'vault': '/root/lightsem-vault', 'rows': rows, 'source_sha256': digest(__file__),
              'unrelated_vault_unchanged': True, 'unrelated_markdown_count': len(before)}
    authenticated_json(report, result)
    return result


if __name__ == '__main__':
    print(json.dumps(register()), flush=True)
