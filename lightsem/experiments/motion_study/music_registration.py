"""Commit the native music-generation claim before scientific training."""
import json
import sys
from pathlib import Path

from .registration import PROBLEM
from ..stage11.stage11_common import read_authenticated,authenticated_json,digest

ROOT = Path('/mnt/lightsem-data/experiments/lightsem/motion_native_audio_20260910')
HYPOTHESIS = 'h-motion-native-audio-20260910'


def main():
    path = ROOT/'music/registration.json'
    if path.exists():
        return read_authenticated(path)
    if not read_authenticated(ROOT/'codec/final_acceptance.json')['audio_training_authorized']:
        raise ValueError('Native representation is not accepted')
    sys.path.insert(0,'/root/vibescience-mcp/src')
    from vibescience_mcp.core import Store
    store = Store('/root/lightsem-vault')
    intervention = 'motion-native-audio-flow-20260910'
    store.register_intervention(id=intervention,name=intervention,
        description='Matched Base92-to-native139 temporal regression and conditional flow. Uncompressed frozen motion representation,32s contexts,one stationary noise field and global Euler state,30k matched updates,seed0. Six inference draws; direct audio lighting dynamics disabled.')
    store.register_diagnostic(id='motion-native-flow-vs-regression-energy-score',name='motion-native-flow-vs-regression-energy-score',
        direction='lower_better',unit='standardized_rms',description='Correct-audio probabilistic energy score in frozen18-channel physical motion summary space; trained regression before, trained flow after. Separate from within-model audio controls.')
    if not store.vault.exists('hypotheses',HYPOTHESIS):
        store.propose_hypothesis(id=HYPOTHESIS,problem_id=PROBLEM,
            statement='A conditional temporal flow on full native motion uses the correct music and preserves coherent dynamics better than a matched deterministic regressor.',
            rationale='Both compressed codec recipes failed. An explicitly uncompressed139-channel native reference now passes the original fidelity gates and a fresh12-clip visual review. Its acceptance is engineering identity, not learned compression. Test actual music conditioning independently of direct brightness/activity and assess generated geometry/dynamics separately.',
            interventions=[intervention],predicted_effects=[
                {'diagnostic_id':'motion-correct-audio-energy-score','direction':'down','magnitude_note':'Primary trained flow donor-audio ES before versus correct-audio ES after. Correct ES >=5% lower than BOTH donor and contiguous within-track-shift controls; paired canonical-song95%bootstrap CI for improvement excludes0.'},
                {'diagnostic_id':'motion-native-flow-vs-regression-energy-score','direction':'down','magnitude_note':'Secondary matched trained regression correct-audio ES before versus flow correct-audio ES after. Must not replace primary audio-control comparison.'}],
            gates=[{'id':g,'description':'Independent frozen gate; no seed1, TEST or promotion unless all pass.'}
                   for g in ('representation_prerequisite','conditioning_magnitude','conditioning_ci','continuity','generated_quality','visual_review','integrity')],
            plan=str(ROOT)+'/music. Freeze source, inputs, exact32sVAL windows/donors/shifts and schedule before training. Scoring is a fixed physical18-channel summary at2Hz, TRAIN-normalized, no learned scoring head: speeds,regional energies,expansion,height,lean,extension and contacts. Same final-step architecture/crops/initial weights and optimizer runtime across arms. Separate real untrained states. Source/native139 has no training seed. Lighting experiments remain independently gated.',
            ambition_class='mechanism')
    result = {'hypothesis_id':HYPOTHESIS,'vault':'/root/lightsem-vault',
        'native_acceptance_sha256':digest(ROOT/'codec/final_acceptance.json'),'source_sha256':digest(__file__),
        'recall':store.recall(problem_id=PROBLEM)}
    authenticated_json(path,result)
    print(HYPOTHESIS)
    return result


if __name__=='__main__':
    main()
