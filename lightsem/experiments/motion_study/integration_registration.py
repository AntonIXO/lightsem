"""Preregister the requested incumbent8-versus32-step diagnostic."""
import sys
from pathlib import Path

from .registration import PROBLEM
from ..stage11.stage11_common import read_authenticated,authenticated_json,digest

ROOT = Path('/mnt/lightsem-data/experiments/lightsem/motion_native_audio_20260910')


def main():
    path = ROOT/'music/review/integration_experiment.json'
    if path.exists():
        return read_authenticated(path)
    sys.path.insert(0,'/root/vibescience-mcp/src')
    from vibescience_mcp.core import Store
    store = Store('/root/lightsem-vault')
    did = 'motion-incumbent-pose-high-frequency'
    intervention = 'motion-incumbent-euler32-20260910'
    hid = 'h-motion-incumbent-euler32-20260910'
    store.register_diagnostic(id=did,name=did,direction='lower_better',unit='fraction',
        description='Full-sequence COCO2D pose coordinate AC spectral power above6Hz at40Hz, averaged equally by canonical song; same incumbent checkpoint, audio and noise seed101.')
    store.register_intervention(id=intervention,name=intervention,
        description='Increase Euler integration from8 to32 steps without retraining the incumbent BeatConditionedModel or descriptor256 pose decoder. Same full-track audio/noise; twelve fixed FineDance review songs, no TEST.')
    if not store.vault.exists('hypotheses',hid):
        store.propose_hypothesis(id=hid,problem_id=PROBLEM,
            statement='More accurate integration of the incumbent flow reduces excess high-frequency pose motion while retaining active movement.',
            rationale='The accepted plan requests a separate8-versus32-step check to distinguish numerical integration from learned dynamics. Directional movement is not enough: a tiny change does not explain the visible jitter complaint.',
            interventions=[intervention],predicted_effects=[{'diagnostic_id':did,'direction':'down',
                'magnitude_note':'At least10% lower high-frequency power fraction and paired song95%bootstrap CI excludes0; retain80–125% of8-step total pose-velocity energy. Full12 prespecified review recordings, equal canonical weights, no example selection.'}],
            gates=[{'id':g,'description':'Independent diagnostic gate, not model-promotion authorization.'} for g in ('paired_inputs','magnitude','bootstrap','active_motion_retained','integrity')],
            plan=str(ROOT)+'/music/review. Both poses are shown in the incumbent column of every music review.2D output is not metrically equivalent to the new fixed-bone3D representation. No retraining, no second seed and no TEST.',
            ambition_class='mechanism')
    eid = 'exp-motion-incumbent-euler32-20260910'
    if not store.vault.exists('experiments',eid):
        store.start_experiment(id=eid,hypothesis_id=hid,parent_experiment_id='exp-motion-native-audio-20260910',
            external_run=str(ROOT/'music/review'),config_note='Registered before generating32-step incumbent outputs; exact renderer/cohort/checkpoint manifest must freeze before execution. Main music training is independent.')
    result = {'experiment_id':eid,'hypothesis_id':hid,'diagnostic_id':did,'source_sha256':digest(__file__),
              'bootstrap_seed':20260910,'bootstrap_draws':10000,'minimum_relative_reduction':.10,'velocity_energy_ratio':[.8,1.25]}
    authenticated_json(path,result)
    return result


if __name__=='__main__':
    main()
