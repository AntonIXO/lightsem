"""Pre-outcome scientific identity for the documented codec revision."""
import json
import sys
from pathlib import Path

from . import ROOT as PARENT
from .registration import PROBLEM
from ..stage11_common import authenticated_json, read_authenticated, digest

ROOT = Path('/mnt/lightsem-data/experiments/lightsem/motion_representation_phasecodec_20260910')
HYPOTHESIS = 'h-motion-native-phase-codec-20260910'


def main():
    if (ROOT/'registration.json').exists():
        print(json.dumps(read_authenticated(ROOT/'registration.json')))
        return
    selection = read_authenticated(PARENT/'codec/final_selection.json')
    if selection['eligible_arms']:
        raise RuntimeError('This revision requires failure of the full initial codec study')
    sys.path.insert(0,'/root/vibescience-mcp/src')
    from vibescience_mcp.core import Store
    store = Store('/root/lightsem-vault')
    intervention = 'motion-native-phase-codec-20260910'
    store.register_intervention(id=intervention,name=intervention,
        description='Four-native-frame orthonormal Haar transform, TRAIN-fitted linear projections per temporal band, explicit generated root/contact latent channels. No interpolation; fixed zero AC offset preserves static motion. Compare native frame24 PCA to phase64/128/256 rotation ranks, counting all side channels.')
    if not store.vault.exists('hypotheses',HYPOTHESIS):
        store.propose_hypothesis(id=HYPOTHESIS,problem_id=PROBLEM,
            statement='A phase-preserving linear codec with explicit root/contact channels can satisfy clean-motion fidelity gates where the 20k nonlinear temporal codec recipe failed.',
            rationale='All original codec arms miss velocity, full-track and physical fidelity. Root-oracle diagnostics show limb velocity errors persist; initial 16s quiet coverage was zero. This revision tests native temporal-detail preservation and separates analytically fitted compression from neural optimization. It is a compound engineering revision, not attribution of failure solely to interpolation or BF16.',
            interventions=[intervention],predicted_effects=[{'diagnostic_id':'motion-codec-velocity-nrmse','direction':'down',
                'magnitude_note':'Primary phase128 full-VAL velocity NRMSE versus the frozen original temporal128. Absolute <=.15, joint <=.02 body height, contact>=.90, energy/travel .9..1.1; physical and visual gates also required. Other ranks are prespecified selection candidates, never substituted as primary after outcomes.'}],
            gates=[{'id':g,'description':'Independent gate; all candidates retain original numerical limits, plus directly covered 2s quiet/stationary windows and native phase continuity.'}
                   for g in ('source_quality','joint_fidelity','velocity_fidelity','contacts','travel_energy','continuity','quiet_holds','visual_review','integrity')],
            plan=str(ROOT)+'. One analytic fit to identical full TRAIN records for every rank; equal canonical-song weights, no random training seed or checkpoint selection. Random orthogonal projection initialization is a descriptive control. All four arms evaluated before selection. Music training only after full qualification; no TEST. Frozen original temporal128 is historical baseline, not an exposure-matched optimization comparison.',
            ambition_class='mechanism')
    authenticated_json(ROOT/'registration.json',{'hypothesis_id':HYPOTHESIS,'vault':'/root/lightsem-vault',
        'parent_selection_sha256':digest(PARENT/'codec/final_selection.json'),'primary_arm':'phase128',
        'primary_baseline':'original temporal128','source_sha256':digest(__file__)})
    print(HYPOTHESIS)


if __name__ == '__main__':
    main()
