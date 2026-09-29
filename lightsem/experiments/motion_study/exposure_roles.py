"""Distinct fixed-role optical revision; original ramps remain immutable."""
import argparse
import json
import sys
from pathlib import Path

from .registration import PROBLEM
from ..stage11.stage11_common import authenticated_json,read_authenticated,digest,verify_sources,gpu_lock

ROOT = Path('/mnt/lightsem-data/experiments/lightsem/motion_native_audio_20260910')
OUT = ROOT/'exposure_roles'
EID = 'exp-motion-exposure-roles-20260910'
HID = 'h-motion-exposure-roles-20260910'
DID = 'motion-render-dimmer-response'


def register():
    if (OUT/'experiment.json').exists():return read_authenticated(OUT/'experiment.json')
    p=read_authenticated(OUT/'protocol.json')
    sys.path.insert(0,'/root/vibescience-mcp/src')
    from vibescience_mcp.core import Store
    store=Store('/root/lightsem-vault')
    iid='motion-fixed-role-radiance-20260910'
    store.register_intervention(id=iid,name=iid,description='Fixed exposure0.4 with static beam/wash/flash/projector radiance gains1/.3/.03/.2. No time-varying intensity normalization; legacy default unchanged.')
    if not store.vault.exists('hypotheses',HID):
        store.propose_hypothesis(id=HID,problem_id=PROBLEM,
            statement='Fixed role radiance restores dimmer response while retaining single-beam visibility.',
            rationale='The completed scalar-only exposure study failed single-beam visibility; increasing the scalar compressed whole-rig response. Static role gains separate these competing radiance requirements.',
            interventions=[iid],predicted_effects=[{'diagnostic_id':DID,'direction':'up','magnitude_note':'Fixed primary whole-rig80/20 bulb response must reach2.0. Every original gate remains unchanged, including single-head nonbulb increment>=.002. No selection of a favorable primary after seeing outcomes.'}],
            gates=[{'id':g,'description':'Original optical gate applied across all three scenes.'} for g in ('ramp_monotonicity','dynamic_range','quiet_visibility','clipping','beam_visibility','integrity')],
            plan=str(OUT/'protocol.json'),ambition_class='mechanism')
    if not store.vault.exists('experiments',EID):
        store.start_experiment(id=EID,hypothesis_id=HID,parent_experiment_id='exp-motion-exposure-20260910',external_run=str(OUT),config_note=json.dumps(p))
    result={'experiment_id':EID,'hypothesis_id':HID,'protocol_sha256':digest(OUT/'protocol.json')}
    authenticated_json(OUT/'experiment.json',result)
    return result


def close():
    verify_sources(OUT)
    p=read_authenticated(OUT/'protocol.json');r=read_authenticated(OUT/'report.json')
    if r['protocol_sha256']!=digest(OUT/'protocol.json'):raise ValueError('Protocol changed')
    candidate=r['candidates'][0]
    gates={k:all(s['gates'][k] for s in candidate['scenes'].values()) for k in candidate['scenes']['whole_rig']['gates']}
    gates['integrity']=r['default_pixel_max_abs']==0
    sys.path.insert(0,'/root/vibescience-mcp/src')
    from vibescience_mcp.core import Store
    store=Store('/root/lightsem-vault');exp=store.vault.read_experiment(EID)
    if not exp.closed:
        store.record_diagnostics(EID,[{'diagnostic_id':DID,'before':r['legacy_response'],'after':candidate['scenes']['whole_rig']['response']}])
        store.record_gate_results(EID,[{'gate_id':k,'passed':v,'evidence':json.dumps(candidate)} for k,v in gates.items()])
        store.close_experiment(EID,notes='Fixed primary measured regardless of eligibility. Optical qualification is separate from native-music failure and full-track visual acceptance.',artifacts=[str(OUT/'report.json')],key_numbers={'before':r['legacy_response'],'after':candidate['scenes']['whole_rig']['response'],'qualified':int(all(gates.values()))})
        exp=store.vault.read_experiment(EID)
    result={'verdict':exp.verdict.value,'gates':gates,'qualified':all(gates.values()),'report_sha256':digest(OUT/'report.json'),
            'calibration':store.calibration(diagnostic_id=DID),'causal_map':store.causal_map(problem_id=PROBLEM),'recall':store.recall(problem_id=PROBLEM)}
    authenticated_json(OUT/'vibescience_results.json',result)
    print(json.dumps({'verdict':result['verdict'],'gates':gates}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','register','run','close']);args=parser.parse_args()
    if args.stage=='register':register()
    elif args.stage=='close':close()
    else:
        from . import exposure_study as study
        study.OUT=OUT;study.ROLE_REVISION=True
        if args.stage=='freeze':study.freeze()
        else:
            with gpu_lock(ROOT):study.run()
