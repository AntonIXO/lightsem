"""Preregister the accepted overnight questions in the LightSem vault only."""
from pathlib import Path
import sys,json
from .night_common import ROOT,ARMS,SEEDS
from ..stage11.stage11_common import authenticated_json,read_authenticated,digest

STUDIES=[
 ('representation','night-pose-reconstruction-error','lower_better','torso_lengths',
  'A learned 24-dimensional pose code reconstructs observed 2D joints better than the hand-designed 24 motion/event descriptors.',
  'At least20% lower confidence-masked joint error, canonical-song95%CI excludes0; compare descriptor decoder widths128/256 and mean/donor controls. This is reconstruction, not audio-only deployment.',
  ['pose_error','decoder_capacity','confidence_masks','integrity']),
 ('rhythm','night-annotated-beat-f1','higher_better','fraction',
  'Local periodicity recovers annotated beats better than the incumbent constant tempo grid.',
  'GuitarSet358 microphone clips after two predefined timing exclusions: F1 gain>=.05, CMLt drop<=.02. Synthetic tempo/meter/syncopation/silence stress tests are separate engineering evidence.',
  ['beat_f1','continuity','real_annotations','integrity']),
 ('fourier','night-fourier-conditional-energy-score','lower_better','standardized_distance',
  'Local Fourier coordinates of the existing spectral-flux streams improve music-conditioned motion generation.',
  'At least5% lower conditional energy score versus Base at matched architecture/exposure; direction agrees in both training seeds. All independent output/conditioning gates remain necessary.',
  ['energy_score','both_seeds','conditioning','visible_motion','brightness','integrity']),
 ('muq','night-muq-conditional-energy-score','lower_better','standardized_distance',
  'The pinned released MuQ representation improves native conditional motion beyond the92-feature baseline.',
  'At least5% lower conditional energy score versus Base at matched architecture/exposure in both seeds; within-song and donor controls must exclude a song-mean-only benefit. Public MSD weights, not the closed160K-hour model.',
  ['energy_score','both_seeds','conditioning','visible_motion','brightness','integrity']),
 ('conditioning','night-audio-controlled-song-fraction','higher_better','fraction',
  'Repeated audio FiLM in the temporal flow increases meaningful audio control relative to single-entry conditioning.',
  'Increase>=.25 controlled-song fraction versus matched single-entry arm, absolute>=.75; response>=.15span and>=.5noise response, with independently correct-signed native alignment. FiLM adds a low-rank module; parameter count disclosed.',
  ['conditioning_effect','absolute_conditioning','native_alignment','both_seeds','integrity']),
 ('composition','night-expressive-group-clip-fraction','higher_better','fraction',
  'The fixed group-expression adapter yields expressive dimmers and distinguishable conditional fixture groups while preserving mirror geometry and visibility.',
  '>=.75 of8fixed clips: >=4heads dimmer p95-p05>=.4, actual off intervals; independent unrelated velocity |corr|<.85; mirror targets correct; physical/visible movement preserved. Shpongle ending border brightness>=30%lower under same renderer without a timer/fade.',
  ['brightness','group_independence','mirror_geometry','visible_motion','ending_glare','integrity'])]

def register(root=ROOT):
    root=Path(root);rp=root/'vibescience_registration.json'
    if rp.exists():return read_authenticated(rp)
    sys.path.insert(0,'/root/vibescience-mcp/src')
    from vibescience_mcp.core import Store
    store=Store('/root/lightsem-vault');rows=[]
    for suffix,diag,direction,unit,statement,bar,gates in STUDIES:
        intervention='night-expression-'+suffix+'-20260909'
        store.register_diagnostic(id=diag,name=diag,direction=direction,unit=unit,description=statement)
        store.register_intervention(id=intervention,name=intervention,description=statement+' '+bar)
        gate_defs=[{'id':g,'description':bar+' Gate component: '+g} for g in gates]
        h=store.propose_hypothesis(id='h-night-'+suffix+'-20260909',problem_id='dancing-light-generative-20260903',
            statement=statement,rationale='Accepted plan2026-09-09. Preserve prior directional_only and failed product gates. Separate representation sufficiency, conditional use, output geometry and emitted brightness. No TEST access.',
            interventions=[intervention],predicted_effects=[{'diagnostic_id':diag,'direction':'down' if direction=='lower_better' else 'up','magnitude_note':bar}],
            gates=gate_defs,plan='6cells x2seeds, oneGPU, finalscheduled checkpoints. '+str(root)+'. Performance-only common step calibration before scientific training. Diagnostic and main protocol/source hashes are attached before the corresponding execution.',ambition_class='mechanism')
        e=store.start_experiment(id='exp-night-'+suffix+'-20260909',hypothesis_id=h.id,external_run=str(root),
            config_note='Registered before diagnostic/model outcomes. Feature extraction and synthetic/TRAIN rig calibration are engineering only. Preregistered source/protocol manifests follow before scientific runs.',
            parent_experiment_id='exp-beat-conditioned-visible-motion-20260909')
        rows.append({'hypothesis_id':h.id,'experiment_id':e.id,'diagnostic_id':diag,'gates':gate_defs})
    result={'vault':'/root/lightsem-vault','rows':rows,'arms':list(ARMS),'seeds':list(SEEDS),'source_sha256':digest(__file__)}
    authenticated_json(rp,result);print(json.dumps(result));return result

if __name__=='__main__':register()
