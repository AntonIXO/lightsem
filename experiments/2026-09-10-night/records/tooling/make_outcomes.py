"""Compute scientific effect sizes and frozen gate outcomes after independent audits."""
from pathlib import Path
import itertools,json,sys
import numpy as np
ROOT=Path('/mnt/lightsem-data/experiments/lightsem/night_expression_20260909')
sys.path.insert(0,str(ROOT/'source'))
from lightsem.experiments.night_common import ARMS,SEEDS
from lightsem.experiments.stage11_common import read_authenticated,authenticated_json,digest

def boot(values):
    x=np.asarray(values,float)
    if not len(x) or not np.isfinite(x).all():raise ValueError('Finite paired observations required')
    rng=np.random.default_rng(991);means=x[rng.integers(len(x),size=(10000,len(x)))].mean(1)
    return {'mean':float(x.mean()),'ci95':np.quantile(means,[.025,.975]).tolist(),'songs':len(x)}

def physical(v):
    b,s=v['beam'],v['screen']
    return b['lit_motion_fraction']>=.5 and b['span_2s_median_deg']>=8 and b['upward_light_fraction'] is not None and b['upward_light_fraction']<=.01 and b['direction_hf_above4hz_fraction']<=.15 and s['visible_motion_frame_fraction']>=.5 and s['mean_brightness']>=.002 and s['support_fraction']>=.01

def expressive(v):
    return v['composition']['heads_with_span_ge_point4']>=4 and v['off_intervals']['heads_with_off_at_least_150ms']>=2

def compute():
    audit_names=['independent_input_audit','independent_pose_audit','independent_rhythm_audit','independent_evaluation_audit','independent_video_audit']
    audits={name:read_authenticated(ROOT/'diagnostics'/(name+'.json')) for name in audit_names}
    if not all(r['passed'] for r in audits.values()):raise RuntimeError('Independent audit failed')
    training_audit=read_authenticated(ROOT/'diagnostics/independent_training_audit.json')
    if not training_audit['complete'] or not training_audit['passed_for_completed_cells'] or len(training_audit['rows'])!=12:raise RuntimeError('Incomplete audited training')
    pose=read_authenticated(ROOT/'diagnostics/supplementary_pose.json');rhythm=read_authenticated(ROOT/'rhythm/report.json')
    main=read_authenticated(ROOT/'evaluation/report.json');cells={(r['arm'],r['seed']):r for r in main['cells']}
    inf={(a,s):read_authenticated(ROOT/'evaluation'/a/f'seed_{s}'/'inference_report.json') for a,s in itertools.product(ARMS,SEEDS)}
    outcomes={};before=[];after=[]
    for seed in SEEDS:
        before.append(pose['summary'][f'descriptor256/seed_{seed}']['full']['joint_error_tl']['mean'])
        after.append(pose['summary'][f'pose_code24/seed_{seed}']['full']['joint_error_tl']['mean'])
    representation_details={'descriptor256_means':before,'pose_code24_means':after,'paired_by_seed':pose['relative_pose_code_improvement_over_descriptor256'],
        'descriptor128_means':[pose['summary'][f'descriptor128/seed_{s}']['full']['joint_error_tl']['mean'] for s in SEEDS]}
    representation_gates={'pose_error':all(v['mean']>=.2 and v['ci95'][0]>0 for v in pose['relative_pose_code_improvement_over_descriptor256'].values()),
        'decoder_capacity':all(after[i]<=.8*min(before[i],representation_details['descriptor128_means'][i]) for i in SEEDS),
        'confidence_masks':all(r['valid_joints']>0 and r['cases']==40 for r in audits['independent_pose_audit']['rows']),'integrity':True}
    outcomes['representation']={'before':float(np.mean(before)),'after':float(np.mean(after)),'gates':representation_gates,'details':representation_details,
        'lesson':'The learned24-dimensional pose-conditioned code retains substantially more detected2Dgeometry than the fixed descriptors at both tested decoder widths. This is an oracle reconstruction result; direct audio-to-pose probes did not establish deployable recovery of a specific choreography.'}
    outcomes['rhythm']={'before':rhythm['means']['constant_grid']['f1'],'after':rhythm['means']['plp']['f1'],'gates':rhythm['gates'],
        'details':{'f1_delta':rhythm['f1_delta'],'cmlt_delta':rhythm['cmlt_delta'],'means':rhythm['means']},
        'lesson':'The fixed PLP recipe underperforms the constant grid on this358-clip GuitarSet benchmark despite succeeding on some synthetic tempo changes. Keep local periodicity as a representation/control question; do not replace the rhythm tracker on this evidence or generalize the failure to all local methods.'}
    local=read_authenticated(ROOT/'diagnostics/muq_local_controls/report.json')
    if len(local['cells'])!=4 or {(r['arm'],r['seed']) for r in local['cells']}!={(a,s) for a,s in itertools.product(('muq_once','muq_film'),SEEDS)}:raise ValueError('Incomplete local MuQ matrix')
    for family in ('fourier','muq'):
        comparisons=[]
        for architecture,seed in itertools.product(('once','film'),SEEDS):
            a=family+'_'+architecture;b='base_'+architecture
            original={r['canonical_id']:r['metrics']['energy_score']['full'] for r in inf[(b,seed)]['rows']}
            candidate={r['canonical_id']:r['metrics']['energy_score']['full'] for r in inf[(a,seed)]['rows']}
            if set(original)!=set(candidate) or len(original)!=40:raise ValueError('Unmatched feature comparison')
            relative=boot([(original[k]-candidate[k])/original[k] for k in sorted(original)])
            comparisons.append({'architecture':architecture,'seed':seed,'before':float(np.mean(list(original.values()))),'after':float(np.mean(list(candidate.values()))),
                'relative_improvement_macro_song':relative,'relative_improvement_of_means':float((np.mean(list(original.values()))-np.mean(list(candidate.values())))/np.mean(list(original.values()))),
                'main_gates':cells[(a,seed)]['gates']})
        gates={'energy_score':all(r['relative_improvement_of_means']>=.05 for r in comparisons),
            'both_seeds':all(r['after']<r['before'] for r in comparisons),
            'conditioning':all(all(r['main_gates'][k] for k in ('conditioning','native_alignment','local_alignment','event_selectivity')) for r in comparisons),
            'visible_motion':all(r['main_gates']['visible_motion'] for r in comparisons),'brightness':all(r['main_gates']['brightness'] for r in comparisons),'integrity':True}
        details={'comparisons':comparisons}
        if family=='muq':
            details['centered_embedding_controls']=[{'arm':r['arm'],'seed':r['seed'],'advantages':r['centered_energy_advantages']} for r in local['cells']]
            local_supported=all(v['mean']>0 and v['ci95'][0]>0 for r in local['cells'] for v in r['centered_energy_advantages'].values())
            gates['conditioning']=bool(gates['conditioning'] and local_supported)
            details['within_song_mean_only_explanation_excluded']=local_supported
            details['centering_interpretation']='Operationalization fixed before any main candidate inference: both centered-versus-mean-only and centered-versus-centered-donor score advantages must have positive95%canonical-song bootstrap lower bounds in all four MuQ cells to establish the already-committed local-use claim. These distribution-shifting controls can leave the mechanism unresolved; they do not change main candidate selection.'
        outcomes[family]={'before':float(np.mean([r['before'] for r in comparisons])),'after':float(np.mean([r['after'] for r in comparisons])),
            'gates':gates,'details':details,'lesson':f'Inspect every matched {family}/Base comparison across both architectures and seeds. A lower joint trajectory score alone does not establish musical correspondence, local embedding use, expressive brightness or product acceptance; all frozen gates and supplementary controls are reported separately.'}
    comparisons=[]
    for family,seed in itertools.product(('base','fourier','muq'),SEEDS):
        b=cells[(family+'_once',seed)];a=cells[(family+'_film',seed)]
        comparisons.append({'family':family,'seed':seed,'before':b['controlled_song_fraction'],'after':a['controlled_song_fraction'],
            'change':a['controlled_song_fraction']-b['controlled_song_fraction'],'film_gates':a['gates']})
    gates={'conditioning_effect':all(r['change']>=.25 for r in comparisons),'absolute_conditioning':all(r['after']>=.75 for r in comparisons),
        'native_alignment':all(r['film_gates']['native_alignment'] and r['film_gates']['local_alignment'] for r in comparisons),
        'both_seeds':all(r['change']>0 for r in comparisons),'integrity':True}
    outcomes['conditioning']={'before':float(np.mean([r['before'] for r in comparisons])),'after':float(np.mean([r['after'] for r in comparisons])),
        'gates':gates,'details':{'comparisons':comparisons},'lesson':'Repeated FiLM changes how audio enters each flow block and adds parameters; the matched factorial comparison tests this recipe rather than a pure parameter-count intervention. Swap response must be accompanied by correct-audio alignment and preserved native/output quality.'}
    inc=read_authenticated(ROOT/'evaluation/incumbent_adapter/report.json');obs=read_authenticated(ROOT/'delivery/shpongle/observation_report.json')
    details=[]
    mirror={r['key']:r for r in audits['independent_video_audit']['incumbent_mirror']}
    for r in inc['rows']:
        independent=r['independent']['unrelated_velocity_abs_correlation'];old=r['modes']['before'];new=r['modes']['after']
        oldcorr=old['composition']['unrelated_velocity_abs_correlation']
        details.append({'key':r['key'],'before_expressive':bool(expressive(old) and physical(old) and oldcorr is not None and oldcorr<.85),
            'after_expressive':bool(expressive(new) and physical(new) and independent is not None and independent<.85),
            'brightness':bool(expressive(new)),'visible_motion':bool(physical(new)),'independent_correlation':independent,
            **mirror[r['key']]})
    if len(details)!=8:raise ValueError('Incomplete composition comparison')
    gates={'brightness':np.mean([r['brightness'] for r in details])>=.75,'group_independence':np.mean([r['independent_correlation'] is not None and r['independent_correlation']<.85 for r in details])>=.75,
        'mirror_geometry':all(r['mirror_ray_error_deg']<.05 and r['mirror_target_error_m']<1e-8 for r in details),
        'visible_motion':np.mean([r['visible_motion'] for r in details])>=.75,
        'ending_glare':audits['independent_video_audit']['incumbent_ending_border_luma_reduction']>=.3,'integrity':True}
    gates={k:bool(v) for k,v in gates.items()}
    outcomes['composition']={'before':float(np.mean([r['before_expressive'] for r in details])),'after':float(np.mean([r['after_expressive'] for r in details])),
        'gates':gates,'details':{'clips':details,'ending_border_luma_reduction':audits['independent_video_audit']['incumbent_ending_border_luma_reduction'],
            'delivered_demo_border_luma_reduction':obs['border_luma_reduction'],'demo_uses_incumbent':obs['uses_incumbent']},
        'lesson':'The same incumbent checkpoint isolates the adapter change from retraining. Several conditional draws and a fixed geometric/brightness policy are not a learned fixture-level lighting director; engineering gate outcomes do not override failed musical-control gates or establish human aesthetic preference.'}
    for name,r in outcomes.items():r['all_gates_pass']=all(r['gates'].values())
    result={'studies':outcomes,'chosen_arm':main['chosen_arm'],'source_sha256':digest(__file__),'interpretation':'Effect-size and magnitude decisions, not hand-asserted Vibescience verdicts. Component regressions are retained. Main candidate selection remains the frozen main evaluation decision.'}
    authenticated_json(ROOT/'delivery/scientific_outcomes.json',result);return result

if __name__=='__main__':compute()
