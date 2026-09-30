#!/usr/bin/env python3
"""Complete7x3 paired-family H3 proxy summary; no selected seed or TEST."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from lightsem.model.light_layers.visual import sha256,write_json
from lightsem.experiments.visual_next.features import ARMS
from lightsem.experiments.visual_next.instruments import paired_summary


def holm(values):
    names=sorted(values,key=lambda x:values[x]);result={};previous=0.
    for i,name in enumerate(names):
        previous=max(previous,min(1.,(len(names)-i)*values[name]));result[name]=previous
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--freeze',required=True,type=Path)
    args=ap.parse_args();f=json.loads(args.freeze.read_text())
    for p,d in f['files'].items():assert sha256(p)==d,p
    qualification_ref=f['representation_qualification']
    if f['files'].get(qualification_ref['path'])!=qualification_ref['sha256']:
        raise RuntimeError('Representation qualification is not frozen with the aggregate')
    qualification=json.loads(Path(qualification_ref['path']).read_text())
    if qualification['status']!='controlled_retention_probes_complete':
        raise RuntimeError('Missing completed representation qualification')
    cells={};anchors=[]
    for item in f['reports']:
        report=json.loads(Path(item['path']).read_text())
        if report['status']!='all31historicalDEV_conditions_scored' or len(report['rows'])!=31:raise RuntimeError('Incomplete evaluation')
        ck=report['checkpoint'];rows={r['canonical_id']:r for r in report['rows']}
        if len(rows)!=31:raise RuntimeError('Duplicate family')
        if ck['role']=='final':
            key=(ck['arm'],ck['seed'])
            if key in cells:raise RuntimeError('Duplicate study cell')
            cells[key]=rows
        else:anchors.append({'checkpoint':ck,'rows':rows})
    if set(cells)!={(arm,seed) for arm in ARMS for seed in (0,1,2)}:raise RuntimeError('All21 fixed study cells required')
    first=cells[('R0',0)];families=sorted(first)
    for rows in cells.values():
        if set(rows)!=set(first):raise RuntimeError('Family membership drift')
        for cid in families:
            if rows[cid]['reference_hash']!=first[cid]['reference_hash']:raise RuntimeError('Reference/clock drift across arms')
    eligible=[cid for cid in families if first[cid]['source_coverage']>=.8]
    events=[cid for cid in eligible if first[cid]['scores']['model']['average_precision'] is not None]
    def mean_score(arm,cid,condition='model',common=False):
        key='same_shift_support_scores' if common else 'scores'
        values=[cells[(arm,s)][cid][key][condition]['average_precision'] for s in (0,1,2)]
        if any(v is None for v in values):return None
        return float(np.mean(values))
    contrasts={}
    for name,other in [('R3_minus_R0','R0'),('R3_minus_Rrand','Rrand'),('R3_minus_R2','R2'),
                       ('R3_minus_shuffle','R3-shuffle'),('R3_minus_reverse','R3-reverse'),('R1_minus_R0','R0')]:
        arm='R1' if name=='R1_minus_R0' else 'R3'
        contrasts[name]=paired_summary([mean_score(other,c) for c in events],[mean_score(arm,c) for c in events]) if len(events)>=2 else None
    control_names=['R3_minus_Rrand','R3_minus_R2','R3_minus_shuffle','R3_minus_reverse']
    adjusted=holm({k:contrasts[k]['p'] for k in control_names if contrasts[k] is not None})
    qualified_null=[c for c in eligible if first[c]['shift_plan']['qualified'] and first[c]['shift_plan']['shift_eligible_duration_s']/64>=.8]
    null_events=[c for c in qualified_null if first[c]['same_shift_support_scores'] and first[c]['same_shift_support_scores']['model']['average_precision'] is not None]
    audio_controls={}
    for condition in ('beat_only','donor','shift'):
        before=[];after=[]
        for cid in null_events:
            real=mean_score('R3',cid,common=True)
            if condition=='shift':
                vals=[score['average_precision'] for seed in (0,1,2) for score in cells[('R3',seed)][cid]['shift_scores']]
                if len(vals)!=60 or any(v is None for v in vals):raise RuntimeError('Missing paired20shift outputs')
                control=float(np.mean(vals))
            else:control=mean_score('R3',cid,condition,common=True)
            before.append(control);after.append(real)
        audio_controls[condition]=paired_summary(before,after) if len(before)>=2 else None
    audio_holm=holm({k:v['p'] for k,v in audio_controls.items() if v is not None})
    nonevents=[c for c in eligible if first[c]['reference_count']==0]
    false_rates={}
    for arm in ('R0','R3'):
        values=[cells[(arm,seed)][c]['scores']['model']['false_cues_per_minute'] for c in nonevents for seed in (0,1,2)]
        false_rates[arm]=float(np.mean(values)) if values else None
    primary=contrasts['R3_minus_R0']
    gates={'complete21':True,'source_support':len(eligible)>=30,'event_bearing_groups':len(events)>=10,
        'primary_magnitude':primary is not None and primary['benefit_mean']>=.03,
        'primary_interval_and_four_test_p':primary is not None and primary['benefit_ci95'][0]>0 and primary['four_test_bonferroni_p']<.05,
        'qualified_shift_support':len(qualified_null)>=30,
        'nonevent_false_cues':false_rates['R0'] is not None and false_rates['R3'] is not None and false_rates['R3']<=1.05*false_rates['R0']}
    for k in control_names:
        v=contrasts[k];gates[k]=v is not None and v['benefit_mean']>=.01 and v['benefit_ci95'][0]>0 and adjusted[k]<.05
    for k,v in audio_controls.items():
        gates['correct_audio_over_'+k]=v is not None and v['benefit_mean']>=.02 and v['benefit_ci95'][0]>0 and audio_holm[k]<.05
    result={'status':'complete21_cell_automatic_proxy_summary','freeze_sha256':sha256(args.freeze),
        'eligible_families':eligible,'event_bearing_families':events,'qualified_shift_families':qualified_null,
        'shift_event_families':null_events,'nonevent_families':nonevents,'nonevent_false_rates':false_rates,
        'contrasts':contrasts,'representation_control_Holm_p':adjusted,'audio_control_contrasts':audio_controls,
        'audio_control_Holm_p':audio_holm,'proxy_gate_results':gates,'all_numeric_proxy_gates_pass':all(gates.values()),
        'arm_means':{arm:float(np.mean([mean_score(arm,c) for c in events])) if events else None for arm in ARMS},
        'anchor_evaluations':anchors,'original_independent_content_AP':'unmeasured',
        'representation_qualification':qualification_ref,
        'representation_qualification_gates':qualification['gates'],
        'original_representation_fit_gate_pass':qualification['original_fit_gate_pass'],
        'diagnostic_only_due_to_failed_qualification':not qualification['original_fit_gate_pass'],
        'reverse_control':'Fixed feature-coordinate permutation; superiority is not an identifiable chronology test. Preserve the number without a causal temporal-order claim.',
        'formal_original_hypothesis_confirmed':False,'test_access':False,
        'scope':'Paired seeds averaged within canonical family; automatic observer64s endpoint on exposed historicalDEV. Original31/40 coverage and29/30 historical null failures remain unchanged.'}
    write_json(Path(f['output']),result)


if __name__=='__main__':main()
