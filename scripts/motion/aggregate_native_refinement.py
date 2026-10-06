"""Original repair gates, paired continuation integrity and frozen selection rule."""
from pathlib import Path
from collections import defaultdict
import argparse,json
import numpy as np
from lightsem.experiments.native_repair.assessment import pair_metrics,raw_metrics
from lightsem.experiments.native_repair.reporting import assess_pair
from lightsem.experiments.dynamic_dance.contracts import DanceResult
from lightsem.experiments.dynamic_dance_combo.campaign import save,sha

def aggregate(root):
    root=Path(root);old=root.parent;first=old/'repair_v1';recipes={};identities=[]
    for seed in (0,1):
        pair=[json.loads((root/k/'learning/runs/joint'/f'seed_{seed}'/'REPORT.json').read_text()) for k in ('continued','strong')]
        a,b=pair;checks={k:a[k]==b[k] for k in ('steps','exposure_sha256','counts','valid_frames')};checks['initial_arrays']=a['identity']['initial_model_array_sha256']==b['identity']['initial_model_array_sha256'];checks['parent_identity']=a['identity']['initial']==b['identity']['initial']
        if not all(checks.values()) or a['steps']!=json.loads((root/'REFINEMENT_PROTOCOL.json').read_text())['phase_updates']:raise ValueError('Unmatched continuation: '+str(checks))
        identities.append({'seed':seed,'checks':checks})
    for recipe in ('continued','strong'):
        r=root/recipe;rows=[];dev=[]
        for seed in (0,1):
            suffix='' if seed==0 else '_seed1'
            for song in ('shpongle','cure_for_me'):
                cp=r/'bodies'/('C4'+suffix)/song/'body.npz';c=DanceResult.load(cp);a=DanceResult.load(old/'bodies'/('D7_reference'+suffix)/song/'body.npz');historical=DanceResult.load(first/'bodies/D7_reference'/song/'body.npz')
                for window,s in [('full_song',np.ones(len(c.time),bool)),('90_150',(c.time>=90)&(c.time<150))]:
                    m=pair_metrics(c.world[s],a.world[s]);g=assess_pair(m)
                    if seed==0:g['historical_old_D7_jerk']=m['candidate']['raw_relative_jerk_p95_m_s3']<=1.25*raw_metrics(historical.world[s])['raw_relative_jerk_p95_m_s3']
                    rows.append({'seed':seed,'song':song,'window':window,**m,'gates':g,'all_pass':all(g.values()),'body_sha256':sha(cp)})
            ep=r/'evaluation'/f'seed_{seed}'/'REPORT.json';e=json.loads(ep.read_text());groups=defaultdict(list)
            for x in e['rows']:groups[x['domain'],x['group']].append(x)
            fd=[];jd=[]
            for (domain,group),items in groups.items():
                if domain=='fd':fd.append(float(np.mean([x['raw3d']['ratios']['raw_relative_jerk_p95_m_s3'] for x in items])))
                else:jd.append((float(np.mean([x['profile_error'] for x in items])),float(np.mean([x['baseline_profile_error'] for x in items]))))
            f=float(np.mean(fd));j=float(np.mean([x[0] for x in jd])/np.mean([x[1] for x in jd]));g={'FD_raw_jerk_macro':f<=1.25,'JD_available_profile_macro':j<=1.05}
            dev.append({'seed':seed,'FD_jerk_ratio':f,'JD_available_profile_ratio':j,'FD_groups':len(fd),'JD_groups':len(jd),'gates':g,'all_pass':all(g.values()),'report_sha256':sha(ep)})
        recipes[recipe]={'showcases':rows,'development':dev,'all_numeric_gates_pass':all(x['all_pass'] for x in rows+dev),'primary_worst_fast_residual_ratio':max(x['ratios']['high_frequency_residual_RMS_m'] for x in rows)}
    selected='continued' if recipes['continued']['all_numeric_gates_pass'] else 'strong' if recipes['strong']['all_numeric_gates_pass'] else None
    result={'recipes':recipes,'paired_training':identities,'selected_recipe':selected,'selection_rule':'Precommitted continued-first; bothseeds/alloriginalgates required; alwaysdisplayseed0.','status':'numeric_gates_pass_source_and_media_checks_pending' if selected else 'failure_diagnostic','baseline_promoted':False,'no_posthoc_output_processing':True}
    save(root/'reports/SELECTION.json',result);return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();print(json.dumps(aggregate(a.root),indent=2))
