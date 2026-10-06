"""Frozen repair gates: raw stability AND activity, never smoothedscore alone."""
from pathlib import Path
import json,hashlib
from collections import defaultdict
import numpy as np
from ..dynamic_dance.contracts import DanceResult
from ..dynamic_dance_combo.campaign import save,sha
from .assessment import pair_metrics,raw_metrics

LIMITS={'raw_relative_jerk_p95_m_s3':1.25,'high_frequency_residual_RMS_m':1.25,'root_vertical_fast_residual_RMS_m':1.25}

def assess_pair(pair):
    a,b,r=pair['candidate'],pair['continued_A'],pair['ratios']
    checks={k:r[k]<=limit for k,limit in LIMITS.items()}
    checks.update(activity=r['raw_relative_speed_mean_m_s']>=.8,variation=r['low_frequency_speed_spread_m_s']>=.8,finite=a['finite'],not_constant=not a['constant_body'])
    av,bv=a['geometric_slide_p95_m_s'],b['geometric_slide_p95_m_s']
    checks['support_slide']=av is not None and bv is not None and av<=1.2*bv
    return checks

def aggregate(root):
    root=Path(root);old=root.parent;rows=[];identities=[]
    for seed in (0,1):
        rp=root/'learning/runs/joint'/f'seed_{seed}'/'REPORT.json';cp=old/'learning/runs/joint'/f'seed_{seed}'/'REPORT.json';a=json.loads(rp.read_text());b=json.loads(cp.read_text())
        same={k:a[k]==b[k] for k in ('steps','exposure_sha256','counts','valid_frames')};same['initial_arrays']=a['identity']['initial_model_array_sha256']==b['identity']['initial_model_array_sha256'];same['initial_checkpoint']=a['identity']['initial']==b['identity']['initial']
        if not all(same.values()):raise ValueError('Pairedtrainingexposure/init drift '+str(same))
        identities.append({'seed':seed,'matched_to_original_C':same,'candidate_report_sha256':sha(rp),'control_report_sha256':sha(cp)})
        alias='C4' if seed==0 else 'C4_seed1';base='D7_reference' if seed==0 else 'D7_reference_seed1'
        for song in ('shpongle','cure_for_me'):
            p=root/'bodies'/alias/song/'body.npz';candidate=DanceResult.load(p);reference=DanceResult.load(old/'bodies'/base/song/'body.npz');historical=DanceResult.load(root/'bodies/D7_reference'/song/'body.npz')
            for window,s in [('full_song',np.ones(len(candidate.time),bool)),('90_150',(candidate.time>=90)&(candidate.time<150))]:
                measured=pair_metrics(candidate.world[s],reference.world[s]);gates=assess_pair(measured)
                if seed==0:
                    h=raw_metrics(historical.world[s]);gates['historical_oldD7_jerk']=measured['candidate']['raw_relative_jerk_p95_m_s3']<=1.25*h['raw_relative_jerk_p95_m_s3']
                rows.append({'seed':seed,'song':song,'window':window,**measured,'gates':gates,'all_pass':all(gates.values()),'body_sha256':sha(p)})
    development=[]
    for seed in (0,1):
        report=json.loads((root/'evaluation'/f'seed_{seed}'/'REPORT.json').read_text());grouped=defaultdict(list)
        for row in report['rows']:grouped[row['domain'],row['group']].append(row)
        fd=[];jd=[]
        for (domain,group),items in grouped.items():
            if domain=='fd':fd.append(float(np.mean([x['raw3d']['ratios']['raw_relative_jerk_p95_m_s3'] for x in items])))
            else:
                a=np.mean([x['profile_error'] for x in items]);b=np.mean([x['baseline_profile_error'] for x in items]);jd.append((float(a),float(b)))
        fd_ratio=float(np.mean(fd));jd_ratio=float(np.mean([x[0] for x in jd])/np.mean([x[1] for x in jd]));checks={'FD_raw_jerk_macro':fd_ratio<=1.25,'JD_available_observation_profile_macro':jd_ratio<=1.05}
        development.append({'seed':seed,'FD_jerk_ratio':fd_ratio,'JD_available_profile_ratio':jd_ratio,'FD_groups':len(fd),'JD_groups':len(jd),'gates':checks,'all_pass':all(checks.values()),'report_sha256':sha(root/'evaluation'/f'seed_{seed}'/'REPORT.json'),'observational_scope':'Sameoriginal masks, availablecells descriptive, no claim missingobservations aremeasured.'})
    worst=max(x['ratios']['raw_relative_jerk_p95_m_s3'] for x in rows);passed=all(x['all_pass'] for x in rows+development)
    result={'status':'raw_stability_and_activity_gates_pass' if passed else 'failure_diagnostic','all_required_gates_pass':passed,'primary_worst_raw_jerk_ratio':worst,'showcases':rows,'development':development,'training_identity':identities,'baseline_promoted':False,'musical_or_artistic_acceptance':'User review; these gates establish engineeringrepair only. No musical superiority inferred from rawregularity.'}
    save(root/'reports/REPAIR_VERDICT.json',result);return result
