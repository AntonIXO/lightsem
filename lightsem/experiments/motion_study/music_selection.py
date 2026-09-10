"""Magnitude, uncertainty, generated quality and external visual gates stay separate."""
import argparse
import json

import numpy as np

from .audio_prepare import ROOT
from .music_training import context
from ..stage11_common import read_authenticated,authenticated_json,digest


def comparison(per_song,control,config):
    values = np.asarray([[per_song[g][control+'_es'],per_song[g]['correct_es']] for g in sorted(per_song)],dtype=np.float64)
    before,after = values.mean(0)
    rng = np.random.default_rng(config['bootstrap_seed'])
    means = values[rng.integers(len(values),size=(config['bootstrap_draws'],len(values)))].mean(1)
    relative = (means[:,0]-means[:,1])/np.maximum(means[:,0],1e-12)
    return {'before':float(before),'after':float(after),'relative_improvement':float((before-after)/max(before,1e-12)),
            'relative_ci95':np.quantile(relative,[.025,.975]).tolist(),'canonical_groups':len(values),
            'method':'paired canonical-song percentile bootstrap, equal recording/window reduction before resampling'}


def numerical(report,full,p):
    m = report['summary']['metrics']
    g = p['gates']
    comparisons = {c:comparison(report['summary']['per_song'],c,p['evaluation']) for c in ('donor','shift')}
    energy = m['prediction_kinetic']/max(m['source_kinetic'],1e-8)
    travel = m['prediction_travel']/max(m['source_travel'],1e-8)
    coverage = m['prediction_contact_coverage']/max(m['source_contact_coverage'],1e-8)
    quality = {'kinetic_ratio':energy,'travel_ratio':travel,'contact_coverage_ratio':coverage,
        'penetration_excess':max(0,m['prediction_penetration']-m['source_penetration']),
        'sliding_excess':max(0,m['prediction_sliding']-m['source_sliding']),
        'high_frequency_excess':max(0,m['prediction_high_frequency']-m['source_high_frequency'])}
    gates = {'representation_prerequisite':read_authenticated(ROOT/'codec/final_acceptance.json')['passed'],
        'conditioning_magnitude':all(v['relative_improvement']>=g['conditioning_relative_improvement_min'] for v in comparisons.values()),
        'conditioning_ci':all(v['relative_ci95'][0]>0 for v in comparisons.values()),
        'continuity':full['chunk_max_abs']<=g['chunk_max_abs'] and quality['high_frequency_excess']<=g['high_frequency_fraction_excess_max'],
        'generated_quality':g['generated_energy_ratio'][0]<=energy<=g['generated_energy_ratio'][1] and
            g['generated_root_travel_ratio'][0]<=travel<=g['generated_root_travel_ratio'][1] and
            g['generated_contact_coverage_ratio'][0]<=coverage<=g['generated_contact_coverage_ratio'][1] and
            quality['penetration_excess']<=g['floor_penetration_p95_excess_max_m'] and
            quality['sliding_excess']<=g['contact_sliding_excess_max_m_s'],
        'integrity':True}
    return gates,comparisons,quality


def select(seed=0):
    p,_,_,_ = context()
    rows = []
    for arm in p['arms']:
        base = ROOT/'music/evaluation'/arm/f'seed_{seed}'
        initial = read_authenticated(base/'initial/report.json')
        final = read_authenticated(base/'final/report.json')
        full = read_authenticated(ROOT/'music/full_sequences'/arm/f'seed_{seed}/report.json')
        visual = read_authenticated(ROOT/'music/reviews'/f'{arm}_seed_{seed}.json')
        if visual['evaluation_sha256']!=digest(base/'final/report.json'):
            raise ValueError('Review belongs to another evaluation')
        gates,comparisons,quality = numerical(final,full,p)
        gates['visual_review'] = visual['passed']
        rows.append({'arm':arm,'gates':gates,'eligible':all(gates.values()),'controls':comparisons,'quality':quality,
                     'initial':initial['summary']['metrics'],'final':final['summary']['metrics'],
                     'evaluation_sha256':digest(base/'final/report.json')})
    flow = next(r for r in rows if r['arm']=='flow')
    result = {'rows':rows,'selected':'flow' if flow['eligible'] else None,'flow_qualified':flow['eligible'],
              'eligible_arms':[r['arm'] for r in rows if r['eligible']],
              'protocol_sha256':digest(ROOT/'music/protocol.json'),'test_access':False,'confirmation_authorized':False,
              'note':'Pipeline lighting gates still required before seed1; no conditioning-only promotion.'}
    authenticated_json(ROOT/'music'/f'selection_seed_{seed}.json',result)
    print(json.dumps({'eligible':result['eligible_arms'],'flow_qualified':flow['eligible']}))
    return result


if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--seed',type=int,default=0)
    select(p.parse_args().seed)
