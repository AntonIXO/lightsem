"""Expose coverage and matched-donor limitations of completed supplementary probes."""
from pathlib import Path
import sys,json
import numpy as np
ROOT=Path('/mnt/lightsem-data/experiments/lightsem/night_expression_20260909')
sys.path.insert(0,str(ROOT/'source'))
from lightsem.experiments.stage11_common import read_authenticated,authenticated_json,digest

def boot(values):
    x=np.asarray(values,float);rng=np.random.default_rng(991)
    if len(x)!=40 or not np.isfinite(x).all():raise ValueError('Forty finite canonical-song values required')
    bs=x[rng.integers(40,size=(10000,40))].mean(1)
    return {'mean':float(x.mean()),'ci95':np.quantile(bs,[.025,.975]).tolist(),'canonical_songs':40}

def main():
    path=ROOT/'diagnostics/supplementary_pose.json';report=read_authenticated(path);coverage={}
    reference=[r for r in report['rows'] if r['name']=='descriptor256' and r['seed']==0]
    for mode in ('motion_event_context','outside_motion_event_context','audio_onset_context','outside_audio_onset_context'):
        count=[r['extra']['full']['segments'][mode]['frames'] for r in reference]
        coverage[mode]={'minimum_frames':min(count),'median_frames':float(np.median(count)),'maximum_frames':max(count),
            'mean_fraction':float(np.mean(count)/2400),'median_fraction':float(np.median(count)/2400)}
    donors={}
    for name in ('descriptor256','pose_code24','audio_base','audio_fourier','audio_muq'):
        source=[r for r in report['rows'] if r['name']==name];by_song={}
        for r in source:by_song.setdefault(r['canonical_id'],[]).append(r)
        if any(len(v)!=2 for v in by_song.values()):raise ValueError('Both seeds required per song')
        rows=[]
        for key,rr in sorted(by_song.items()):
            a=np.mean([r['donor_control']['matched_full_error_tl'] for r in rr]);b=np.mean([r['donor_control']['saved_donor_prediction_error_tl'] for r in rr])
            rows.append({'canonical_id':key,'full_error_tl':float(a),'saved_donor_error_tl':float(b),'donor_minus_full_error_tl':float(b-a)})
        donors[name]={'rows':rows,'difference':boot([r['donor_minus_full_error_tl'] for r in rows]),
            'mean_prediction_std':float(np.mean([r['primary_errors']['full']['prediction_std'] for r in source])),
            'mean_truth_std':float(np.mean([r['primary_errors']['full']['truth_std'] for r in source]))}
    result={'coverage':coverage,'donor_comparisons':donors,'input_sha256':digest(path),'source_sha256':digest(__file__),
        'interpretation':'Exploratory coverage audit, not a new selection rule. Dense audio-onset neighborhoods cover nearly all frames and provide weak event/non-event separation; retain them without treating that comparison as positive evidence. Paired donor differences average the two seeds within canonical song before bootstrap; they test a saved conditional-mean probe, not an information-theoretic limit.'}
    authenticated_json(ROOT/'diagnostics/supplementary_quality_notes.json',result)
    print(json.dumps({'coverage':coverage,'audio_donor_differences':{k:v['difference'] for k,v in donors.items() if k.startswith('audio')}}))

if __name__=='__main__':main()
