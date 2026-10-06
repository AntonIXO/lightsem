"""Run individual immutable CPU stages of the dynamic-dance campaign."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re

for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_key,'1')

import numpy as np

from lightsem.experiments.dynamic_dance.campaign import (ROOT,METHODS,initialize,read,
    write_new,sha,verify_files,disk_guard,freeze_source)
from lightsem.experiments.dynamic_dance.contracts import MusicTimeline,DanceResult,DancePlan


def select_tracks(plan,song):
    result=[t for t in plan['tracks'] if song is None or t['id']==song]
    if not result:
        raise ValueError('Unknown song: '+str(song))
    return result


def source_baseline(root,track):
    from lightsem.experiments.dynamic_dance.mapping import baseline_body
    folder=root/'bodies/baseline'/track['id']
    if folder.exists():
        body=DanceResult.load(folder/'body.npz')
        if body.metadata['source_sha256']!=sha(track['body']):
            raise RuntimeError('Original body changed')
        return
    body=baseline_body(track)
    body.save(folder/'body.npz')
    DancePlan(time=body.time,source_time=body.time.copy(),effort=np.ones((len(body.time),5)),
              metadata={'method':'baseline','scope':'Identity timing of accepted per-song body'}).save(folder/'plan.npz')


def select_audio_intervals(root,track,timeline):
    """Freeze event observations before candidates, without motion-dependent selection."""
    from scipy.signal import find_peaks
    path=root/'records/audio_intervals'/(track['id']+'.json')
    if path.exists():
        prior=read(path)
        if prior['music_sha256']!=sha(track['music_timeline']):
            raise RuntimeError('Frozen audio-only interval input changed')
        from lightsem.experiments.dynamic_dance.evaluation import freeze_release_intervals
        freeze_release_intervals(root,track['id'])
        return prior
    t=timeline.time
    envelope=timeline.mixture[:,1]
    positive=envelope[envelope>0]
    threshold=float(np.quantile(positive,.75)) if len(positive) else 0.
    peaks,_=find_peaks(envelope,height=threshold,distance=round(.25*30),prominence=max(1e-12,threshold*.2))
    events=t[peaks]
    events=events[(events>=60)&(events<180)]
    boundaries=[float(s['end']) for s in timeline.sections if 60<=float(s['end'])<180]
    selection={'song':track['id'],'music_sha256':sha(track['music_timeline']),
        'review_start_s':60.,'review_end_s':180.,'attack_times':events.tolist(),
        'section_boundaries':boundaries,
        'rules':{'onset_positive_quantile':.75,'minimum_gap_s':.25,'relative_prominence':.2},
        'scope':'Audio-only automatic observations frozen before candidate review, not independent human semantic annotations.',
        'human_reference_status':'not_available','artistic_acceptance':'pending user review'}
    write_new(path,selection)
    from lightsem.experiments.dynamic_dance.evaluation import freeze_release_intervals
    freeze_release_intervals(root,track['id'])
    return selection


def generate(root,plan,method,track,control=None,music_path=None,output_id=None):
    from lightsem.experiments.dynamic_dance.mapping import baseline_body
    if method not in METHODS:
        raise ValueError('Unknown method '+method)
    if method in ('D6','D7','D8'):
        raise ValueError('Use train_dynamic_dance infer for learned models, never substitute a controller')
    identity=output_id or (method if control is None else method+'__'+control)
    if not re.fullmatch('[A-Za-z0-9_-]+',identity):
        raise ValueError('Unsafe output id')
    music_path=Path(music_path or track['music_timeline'])
    folder=root/'bodies'/identity/track['id']
    pins={'plan_sha256':sha(root/'PLAN.json'),'source_body_sha256':sha(track['body']),
          'music_npz_sha256':sha(music_path),'music_metadata_sha256':sha(music_path.with_suffix('.json')),
          'method':method,'control':control,'output_id':identity}
    if folder.exists():
        receipt=read(folder/'COMPLETE.json')
        if receipt['inputs']!=pins:
            raise RuntimeError('Changed candidate inputs')
        verify_files(receipt['artifacts'])
        return receipt
    music=MusicTimeline.load(music_path)
    baseline=baseline_body(track)
    if music_path.resolve()==Path(track['music_timeline']).resolve():
        select_audio_intervals(root,track,music)
    if method=='D5':
        if control is not None:
            raise ValueError('D5 controls must supply an explicit counterfactual MusicTimeline')
        from lightsem.experiments.dynamic_dance.refinement import refine_trajectory
        body,dance_plan=refine_trajectory(baseline.world,music,timestamps=baseline.time)
    else:
        from lightsem.experiments.dynamic_dance.retiming import generate as run_method
        body,dance_plan=run_method(method,baseline.world,music,times=baseline.time,control=control)
    body.metadata.update({'campaign_inputs':pins,'song':track['id'],'music_path':str(music_path),
                          'source_body':track['body'],'output_id':identity})
    dance_plan.metadata.update({'campaign_inputs':pins,'song':track['id']})
    body.validate().save(folder/'body.npz')
    dance_plan.validate().save(folder/'plan.npz')
    artifacts={str(p):sha(p) for p in sorted(folder.iterdir()) if p.is_file()}
    receipt={'inputs':pins,'artifacts':artifacts,'status':'complete',
              'scientific_status':'pending controlled comparison; finite output is not success'}
    write_new(folder/'COMPLETE.json',receipt)
    return receipt


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('initialize','freeze','baseline','audio-intervals','generate','map','preflight'))
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--song')
    parser.add_argument('--method',default='baseline')
    parser.add_argument('--control')
    parser.add_argument('--music-timeline',type=Path)
    parser.add_argument('--output-id')
    parser.add_argument('--revision',default='implementation_v1')
    args=parser.parse_args(argv)
    if args.stage=='initialize':
        plan=initialize(args.root)
        print('INITIALIZED',plan['id'],flush=True)
        return
    disk_guard(args.root)
    plan=read(args.root/'PLAN.json')
    verify_files(plan['files'])
    if args.stage=='freeze':
        frozen=freeze_source(args.root,args.revision)
        print('FROZEN',frozen['source_root'],len(frozen['files']),flush=True)
        return
    for track in select_tracks(plan,args.song):
        if args.stage=='baseline':
            source_baseline(args.root,track)
        elif args.stage=='audio-intervals':
            select_audio_intervals(args.root,track,MusicTimeline.load(track['music_timeline']))
        elif args.stage=='generate':
            generate(args.root,plan,args.method,track,args.control,args.music_timeline,args.output_id)
        elif args.stage=='map':
            from lightsem.experiments.dynamic_dance.mapping import map_result
            map_result(args.root,plan,args.method,track)
        elif args.stage=='preflight':
            from lightsem.experiments.dynamic_dance.mapping import source_preflight
            source_preflight(args.root,plan,track)
        print('COMPLETE',args.stage,args.method,track['id'],flush=True)


if __name__=='__main__':
    main()
