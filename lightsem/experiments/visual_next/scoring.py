"""Full-window historical observer scores with fixed support and audio nulls.

These endpoints are automatic-source proxies, never independently annotated
content-event accuracy. Original center8 results are not silently overwritten.
"""
from __future__ import annotations
import hashlib
import numpy as np
from lightsem.model.light_layers.evaluation import eligible_intervals,candidate_peaks,shift_plan
from .instruments import content_event_score
from .shifts import stable_shifted_candidates


def in_support(times,intervals):
    t=np.asarray(times,float);use=np.zeros(len(t),bool)
    for a,b in np.asarray(intervals,float).reshape(-1,2):use|=(t>=a)&(t<b)
    return use


def uniform_events(count,intervals,rng):
    intervals=np.asarray(intervals,float).reshape(-1,2)
    length=intervals[:,1]-intervals[:,0]
    if not len(length) or length.sum()<=0:
        if count:raise ValueError('Random control has no source support')
        return np.empty(0)
    position=rng.uniform(0,length.sum(),count);ends=np.cumsum(length)
    which=np.searchsorted(ends,position,side='right')
    starts=np.r_[0,ends[:-1]]
    return intervals[which,0]+position-starts[which]


def score_color_window(labels,curves):
    labels.validate();start=float(labels.requested_times10_s[0])
    intervals=eligible_intervals(labels,'color',center=False)
    requested=labels.requested_times10_s;source=labels.source_times10_s
    refs=source[(labels.color>0)&in_support(source,intervals)]
    def measure(reference,times,scores,support):
        return content_event_score(reference-start,times-start,scores,duration=64.,intervals=np.asarray(support).reshape(-1,2)-start)
    scores={};candidates={}
    for name,curve in curves.items():
        values=np.asarray(curve,float)
        if values.shape!=(640,) or not np.isfinite(values).all():raise ValueError('Expected finite640-frame color curve')
        times,confidence=candidate_peaks(requested,values,intervals)
        candidates[name]=(times,confidence);scores[name]=measure(refs,times,confidence,intervals)
    if 'model' not in curves:raise ValueError('Real model curve required for paired nulls')
    plan=shift_plan(requested,intervals,10)
    plan['schema']={**plan['schema'],'segment':'each_contiguous_eligible_requested_grid_segment_in_full64s'}
    shifted=stable_shifted_candidates(*candidates['model'],plan)
    common=np.asarray(plan['eligible_intervals_s'],float).reshape(-1,2)
    common_refs=refs[in_support(refs,common)];nulls=None;common_scores={}
    if shifted['draws'] is not None:
        for name,(times,confidence) in candidates.items():
            use=in_support(times,common);common_scores[name]=measure(common_refs,times[use],confidence[use],common)
        nulls=[measure(common_refs,d['times_s'],d['confidences'],common) for d in shifted['draws']]
    seed=int.from_bytes(hashlib.sha256((labels.canonical_id+'20260929').encode()).digest()[:8],'big')
    rng=np.random.default_rng(seed);times,confidence=candidates['model'];random_scores=[]
    for _ in range(20):
        random_times=uniform_events(len(times),intervals,rng)
        random_scores.append(measure(refs,random_times,confidence,intervals))
    return {'canonical_id':labels.canonical_id,'recording_id':labels.recording_id,
        'scope':'historical automatic-observer64s event proxy, not independent content accuracy',
        'scores':scores,'eligible_intervals_s':intervals.tolist(),'source_coverage':float(np.diff(intervals,axis=1).sum()/64),
        'reference_count':len(refs),'shift_plan':plan,'same_shift_support_scores':common_scores,'shift_scores':nulls,
        'shift_numeric_endpoint_corrections':None if shifted['draws'] is None else [d['half_open_endpoint_roundoff_corrections'] for d in shifted['draws']],
        'shift_numeric_rule':'Modulo output rounded onto an excluded segment end is bounded to nextafter(end,start); at most two floating-point spacings, no support expansion.',
        'rate_matched_random_scores':random_scores,'random_control_seed':seed,
        'random_control_scope':'Same candidate count/confidences, uniform on source-valid time; does not preserve .2s spacing. Circular-shift control separately preserves the original process aside from wrap pairs.',
        'reference_hash':labels.content_hash()}
