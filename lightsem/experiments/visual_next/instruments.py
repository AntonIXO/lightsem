"""Independent-label instruments for the September 29 preregistration.

No source data, predictions, thresholds, or reference support are selected here.
Undefined scores stay undefined. Candidate vetoes never alter reference support.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import linear_sum_assignment
from lightsem.model.light_layers.metrics import visual_event_ap


def matched_events(reference, candidate, tolerance=.2):
    """Maximum cardinality, then minimum absolute timing distance.

Sorted timestamps/indices determine exact ties reproducibly. Dummy columns cost
more than every possible sum of admissible distances, so timing minimization
cannot sacrifice a true match. Returned indices refer to the original arrays.
Call separately in each independently frozen continuous support interval.
"""
    ref, pred = np.asarray(reference, float), np.asarray(candidate, float)
    if any(x.ndim != 1 or not np.isfinite(x).all() for x in (ref,pred)):
        raise ValueError('Event times must be finite vectors')
    if len(np.unique(ref)) != len(ref) or not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('Unique reference events and positive tolerance required')
    if not len(ref) or not len(pred):return np.empty((0,2),int)
    ri,pi=np.argsort(ref,kind='stable'),np.argsort(pred,kind='stable')
    distance=np.abs(pred[pi,None]-ref[ri][None])
    penalty=(min(len(ref),len(pred))+1)*(tolerance+1)
    cost=np.full((len(pred),len(ref)+len(pred)),penalty)
    cost[:,:len(ref)]=np.where(distance <= tolerance+1e-12,distance,penalty*2)
    rows,cols=linear_sum_assignment(cost)
    keep=cols<len(ref)
    rows,cols=rows[keep],cols[keep]
    return np.stack((ri[cols],pi[rows]),axis=1)


def content_event_score(reference, candidate, confidence, *, duration=64., intervals=None):
    """Grouped-tie AP and fixed .5 F1, with explicit no-event behavior."""
    score=visual_event_ap(reference,candidate,confidence,duration_s=duration,
        deployment_threshold=.5,channel='color',eligible_intervals_s=intervals)
    n,k,tp=score.reference_count,score.emitted_count,score.matched_count
    return {'average_precision':score.average_precision,
            'f1':2*tp/(n+k) if n+k else None,
            'recall':tp/n if n else None,
            'precision':tp/k if k else None,
            'false_cues_per_minute':score.false_cues_per_minute,
            'reference_count':n,'emitted_count':k,'matched_count':tp,
            'eligible_duration_s':score.eligible_duration_s,'coverage':score.coverage}


def edit_veto(candidate_times, edit_intervals, margin=.2):
    """Boolean keep mask; inclusive edit edges; no mutation of reference times."""
    times=np.asarray(candidate_times,float);edits=np.asarray(edit_intervals,float).reshape(-1,2)
    if times.ndim!=1 or not np.isfinite(times).all() or not np.isfinite(edits).all():
        raise ValueError('Finite times/intervals required')
    if np.any(edits[:,1]<edits[:,0]) or not np.isfinite(margin) or margin<0:
        raise ValueError('Invalid edit interval/margin')
    return ~np.any((times[:,None]>=edits[None,:,0]-margin-1e-12)
                  &(times[:,None]<=edits[None,:,1]+margin+1e-12),axis=1)


def hs_distance(reference, prediction):
    """Base2 sqrt(JS). Missing prediction mass has maximal error=1.

    Zero reference mass is undefined, not score zero. Caller freezes reference
    validity independently; never make support conditional on a prediction.
    """
    p,q=np.asarray(reference,float),np.asarray(prediction,float)
    if p.shape!=(72,) or q.shape!=(72,) or not np.isfinite([p,q]).all() or min(p.min(),q.min())<0:
        raise ValueError('Finite nonnegative 72-bin histograms required')
    if p.sum()==0:return None
    if q.sum()==0:return 1.
    p,q=p/p.sum(),q/q.sum();m=(p+q)/2
    a,b=p>0,q>0
    divergence=(np.sum(p[a]*np.log2(p[a]/m[a]))+np.sum(q[b]*np.log2(q[b]/m[b])))/2
    return float(np.sqrt(np.clip(divergence,0,1)))


def mask_scores(reference_foreground, predicted_foreground, reference_background):
    arrays=[np.asarray(x) for x in (reference_foreground,predicted_foreground,reference_background)]
    if any(x.dtype!=bool or x.ndim!=2 or x.shape!=arrays[0].shape for x in arrays):
        raise ValueError('Three matching boolean image masks required')
    ref,pred,bg=arrays
    if np.any(ref&bg):raise ValueError('Reference foreground/background overlap')
    # Unknown/UI reference pixels do not enter either mask endpoint.
    pred=pred&(ref|bg)
    union=np.count_nonzero(ref|pred);bg_count=np.count_nonzero(bg)
    return {'foreground_iou':float(np.count_nonzero(ref&pred)/union) if union else None,
            'background_removal':float(np.count_nonzero(pred&bg)/bg_count) if bg_count else None}


def paired_summary(before, after, *, direction='up', seed=20260929, draws=10000):
    """One paired row per canonical family, after averaging seeds/frames.

    Positive benefit always means improvement. Finite full pairs are required;
    missingness must be resolved by independently frozen support upstream.
    """
    a,b=np.asarray(before,float),np.asarray(after,float)
    if a.ndim!=1 or a.shape!=b.shape or len(a)<2 or not np.isfinite([a,b]).all():
        raise ValueError('At least two finite paired family scores required')
    if direction not in ('up','down') or draws<1000:raise ValueError('Invalid inference settings')
    delta=(b-a)*(1 if direction=='up' else -1)
    rng=np.random.default_rng(seed)
    boot=delta[rng.integers(len(delta),size=(draws,len(delta)))].mean(1)
    observed=abs(delta.mean())
    if len(delta)<=16:
        bits=(np.arange(2**len(delta))[:,None]>>np.arange(len(delta)))&1
        null=((2*bits-1)*delta).mean(1)
        p=float(np.mean(np.abs(null)>=observed-1e-12));method='exact_two_sided_sign_flip'
    else:
        signs=rng.integers(0,2,size=(draws,len(delta)))*2-1
        null=(signs*delta).mean(1)
        p=float((1+np.count_nonzero(np.abs(null)>=observed-1e-12))/(draws+1))
        method='monte_carlo_two_sided_sign_flip_plus_one'
    return {'families':len(a),'before_mean':float(a.mean()),'after_mean':float(b.mean()),
        'benefit_mean':float(delta.mean()),'benefit_ci95':np.quantile(boot,[.025,.975]).tolist(),
        'p':p,'four_test_bonferroni_p':min(1.,4*p),'sign_flip_method':method,
        'seed':seed,'bootstrap_draws':draws,'scope':'conditional_enrolled_families_not_fresh_confirmation'}
