"""Read-only frozen candidate/shift reproduction. Never computes event metrics."""
import json, math, hashlib, os
from pathlib import Path
from types import SimpleNamespace
from fractions import Fraction
import numpy as np
from lightsem.experiments.light_layers import evaluation as ev

DOCS = Path(__file__).resolve().parent
STATE = json.loads((DOCS/'SESSION_STATE.json').read_text())
def forbidden(*args, **kwargs):
    raise AssertionError('Scoring forbidden in boundary diagnostic')
ev.score_record = forbidden
ev.summarize_campaign = forbidden
ev.metrics.visual_event_ap = forbidden

def scalar(x):
    x=float(x)
    return {'decimal':repr(x), 'hex':x.hex()}
def save_new(path, obj):
    with path.open('x') as out: json.dump(obj,out,indent=2,sort_keys=True,allow_nan=False);out.write('\n')
def trace(candidates, confidence, plan):
    intervals=np.asarray(plan['eligible_intervals_s'],float).reshape(-1,2)
    real=np.asarray(candidates)[ev._regions(np.asarray(candidates),intervals)>=0]
    for draw in range(20):
        for si,row in enumerate(plan['segments']):
            if row['offsets_s'] is None: continue
            a,b,d=row['start_s'],row['stop_s'],row['duration_s']
            take=(real>=a)&(real<b)
            local=real[take]-a+row['offsets_s'][draw]
            residue=np.mod(local,d)
            shifted=a+residue
            for ci in np.flatnonzero((shifted<a)|(shifted>=b)):
                original=float(real[take][ci]);offset=float(row['offsets_s'][draw])
                exact_d=Fraction.from_float(b)-Fraction.from_float(a)
                exact_local=Fraction.from_float(original)-Fraction.from_float(a)+Fraction.from_float(offset)
                exact=Fraction.from_float(a)+exact_local%exact_d
                return {'draw_zero_based':draw,'draw_one_based':draw+1,'segment_zero_based':si,
                    'candidate_index_in_segment':int(ci),'row':row,
                    'original':scalar(original),'a':scalar(a),'b':scalar(b),'duration':scalar(d),
                    'offset':scalar(offset),'local':scalar(local[ci]),'residue':scalar(residue[ci]),
                    'shifted':scalar(shifted[ci]),'distance_from_stop_s':float(shifted[ci]-b),
                    'distance_from_largest_in_domain_float_s':float(shifted[ci]-np.nextafter(b,a)),
                    'distance_from_largest_in_domain_float_ulps':float((shifted[ci]-np.nextafter(b,a))/math.ulp(b)),
                    'endpoint_ulp_s':math.ulp(b),'exact_real_result_numerator':exact.numerator,
                    'exact_real_result_denominator':exact.denominator,'exact_real_result_rounded':scalar(float(exact)),
                    'exact_real_distance_below_stop_s':float(Fraction.from_float(b)-exact),
                    'exact_real_distance_below_stop_ulps':float((Fraction.from_float(b)-exact)/Fraction.from_float(math.ulp(b)))}
    return None

frozen_before=ev.implementation_hashes()
results=[]
for role in ('selected','final'):
    args=SimpleNamespace(registration=STATE['score_bindings'][role],manifest=STATE['observer_training_manifest'],
        transforms=STATE['observer_transforms'])
    registration,manifest,transforms,store=ev._inputs(args,'score')
    index_ref=registration['prediction_index']
    predictions=json.loads(ev.checked_ref(index_ref).read_text())
    label_cache={}; visited=0;found=False
    for pi,pref in enumerate(predictions):
        receipt=json.loads(ev.checked_ref(pref).read_text())
        checkpoint=ev.validate_prediction_receipt(receipt,registration,registration['baselines'])
        ev.checked_ref(checkpoint)
        ev.validate_training_receipt(checkpoint,registration)
        for ri,record in enumerate(receipt['records']):
            rid=record['recording_id']
            if rid not in label_cache: label_cache[rid]=ev.freeze_labels(store.visual(rid),transforms)
            label=label_cache[rid]
            assert label.content_hash()==record['target_sha256']==registration['target_hashes'][rid]
            assert record['source_row']==store.rows[rid]
            with np.load(ev.checked_ref(record['artifact']),allow_pickle=False) as z:
                for method in ('model','donor'):
                    assert np.array_equal(z[method+'__requested_times10_s'],label.requested_times10_s)
                    assert np.array_equal(z[method+'__source_times10_s'],label.source_times10_s)
                for head,rate in ev.RATES.items():
                    intervals=ev.eligible_intervals(label,head)
                    full=ev.eligible_intervals(label,head,center=False)
                    times=label.requested_times10_s[0]+np.arange(64*rate)/rate
                    candidate,confidence=ev.candidate_peaks(times,z['model__'+head],full)
                    keep=ev._regions(candidate,intervals)>=0
                    candidate,confidence=candidate[keep],confidence[keep]
                    plan=ev.shift_plan(times,intervals,rate)
                    visited+=1
                    try: ev.shifted_candidates(candidate,confidence,plan)
                    except ev.ContractError as error:
                        assert str(error)=='Circular shift escaped its valid segment'
                        details=trace(candidate,confidence,plan)
                        assert details is not None
                        fixture_path=DOCS/('observer_shift_boundary_fixture_'+role+'_v1.json')
                        fixture={'schema':1,'role':role,'recording_id':rid,'canonical_id':record['canonical_id'],
                            'head':head,'candidate_times_s':candidate.tolist(),'confidences':confidence.tolist(),
                            'plan':plan,'expected_original_exception':str(error),'first_failure':details,
                            'provenance':{'prediction_receipt':pref,'raw_prediction':record['artifact'],
                                'checkpoint':{k:checkpoint[k] for k in ('path','sha256','recipe','seed','role')},
                                'target_sha256':label.content_hash(),'source_row':record['source_row'],
                                'manifest':ev.reference(args.manifest),'transforms':ev.reference(args.transforms),
                                'registration':ev.reference(args.registration),'prediction_index':index_ref,
                                'prediction_index_position':pi,'record_index_position':ri}}
                        save_new(fixture_path,fixture)
                        results.append({'role':role,'fixture':ev.reference(fixture_path),'head':head,'recording_id':rid,
                            'checkpoint':fixture['provenance']['checkpoint'],'visited_record_heads':visited,
                            'first_failure':details})
                        print(json.dumps(results[-1],indent=2),flush=True)
                        found=True;break
            if found:break
        if found:break
    assert found
assert ev.implementation_hashes()==frozen_before
save_new(DOCS/'observer_shift_boundary_diagnostic_receipt_v1.json',{'schema':1,
    'scope':'Candidate geometry only; score_record, summarize_campaign, visual_event_ap blocked; no AP outcomes computed or inspected',
    'environment':{'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),'TORCH_ENV':os.environ.get('TORCH_ENV')},
    'diagnostic_script':ev.reference(__file__),'original_implementation_hashes_before':frozen_before,
    'original_implementation_hashes_after':ev.implementation_hashes(),'results':results})
