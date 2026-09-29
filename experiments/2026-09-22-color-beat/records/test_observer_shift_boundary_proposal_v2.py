"""Geometry-only tests of an isolated, in-memory endpoint correction proposal.
No production code is edited, and no event metric is called.
"""
import copy, inspect, json, math, os
from pathlib import Path
import numpy as np
from lightsem.experiments.light_layers import evaluation as ev

DOCS=Path(__file__).resolve().parent
def forbidden(*args, **kwargs): raise AssertionError('No scores in geometry tests')
ev.score_record=forbidden;ev.summarize_campaign=forbidden;ev.metrics.visual_event_ap=forbidden
before=ev.implementation_hashes()
original=inspect.getsource(ev.shifted_candidates)
old='            shifted[take] = a+np.mod(local, duration)'
replacement=old+'\n            shifted[take] = np.where(shifted[take] == b, np.nextafter(b, a), shifted[take])'
assert original.count(old)==1
namespace=dict(ev.__dict__)
exec(original.replace(old,replacement),namespace)
proposal=namespace['shifted_candidates']
checks=[]
corrected_fixture_counts={}
for role in ('selected','final'):
    fixture=json.loads((DOCS/f'observer_shift_boundary_fixture_{role}_v1.json').read_text())
    t=np.array(fixture['candidate_times_s']);c=np.array(fixture['confidences']);plan=fixture['plan']
    saved=copy.deepcopy(plan)
    try:ev.shifted_candidates(t,c,plan)
    except ev.ContractError as error: assert str(error)=='Circular shift escaped its valid segment'
    else:raise AssertionError('Original must fail')
    result=proposal(t,c,plan)
    np.testing.assert_array_equal(result['real_times_s'],t)
    np.testing.assert_array_equal(result['confidences'],c)
    assert len(result['draws'])==20 and plan==saved
    count=0
    for draw,resultrow in enumerate(result['draws']):
        np.testing.assert_array_equal(resultrow['confidences'],c)
        assert resultrow['times_s'].shape==t.shape
        for row in plan['segments']:
            if row['offsets_s'] is None:continue
            a,b,d=row['start_s'],row['stop_s'],row['duration_s']
            take=(t>=a)&(t<b)
            local=t[take]-a+row['offsets_s'][draw]
            oldtimes=a+np.mod(local,d)
            newtimes=resultrow['times_s'][take]
            changed=oldtimes!=newtimes
            np.testing.assert_array_equal(changed,oldtimes==b)
            np.testing.assert_array_equal(newtimes[~changed],oldtimes[~changed])
            assert np.all(newtimes[changed]==np.nextafter(b,a))
            assert np.all((newtimes>=a)&(newtimes<b))
            assert resultrow['wrapped_candidate_count']==int(np.sum(local>=d))
            count+=int(changed.sum())
    corrected_fixture_counts[role]=count
checks.append('both_frozen_fixture_original_fail_proposal_pass_20draws_exact_boundary_only')

# First_failure should move down exactly one endpoint ULP, not wrap to start.
f=fixture['first_failure'];a=float.fromhex(f['a']['hex']);b=float.fromhex(f['b']['hex'])
assert np.nextafter(b,a).hex()=='0x1.0d33333333332p+7'
assert b-np.nextafter(b,a)==math.ulp(b)
checks.append('one_ulp_inward_not_start_wrap')

# Disjoint segments retain membership for every individual candidate. No redetection/NMS.
for rate in (10,40):
    times=100+np.arange(64*rate)/rate
    plan=ev.shift_plan(times,[[128.013,131.026],[132.037,135.999]],rate)
    t=np.array([128.05,130.95,132.1,135.9]);c=np.array([.8,.4,.2,.9])
    orig=ev.shifted_candidates(t,c,plan);new=proposal(t,c,plan)
    for r1,r2 in zip(orig['draws'],new['draws']):
        np.testing.assert_array_equal(r1['times_s'],r2['times_s'])
        np.testing.assert_array_equal(r2['confidences'],c)
        np.testing.assert_array_equal(ev._regions(r2['times_s'],np.array(plan['eligible_intervals_s'])),[0,0,1,1])
checks.append('disjoint_actual_source_segments_counts_confidences_native_10_40Hz_noop_when_no_endpoint')

# Undefined short-segment behavior and empty candidates do not change.
plan=ev.shift_plan(np.arange(80)/10,[[0,1],[2,3]],10)
assert proposal([.5,2.5],[1,2],plan)['draws'] is None
plan=ev.shift_plan(np.arange(80)/10,[[0,8]],10)
assert all(len(r['times_s'])==0 for r in proposal([],[],plan)['draws'])
checks.append('undefined_short_segments_and_empty_candidates_unchanged')

# Retain hard failure for actual escapes. Corrupt geometry is NOT swallowed.
plan=ev.shift_plan(np.arange(80)/10,[[0,8]],10)
plan['segments'][0]['duration_s']=16.
try:proposal([7.9],[1],plan)
except ev.ContractError as error:assert str(error)=='Circular shift escaped its valid segment'
else:raise AssertionError('Broad clipping would hide a genuine invalid plan')
checks.append('genuine_escape_still_ContractError')

# Deterministic broad numeric property over nonnegative source PTS, center spans <=8s.
# 1e6 samples include arbitrary residues and deliberate sums close to the upper boundary.
rng=np.random.default_rng(20260922);n=1_000_000
starts=rng.uniform(0,100000,n);stops=starts+rng.uniform(.001,8.,n)
durations=stops-starts
candidate=starts+rng.uniform(0,1,n)*durations
candidate=np.minimum(candidate,np.nextafter(stops,starts))
offsets=rng.uniform(0,1,n)*durations
# Force half the samples close to wrap boundaries without changing binary64 inputs.
candidate[n//2:]=np.nextafter(stops[n//2:]-offsets[n//2:],starts[n//2:])
local=candidate-starts+offsets
residue=np.mod(local,durations)
oldtimes=starts+residue
newtimes=np.where(oldtimes==stops,np.nextafter(stops,starts),oldtimes)
assert np.all((newtimes>=starts)&(newtimes<stops))
changed=oldtimes!=newtimes
np.testing.assert_array_equal(changed,oldtimes==stops)
np.testing.assert_array_equal(newtimes[~changed],oldtimes[~changed])
assert np.all(oldtimes[changed]-newtimes[changed]==np.spacing(stops[changed]))
checks.append('million_nonnegative_actual_source_numeric_cases_domain_exact_one_ulp_only_bitwise_noop_elsewhere')

assert ev.implementation_hashes()==before
receipt={'schema':1,'scope':'Geometry-only diagnostic proposal; not a scoring amendment or authorization',
    'tested_replacement':{'old':old,'new':replacement},'checks':checks,'check_count':len(checks),
    'fixture_endpoint_corrections_across_20_draws':corrected_fixture_counts,
    'property_cases':n,'property_endpoint_correction_count':int(changed.sum()),
    'property_seed':20260922,'script':ev.reference(__file__),
    'original_implementation_hashes_before':before,'original_implementation_hashes_after':ev.implementation_hashes(),
    'environment':{'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),'TORCH_ENV':os.environ.get('TORCH_ENV')}}
with (DOCS/'observer_shift_boundary_proposal_test_receipt_v2.json').open('x') as out:
    json.dump(receipt,out,indent=2,sort_keys=True);out.write('\n')
print(json.dumps({k:v for k,v in receipt.items() if 'implementation_hashes' not in k},indent=2))
