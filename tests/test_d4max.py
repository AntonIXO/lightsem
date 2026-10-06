import importlib.util
from pathlib import Path
import copy
import numpy as np
import pytest
from lightsem.experiments.d4max.motifs import Config,build_bank,compose,kmedoids,select_path,phrase_nodes
from lightsem.experiments.d4max.pacing import anchored_clock,apply_d6
from lightsem.experiments.dynamic_dance.contracts import DancePlan,DanceResult

spec=importlib.util.spec_from_file_location('retiming_fixtures',Path(__file__).with_name('test_dynamic_dance_retiming.py'))
fx=importlib.util.module_from_spec(spec);spec.loader.exec_module(fx)

def fixture(seconds=48):
    w=fx.source_body(seconds);m=fx.music(seconds)
    # Repeated, phase-dependent embeddings without relying on semantic labels.
    phase=np.floor(m.time/4).astype(int)%3
    m.muq[:]=0
    for i in range(3):m.muq[phase==i,i]=1
    return w,m


def test_kmedoids_identical_features_do_not_invent_six_classes():
    x=np.ones((12,4))/2
    centers,labels,_=kmedoids(x,6)
    assert len(centers)==1 and np.all(labels==0)


def test_bank_is_bounded_and_deterministic():
    w,m=fixture();a=build_bank(w,m);b=build_bank(w,m)
    assert len(a['medoids'])<=6 and all(len(x)<=3 for x in a['takes'].values())
    np.testing.assert_array_equal(a['medoids'],b['medoids']);assert a['takes']==b['takes']
    assert all(i not in [0,len(a['nodes'])-1] for v in a['takes'].values() for i in v)


def test_const_body_cannot_generate_motion():
    w,m=fixture();w[:]=w[0]
    result,plan,_=compose(w,m)
    assert np.array_equal(result.world,w) and np.array_equal(result.source_time,result.time)


def test_complete_endpoint_and_fixed_bones_and_provenance():
    w,m=fixture();result,plan,bank=compose(w,m)
    result.validate();plan.validate();assert result.world.shape==w.shape
    assert plan.metadata['source_order_preserved'] is False
    assert all(.5<=x['local_rate']<=1.5 for x in plan.metadata['fragment_provenance'])
    assert max(abs(x['root_translation_m'][1]) for x in plan.metadata['fragment_provenance'])==0
    np.testing.assert_allclose(fx.lengths(result.world),fx.lengths(w),atol=1e-6)
    assert plan.metadata['fragment_provenance'][0]['source_seconds'][0]==0
    assert plan.metadata['fragment_provenance'][-1]['source_seconds'][-1]==result.time[-1]


def test_chronological_control_is_exact_identity():
    w,m=fixture();result,plan,_=compose(w,m,control='chronological')
    assert np.array_equal(w,result.world);assert plan.metadata['recomposed_fragment_count']==0


def test_family_shuffle_preserves_distribution_and_bank():
    w,m=fixture();bank=build_bank(w,m);a=select_path(w,bank,m);b=select_path(w,bank,m,control='shuffle')
    assert sorted(a[3])==sorted(b[3]);assert not np.array_equal(a[3],b[3]);assert a[0]==b[0]


def test_library_not_changed_by_target_audio():
    w,m=fixture();bank=build_bank(w,m);before=bank['features'].copy();target=copy.deepcopy(m);target.muq=np.roll(target.muq,150,axis=0)
    select_path(w,bank,target);assert np.array_equal(bank['features'],before)

@pytest.mark.parametrize('scale',[0.,1e-12,1.,100.])
def test_anchor_projection_constant_proposals_identity(scale):
    n=1200;clock,rate,meta=anchored_clock(np.full((n,5),scale),[0,150,301,510,n-1])
    assert np.max(abs(clock-np.arange(n)/30))<1e-10
    assert rate.min()>=0 and rate.max()<=1.5+1e-9
    assert meta['anchor_max_error_s']==0


def test_anchored_rate_has_no_reverse_and_never_moves_motif_boundary():
    n=1200;r=np.r_[np.zeros(400),np.full(400,4),np.linspace(0,3,400)]
    anchors=np.array([0,150,300,500,800,1199]);clock,rate,meta=anchored_clock(r,anchors)
    assert np.array_equal(clock[anchors],anchors/30)
    assert np.diff(clock).min()>=0 and rate.max()<=1.5+1e-9
    assert max(abs(x['edge_rate_start']-1) for x in meta['segments'])==0
    assert max(abs(x['edge_rate_end']-1) for x in meta['segments'])==0
    assert np.max(abs(clock-np.arange(n)/30))>.05


def test_reject_nonfinite_clock_proposals():
    with pytest.raises(ValueError):anchored_clock(np.full(100,np.nan),[0,99])


def test_zero_features_have_one_family():
    centers,labels,_=kmedoids(np.zeros((12,4)),6)
    assert len(centers)==1 and np.all(labels==0)


def test_positive_pacing_preserves_anchors_without_forced_holds():
    n=1200;proposals=np.r_[np.zeros(500),np.full(700,100.)];anchors=np.array([0,400,800,1199])
    clock,rate,audit=anchored_clock(proposals,anchors,minimum_rate=.25)
    assert rate.min()>=.25-1e-9 and rate.max()<=1.5+1e-9
    assert np.array_equal(clock[anchors],anchors/30)
    assert np.min(np.diff(clock))>0
    assert audit['minimum_rate']==.25
