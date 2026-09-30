import numpy as np
from lightsem.model.light_layers.evaluation import FrozenLabels
from lightsem.experiments.visual_next.scoring import score_color_window,uniform_events


def fixture(events):
    t=np.arange(640)/10;y=np.zeros(640);y[events]=1;valid=np.ones(640,bool)
    return FrozenLabels('record','family','val',t,t,y.copy(),y.copy(),valid.copy(),valid.copy())


def test_proxy_scoring_recovers_known_events_and_keeps_null_support_paired():
    labels=fixture([100,200,400]);curve=np.zeros(640);curve[[100,200,400]]=1
    r=score_color_window(labels,{'model':curve,'constant':np.full(640,.1)})
    assert r['scores']['model']['average_precision']==1
    assert r['scores']['constant']['average_precision']==0
    assert len(r['shift_scores'])==20 and len(r['rate_matched_random_scores'])==20
    assert r['same_shift_support_scores']['model']['reference_count']==3


def test_nonevent_AP_stays_undefined_and_random_events_never_cross_gaps():
    r=score_color_window(fixture([]),{'model':np.zeros(640)})
    assert r['scores']['model']['average_precision'] is None
    x=uniform_events(1000,[[0,2],[10,11]],np.random.default_rng(4))
    assert np.all(((x>=0)&(x<2))|((x>=10)&(x<11)))
