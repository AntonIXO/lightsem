import numpy as np
import pytest
from lightsem.experiments.visual_next.instruments import (
    matched_events, content_event_score, edit_veto, hs_distance, mask_scores, paired_summary)
from lightsem.model.light_layers.targets import extract_color_targets


def histogram(rgb):
    a=np.tile(np.asarray(rgb,dtype=np.uint8),(8,8,1))
    return extract_color_targets(a,np.ones(a.shape[:2],bool)).hs_histogram


def test_global_assignment_avoids_greedy_undercount_and_minimizes_timing():
    # Nearest-first would spend reference .2 on .19 and strand .39.
    ref=np.array([0.,.2]);pred=np.array([.19,.39])
    pairs=matched_events(ref,pred)
    assert len(pairs)==2
    assert np.abs(ref[pairs[:,0]]-pred[pairs[:,1]]).sum()==pytest.approx(.38)
    pairs=matched_events([0.,.2],[.01,.19,.195])
    assert set(pairs[:,1])=={0,2}


def test_ties_boundaries_duplicates_and_missing_intervals():
    s=content_event_score([1,2],[.8,1,2],[.8,.8,.8])
    assert s['average_precision']==pytest.approx(2/3)
    assert s['f1']==pytest.approx(.8)
    assert content_event_score([1],[1,1],[1,1])['matched_count']==1
    assert len(matched_events([1],[1.2]))==1
    assert len(matched_events([1],[1.20001]))==0
    assert content_event_score([.99],[1.01],[1],intervals=[[0,1],[1.001,64]])['matched_count']==0
    with pytest.raises(ValueError):content_event_score([1],[2],[1],intervals=[[2,64]])


def test_cut_veto_cannot_erase_genuine_cut_coincident_truth():
    times=np.array([1.,2.,3.]);keep=edit_veto(times,[[1,1],[2.8,2.8]])
    assert keep.tolist()==[False,True,False]
    before=content_event_score([1,2],times,[1,1,1])
    after=content_event_score([1,2],times[keep],[1])
    assert after['recall']<before['recall']
    assert after['eligible_duration_s']==before['eligible_duration_s']==64


def test_no_event_and_empty_predictions_are_not_fabricated_success():
    s=content_event_score([],[],[])
    assert s['average_precision'] is None and s['f1'] is None and s['recall'] is None
    assert s['false_cues_per_minute']==0
    assert content_event_score([1],[],[])['average_precision']==0
    assert content_event_score([], [1], [1])['false_cues_per_minute']==pytest.approx(60/64)


def test_chromatic_instrument_separates_hue_from_value_and_darkness():
    red=histogram([255,0,0]);green=histogram([0,255,0])
    assert hs_distance(red,red)==0
    assert hs_distance(red,histogram([100,0,0]))==pytest.approx(0)
    assert hs_distance(red,green)==1
    assert hs_distance(red,histogram([0,0,0]))==1
    assert hs_distance(histogram([128,128,128]),red) is None
    # Circular red boundary is symmetric; JS distance scales as sqrt(bin mass).
    assert hs_distance(red,histogram([255,0,1]))==pytest.approx(
        hs_distance(red,histogram([255,1,0])),rel=1e-5)


def test_wrong_spatial_assignment_detectable_with_identical_global_colors():
    r,g=histogram([255,0,0]),histogram([0,255,0])
    assert hs_distance((r+g)/2,(g+r)/2)==0
    assert (hs_distance(r,g)+hs_distance(g,r))/2==1


def test_masks_detect_background_damage_and_ignore_unknown_ui():
    ref=np.zeros((8,8),bool);ref[2:6,2:6]=True
    bg=~ref;bg[0]=False
    pred=ref.copy();pred[0]=True
    assert mask_scores(ref,pred,bg)=={'foreground_iou':1.,'background_removal':0.}
    pred[1]=True
    s=mask_scores(ref,pred,bg)
    assert s['foreground_iou']<1 and s['background_removal']>0


def test_paired_zero_and_direction_correction():
    z=paired_summary([1,2,3],[1,2,3]);assert z['p']==1 and z['benefit_ci95']==[0,0]
    a=paired_summary([1]*8,[.8]*8,direction='down')
    assert a['benefit_mean']==pytest.approx(.2)
    assert a['four_test_bonferroni_p']==pytest.approx(4*2/256)
    with pytest.raises(ValueError):paired_summary([1,np.nan],[1,2])
