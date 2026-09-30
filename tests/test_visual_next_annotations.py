import copy
import pytest
from scripts.light_layers.serve_visual_next_annotations import validate_draft


PACK={'enrollment_sha256':'frozen','records':[{'canonical_id':'family1'}]}


def draft():
    return {'schema_version':1,'annotator':'reviewer_a','enrollment_sha256':'frozen',
        'clips':{'family1':{'events':[{'kind':'content_color','start':1.,'end':1.}],
            'masks':{'0':{'reviewed':True,'polygons':[
                {'kind':'foreground','points':[[.1,.1],[.1,.7],[.6,.8]]}]}}}}}


def test_draft_bound_to_person_cohort_and_known_source():
    assert validate_draft(draft(),PACK)=='reviewer_a'
    for key,value in [('annotator','../../other'),('enrollment_sha256','wrong'),('clips',{'other':{}})]:
        d=draft();d[key]=value
        with pytest.raises(ValueError):validate_draft(d,PACK)


@pytest.mark.parametrize('event',[
    {'kind':'ambiguous','start':1,'end':1},
    {'kind':'hard_edit','start':-1,'end':1},
    {'kind':'content_color','start':64,'end':64},
    {'kind':'content_color','start':float('nan'),'end':2},
    {'kind':'unsupported','start':1,'end':2},['wrong'],
])
def test_invalid_events_cannot_be_saved(event):
    d=draft();d['clips']['family1']['events']=[event]
    with pytest.raises(ValueError):validate_draft(d,PACK)


def test_polygon_bounds_and_frame_support():
    d=draft();d['clips']['family1']['masks']['0']['polygons'][0]['points'][0]=[2,.1]
    with pytest.raises(ValueError):validate_draft(d,PACK)
    d=draft();d['clips']['family1']['masks']['10']={'polygons':[]}
    with pytest.raises(ValueError):validate_draft(d,PACK)
