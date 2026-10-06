from pathlib import Path
import numpy as np
from lightsem.experiments.native_dynamics.evaluation import metrics,metric_controls
from lightsem.experiments.native_dynamics.showcase import ALIASES
from scripts.rendering import render_native_dynamics as renderer

def test_missing_region_not_invented_as_zero():
    q=np.zeros((480,13,2),np.float32);q[:,:,0]=np.arange(480)[:,None]/30
    mask=np.ones((480,13),bool);mask[:,0]=False
    r,_=metrics(q,q,mask)
    assert not r['registered_complete_coverage']
    assert r['profile_error_by_region_scale'][2]==[None,None,None]
    assert r['distribution_by_region'][2] is None
    import json
    json.dumps(metric_controls(q,mask),allow_nan=False)

def test_native_renderer_labels_and_no_baseline_promotion():
    assert ALIASES=={'continued':'D7_reference','dynamics':'C3','joint':'C4'}
    assert 'A continued training' in renderer.comparison_note({},'C3')
    assert 'B dynamics losses' in renderer.comparison_note({},'C4')
    text=renderer.TEMPLATE.read_text()
    assert 'matched continued training' in text and 'no new baseline' in text
    assert 'D6 off' not in text and 'D7 reference' not in text

def test_missing_all_observations_are_unmeasurable():
    q=np.zeros((480,13,2),np.float32);m=np.zeros((480,13),bool)
    r,_=metrics(q,q,m)
    assert r['profile_error'] is None and r['distribution_error'] is None
    import json
    json.dumps(r,allow_nan=False)

def test_root_trace_does_not_invent_a_terminal_stop():
    from types import SimpleNamespace
    t=np.arange(10)/40;body=np.zeros((10,22,3));body[:,:,0]=t[:,None]
    pair=SimpleNamespace(body=body,times=t,patch=[{'kind':'beam'}],controls={'angles_deg':np.zeros((10,1,2)),'dimmer':np.zeros((10,1))})
    values=renderer.trace_values(pair)
    np.testing.assert_allclose(values['root'],1.)
    np.testing.assert_allclose(values['upper'],0.)
