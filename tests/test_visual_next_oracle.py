import numpy as np
from lightsem.experiments.visual_next.oracle import ColorState,assignments,regional_cost,choose_assignment,region_histograms_from_grid
from lightsem.experiments.visual_next.observation import FrameObservation
from lightsem.model.light_layers.targets import extract_color_targets
from lightsem.model.light_layers.transitions import fade_rgb_targets
from lightsem.experiments.visual_next.oracle import oracle_gate_report,palette_quantization_reference


def test_group_fades_match_accepted_repeated_and_interrupted_transition_law():
    t=np.arange(80)/40;track=np.zeros((80,3));track[:]=[1,0,0]
    track[8:]=[0,0,1];track[13:]=[0,1,0];track[40:]=[1,0,0]
    reference=fade_rgb_targets(track,t,duration_s=.3)
    state=ColorState.constant(np.tile(track[0],(4,1)),0);result=[]
    for i,time in enumerate(t):
        state=state.adopt(np.tile(track[i],(4,1)),time)
        result.append(state.at([time])[0,0])
    assert np.array_equal(np.asarray(result),reference)


def test_mass_weighted_cells_recover_unequal_coverage_quadrants():
    rng=np.random.default_rng(22);rgb=rng.integers(0,256,(28,36,3),dtype=np.uint8);mask=rng.random((28,36))>.4
    grid=FrameObservation(rgb).grid(mask)
    h,v=region_histograms_from_grid(grid,(36,28))
    for k,(r,c) in enumerate(((0,0),(0,18),(14,0),(14,18))):
        expected=extract_color_targets(rgb[r:r+14,c:c+18],mask[r:r+14,c:c+18]).hs_histogram
        assert np.allclose(h[0,k],expected,atol=1e-7)
    assert v.all()


def test_oracle_metric_rewards_correct_spatial_colors_and_penalizes_darkness():
    ref=np.zeros((1,4,72));ref[0,0,5]=1;ref[0,1,29]=1;ref[0,2,5]=1;ref[0,3,29]=1
    global_color=np.zeros_like(ref);global_color[:,:,5]=1
    pred=np.stack([ref,global_color,np.zeros_like(ref)])
    costs=regional_cost(ref,pred,np.ones((1,4),bool))
    assert np.allclose(costs,[0,.5,1])
    assert len(assignments('O0'))==4 and len(assignments('O2'))==16 and len(assignments('O4'))==256
    chosen,_,_=choose_assignment(np.full(256,np.nan),assignments('O4'),[1,2,3,0])
    assert chosen.tolist()==[1,2,3,0]


def test_positive_direction_cannot_override_luma_guard_or_zero_denominator():
    rows=[]
    for i in range(31):
        scores={arm:{'error':1.,'mean_luma':10.,'clipping':0.} for arm in ('O0','O4','permuted','donor')}
        scores['O4'].update(error=.5,mean_luma=12.)
        rows.append({'source_frame_coverage':1.,'scores':scores})
    summary={'benefit_ci95':[.4,.6],'four_test_bonferroni_p':.001}
    r=oracle_gate_report(rows,summary)
    assert r['gates']['relative_error_improvement'] and not r['gates']['display_luma']
    assert not r['automated_proxy_gates_pass']
    for row in rows:row['scores']['O0']['error']=0.
    assert oracle_gate_report(rows,summary)['relative_improvement'] is None


def test_palette_reference_keeps_a_representable_pure_bin():
    reference=np.zeros((4,72));reference[:,5]=1
    palette=np.array([[255,0,0],[0,255,0],[0,0,255],[255,255,0]],np.uint8)
    assert np.array_equal(palette_quantization_reference(reference,palette),reference)
