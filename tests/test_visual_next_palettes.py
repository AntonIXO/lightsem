import numpy as np
import pytest
from lightsem.conductor.rig import canonical_rig
from lightsem.experiments.visual_next.palettes import fixture_groups,prototype_representatives,group_rgb_mask


def test_fixed_rig_partition_protects_479_slots_and_assigns_center_right():
    groups=fixture_groups(canonical_rig());mask=group_rgb_mask(groups)
    assert np.count_nonzero(~mask)==479
    assert {r['o4'] for r in groups}=={'left_heads','right_heads','left_other','right_other'}
    assert all(r['o4']=='right_other' for r in groups if r['kind']=='projector')


def test_modes_preserve_three_distinct_hues_and_handle_sparse_prototype():
    p=np.zeros(72);p[[5,29,53]]=[.5,.3,.2]
    modes=prototype_representatives(p)
    assert np.allclose([x['rgb'] for x in modes],np.eye(3))
    p=np.zeros(72);p[5]=1
    assert np.allclose([x['rgb'] for x in prototype_representatives(p)],[[1,0,0]]*3)
    with pytest.raises(ValueError):prototype_representatives(np.zeros(72))


def test_circular_wrap_mode_and_deterministic_ties():
    p=np.ones(72)
    assert prototype_representatives(p)==prototype_representatives(p.copy())
    assert all(np.isfinite(m['rgb']).all() for m in prototype_representatives(p))
