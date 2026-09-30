import numpy as np
import pytest
from lightsem.experiments.visual_next.data import common_history_support,train_factory


def test_common_support_is_shared_with_static_controls_without_bridging_gaps():
    valid=np.ones(20,bool);valid[8]=False
    result=common_history_support(valid)
    assert np.flatnonzero(result).tolist()==[5,6,7,14,15,16,17,18,19]
    assert not common_history_support(np.ones(4,bool)).any()


def test_dev_cannot_enter_projection_fitting():
    with pytest.raises(ValueError):train_factory([{'partition':'val'}],'/unused','R1')
    with pytest.raises(ValueError):train_factory([], '/unused','R1')
