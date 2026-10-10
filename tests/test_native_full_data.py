import json

import numpy as np
import pytest

from lightsem.experiments.native_dynamics.data import pose_observations
from test_native_dynamics_data import fixture


def multicoach(path):
    k,c,b,t,_=fixture(path)
    k=np.repeat(k,2,axis=1);c=np.repeat(c,2,axis=1);b=np.repeat(b,2,axis=1)
    # Different gestures and missing observations belong to different people.
    k[:,1,9,0]+=100
    c[100:120,0,9]=0
    np.savez(path,K=k,C=c,box=b,t=t,meta=json.dumps({'height':1000,'width':1000}))


def test_explicit_track_isolation_and_historical_policy(tmp_path):
    path=tmp_path/'pose.npz';multicoach(path)
    with pytest.raises(ValueError,match='Single-coach pilot'):
        pose_observations(path,20)
    first,_=pose_observations(path,20,coach_index=0)
    second,meta=pose_observations(path,20,coach_index=1)
    assert not first['mask'][100:120,5].any()
    assert second['mask'][100:120,5].all()
    assert np.isclose(second['pose'][150,5,0]-first['pose'][150,5,0],.5)
    assert meta['coach_count']==2 and meta['coach_index']==1


@pytest.mark.parametrize('index',[-1,2,True,1.5])
def test_invalid_coach_index_fails_closed(tmp_path,index):
    path=tmp_path/'pose.npz';multicoach(path)
    with pytest.raises(ValueError,match='Invalid coach index'):
        pose_observations(path,20,coach_index=index)
