import numpy as np
import pytest
from lightsem.experiments.visual_next.projection import fit_pca,project,effective_rank


def test_streamed_projection_recovers_known_low_rank_geometry_and_global_scale():
    rng=np.random.default_rng(7);q=np.linalg.qr(rng.normal(size=(60,5)))[0]
    latent=rng.normal(size=(400,5))*np.arange(1,6)
    x=(latent@q.T+rng.normal(size=60)*3).astype(np.float32)
    factory=lambda:(x[i:i+37] for i in range(0,len(x),37))
    state=fit_pca(factory,60,components=5,oversample=5,power_iterations=2)
    z=project(x,state)
    reconstruction=project(x,state,normalize=False)@state['components']+state['input_mean']
    assert np.max(np.abs(reconstruction-x))<1e-4
    assert np.allclose(z.mean(0),0,atol=1e-5)
    assert np.allclose(z.std(0),1,atol=1e-5)
    assert state['retained_variance_fraction']>.99999
    # Transformation is fixed: future observations do not refit their own mean.
    assert np.linalg.norm(project(x+1,state).mean(0))>.01


def test_determinism_and_constant_representation_rejection():
    rng=np.random.default_rng(99);x=rng.normal(size=(200,12))
    a=fit_pca(lambda:iter([x]),12,components=6,oversample=3)
    b=fit_pca(lambda:iter([x]),12,components=6,oversample=3)
    assert np.array_equal(a['components'],b['components'])
    assert effective_rank([1,0,0])==1
    assert effective_rank([1,1,1])==pytest.approx(3)
    with pytest.raises(ValueError):fit_pca(lambda:iter([np.ones((100,12))]),12,components=6)
