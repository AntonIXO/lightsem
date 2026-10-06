import numpy as np
from lightsem.experiments.native_repair.assessment import raw_metrics,pair_metrics
from lightsem.experiments.native_repair.reporting import assess_pair

def moving_coordinates():
    # A numerical derivative fixture, not a rendered dance or training sample.
    t=np.arange(180)/30;w=np.zeros((180,22,3));w[...,1]=1.;w[:,[7,8,10,11],1]=0.;w[:,20,0]=.02*t**3
    return w

def test_raw_jerk_retains_physical_units():
    m=raw_metrics(moving_coordinates())
    np.testing.assert_allclose(m['raw_relative_jerk_p95_m_s3'],.12/np.sqrt(22),rtol=1e-6)
    assert m['raw_relative_speed_mean_m_s']>0 and not m['constant_body']

def test_stopping_the_body_cannot_pass_repair_gates():
    ref=moving_coordinates();stopped=np.broadcast_to(ref[:1],ref.shape).copy();p=pair_metrics(stopped,ref);g=assess_pair(p)
    assert g['raw_relative_jerk_p95_m_s3']
    assert not g['activity'] and not g['variation'] and not g['not_constant']

def test_translations_do_not_change_articulation_metrics():
    w=moving_coordinates();a=raw_metrics(w);b=raw_metrics(w+np.array([2.,0.,-4.]))
    for k in ['raw_relative_jerk_p95_m_s3','raw_relative_speed_mean_m_s','high_frequency_residual_RMS_m']:
        np.testing.assert_allclose(a[k],b[k],rtol=1e-6,atol=1e-10)
