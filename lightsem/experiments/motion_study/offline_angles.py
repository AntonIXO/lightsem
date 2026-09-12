"""Whole-track convex target fitting under quantization-safe rig constraints.

No clock/chase/motif generator: optimize closeness to supplied angular targets.
An inscribed octagon conservatively bounds each 2D speed/acceleration norm.
Reserve worst-case tick-rounding error before solving; verify emitted DMX after.
Requires clarabel in the dedicated LightSem torch environment.
"""
import numpy as np
from scipy import sparse


def project_angles(targets,ranges,activity,cfg):
    import clarabel
    targets=np.asarray(targets,float);ranges=np.asarray(ranges,float)
    n=len(targets)
    if targets.shape!=(n,2) or n<1 or not np.isfinite(targets).all():
        raise ValueError('Expected finite nonempty (time,2) angles')
    if n<3:
        # No useful optimizer boundary context; constant only is unambiguous.
        if not np.all(targets==targets[:1]):
            raise ValueError('Nonconstant offline path needs at least three frames')
        return np.rint((.5+targets/ranges)*65535).astype(np.int64)
    tick=ranges/65535
    normal=np.stack((np.cos(np.arange(4)*np.pi/4),np.sin(np.arange(4)*np.pi/4)),1)
    inscribed=np.cos(np.pi/8)
    d1=sparse.diags([-np.ones(n-1),np.ones(n-1)],[0,1],shape=(n-1,n),format='csc')
    d2=sparse.diags([np.ones(n-2),-2*np.ones(n-2),np.ones(n-2)],[0,1,2],shape=(n-2,n),format='csc')
    # Components use interleaved time-major [pan0,tilt0,pan1,tilt1,...].
    a1=sparse.kron(d1,sparse.csc_matrix(normal),format='csc')
    a2=sparse.kron(d2,sparse.csc_matrix(normal),format='csc')
    local=cfg.quiet_target_speed_deg_s+(cfg.maximum_speed_deg_s-cfg.quiet_target_speed_deg_s)*np.asarray(activity)**2
    round_error=np.abs(normal)@tick
    tolerance=2e-5  # extra degrees per difference beyond solver residual tolerance
    b1=(local[1:,None]*inscribed/cfg.fps-round_error[None]-tolerance).ravel()
    b2=np.tile(cfg.maximum_acceleration_deg_s2*inscribed/cfg.fps**2-2*round_error-tolerance,n-2)
    if min(b1.min(),b2.min())<=0:
        raise ValueError('Requested limits cannot support this quantization-safe bound')
    # Mechanical position bounds reserve half a tick on either end.
    lo=np.tile(-ranges/2+tick/2,n);hi=np.tile(ranges/2-tick/2,n)
    matrix=sparse.vstack((sparse.eye(2*n,format='csc'),a1,a2),format='csc')
    lower=np.r_[lo,-b1,-b2];upper=np.r_[hi,b1,b2]
    settings=clarabel.DefaultSettings()
    settings.verbose=False
    settings.max_iter=200
    settings.tol_gap_abs=1e-7
    settings.tol_gap_rel=1e-7
    settings.tol_feas=1e-9
    constraints=sparse.vstack((matrix,-matrix),format='csc')
    bound=np.r_[upper,-lower]
    solver=clarabel.DefaultSolver(sparse.eye(2*n,format='csc')*2,
        -2*targets.ravel(),constraints,bound,
        [clarabel.NonnegativeConeT(len(bound))],settings)
    result=solver.solve()
    if str(result.status)!='Solved':
        raise RuntimeError('Offline angular fit failed: '+str(result.status))
    ticks=np.rint((.5+np.asarray(result.x).reshape(n,2)/ranges)*65535).astype(np.int64)
    angles=(ticks/65535-.5)*ranges
    speed=np.linalg.norm(np.diff(angles,axis=0)*cfg.fps,axis=-1)
    accel=np.linalg.norm(np.diff(angles,n=2,axis=0)*cfg.fps**2,axis=-1)
    if ticks.min()<0 or ticks.max()>65535 or speed.max()>cfg.maximum_speed_deg_s+1e-8 or accel.max()>cfg.maximum_acceleration_deg_s2+1e-8:
        raise RuntimeError('Quantized offline trajectory failed hard mechanics')
    return ticks
