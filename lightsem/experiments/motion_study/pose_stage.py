"""Opt-in stateless pose chart, ported from the reviewed outputs-only prototype.

Engineering candidate, not a promoted dance mapping. Region ownership is fixed
but x comes from local anatomical position, NOT fixture lanes. This preserves
crossed endpoint ordering as well as the prototype's tiny pair overlaps and
3D-to-2D nullspace. It proves nothing about all emitted beam intersections.

All geometry constants are fixed here. In particular anchors are (7.7,3.2,8.0)
and signed lean gain is 1.7 m. LightingConfig.stage_base_unfolded_m and
stage_lean_gain_m configure ONLY articulated_stage; they do not configure this
mode. surface_gain and torso_gain also do not apply. There is no phase, time,
history, source edit, calibration, or sequence statistic in this chart.
"""
from __future__ import annotations

import numpy as np

from .articulated_stage import REGIONS, NEUTRAL, infer_root_rotation, unfolded_targets

X_GAIN = 6. / .75
X_BOUND = 7.5
U_HALF = 6.25
LEAN_M = 1.7
BASE = np.array([7.7, 3.2, 8.0])
BASE.setflags(write=False)


def pose_chart(body, activity=.7, *, root_rotation=None):
    """Return targets (T,6,3), chart (T,6,2), and LOCAL deltas (T,6,3).

    Body is canonical world FK (T,22,3). Optional proper rotation Q acts on
    column vectors and has shape (3,3) or (T,3,3). Otherwise hip/spine1 anchors
    infer it, exactly as in the prototype and articulated_stage. Outputs stay
    in x=[-7.5,7.5], u=[0,12.5]; u=7 is the floor/back-wall seam.
    """
    body = np.asarray(body, float)
    if body.ndim != 3 or body.shape[1:] != (22,3) or len(body) < 1 or not np.isfinite(body).all():
        raise ValueError("Expected nonempty finite canonical FK body (T,22,3)")
    activity = np.broadcast_to(np.asarray(activity,float),(len(body),))
    if not np.isfinite(activity).all():
        raise ValueError("Finite activity required")
    a = .25+.75*np.clip(activity,0,1)
    q = (infer_root_rotation(body) if root_rotation is None else
         np.broadcast_to(np.asarray(root_rotation,float),(len(body),3,3)))
    if (not np.isfinite(q).all() or
            not np.allclose(q @ np.swapaxes(q,-1,-2),np.eye(3),atol=1e-5) or
            not np.allclose(np.linalg.det(q),1.,atol=1e-5)):
        raise ValueError("root_rotation must contain proper rotation matrices")
    root = body[:,0]
    local = np.einsum('tji,tkj->tki',q,body-root[:,None])
    forward, up = q[:,:,2], q[:,:,1]
    shared_x = .35*forward[:,0]+.25*np.tanh(root[:,0]/3)+a*LEAN_M*up[:,0]
    shared_u = (.35*(1-forward[:,2])+.8*np.tanh((root[:,1]-.95)/.4)
                -.4*np.tanh(root[:,2]/3)-a*LEAN_M*up[:,2])
    charts, targets, deltas = [], [], []
    for rank, (_,ids,weights) in enumerate(REGIONS):
        rest = np.sum(NEUTRAL[list(ids)]*np.asarray(weights)[:,None],axis=0)
        region = np.sum(local[:,ids]*np.asarray(weights)[None,:,None],axis=1)
        delta = region-rest
        v = X_GAIN*region[:,0]+shared_x
        x = X_BOUND*np.tanh(v/X_BOUND)
        du = a*(4*delta[:,1]-3*delta[:,2])+shared_u
        base = BASE[min(rank,5-rank)]
        u = U_HALF+U_HALF*np.tanh(np.arctanh((base-U_HALF)/U_HALF)+du/U_HALF)
        charts.append(np.stack((x,u),-1))
        targets.append(unfolded_targets(x,u))
        deltas.append(delta)
    result = np.stack(targets,1),np.stack(charts,1),np.stack(deltas,1)
    if not all(np.isfinite(v).all() for v in result):
        raise ValueError("Pose chart arithmetic must remain finite")
    return result


def stage_targets(body, fixture_x, rank, activity, cfg, *, root_rotation=None):
    """Mapper adapter. Fixture x validates the rig; it does NOT set a lane.

    cfg intentionally does not tune geometry (see module docstring). Existing
    smoothing, photometry, decoder, and controller remain mapper concerns.
    """
    if not isinstance(rank,(int,np.integer)) or not 0 <= rank < 6:
        raise ValueError("Fixture rank must be an integer in 0..5")
    if not np.isfinite(fixture_x) or abs(fixture_x) > 6.5:
        raise ValueError("Candidate supports fixture x within +/-6.5 m")
    targets, _, deltas = pose_chart(body,activity,root_rotation=root_rotation)
    return targets[:,rank],deltas[:,rank]
