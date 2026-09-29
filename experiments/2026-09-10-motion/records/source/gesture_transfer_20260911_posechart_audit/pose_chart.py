"""Outputs-only stateless geometry prototype. Not a production mapper."""
import numpy as np
from lightsem.experiments.motion_study.articulated_stage import REGIONS, NEUTRAL, infer_root_rotation, unfolded_targets

# Engineering constants, fixed before synthetic measurements.
# 0.75 m canonical half arm span -> 6 m linear half-stage extent.
X_GAIN = 6. / .75
X_BOUND = 7.5
U_HALF = 6.25
LEAN_M = 1.7  # canonical body height, not estimated from any recording
BASE = np.array([7.7, 3.2, 8.0])  # stage02 role anchor comparator

def pose_chart(body, activity=.7):
    body = np.asarray(body, float)
    if body.ndim != 3 or body.shape[1:] != (22,3) or not np.isfinite(body).all():
        raise ValueError("Expected finite canonical FK body")
    a = .25+.75*np.clip(np.broadcast_to(np.asarray(activity,float),(len(body),)),0,1)
    if not np.isfinite(a).all():
        raise ValueError("Finite activity required")
    q = infer_root_rotation(body)
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
        # Absolute local regional x supplies anatomical composition; no fixture lane.
        v = X_GAIN*region[:,0]+shared_x  # Do not blend crossed limbs back toward rest.
        x = X_BOUND*np.tanh(v/X_BOUND)
        du = a*(4*delta[:,1]-3*delta[:,2])+shared_u
        base = BASE[min(rank,5-rank)]
        u = U_HALF+U_HALF*np.tanh(np.arctanh((base-U_HALF)/U_HALF)+du/U_HALF)
        charts.append(np.stack((x,u),-1))
        targets.append(unfolded_targets(x,u))
        deltas.append(delta)
    return np.stack(targets,1),np.stack(charts,1),np.stack(deltas,1)
