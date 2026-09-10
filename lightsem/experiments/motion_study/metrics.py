"""Source-relative physical fidelity; static predictions cannot pass by smoothing."""
import numpy as np
import torch

from .geometry import FPS, BODY_HEIGHT, FEET, joints_from_features


def compare(reference, prediction, velocity_rms):
    reference, prediction = np.asarray(reference), np.asarray(prediction)
    if reference.shape != prediction.shape or reference.ndim != 2 or reference.shape[1] != 139:
        raise ValueError('Expected matched (T,139) arrays')
    if len(reference) < 3 or not np.isfinite(prediction).all() or not np.isfinite(reference).all():
        raise ValueError('Nonfinite or too-short motion')
    with torch.inference_mode():
        truth, pred = [joints_from_features(torch.from_numpy(x.astype(np.float32))).numpy() for x in (reference, prediction)]
    tv, pv = [np.diff(x, axis=0)*FPS for x in (truth, pred)]
    ta, pa = [np.diff(x, axis=0)*FPS for x in (tv, pv)]
    tc, pc = reference[:, 135:] >= .5, prediction[:, 135:] >= .5
    tp = (tc & pc).sum(0)
    fp = (~tc & pc).sum(0)
    fn = (tc & ~pc).sum(0)
    denom = 2*tp + fp + fn
    f1 = np.divide(2*tp, denom, out=np.ones(4), where=denom > 0)
    kinetic = [float(np.square(v).sum(-1).mean()) for v in (tv, pv)]
    travel = [float(np.linalg.norm(np.diff(j[:, 0, [0, 2]], axis=0), axis=-1).sum()) for j in (truth, pred)]
    physical = []
    for joints, velocity in ((truth, tv), (pred, pv)):
        feet = joints[:, FEET]
        speed = np.linalg.norm(velocity[:, FEET], axis=-1)
        physical.append({'penetration_p95_m': float(np.quantile(np.maximum(-feet[..., 1], 0), .95)),
                         'sliding_on_reference_contacts_m_s': float(speed[tc[:-1]].mean()) if tc[:-1].any() else 0.})
    return {'joint_error_height': float(np.linalg.norm(pred-truth, axis=-1).mean()/BODY_HEIGHT),
            'velocity_nrmse': float(np.sqrt(np.square(pv-tv).sum(-1).mean())/max(velocity_rms, 1e-8)),
            'acceleration_rmse': float(np.sqrt(np.square(pa-ta).sum(-1).mean())),
            'contact_f1': float(f1.mean()), 'contact_counts': {'tp': tp.tolist(), 'fp': fp.tolist(), 'fn': fn.tolist()},
            'kinetic_ratio': kinetic[1]/kinetic[0] if kinetic[0]>1e-8 else (1. if kinetic[1]<=1e-8 else kinetic[1]/1e-8),
            'kinetic_reference': kinetic[0], 'kinetic_prediction': kinetic[1],
            'travel_ratio': travel[1]/travel[0] if travel[0]>1e-8 else (1. if travel[1]<=1e-8 else travel[1]/1e-8),
            'travel_reference_m': travel[0], 'travel_prediction_m': travel[1],
            'physical_reference': physical[0], 'physical_prediction': physical[1]}


def fidelity_gates(metrics):
    energy = (.9 <= metrics['kinetic_ratio'] <= 1.1 if metrics['kinetic_reference'] >= .01 else
              np.sqrt(metrics['kinetic_prediction']) <= np.sqrt(metrics['kinetic_reference'])+.02)
    travel = (.9 <= metrics['travel_ratio'] <= 1.1 if metrics['travel_reference_m'] >= .1 else
              abs(metrics['travel_prediction_m']-metrics['travel_reference_m']) <= .034)
    return {'joint_fidelity': metrics['joint_error_height'] <= .02,
            'velocity_fidelity': metrics['velocity_nrmse'] <= .15,
            'contacts': metrics['contact_f1'] >= .90,
            'travel_energy': bool(energy and travel)}
