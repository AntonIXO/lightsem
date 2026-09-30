"""Pre-render output inactivity, degeneracy and cross-input checks.

These are engineering diagnostics, not music-fit or generalization scores.
"""
import numpy as np


def finite_array(value, name):
    result = np.asarray(value)
    if not result.size or not np.isfinite(result).all():
        raise ValueError(f'{name} must be nonempty and finite')
    return result


def motion_activity(angles, *, fps=40., audio_active=None):
    a = finite_array(angles, 'angles').astype(float)
    if a.ndim != 3 or a.shape[1:] != (6, 2) or len(a) < 3:
        raise ValueError('Expected T x six heads x pan/tilt degrees')
    speed = np.linalg.norm(np.diff(a, axis=0)*fps, axis=-1)
    acceleration = np.linalg.norm(np.diff(a, n=2, axis=0)*fps**2, axis=-1)
    active = np.ones(len(a), bool) if audio_active is None else np.asarray(audio_active)
    if active.dtype != bool or active.shape != (len(a),):
        raise ValueError('Audio activity must match the angle clock')
    edges = active[1:] & active[:-1]
    return {
        'pan_tilt_range_deg_by_head': np.ptp(a, axis=0).tolist(),
        'pan_tilt_std_deg_by_head': a.std(axis=0).tolist(),
        'median_speed_deg_s': float(np.median(speed)),
        'p95_speed_deg_s': float(np.quantile(speed, .95)),
        'maximum_speed_deg_s': float(speed.max()),
        'maximum_acceleration_deg_s2': float(acceleration.max()),
        'fraction_head_frames_below_1_deg_s': float(np.mean(speed < 1.)),
        'audio_active_frame_fraction': float(active.mean()),
        'active_median_speed_deg_s': float(np.median(speed[edges])) if edges.any() else None,
        'constant_motion': bool(np.all(a == a[:1])),
        'scope': 'Mechanical activity only; shared procedural vocabulary may remain.',
    }


def palette_activity(ids, cues, scores):
    ids, cues, scores = [finite_array(v, n) for v, n in zip((ids, cues, scores), ('ids', 'cues', 'scores'))]
    if ids.ndim != 1 or ids.shape != cues.shape or ids.shape != scores.shape or cues.dtype != bool:
        raise ValueError('Matching palette ids, boolean cues and probabilities required')
    if ids.dtype.kind not in 'iu' or np.any((ids < 0) | (ids >= 32)) or np.any((scores < 0) | (scores > 1)):
        raise ValueError('Invalid palette identities or probabilities')
    indices = np.flatnonzero(cues)
    if len(indices):
        active_ids = ids[indices[0]:]
        unique, counts = np.unique(active_ids, return_counts=True)
        probability = counts/counts.sum()
        occupancy = {str(int(k)): float(v) for k, v in zip(unique, probability)}
        entropy = float(-np.sum(probability*np.log2(probability)))
        changes = int(np.count_nonzero(np.diff(active_ids)))
    else:
        occupancy, entropy, changes = {}, None, 0
    return {
        'cue_count': int(cues.sum()), 'first_cue_index10': int(indices[0]) if len(indices) else None,
        'subsequent_palette_changes': changes, 'active_palette_occupancy': occupancy,
        'active_palette_entropy_bits': entropy,
        'dominant_active_palette_fraction': max(occupancy.values()) if occupancy else None,
        'color_probability_std': float(scores.std()),
        'color_probability_range': [float(scores.min()), float(scores.max())],
        'inactive_learned_color_head': not bool(len(indices)),
        'single_palette_after_activation': bool(len(indices) and changes == 0),
        'scope': 'Cue inactivity or palette concentration does not alone prove representation collapse.',
    }


def representation_activity(context):
    x = finite_array(context, 'context').astype(float)
    if x.ndim != 2 or len(x) < 2:
        raise ValueError('Expected time by feature context')
    centered = x-x.mean(0)
    eigenvalues = np.linalg.eigvalsh(centered.T@centered/len(x)).clip(0)
    energy = eigenvalues.sum()
    weights = eigenvalues[eigenvalues > 0]/energy if energy else np.empty(0)
    return {'mean_coordinate_std': float(x.std(0).mean()),
            'effective_covariance_rank': float(np.exp(-np.sum(weights*np.log(weights)))) if len(weights) else 0.,
            'constant_context': bool(np.all(x == x[:1])),
            'scope': 'Native inference context; descriptive, not a standalone proof of music conditioning.'}


def cross_input_motion(first, second):
    a, b = finite_array(first, 'first').astype(float), finite_array(second, 'second').astype(float)
    if a.shape != b.shape:
        raise ValueError('Compare the same clock, duration and fixture ordering')
    ac, bc = a-a.mean(0), b-b.mean(0)
    denominator = np.linalg.norm(ac)*np.linalg.norm(bc)
    return {'angle_rms_difference_deg': float(np.sqrt(np.mean((a-b)**2))),
            'mean_absolute_angle_difference_deg': float(np.mean(np.abs(a-b))),
            'centered_angle_cosine': float(np.sum(ac*bc)/denominator) if denominator else None,
            'exactly_identical_angles': bool(np.array_equal(a, b)),
            'scope': 'Different angles establish input sensitivity, not a new motion vocabulary or music-fit.'}
