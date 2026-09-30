"""Circular-shift controls with representable half-open segment bounds.

The historical helper can round ``a + remainder`` to ``b`` even when the
remainder is strictly below ``b-a``. Keep such a result one representable float
inside the same interval. Offsets, source support, candidate counts and scores
are unchanged; every numerical correction is counted in the returned receipt.
"""
from __future__ import annotations

import numpy as np


def stable_shifted_candidates(candidate_times, confidences, plan):
    candidates = np.asarray(candidate_times, float)
    confidence = np.asarray(confidences, float)
    if candidates.ndim != 1 or confidence.shape != candidates.shape:
        raise ValueError('Shift candidate shape mismatch')
    if not np.isfinite(candidates).all() or not np.isfinite(confidence).all():
        raise ValueError('Nonfinite shift candidates')
    intervals = np.asarray(plan['eligible_intervals_s'], float).reshape(-1, 2)

    def supported(times):
        mask = np.zeros(len(times), bool)
        for a, b in intervals:
            mask |= (times >= a) & (times < b)
        return mask

    keep = supported(candidates)
    real, scores = candidates[keep], confidence[keep]
    if not plan['defined']:
        return {'real_times_s': real, 'confidences': scores, 'draws': None}
    draws = []
    for draw in range(20):
        shifted = real.copy()
        wrapped = close_pairs = corrections = 0
        jitter = []
        for row in plan['segments']:
            if row['offsets_s'] is None:
                continue
            a, b, duration = row['start_s'], row['stop_s'], row['duration_s']
            if b <= a or duration != b - a:
                raise ValueError('Invalid fixed shift segment')
            take = (real >= a) & (real < b)
            local = real[take] - a + row['offsets_s'][draw]
            wrapped += int(np.sum(local >= duration))
            raw = a + np.mod(local, duration)
            bounded = np.clip(raw, a, np.nextafter(b, a))
            maximum_roundoff = 2 * max(abs(np.spacing(a)), abs(np.spacing(b)))
            if np.any(np.abs(bounded - raw) > maximum_roundoff):
                raise ValueError('Shift escaped segment beyond floating-point roundoff')
            corrections += int(np.count_nonzero(raw != bounded))
            shifted[take] = bounded
            displacement = bounded - real[take]
            jitter.extend(np.abs(displacement * plan['rate'] - np.rint(displacement * plan['rate'])) / plan['rate'])
            sorted_times = np.sort(bounded)
            close_pairs += sum(int(np.sum(sorted_times[i + 1:] - t < .2 - 1e-12))
                               for i, t in enumerate(sorted_times))
        if not supported(shifted).all():
            raise ValueError('Circular shift escaped its fixed source support')
        draws.append({
            'times_s': shifted, 'confidences': scores.copy(),
            'wrapped_candidate_count': wrapped, 'close_pair_count_lt_0p2s': close_pairs,
            'wrap_grid_jitter_max_abs_s': float(max(jitter, default=0.)),
            'off_requested_grid_candidate_count': int(np.sum(np.asarray(jitter) > 1e-10)),
            'half_open_endpoint_roundoff_corrections': corrections,
        })
    return {'real_times_s': real, 'confidences': scores, 'draws': draws}
