"""Canonicalize negligible negative *display rates*, never saved body clocks.

The regional clock builder combines accumulated prefixes, endpoint assignments
and affine blends. A mathematically held interval can therefore end a few
float64 ULPs below its start. Subtracting those two stored endpoints may be exact;
this module does not attribute the error to finite differencing itself.

Eight LOCAL endpoint ULPs is a strict numerical admission budget, not a proof of
the upstream algorithm's complete forward-error bound. Larger backwards steps
remain errors. No source timestamp, pose, contact, native channel or positive
rate is modified. This helper is only for explicitly classified pure retiming;
recomposition's genuine fragment jumps must retain their separate provenance.
"""
from __future__ import annotations

import hashlib

import numpy as np

LOCAL_ULP_BUDGET = 8


def nonnegative_effort(source_time, time, fps=None):
    """Return ``(rates[T,R], audit)`` with eligible negative rates set to zero.

    With ``fps`` provided, preserve the existing plan arithmetic
    ``diff(source_time) * fps`` exactly. Without it, preserve the mapper's
    ``diff(source_time) / diff(time)`` arithmetic. The last interval is repeated
    as the last displayed rate. Results remain float64; callers may apply their
    original serialization cast afterward. Nonnegative paths are bit exact.
    """
    source = np.asarray(source_time)
    clock = np.asarray(time)
    if source.dtype != np.float64 or clock.dtype != np.float64:
        raise ValueError('Telemetry roundoff admission requires original float64 clocks')
    if source.ndim not in (1, 2) or (source.ndim == 2 and source.shape[1] < 1):
        raise ValueError('Expected source timestamps [T] or [T,R]')
    if clock.ndim != 1 or len(clock) != len(source) or len(clock) < 2:
        raise ValueError('Expected matching clocks with at least two frames')
    if not np.isfinite(source).all() or not np.isfinite(clock).all() or np.any(source < 0) or np.any(clock < 0):
        raise ValueError('Clocks must be finite and nonnegative')
    dt = np.diff(clock)
    if np.any(dt <= 0):
        raise ValueError('Output timestamps must be strictly increasing')
    if fps is not None:
        if not np.isscalar(fps) or not np.isfinite(fps) or fps <= 0:
            raise ValueError('FPS must be finite and positive')
        if not np.allclose(dt, 1./fps, rtol=0, atol=1e-8):
            raise ValueError('Declared FPS does not match the uniform output clock')
    values = source[:, None] if source.ndim == 1 else source
    delta = np.diff(values, axis=0)
    # A global maximum could admit many thousands of ULPs in an early, small
    # source timestamp. Each interval gets only its own endpoint-scale budget.
    magnitude = np.maximum(np.abs(values[:-1]), np.abs(values[1:]))
    local_ulp = np.spacing(magnitude)
    bound_seconds = LOCAL_ULP_BUDGET * local_ulp
    negative = delta < 0
    inadmissible = negative & (-delta > bound_seconds)
    if np.any(inadmissible):
        frame, region = np.argwhere(inadmissible)[0]
        raise ValueError('Backward source-clock step exceeds local float64 roundoff budget: '
                         f'frame={int(frame)}, region={int(region)}, '
                         f'delta_s={delta[frame,region]!r}, bound_s={bound_seconds[frame,region]!r}')
    interval_rates = delta*fps if fps is not None else delta/dt[:, None]
    if not np.isfinite(interval_rates).all():
        raise ValueError('Nonfinite derived source-clock rates')
    entries = []
    for frame, region in np.argwhere(negative):
        bound_rate = bound_seconds[frame, region]*fps if fps is not None else bound_seconds[frame, region]/dt[frame]
        entries.append({'frame': int(frame), 'region': int(region),
                        'source_start_s': float(values[frame, region]),
                        'source_end_s': float(values[frame+1, region]),
                        'delta_s': float(delta[frame, region]),
                        'local_endpoint_ulp_s': float(local_ulp[frame, region]),
                        'negative_step_ulps': float(-delta[frame, region]/local_ulp[frame, region]),
                        'bound_s': float(bound_seconds[frame, region]),
                        'original_rate': float(interval_rates[frame, region]),
                        'rate_bound': float(bound_rate), 'display_rate': 0.})
    original_min = float(interval_rates.min())
    # Assignment touches only admitted negative entries. In particular, do not
    # clamp the upper cap or round/cast successful nonnegative values.
    interval_rates[negative] = 0.
    rates = np.concatenate((interval_rates, interval_rates[-1:]), axis=0)
    audit = {'kind': 'derived_clock_rate_telemetry_only',
             'status': 'canonicalized_local_float64_roundoff' if entries else 'unchanged',
             'source_time_modified': False, 'body_modified': False,
             'source_time_bytes_sha256': hashlib.sha256(np.ascontiguousarray(source).tobytes()).hexdigest(),
             'time_bytes_sha256': hashlib.sha256(np.ascontiguousarray(clock).tobytes()).hexdigest(),
             'clock_dtype': 'float64', 'local_ulp_budget': LOCAL_ULP_BUDGET,
             'bound_definition': '8 * spacing(max(abs(source[i,r]),abs(source[i+1,r]))), independently per interval',
             'bound_interpretation': 'strict local clock-construction roundoff admission budget; not a universal forward-error proof',
             'rate_arithmetic': 'diff(source_time)*fps' if fps is not None else 'diff(source_time)/diff(time)',
             'fps': None if fps is None else float(fps),
             'negative_interval_count': len(entries),
             'negative_display_entry_count': int(negative.sum()+negative[-1].sum()),
             'minimum_original_interval_rate': original_min,
             'minimum_original_interval_delta_s': float(delta.min()),
             'maximum_admitted_negative_delta_s': max((-entry['delta_s'] for entry in entries), default=0.),
             'maximum_applied_bound_s': max((entry['bound_s'] for entry in entries), default=0.),
             'entries': entries}
    return rates, audit
