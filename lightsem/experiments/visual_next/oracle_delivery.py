"""Full-track presentation of fixed, explicitly privileged oracle decisions."""
from __future__ import annotations

import numpy as np

from lightsem.conductor.rig import canonical_rig
from lightsem.model.light_layers.overlay import apply_overlay
from lightsem.model.light_layers.transitions import fade_rgb_targets
from .oracle import GROUP_ORDER
from .palettes import fixture_groups, group_rgb_mask


def oracle_color_delivery(baseline, palette_ids40, cue_indices, assignments, palette_candidates):
    base = np.asarray(baseline)
    ids = np.asarray(palette_ids40)
    cues = np.asarray(cue_indices)
    choices = np.asarray(assignments)
    palette = np.asarray(palette_candidates)
    if base.dtype != np.uint8 or base.ndim != 2 or base.shape[1] != 512:
        raise ValueError('Expected saved C universe stream')
    if ids.shape != (len(base),) or ids.dtype.kind not in 'iu':
        raise ValueError('Expected the original C palette timeline')
    if cues.ndim != 1 or not len(cues) or cues.dtype.kind not in 'iu' or np.any(np.diff(cues) <= 0):
        raise ValueError('Ordered palette cues required')
    if cues[0] < 1 or cues[-1] >= len(base) or choices.shape != (len(cues), 4):
        raise ValueError('Oracle schedule does not fit the full stream')
    if choices.dtype.kind not in 'iu' or np.any((choices < 0) | (choices > 3)):
        raise ValueError('Only the four frozen palette modes are allowed')
    if palette.shape != (32, 4, 3) or palette.dtype != np.uint8:
        raise ValueError('Frozen uint8 palette candidates required')
    times = np.arange(len(base)) / 40
    active = np.arange(len(base)) >= cues[0]
    if np.any((ids[active] < 0) | (ids[active] >= 32)):
        raise ValueError('Invalid active palette identity')
    # The inherited first decision is held before110s, and the final decision
    # is held after174s. Actual C palette changes still supply all later colors.
    decision = np.clip(np.searchsorted(cues, np.arange(len(base)), side='right') - 1, 0, len(cues) - 1)
    modes = choices[decision]
    rig = canonical_rig()
    groups = fixture_groups(rig)
    tracks = {}
    for fixture in groups:
        slots = fixture['rgb_slots_zero_based']
        group = GROUP_ORDER.index(fixture['o4'])
        target = base[:, slots].astype(float) / 255
        target[active] = palette[ids[active], modes[active, group]] / 255
        # Initialize each fixture from its actual pre-adoption RGB. A pooled
        # group origin would be wrong during the first fade of the full video.
        tracks[fixture['name']] = fade_rgb_targets(target, times, duration_s=.3, active=active)
    stream = apply_overlay(base, rig, times, rgb=tracks)
    mask = group_rgb_mask(groups)
    if not np.array_equal(stream[:, ~mask], base[:, ~mask]):
        raise RuntimeError('Oracle presentation changed protected non-RGB slots')
    if not np.array_equal(stream[~active], base[~active]):
        raise RuntimeError('Oracle presentation changed pre-adoption frames')
    return stream, {
        'protected_nonRGB_slots': int((~mask).sum()), 'fade_s': .3,
        'optimized_interval_s': [110., 174.], 'reference_information_used': True,
        'outside_interval': 'First/last group-mode decisions are carried outside the optimized interval; C palette IDs and cue timing continue unchanged.',
        'changed_RGB_bytes_vs_C': int(np.count_nonzero(stream != base)),
        'scope': 'Privileged reference-guided color-capacity example, not an audio-learned spatial model.',
    }
