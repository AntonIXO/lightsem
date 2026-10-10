"""Admission-package utilities kept independent from model training."""
from __future__ import annotations

from collections import defaultdict
import numpy as np


def group_balanced_schedule(entries, *, steps: int, batch_size: int, seed: int = 0):
    """Return deterministic coach entries sampled uniformly by canonical group.

    A batch contains at most one entry from each group.  Groups are shuffled
    in cycles, so a group is selected once per cycle before any group repeats;
    recording/coach choice is uniform within that group.  This is a sampling
    schedule only and never averages coaches or creates pose data.
    """
    if steps < 0 or batch_size < 1:
        raise ValueError('steps must be nonnegative and batch_size positive')
    groups = defaultdict(list)
    for row in entries:
        group = row.get('canonical_id', row.get('group'))
        if not group:
            raise ValueError('every entry needs a canonical_id/group')
        groups[group].append(row)
    if not groups:
        raise ValueError('at least one group is required')
    rng = np.random.default_rng(seed)
    names = sorted(groups)
    batches = []
    order = []
    cursor = 0
    for _ in range(steps):
        chosen_groups = []
        while len(chosen_groups) < min(batch_size, len(names)):
            if cursor >= len(order):
                order = list(rng.permutation(names))
                cursor = 0
            group = order[cursor]; cursor += 1
            if group not in chosen_groups:
                chosen_groups.append(group)
        batches.append([groups[g][int(rng.integers(len(groups[g])))] for g in chosen_groups])
    return batches
