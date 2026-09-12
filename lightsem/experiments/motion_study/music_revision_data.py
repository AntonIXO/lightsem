"""Opt-in TRAIN resident with zero-padding and canonical-equal crop sampling.

Original MusicResident is untouched. b stores a BOOLEAN frame mask, so the
existing BatchStream gather interface fits without changes. CPU supports depth=0;
BatchStream depth=2 is CUDA-only and still needs separately authorized preflight.
"""
from collections import defaultdict

import numpy as np
import torch

from ..stage11_common import digest


class MusicRevisionResident:
    def __init__(self, manifest, audio_norms, frames=960, device='cpu'):
        records = []
        for entry in sorted((e for e in manifest['entries'] if e['partition'] == 'train'),
                            key=lambda e: e['key']):
            if digest(entry['path']) != entry['sha256'] or digest(entry['audio_path']) != entry['audio_sha256']:
                raise ValueError('Paired TRAIN artifact changed')
            with np.load(entry['audio_path']) as data:
                audio = data['x'].copy()
            with np.load(entry['path']) as data:
                features = data['features'].copy()
            if len(features) != entry['n_frames']:
                raise ValueError('Manifest frame count mismatch')
            records.append({**entry, 'audio': audio, 'features': features})
        self._initialize(records, audio_norms, frames, device)

    @classmethod
    def from_recordings(cls, records, audio_norms, frames=960, device='cpu'):
        """In-memory interface for synthetic tests or already authenticated arrays.

        TRAIN records require key/canonical_id/partition/audio/features. This does
        not authenticate disk artifacts: production should use the manifest loader.
        Non-TRAIN arrays are not inspected or sampled.
        """
        instance = cls.__new__(cls)
        instance._initialize(records, audio_norms, frames, device)
        return instance

    def _initialize(self, records, audio_norms, frames, device):
        if not isinstance(frames, int) or frames < 1:
            raise ValueError('frames must be a positive integer')
        mean, std = (np.asarray(audio_norms[k], np.float32) for k in ('mean', 'std'))
        if mean.shape != (92,) or std.shape != (92,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or not (std > 0).all():
            raise ValueError('Expected finite Base92 normalizers and positive std')
        records = sorted((e for e in records if e['partition'] == 'train'), key=lambda e: e['key'])
        if not records or len({e['key'] for e in records}) != len(records):
            raise ValueError('Need nonempty TRAIN with unique recording keys')
        self.entries, grouped = [], defaultdict(list)
        arrays, targets, masks, starts, lengths, contacts = [], [], [], [], [], []
        cursor = 0
        for index, record in enumerate(records):
            audio, features = np.asarray(record['audio'], np.float32), np.asarray(record['features'], np.float32)
            if features.ndim != 2 or features.shape[1] != 139 or len(features) < 1:
                raise ValueError('Every recording needs at least one native139 frame')
            if audio.shape != (len(features), 92) or not np.isfinite(audio).all() or not np.isfinite(features).all():
                raise ValueError('Expected finite, frame-aligned audio/motion')
            n, capacity = len(features), max(frames, len(features))
            x, y = np.zeros((capacity, 92), np.float32), np.zeros((capacity, 139), np.float32)
            x[:n], y[:n] = (audio-mean)/std, features
            if not np.isfinite(x).all():
                raise ValueError('Nonfinite normalized audio')
            arrays.append(x)
            targets.append(y)
            masks.append(np.arange(capacity) < n)
            starts.append(cursor)
            lengths.append(n)
            contacts.append(int((features[:, 135:] >= .5).any(axis=1).sum()))
            cursor += capacity
            grouped[record['canonical_id']].append(index)
            self.entries.append({k: v for k, v in record.items() if k not in ('audio', 'features')})
        self.frames, self.device = frames, torch.device(device)
        self.host_aux, self.aux_dim = None, 0
        self.groups = [grouped[key] for key in sorted(grouped)]
        self.starts, self.lengths = np.asarray(starts, np.int64), np.asarray(lengths, np.int64)
        self.x = torch.from_numpy(np.concatenate(arrays)).to(self.device)
        self.y = torch.from_numpy(np.concatenate(targets)).to(self.device)
        self.b = torch.from_numpy(np.concatenate(masks)).to(self.device)
        probabilities = {i: 1/(len(self.groups)*len(group)) for group in self.groups for i in group}
        self.coverage = {
            'recordings': len(records), 'canonical_songs': len(self.groups),
            'short_recordings': int((self.lengths < frames).sum()),
            'real_frames': int(self.lengths.sum()), 'padding_frames': int(cursor-self.lengths.sum()),
            'expected_valid_frames_per_crop': sum(probabilities[i]*min(n, frames) for i, n in enumerate(lengths)),
            'expected_valid_edges_per_crop': sum(probabilities[i]*(min(n, frames)-1) for i, n in enumerate(lengths)),
            'hold_coverage': 'not measured: any reference contact is not a validated hold label',
            'rows': [{'key': e['key'], 'canonical_id': e['canonical_id'], 'real_frames': lengths[i],
                      'valid_frames_per_crop': min(lengths[i], frames), 'sampling_probability': probabilities[i],
                      'frames_with_any_reference_contact': contacts[i]}
                     for i, e in enumerate(self.entries)],
            'sampling': 'uniform canonical song, uniform recording within song, uniform contiguous crop; short recordings start at zero without repetition',
        }

    def draw_indices(self, size, frames, rng):
        """Indices replay exactly from caller's NumPy Generator state.

        b[indices] is the validity mask. Padding occupies its own resident region;
        no gather wraps or spills into another recording. Counts are equal-song
        probabilities, not a guarantee every finite batch contains every song.
        """
        if frames != self.frames or not isinstance(size, int) or size < 1:
            raise ValueError('Use positive batch size and the resident crop length')
        selected = []
        for _ in range(size):
            group = self.groups[int(rng.integers(len(self.groups)))]
            selected.append(group[int(rng.integers(len(group)))])
        offsets = np.asarray([rng.integers(max(1, self.lengths[i]-frames+1)) for i in selected])
        return self.starts[selected, None] + offsets[:, None] + np.arange(frames)[None]

    def batch(self, size, frames, rng):
        """Return normalized audio (B,T,92), raw motion (B,T,139), bool valid (B,T)."""
        indices = torch.as_tensor(self.draw_indices(size, frames, rng), device=self.device)
        return self.x[indices], self.y[indices], self.b[indices]
