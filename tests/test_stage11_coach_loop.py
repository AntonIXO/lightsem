import numpy as np

from lightsem.experiments.stage11.stage11_data import _extract_coach_tracks


class _Bus:
    beats = np.asarray([0.0, 0.5, 1.0])
    tempo = 120.0


class _Descriptor:
    def __init__(self):
        self.channels = {'speed': np.ones(4, dtype=np.float32)}
        self.events = {'footfall': np.asarray([0.5])}
        self.meta = {}


def test_coach_failure_does_not_drop_later_tracks(tmp_path):
    calls = []

    def to_skeleton(k, c, fps, *, name, stats):
        calls.append(('skeleton', name))
        if name.endswith('_c0'):
            raise ValueError('synthetic coach-0 failure')
        return type('Skeleton', (), {'duration_s': 1.0})()

    def extract(sk, *, beats, tempo):
        return _Descriptor()

    def phase(events, beats):
        return {'R': 0.5, 'offset_s': 0.0}

    def save(desc, path):
        calls.append(('save', path.name))

    k = np.zeros((4, 2, 17, 2), dtype=np.float32)
    c = np.ones((4, 2, 17), dtype=np.float32)
    job = {
        'recording_id': 'rec', 'canonical_id': 'song', 'partition': 'train',
        'coaches': [
            {'coach_id': 'rec_c0', 'coach_index': 0, 'coverage': 0.9},
            {'coach_id': 'rec_c1', 'coach_index': 1, 'coverage': 0.8},
        ],
    }
    meta = {'fps_eff': 30.0, 'fps_src': 30.0, 'model': 'test', 'imgsz': 640}
    coaches, failures = _extract_coach_tracks(
        job, k, c, 0, 4, 0.0, np.asarray([0.9, 0.8]), meta, _Bus(), tmp_path,
        to_skeleton_fn=to_skeleton, extract_fn=extract, phase_fn=phase, save_fn=save)

    assert [row['coach_id'] for row in coaches] == ['rec_c1']
    assert failures == [{'coach_id': 'rec_c0', 'coach_index': 0,
                         'reason': 'ValueError: synthetic coach-0 failure'}]
    assert calls == [('skeleton', 'rec_c0'), ('skeleton', 'rec_c1'), ('save', 'rec_c1.npz')]
