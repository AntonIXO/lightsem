"""Resume contracts: exact clocks, immutable inputs, independent output roots."""
import json
from pathlib import Path
import numpy as np
import pytest
from lightsem.experiments.native_dynamics.feature_cache import verify_training_clock
from lightsem.experiments.native_dynamics.full_fit_admission import sha


def test_finedance_audio_tail_keeps_exact_motion_prefix():
    time = np.arange(2217, dtype=np.float64) / 30
    target = np.arange(2191, dtype=np.float64) / 30
    verify_training_clock(time, target, 'fd')
    with pytest.raises(ValueError):
        verify_training_clock(time, target, 'jd')


@pytest.mark.parametrize('case', ['short', 'shift', 'nonfinite_tail', 'repeated_tail'])
def test_tail_permission_never_accepts_changed_or_incomplete_clock(case):
    target = np.arange(2191, dtype=np.float64) / 30
    time = np.arange(2217, dtype=np.float64) / 30
    if case == 'short': time = time[:2190]
    if case == 'shift': time += .001
    if case == 'nonfinite_tail': time[-1] = np.nan
    if case == 'repeated_tail': time[-1] = time[-2]
    with pytest.raises(ValueError):
        verify_training_clock(time, target, 'fd')


def test_recovery_reuses_partial_without_overwriting_inputs(tmp_path, monkeypatch):
    from scripts.motion import native_dynamics_features_reuse as m
    root, old, partial = (tmp_path/n for n in ('recovery', 'old', 'partial'))
    (root/'data').mkdir(parents=True)
    wave=tmp_path/'wave'; wave.write_bytes(b'wave')
    aligned=tmp_path/'aligned.npz'; np.savez(aligned, x=np.zeros((2,92)))
    def row(key, domain):
        return dict(key=key,feature_key=key,domain=domain,partition='train',
                    wav_path=str(wave),wav_sha256=sha(wave),frames=2,
                    timeline_path=key+'.npz',audio_path=str(aligned))
    jd, fd = row('jd_song','jd'), row('fd_song','fd')
    for path, data in [('data/MANIFEST.json',{'entries':[jd]}),
                       ('data/FD_MANIFEST.json',{'entries':[fd]}),
                       ('TRAINING_PROTOCOL.json',{'muq':{}})]:
        (root/path).write_text(json.dumps(data))
    before=sha(root/'data/FD_MANIFEST.json')
    monkeypatch.setattr(m,'guard_disk',lambda *a,**k:None)
    seen=[]
    def receipt(cache,key):
        if cache==old and key=='jd_song': raise FileNotFoundError(key)
        return 'wave'
    def verify(r, expected, cache, wave_id):
        seen.append((r['key'],cache))
        product=dict(feature_key=r['key'],timeline_path='verified-timeline',
                     timeline_sha256='timeline',base92_path='verified-base',
                     base92_sha256='base',reused=True)
        if r['domain']=='fd': product['fd_aligned']=str(aligned)
        return product,{'key':r['key']},{'key':r['key']}
    monkeypatch.setattr(m,'_waveform_id_from_receipt',receipt)
    monkeypatch.setattr(m,'verify_cached_features',verify)
    def forbidden(*a,**k): raise AssertionError('Cached streams must not be recomputed')
    monkeypatch.setattr(m,'AudioAnalyzer',forbidden)
    assert m.prepare(root,old,'recovery-test',partial)=={'unique':2,'reused':2,'fresh':0}
    assert seen==[('jd_song',partial),('fd_song',old)]
    assert sha(root/'data/FD_MANIFEST.json')==before
    complete=json.loads((root/'features/COMPLETE.json').read_text())
    assert complete['fd_manifest_sha256']==sha(root/'features/FD_MANIFEST.json')
    from lightsem.experiments.native_dynamics.dataset import fd_manifest_path
    assert fd_manifest_path(root)==root/'features/FD_MANIFEST.json'
    assert fd_manifest_path(root,motion_only=True)==root/'data/FD_MANIFEST.json'
