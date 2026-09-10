"""Run a music-study module strictly from its immutable source snapshot."""
import os
import runpy
import sys
from pathlib import Path

root = Path('/mnt/lightsem-data/experiments/lightsem/motion_native_audio_20260910')
snapshot = root/'music/source'
if not snapshot.is_dir() or 'lightsem' in sys.modules:
    raise RuntimeError('A frozen source snapshot and fresh process are required')
stage = sys.argv.pop(1)
if stage not in ('training','evaluation','selection'):
    raise ValueError('Unknown music study module')
sys.path.insert(0,str(snapshot))
os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR',str(root/'compiler_cache'))
runpy.run_module('lightsem.experiments.motion_study.music_'+stage,run_name='__main__')
