"""Import the immutable native engineering reference before any lightsem import."""
import runpy
import sys
from pathlib import Path

snapshot = Path('/mnt/lightsem-data/experiments/lightsem/motion_native_audio_20260910/codec/source')
if not snapshot.is_dir() or 'lightsem' in sys.modules:
    raise RuntimeError('A frozen snapshot and fresh process are required')
sys.path.insert(0,str(snapshot))
runpy.run_module('lightsem.experiments.motion_study.native_study',run_name='__main__')
