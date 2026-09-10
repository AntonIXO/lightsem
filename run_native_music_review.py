"""Run the separately frozen visual harness without changing the music model."""
import runpy
import sys
from pathlib import Path

snapshot = Path('/mnt/lightsem-data/experiments/lightsem/motion_native_audio_20260910/music/review/source')
if not snapshot.is_dir() or 'lightsem' in sys.modules:
    raise RuntimeError('A frozen visual harness and fresh process are required')
sys.path.insert(0,str(snapshot))
runpy.run_module('lightsem.experiments.motion_study.music_review',run_name='__main__')
