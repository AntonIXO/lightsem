"""Execute the immutable optical calibration protocol."""
import runpy
import sys
from pathlib import Path

snapshot=Path('/mnt/lightsem-data/experiments/lightsem/motion_native_audio_20260910/exposure/source')
if not snapshot.is_dir() or 'lightsem' in sys.modules:
    raise RuntimeError('A frozen exposure snapshot and fresh process are required')
sys.path.insert(0,str(snapshot))
runpy.run_module('lightsem.experiments.motion_study.exposure_study',run_name='__main__')
