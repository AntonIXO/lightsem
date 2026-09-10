"""Execute the frozen motion and lighting diagnostic delivery."""
import sys
import runpy
from pathlib import Path
snapshot=Path('/mnt/lightsem-data/experiments/lightsem/motion_native_audio_20260910/delivery/final_diagnostic/source')
if not snapshot.is_dir() or 'lightsem' in sys.modules:raise RuntimeError('Frozen sources and fresh process required')
sys.path.insert(0,str(snapshot))
runpy.run_module('lightsem.experiments.motion_study.delivery',run_name='__main__')
