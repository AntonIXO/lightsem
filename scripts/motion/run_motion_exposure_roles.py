"""Run immutable fixed-role exposure calibration."""
import runpy
import sys
from pathlib import Path
snapshot=Path('/mnt/lightsem-data/experiments/lightsem/motion_native_audio_20260910/exposure_roles/source')
if not snapshot.is_dir() or 'lightsem' in sys.modules:raise RuntimeError('Fresh process and frozen sources required')
sys.path.insert(0,str(snapshot))
runpy.run_module('lightsem.experiments.motion_study.exposure_roles',run_name='__main__')
