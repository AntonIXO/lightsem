"""Execute only the immutable spatial-correction snapshot."""
import runpy,sys
from pathlib import Path
snapshot=Path('/mnt/lightsem-data/experiments/lightsem/motion_native_audio_20260910/delivery/surface_revision_01/source')
if not snapshot.is_dir():raise RuntimeError('Freeze live module first')
sys.path.insert(0,str(snapshot))
runpy.run_module('lightsem.experiments.motion_study.surface_revision',run_name='__main__')
