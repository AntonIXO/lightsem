"""Reuse the verified sequential guard with a separately bounded campaign identity."""
from pathlib import Path
import importlib.util
import sys

_spec=importlib.util.spec_from_file_location('_lightsem_d4max_guard_implementation',Path(__file__).with_name('dynamic_dance_guard.py'))
implementation=importlib.util.module_from_spec(_spec)
sys.modules[_spec.name]=implementation
_spec.loader.exec_module(implementation)

implementation.GUARD=Path(__file__).resolve()
implementation.CAMPAIGN_ROOT=Path('/mnt/lightsem-data/experiments/lightsem/d4max_20261005_v1')
implementation.STAGE_SECONDS={'features':0,'training':0,'evaluation':0,'rendering':2*3600}
implementation.TOTAL_GPU_SECONDS=2*3600
implementation.PREFIX='LIGHTSEM_D4MAX_'
authenticate_guarded_job=implementation.authenticate_guarded_job
verify_runtime_imports=implementation.verify_runtime_imports
if __name__=='__main__':raise SystemExit(implementation.main())
