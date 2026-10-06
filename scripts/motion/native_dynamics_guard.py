"""One sequential native-dynamics campaign, within remaining user resource cap."""
from pathlib import Path
import importlib.util,sys
_spec=importlib.util.spec_from_file_location('_native_dynamics_guard_impl',Path(__file__).with_name('dynamic_dance_guard.py'))
implementation=importlib.util.module_from_spec(_spec);sys.modules[_spec.name]=implementation;_spec.loader.exec_module(implementation)
implementation.GUARD=Path(__file__).resolve()
implementation.CAMPAIGN_ROOT=Path('/mnt/lightsem-data/experiments/lightsem/native_dynamics_jd_20261005_v1')
implementation.STAGE_SECONDS={'features':4*3600,'training':10*3600,'evaluation':2*3600,'rendering':3600}
implementation.TOTAL_GPU_SECONDS=17*3600
implementation.PREFIX='LIGHTSEM_NATIVE_JD_'
authenticate_guarded_job=implementation.authenticate_guarded_job
verify_runtime_imports=implementation.verify_runtime_imports
if __name__=='__main__':raise SystemExit(implementation.main())
