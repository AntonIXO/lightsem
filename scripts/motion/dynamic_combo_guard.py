"""Reuse the verified sequential guard with a separately bounded campaign identity."""
from pathlib import Path
import importlib.util
import sys

_spec=importlib.util.spec_from_file_location('_lightsem_combo_guard_implementation',Path(__file__).with_name('dynamic_dance_guard.py'))
implementation=importlib.util.module_from_spec(_spec)
sys.modules[_spec.name]=implementation
_spec.loader.exec_module(implementation)

implementation.GUARD=Path(__file__).resolve()
implementation.CAMPAIGN_ROOT=Path('/mnt/lightsem-data/experiments/lightsem/dynamic_dance_combo_20261005_v1')
implementation.STAGE_SECONDS={'features':3600,'training':0,'evaluation':3*3600,'rendering':2*3600}
implementation.TOTAL_GPU_SECONDS=6*3600
implementation.PREFIX='LIGHTSEM_COMBO_'
_current_budget_usage=implementation.budget_usage
def _including_archived_costs(root):
    usage=_current_budget_usage(root)
    archive=Path(root)/'records/ARCHIVED_JOB_COSTS_v2.json'
    if archive.exists():
        for row in implementation.read_verified(archive)['jobs']:
            implementation.verify_files({row['receipt']:row['sha256']},'Archived failure')
            terminal=implementation.read_verified(row['receipt'])
            usage[implementation.budget_group(terminal['identity']['budget_group'])]+=terminal['gpu_elapsed_seconds']
    return usage
implementation.budget_usage=_including_archived_costs
authenticate_guarded_job=implementation.authenticate_guarded_job
verify_runtime_imports=implementation.verify_runtime_imports
if __name__=='__main__':raise SystemExit(implementation.main())
