"""Guard for the new BF16 RTMPose all-data campaign root."""
from pathlib import Path
import importlib.util
import sys

_spec = importlib.util.spec_from_file_location(
    '_native_full_fit_guard_bf16_v7b_impl', Path(__file__).with_name('dynamic_dance_guard.py'))
implementation = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = implementation
_spec.loader.exec_module(implementation)
implementation.GUARD = Path(__file__).resolve()
implementation.CAMPAIGN_ROOT = Path('/root/lightsem/experiments/native_bf16_v7b')
implementation.STAGE_SECONDS = {
    'features': 24 * 3600, 'training': 12 * 3600,
    'evaluation': 4 * 3600, 'rendering': 2 * 3600,
}
implementation.TOTAL_GPU_SECONDS = 42 * 3600
implementation.PREFIX = 'LIGHTSEM_NATIVE_BF16_V7B_'
authenticate_guarded_job = implementation.authenticate_guarded_job
verify_runtime_imports = implementation.verify_runtime_imports

if __name__ == '__main__':
    raise SystemExit(implementation.main())
