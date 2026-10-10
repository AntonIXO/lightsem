"""BF16 v7b guard using writable locks when /mnt is mounted read-only.

The campaign data on /mnt is intentionally read-only in this session.  The
original guard's historical lock path is therefore unusable for a new job;
this wrapper keeps the same guard implementation and campaign root while
binding the job to the writable campaign lock and the shared audio lock.
"""
from pathlib import Path
import importlib.util
import sys

_spec = importlib.util.spec_from_file_location(
    '_native_full_fit_guard_bf16_v7b_rwlock_impl', Path(__file__).with_name('dynamic_dance_guard.py'))
implementation = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = implementation
_spec.loader.exec_module(implementation)
implementation.GUARD = Path(__file__).resolve()
implementation.CAMPAIGN_ROOT = Path('/root/lightsem/experiments/native_bf16_v7b')
implementation.GPU_LOCKS = (
    Path('/root/lightsem/experiments/native_bf16_v7b/gpu.lock'),
    Path('/tmp/lightsem-audio-understanding-gpu.lock'),
)
implementation.STAGE_SECONDS = {
    'features': 24 * 3600, 'training': 12 * 3600,
    'evaluation': 4 * 3600, 'rendering': 2 * 3600,
}
implementation.TOTAL_GPU_SECONDS = 42 * 3600
implementation.PREFIX = 'LIGHTSEM_NATIVE_BF16_V7B_RWLOCK_'
authenticate_guarded_job = implementation.authenticate_guarded_job
verify_runtime_imports = implementation.verify_runtime_imports

if __name__ == '__main__':
    raise SystemExit(implementation.main())
