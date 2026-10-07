"""Compatibility entry point for autonomous training audits.

The implementation lives in :mod:`lightsem.eval.training_health` so library
users and the command-line runner share exactly the same measurements.  Keep
this small re-export importable through ``bash run.sh`` and from tests.
"""

from lightsem.eval.training_health import (  # noqa: F401
    collapse_metrics,
    compare_training_runs,
    dump_json,
    input_sensitivity,
    load_npz_array,
    motion_metrics,
    provenance_report,
    stream_metrics,
)

__all__ = [
    "collapse_metrics",
    "compare_training_runs",
    "dump_json",
    "input_sensitivity",
    "load_npz_array",
    "motion_metrics",
    "provenance_report",
    "stream_metrics",
]
