"""Second preregistered surface diagnostic, using the shared delivery harness.

Scientific reproduction uses scripts/motion/run_surface_bandwidth.py and its immutable
revision02 snapshot. This wrapper avoids duplicating the live harness.
"""
import argparse
from . import surface_revision as study
from ..stage11.stage11_common import gpu_lock

study.OUT = study.ROOT/'delivery/surface_revision_02'

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','map','render']);a=p.parse_args()
    if a.stage=='freeze':study.freeze()
    elif a.stage=='map':study.map_lights()
    else:
        with gpu_lock(study.ROOT):study.render_lights()
