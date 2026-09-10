"""Load only the immutable codec revision before importing scientific modules."""
import argparse
import runpy
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--root',type=Path,default=Path('/mnt/lightsem-data/experiments/lightsem/motion_representation_phasecodec_20260910'))
args,rest = parser.parse_known_args()
snapshot = args.root/'codec/source'
if not snapshot.is_dir() or 'lightsem' in sys.modules:
    raise RuntimeError('A frozen snapshot and fresh process are required')
sys.path.insert(0,str(snapshot))
sys.argv = ['phase_study',*rest,'--root',str(args.root)]
runpy.run_module('lightsem.experiments.motion_study.phase_study',run_name='__main__')
