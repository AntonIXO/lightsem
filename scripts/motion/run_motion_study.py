"""Run scientific stages from the immutable study snapshot, not the live checkout."""
import argparse
import os
import runpy
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('module', choices=['training', 'evaluate', 'selection'])
parser.add_argument('--root', type=Path, default=Path('/mnt/lightsem-data/experiments/lightsem/motion_representation_20260910'))
args, rest = parser.parse_known_args()
snapshot = args.root/'codec/source'
if not snapshot.is_dir():
    raise RuntimeError('Run the engineering preflight and freeze before scientific stages')
if 'lightsem' in sys.modules:
    raise RuntimeError('Launcher must run in a fresh process')
os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR', str(args.root/'compiler_cache'))
sys.path.insert(0, str(snapshot))
sys.argv = [args.module, *rest, '--root', str(args.root)]
runpy.run_module('lightsem.experiments.motion_study.'+args.module, run_name='__main__')
