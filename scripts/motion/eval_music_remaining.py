"""Schedule complete frozen supplemental cells; no parameter/checkpoint selection."""
import argparse,importlib.util,sys
from pathlib import Path
import torch
OUT=Path('/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/music_physical_revision_01/evaluation')
a=argparse.ArgumentParser();a.add_argument('stage',choices=('short','full','shpongle'));args=a.parse_args()
path=OUT/'evaluator.py';s=importlib.util.spec_from_file_location('frozen_music_evaluator',path);m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m)
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
with m.output_lock(OUT):
 p,q=m.context(OUT);m.frozen_modules()
 from lightsem.experiments.stage11.stage11_common import gpu_lock
 with gpu_lock(m.NATIVE):
  for arm in m.ARMS:
   for checkpoint in (m.CHECKPOINTS if args.stage=='short' else ('final',)):
    m.stage_inference(args.stage,arm,checkpoint,'cuda',100000,OUT,p,q)
print('STAGE MATRIX COMPLETE',args.stage,flush=True)
