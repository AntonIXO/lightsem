from pathlib import Path
import argparse
from scripts.motion.native_dynamics_guard import authenticate_guarded_job
from lightsem.experiments.native_repair.assessment import evaluate
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--guarded',action='store_true');a=p.parse_args()
authenticate_guarded_job(__file__);evaluate(a.root,a.seed)
