from pathlib import Path
import argparse
from lightsem.experiments.native_dynamics.data import ROOT
from lightsem.experiments.native_dynamics.evaluation import evaluate,aggregate
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['evaluate','aggregate']);p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--arm');p.add_argument('--seed',type=int,default=0);p.add_argument('--guarded',action='store_true');a=p.parse_args()
    if a.stage=='evaluate':
        if not a.guarded:raise ValueError('GPU guard required')
        from scripts.motion.native_dynamics_guard import authenticate_guarded_job
        authenticate_guarded_job(Path(__file__))
        evaluate(a.root,a.arm,a.seed)
    else:aggregate(a.root)
