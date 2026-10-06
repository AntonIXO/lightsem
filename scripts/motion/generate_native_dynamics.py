from pathlib import Path
import argparse
from lightsem.experiments.native_dynamics.data import ROOT
from lightsem.experiments.native_dynamics.showcase import prepare,generate
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','generate']);p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--arm');p.add_argument('--seed',type=int,default=0);p.add_argument('--song');p.add_argument('--guarded',action='store_true');a=p.parse_args()
    if a.stage=='generate':
        if not a.guarded:raise ValueError('GPU guard required')
        from scripts.motion.native_dynamics_guard import authenticate_guarded_job
        authenticate_guarded_job(Path(__file__))
        generate(a.root,a.arm,a.seed,a.song)
    else:prepare(a.root)
