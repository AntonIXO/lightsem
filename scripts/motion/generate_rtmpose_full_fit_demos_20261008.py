"""Generate independent Shpongle/Cure For Me body and DMX diagnostics."""
from pathlib import Path
import argparse
from lightsem.experiments.native_dynamics import showcase

def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); a=p.parse_args()
    if not (a.root/'learning/runs/joint/seed_0/final.pt').exists():
        raise FileNotFoundError('new seed-0 final checkpoint is required')
    showcase.prepare(a.root)
    for song in ('shpongle','cure_for_me'):
        showcase.generate(a.root,'joint',0,song)
    print('generated independent native body/DMX diagnostics for both songs')

if __name__=='__main__': main()
