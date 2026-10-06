import argparse
from pathlib import Path
from lightsem.experiments.native_dynamics.data import ROOT
from lightsem.experiments.native_dynamics.diagnostics import audit_song
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--song',required=True);a=p.parse_args();audit_song(a.root,a.song)
