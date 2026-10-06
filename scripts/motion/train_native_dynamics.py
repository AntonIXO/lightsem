"""Guard-authenticated CLI for native Just Dance dynamics learning."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from lightsem.experiments.native_dynamics.training import main
if __name__=='__main__':main()
