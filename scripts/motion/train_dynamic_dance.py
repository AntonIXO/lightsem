"""D6--D8 campaign CLI. See --help; never starts a GPU job without admission."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lightsem.experiments.dynamic_dance.training import main

if __name__ == '__main__':
    main()
