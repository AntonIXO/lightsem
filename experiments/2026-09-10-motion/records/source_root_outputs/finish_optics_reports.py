import importlib.util,sys
from pathlib import Path
OUT=Path('/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/optics_full_controls_01')
s=importlib.util.spec_from_file_location('frozen_optics',OUT/'frozen/run_dance_optics_full.py');m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m);loaded=m.load_frozen(OUT)
m.assemble_pair(OUT,loaded);m.report(OUT,loaded)
