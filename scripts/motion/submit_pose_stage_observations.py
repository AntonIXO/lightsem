import importlib.util,json,hashlib,sys
from pathlib import Path
OUT=Path('/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/pose_stage_01')
p=OUT/'frozen/render_pose_stage_study.py';spec=importlib.util.spec_from_file_location('frozen_pose_renderer_submission',p);r=importlib.util.module_from_spec(spec);sys.modules[spec.name]=r;spec.loader.exec_module(r);ev=r.load_evaluator()
registry=json.loads(Path('experiments/2026-09-10-motion/outputs/loose/pose_stage_observation_registry.json').read_text())
for row in registry:
 path=Path(row['path']);assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
 if (OUT/'reviews'/(row['entry_id']+'.json')).exists():
  # Let the frozen report/next-review path authenticate completed receipts; no replacement.
  continue
 r.run(OUT,1,ev,stage='review',observation=path)
print('All scheduled observer submissions processed; sampled coverage only.')
