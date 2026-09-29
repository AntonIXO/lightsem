from pathlib import Path
import shutil,json,hashlib
from lightsem.experiments.stage11_common import freeze_sources
OUT=Path('/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/stage_geometry_02')
assert (OUT/'registration.json').exists()
assert not (OUT/'source').exists()
s=freeze_sources(OUT)
for path in ['outputs/eval_dance_stage_anchors.py','tests/test_stage_lighting_integration.py']:
 p=Path(path);dest=OUT/p.name;shutil.copy2(p,dest)
inputs=[Path('/mnt/lightsem-data/experiments/lightsem/motion_native_audio_20260910')/x for x in ['delivery/final_diagnostic/inference/shpongle.npz','data/shpongle/loudness.npz','delivery/surface_revision_02/articulated_surface.npz','delivery/surface_revision_02/articulated_surface_trace.npz']]
inputs += [OUT.parent/'stage_geometry_01/candidate.npz',OUT.parent/'stage_geometry_01/design.json',OUT/'design.json',OUT/'eval_dance_stage_anchors.py',OUT/'test_stage_lighting_integration.py']
(OUT/'inputs.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},indent=2))
print(s['source_sha256'])
