import json,shutil
from pathlib import Path
from lightsem.experiments.stage11.stage11_common import freeze_sources,digest
ROOT=Path('/mnt/lightsem-data/experiments/lightsem/motion_native_audio_20260910')
OUT=Path('/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/music_physical_revision_01')
assert (OUT/'registration.json').exists();assert not (OUT/'source').exists()
eng=json.loads((OUT/'engineering.json').read_text())
for p,sha in eng['source_sha256'].items():assert digest(p)==sha,'Changed after preflight '+p
assert eng['paired_draws'] and eng['prefetch_parity']
s=freeze_sources(OUT)
for p in [Path('scripts/motion/train_music_physical_revision.py'),Path('tests/test_music_revision.py'),Path('tests/test_music_physical.py')]:shutil.copy2(p,OUT/p.name)
inputs=[ROOT/'data/audio_manifest.json',ROOT/'data/audio_normalizers.json',ROOT/'data/motion_normalizers.json',ROOT/'music/protocol.json',OUT/'registration.json',OUT/'design.json',OUT/'engineering.json',OUT/'coverage.json',OUT/'train_music_physical_revision.py']
protocol={'design':json.loads((OUT/'design.json').read_text()),'source_sha256':s['source_sha256'],'inputs':{str(p):digest(p) for p in inputs},'orchestration':'optional fixed5000-update stops with exact model/optimizer/scheduler/global and sampler RNG checkpoint resume; no interim VAL or selection; schedule remains30000 per arm','source_environment':'dedicated /mnt/lightsem-data/venv-torch; BF16 neural, FP32 FK, compile vector field only; paired32step GPU preflight passed'}
(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2));print(s['source_sha256'])
