import json
from pathlib import Path
import subprocess, sys


def test_full_manifest_is_duplicate_safe_and_uses_all_audited_groups(tmp_path):
    audit = Path('docs/research/2026-10-07-native-full-data/audit_rerun/DATA_AUDIT.json')
    fd = Path('/mnt/lightsem-data/experiments/lightsem/dynamic_dance_20261005_v1/learning/cache/manifest.json')
    out = tmp_path / 'manifest.json'
    subprocess.run([sys.executable, 'scripts/motion/build_native_full_data_manifest.py', '--audit', str(audit), '--fine-dance-manifest', str(fd), '--output', str(out)], check=True)
    data = json.loads(out.read_text())
    groups = [r['group'] for r in data['justdance']]
    assert len(groups) == 272
    assert len(groups) == len(set(groups))
    assert all(r['partition'] == 'train' for r in data['justdance'])
    assert data['counts']['fd_groups'] == 109
