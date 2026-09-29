"""Final read-only media verification plus a compact delivery receipt."""
from pathlib import Path
import hashlib
import json
import os
import subprocess

ROOT = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    manifest = json.loads((ROOT / 'MANIFEST.json').read_text())
    assert manifest['status'] == 'all_ten_variants_verified'
    assert [v['id'] for v in manifest['variants']] == [f'{i:02d}' for i in range(1, 11)]
    assert sha(manifest['audio']) == manifest['audio_sha256']
    pcm_hashes, rows = set(), []
    for variant in manifest['variants']:
        path = Path(variant['path'])
        assert path.stat().st_size == variant['bytes'] and sha(path) == variant['sha256']
        audit_path = Path(variant['audit'])
        audit = json.loads(audit_path.read_text())
        assert audit['decoded_frames'] == 8319 and audit['full_software_AV_decode']
        assert audit['pcm_sha256'] == variant['pcm_sha256']
        pcm_hashes.add(audit['pcm_sha256'])
        probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams', '-of', 'json', str(path)], text=True))
        video = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        audio = next(s for s in probe['streams'] if s['codec_type'] == 'audio')
        assert (video['codec_name'], video['width'], video['height'], video['r_frame_rate']) == ('av1', 1920, 1080, '30/1')
        assert abs(float(video['duration']) - 277.3) < .001
        assert abs(float(audio['duration']) - 277.3) < .06
        rows.append({'id': variant['id'], 'path': str(path), 'sha256': variant['sha256'],
                     'bytes': variant['bytes'], 'audit_sha256': sha(audit_path), 'verified': True})
    assert len(pcm_hashes) == 1
    new_media_bytes = sum(p.stat().st_size for p in (ROOT / 'media').rglob('*') if p.is_file())
    assert new_media_bytes <= manifest['new_media_cap_bytes']
    data = Path('/mnt/lightsem-data/experiments/lightsem/end_to_end_20260928_v1/shpongle_rating_20260929_v1')
    data_bytes = sum(p.stat().st_size for p in data.rglob('*') if p.is_file())
    assert data_bytes <= manifest['new_data_cap_bytes']
    free = {}
    for p in (ROOT, data):
        s = os.statvfs(p)
        free[str(p)] = s.f_bavail * s.f_frsize
        assert free[str(p)] >= manifest['minimum_free_bytes']
    result = {'status': 'PASS_ALL_TEN_READY_FOR_USER_RATING', 'videos': rows,
              'same_source_audio': True, 'identical_decoded_audio_sha256': next(iter(pcm_hashes)),
              'source_audio_sha256': manifest['audio_sha256'], 'combined_variant': '05',
              'new_renders': 3, 'reused_verified_renders': 7,
              'new_media_package_bytes': new_media_bytes, 'new_data_bytes': data_bytes,
              'free_bytes': free, 'human_ratings_collected': False,
              'scientific_or_default_promotion': False}
    with (ROOT / 'DELIVERY_AUDIT.json').open('x') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
