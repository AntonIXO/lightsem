"""Sequential full-track rating renders with exact audio/media verification."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import time

ROOT = Path(__file__).resolve().parent
PLAN = json.loads((ROOT / 'PLAN.json').read_text())
PREP = json.loads((ROOT / 'PREPARATION.json').read_text())
EXPECTED_PCM = '6d5d8917556e5b7247fef054c758e975b9bf9d8ad206dccae755d4b51d794455'


def state(value):
    (ROOT / 'PROGRESS.json').write_text(json.dumps(value, indent=2))


def main():
    started = time.monotonic()
    completed = []
    env = dict(os.environ, TORCH_ENV='/root/lightsem/.venv/torch-access',
               PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1')
    try:
        for entry in PREP['variants']:
            vid = entry['id']
            target = next(v for v in PLAN['variants'] if v['id'] == vid)
            out = Path(target['path'])
            out.parent.mkdir(parents=True, exist_ok=False)
            assert hashlib.sha256(Path(entry['streams']).read_bytes()).hexdigest() == entry['sha256']
            label = {'08': 'Variant 08 - learned colors only',
                     '09': 'Variant 09 - learned accents only',
                     '10': 'Variant 10 - combined with palette history off'}[vid]
            for mode in ('render', 'audit'):
                state({'status': 'running', 'variant': vid, 'phase': mode,
                       'completed': completed, 'elapsed_seconds': time.monotonic() - started})
                cmd = ['bash', '/root/lightsem/run_torch.sh', '-u', str(ROOT / 'render_rating_variant.py'),
                       mode, '--streams', entry['streams'], '--output', str(out), '--label', label,
                       '--audio', PLAN['audio']]
                with (out.parent / (mode + '.log')).open('xb') as log:
                    subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
            audit = json.loads(out.with_name(out.stem + '_audit.json').read_text())
            rendered = json.loads(out.with_name(out.stem + '_render.json').read_text())
            assert audit['decoded_frames'] == 8319 and audit['pts_uniform'] and audit['full_software_AV_decode']
            assert audit['pcm_sha256'] == EXPECTED_PCM
            assert rendered['source_audio_sha256'] == PLAN['audio_sha256']
            target.update(status='rendered_and_verified', sha256=rendered['video_sha256'],
                          bytes=out.stat().st_size, audit=str(out.with_name(out.stem + '_audit.json')),
                          pcm_sha256=audit['pcm_sha256'], dmx=entry)
            completed.append(vid)
            (ROOT / 'MANIFEST.json').write_text(json.dumps(PLAN, indent=2))
        PLAN['status'] = 'all_ten_variants_verified'
        PLAN['human_ratings_collected'] = False
        (ROOT / 'MANIFEST.json').write_text(json.dumps(PLAN, indent=2))
        state({'status': 'complete', 'completed': completed, 'variants_available': 10,
               'elapsed_seconds': time.monotonic() - started})
    except BaseException as exc:
        state({'status': 'failed', 'error': type(exc).__name__ + ': ' + str(exc),
               'completed': completed, 'elapsed_seconds': time.monotonic() - started})
        raise


if __name__ == '__main__':
    main()
