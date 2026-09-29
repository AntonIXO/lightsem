#!/usr/bin/python3 -B
"""Run the frozen 67 synthetic checks with a live systemd envelope and bounded log."""
import hashlib
import json
import os
from pathlib import Path
import re
import selectors
import stat
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
DOCS = Path('/root/lightsem/docs/research/2026-09-28-end-to-end')
ORIGINAL = Path('/mnt/lightsem-data/experiments/lightsem/lighting-cue-model-20260922t094347z')
WORKSPACE = Path('/mnt/lightsem-data/experiments/lightsem/end_to_end_20260928_v1')
UNIT = 'lightsem-lossless-readiness-tests-20260929-v1.service'
AUTH = ROOT / 'NATIVE_LOSSLESS_ARCHIVE_TEST_ADMISSION_V1.json'
PROOF = ROOT / 'NATIVE_LOSSLESS_ARCHIVE_TEST_ENVELOPE_V1.json'
LOG = ROOT / 'native_lossless_archive_tests_v1.log'
RESULT = ROOT / 'NATIVE_LOSSLESS_ARCHIVE_TEST_RESULT_V1.json'
LIMIT = 32768
EXPECTED = {
    str(ROOT / 'measure_lossless_archive_v1.py'): '2c068d0062ae0b9dd24d88b848c80201dffd284221ae4e25d84a5da8d8febf39',
    str(ROOT / 'test_measure_lossless_archive_v1.py'): '79866d5de485812d3b6483a3f0073774fa139772a2d4667622a493358fd1eaa1',
}
ARGV = ['bash', '/root/lightsem/run_torch.sh', '-u', '-m', 'unittest', 'discover',
        '-s', str(ROOT), '-p', 'test_measure_lossless_archive_v1.py', '-v']
ENVIRONMENT = {'CUDA_VISIBLE_DEVICES': '', 'PYTHONDONTWRITEBYTECODE': '1',
               'TORCH_ENV': '/root/lightsem/.venv/torch-access',
               'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def save(path, data):
    raw = json.dumps(data, indent=2).encode() + b'\n'
    assert len(raw) <= 16384
    with path.open('xb') as f:
        f.write(raw)


def inventory(root):
    rows = []
    assert root.is_dir() and root.resolve() == root
    for p in sorted(root.rglob('*')):
        s = p.lstat()
        assert not stat.S_ISLNK(s.st_mode), str(p)
        if stat.S_ISREG(s.st_mode):
            assert s.st_nlink == 1
            rows.append([str(p.relative_to(root)), s.st_size, s.st_mtime_ns])
        else:
            assert stat.S_ISDIR(s.st_mode)
    raw = json.dumps(rows, sort_keys=True, separators=(',', ':')).encode()
    return {'files': len(rows), 'bytes': sum(x[1] for x in rows), 'stat_inventory_sha256': sha(raw)}


def accounting():
    original, workspace = inventory(ORIGINAL), inventory(WORKSPACE)
    comparison = inventory(WORKSPACE / 'comparison_demos_v1')
    packet_dirs = [p for p in WORKSPACE.iterdir() if p.name.startswith('legacy_struct_intensity_v') and p.is_dir()]
    packet = sum(inventory(p)['bytes'] for p in packet_dirs)
    source = inventory(ROOT)['bytes']
    protocols = sum((DOCS / f'PROSPECTIVE_STRUCT_INTENSITY_PROTOCOL_CANDIDATE_V{v}.{ext}').stat().st_size
                    for v in range(3, 7) for ext in ('json', 'md'))
    external = source + protocols
    used = external + packet
    # Preserve the full row8/comparison allocations, all recent metadata
    # reservations and the older row11 candidate allocation conservatively.
    other = 262144 + 65536 + 524288 + 1048576
    committed = original['bytes'] + workspace['bytes'] + external + (12582912 - used) + (33554432 - comparison['bytes']) + other
    free = {}
    for p in (ROOT, WORKSPACE):
        fs = os.statvfs(p)
        free[str(p)] = fs.f_bavail * fs.f_frsize
    assert used + 131072 <= 12582912
    assert comparison['bytes'] <= 33554432
    assert committed <= 1020054732
    assert all(n >= 2**30 + (12582912 - used) for n in free.values())
    assert original['stat_inventory_sha256'] == 'e8305329ebf57603f9cda51ca498826ae9e556e43ce83fcfb2a67bcb5752b359'
    return {'original': original, 'workspace': workspace, 'row8_used_bytes': used,
            'eight_protocol_bytes': protocols, 'other_reserved_bytes': other,
            'combined_committed_bytes': committed, 'free_bytes': free,
            'additional_test_control_log_reserve_inside_row8_bytes': 131072,
            'scientific_payload_read': False}


def entry():
    auth = json.loads(AUTH.read_text())
    assert auth['argv'] == ARGV and auth['environment'] == ENVIRONMENT
    for path, expected in auth['source_sha256'].items():
        assert sha(Path(path).read_bytes()) == expected
    props = subprocess.check_output(['systemctl', 'show', UNIT,
        '-p', 'MemoryMax', '-p', 'MemorySwapMax', '-p', 'PrivateDevices',
        '-p', 'RuntimeMaxUSec', '-p', 'KillMode', '-p', 'DropInPaths', '-p', 'MainPID',
        '-p', 'ControlGroup', '-p', 'ExecStart'], text=True)
    values = dict(line.split('=', 1) for line in props.splitlines() if '=' in line)
    for k, v in {'MemoryMax': '536870912', 'MemorySwapMax': '0', 'PrivateDevices': 'yes',
                 'RuntimeMaxUSec': '50s', 'KillMode': 'control-group', 'DropInPaths': '',
                 'MainPID': str(os.getpid())}.items():
        assert values[k] == v, (k, values[k])
    assert all(os.environ.get(k) == v for k, v in ENVIRONMENT.items())
    assert not list(Path('/dev').glob('nvidia*'))
    assert str(Path(__file__)) in values['ExecStart']
    save(PROOF, {'effective_before_exec': values, 'argv': ARGV, 'environment': ENVIRONMENT,
                 'source_sha256': auth['source_sha256'], 'GPU_devices_visible': False})
    os.execvpe(ARGV[0], ARGV, os.environ)


def controller():
    assert all(not p.exists() for p in (AUTH, PROOF, LOG, RESULT)), 'preserve earlier attempt'
    hashes = {p: sha(Path(p).read_bytes()) for p in EXPECTED}
    assert hashes == EXPECTED
    hashes.update({str(Path(__file__)): sha(Path(__file__).read_bytes()),
                   '/root/lightsem/run_torch.sh': sha(Path('/root/lightsem/run_torch.sh').read_bytes())})
    before = accounting()
    save(AUTH, {'scope': '67_SYNTHETIC_TESTS_ONLY_NO_FULL_CODEC_OR_SCIENCE', 'argv': ARGV,
                'environment': ENVIRONMENT, 'source_sha256': hashes, 'accounting': before,
                'native_registration': str(ROOT / 'READINESS_NATIVE_REGISTRATION_20260929.json'),
                'user_authorization': '2026-09-29 continue readiness and restore memory; no separate memory permission required.'})
    cmd = ['systemd-run', '--quiet', '--wait', '--pipe', '--collect', '--unit=' + UNIT,
           '--property=MemoryMax=536870912', '--property=MemorySwapMax=0',
           '--property=PrivateDevices=yes', '--property=PrivateTmp=yes',
           '--property=RuntimeMaxSec=50', '--property=KillMode=control-group', '--property=TimeoutStopSec=1']
    cmd += ['--setenv=' + k + '=' + v for k, v in ENVIRONMENT.items()]
    cmd += ['/usr/bin/python3', '-B', str(Path(__file__)), '--entry']
    started = time.monotonic()
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    sel = selectors.DefaultSelector()
    sel.register(process.stdout, selectors.EVENT_READ)
    count, stop = 0, None
    with LOG.open('xb') as log:
        try:
            while sel.get_map():
                if time.monotonic() - started > 58:
                    stop = 'controller watchdog'
                    break
                for key, _ in sel.select(timeout=.2):
                    data = os.read(key.fd, 4096)
                    if not data:
                        sel.unregister(key.fileobj)
                        continue
                    keep = min(len(data), LIMIT - count)
                    log.write(data[:keep])
                    count += keep
                    if keep != len(data):
                        stop = 'combined log cap'
                        break
                if stop:
                    break
            if stop:
                subprocess.run(['systemctl', 'kill', '--signal=KILL', UNIT], capture_output=True, timeout=5)
                process.kill()
            code = process.wait(timeout=2)
        finally:
            sel.close()
            if process.poll() is None:
                subprocess.run(['systemctl', 'kill', '--signal=KILL', UNIT], capture_output=True, timeout=5)
                process.kill()
                process.wait(timeout=2)
    elapsed = time.monotonic() - started
    text = LOG.read_text()
    version = re.search(r'CODEC_VERSIONS=(\{[^\n]+\})', text)
    ran = re.search(r'Ran (\d+) tests in ([\d.]+)s', text)
    okay = code == 0 and stop is None and elapsed <= 60 and ran and int(ran[1]) == 67 and text.rstrip().endswith('OK')
    okay = bool(okay and len(re.findall(r'^test_.* \.\.\. ok$', text, re.M)) == 67)
    after = accounting()
    assert before['original'] == after['original'] and before['workspace'] == after['workspace']
    assert all(sha(Path(p).read_bytes()) == h for p, h in hashes.items())
    result = {'exit_code': code, 'plain_OK': okay, 'tests_run': int(ran[1]) if ran else None,
              'failures': 0 if okay else None, 'errors': 0 if okay else None,
              'skipped': 0 if okay else None, 'expected_failures': 0 if okay else None,
              'unexpected_successes': 0 if okay else None,
              'source_sha256': hashes, 'codec_versions': json.loads(version[1]) if version else None,
              'whole_job_seconds': elapsed, 'stop_reason': stop, 'combined_log_bytes': count,
              'log_ref': {'path': str(LOG), 'sha256': sha(LOG.read_bytes()), 'size_bytes': LOG.stat().st_size},
              'effective_envelope_ref': {'path': str(PROOF), 'sha256': sha(PROOF.read_bytes())} if PROOF.exists() else None,
              'post_accounting': after, 'original_and_workspace_stat_inventory_unchanged': True,
              'full40_codec_measurement_run': False, 'model_or_GPU_execution': False,
              'runtime_scope': 'Version text and cached module path only; no binary identity or future size invariance. Changed runtime requires fresh validation.'}
    save(RESULT, result)
    print(json.dumps(result))
    print('\n'.join(text.splitlines()[-8:]))
    raise SystemExit(0 if okay else 1)


if __name__ == '__main__':
    if sys.argv[1:] == ['--entry']:
        entry()
    elif len(sys.argv) == 1:
        controller()
    else:
        raise SystemExit('unsupported arguments')
