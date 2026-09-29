#!/usr/bin/python3 -B
"""Use the existing Client with the isolated repair MCP; independently audit bytes."""
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import resource
import signal
import stat
import subprocess
import sys
import time
import yaml

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
VAULT = Path('/root/lightsem-vault')
FOREIGN = Path('/root/vibescience-vault')
IDS = ('exp-dance-physical-objective-20260911', 'exp-dance-optics-full-20260911')
CAP = 524288


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def save(name, obj, compressed=False):
    raw = json.dumps(obj, ensure_ascii=False, separators=(',', ':')).encode() + b'\n'
    if compressed:
        raw = gzip.compress(raw, mtime=0)
    total = sum(p.stat().st_size for p in BASE.rglob('*') if p.is_file() and not p.is_symlink())
    assert total + len(raw) <= CAP, 'control packet limit'
    with (HERE / name).open('xb') as output:
        output.write(raw)
        output.flush()
        os.fsync(output.fileno())
    return {'path': str(HERE / name), 'sha256': sha(raw), 'bytes': len(raw)}


def file_map(root):
    result = {}
    for p in sorted(root.rglob('*')):
        assert not p.is_symlink(), str(p)
        if p.is_file():
            assert p.stat().st_size < 16 * 1024 * 1024, 'unexpected large vault file'
            result[str(p)] = sha(p.read_bytes())
    return result


def markdown_parts(raw):
    assert raw.startswith(b'---\n')
    end = raw.index(b'\n---\n', 4) + 5
    return raw[:end], raw[end:]


def body_without_interpretation(raw):
    # Separate verifier: cut the one section by heading positions, not the repair diff.
    body = markdown_parts(raw)[1].decode()
    headings = list(re.finditer(r'^## (.+)$', body, re.M))
    found = [i for i, m in enumerate(headings) if m.group(1) == 'Interpretation']
    assert len(found) == 1
    i = found[0]
    start = headings[i].start()
    stop = headings[i + 1].start() if i + 1 < len(headings) else len(body)
    return (body[:start] + body[stop:]).strip()


def installed_writers():
    result = []
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():
            continue
        try:
            args = (p / 'cmdline').read_bytes().split(b'\0')
            if not any(b'vibescience' in a for a in args[:2]):
                continue
            env = dict(x.split(b'=', 1) for x in (p / 'environ').read_bytes().split(b'\0') if b'=' in x)
            if env.get(b'VIBESCIENCE_VAULT', b'').rstrip(b'/') == bytes(VAULT):
                state = (p / 'stat').read_text().rsplit(')', 1)[1].split()
                assert state[0] not in ('T', 't', 'Z'), 'writer already stopped or dead'
                result.append({'pid': int(p.name), 'starttime': state[19]})
        except (FileNotFoundError, ProcessLookupError):
            continue
    return result


def main():
    started = time.monotonic()
    resource.setrlimit(resource.RLIMIT_AS, (268435456, 268435456))
    resource.setrlimit(resource.RLIMIT_CPU, (25, 25))
    resource.setrlimit(resource.RLIMIT_FSIZE, (CAP, CAP))
    def alarm_handler(*_):
        raise TimeoutError('repair wall watchdog')
    signal.signal(signal.SIGALRM, alarm_handler)
    signal.alarm(25)
    fixture = json.loads((HERE / 'FIXTURE_TEST_RESULT_V2.json').read_text())
    assert fixture['exit_code'] == 0 and fixture['stop_reason'] is None
    log = (HERE / 'FIXTURE_TEST_LOG_V2.txt').read_text()
    assert re.search(r'Ran 11 tests in .*\n\nOK\n', log)
    for name, expected in fixture['source_sha256'].items():
        assert sha((HERE / name).read_bytes()) == expected, name
    adapter = HERE / 'restore_notes_mcp.py'
    assert stat.S_IMODE(adapter.stat().st_mode) & 0o111
    client_path = Path('/root/lightsem/log_to_vault.py')
    assert sha(client_path.read_bytes()) == '3dc512f86582022ac074de018c40706478a384780e3c4121ec3361217fe65ca6'
    source_bindings = json.loads((BASE / 'NATIVE_NOTES_PRESERVATION_DIAGNOSIS_V1.json').read_text())['source_bindings']
    server_sources = {p: sha(Path(p).read_bytes()) for p in source_bindings if p.startswith('/root/vibescience-mcp/')}
    assert all(h == source_bindings[p]['sha256'] for p, h in server_sources.items())
    snapshot = json.loads(gzip.decompress((BASE / 'BEFORE_SNAPSHOT_V1.json.gz').read_bytes()))
    manifest_raw = (BASE / 'RESTORATION_CANDIDATES_V1.json').read_bytes()
    manifest = json.loads(manifest_raw)
    assert sha(manifest_raw) == '87431c538e5713ba735b022434e895a1dca33854eee96721fe729aaefe0d9d13'
    total = sum(p.stat().st_size for p in BASE.rglob('*') if p.is_file() and not p.is_symlink())
    assert total + 120000 <= CAP, 'reserve snapshot, receipts and audit before mutation'
    for root in (Path('/root/lightsem'), Path('/mnt/lightsem-data')):
        fs = os.statvfs(root)
        assert fs.f_bavail * fs.f_frsize > 2**30 + 120000
    # Stop only the known installed LightSem writer, briefly. A detached watchdog
    # resumes it on parent failure, timeout or pipe EOF; no server is terminated.
    writers = installed_writers()
    assert len(writers) == 1, 'recheck unexpected writer topology'
    reader, writer = os.pipe()
    watchdog_source = """import os,select,signal,json,sys
from pathlib import Path
fd=int(sys.argv[1]); rows=json.loads(sys.argv[2])
select.select([fd],[],[],30)
for row in rows:
 try:
  s=(Path('/proc')/str(row['pid'])/'stat').read_text().rsplit(')',1)[1].split()
  if s[19]==row['starttime']: os.kill(row['pid'],signal.SIGCONT)
 except ProcessLookupError: pass
 except FileNotFoundError: pass
"""
    watchdog = subprocess.Popen(['/usr/bin/python3', '-B', '-c', watchdog_source, str(reader), json.dumps(writers)],
                                pass_fds=(reader,), start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.close(reader)
    paused = []
    client = None
    result = {'status': 'STARTED', 'authorization': 'User 2026-09-29: memory changes need no separate permission; continue and restore.',
              'source_sha256': {'adapter': sha(adapter.read_bytes()), 'runner': sha(Path(__file__).read_bytes()),
                                'client': sha(client_path.read_bytes())},
              'fixture_result': str(HERE / 'FIXTURE_TEST_RESULT_V2.json'),
              'resource_limits': {'address_space_bytes': 268435456, 'CPU_seconds': 25, 'wall_alarm_seconds': 25},
              'writers': writers, 'partial_failure_preserved': False}
    try:
        for row in writers:
            os.kill(row['pid'], signal.SIGSTOP)
            paused.append(row)
        deadline = time.monotonic() + 2
        for row in writers:
            while (Path('/proc') / str(row['pid']) / 'stat').read_text().rsplit(')', 1)[1].split()[0] not in ('T', 't'):
                assert time.monotonic() < deadline
                time.sleep(.01)
        pre = {'lightsem': file_map(VAULT), 'foreign': file_map(FOREIGN),
               'target_text': {eid: Path(r['path']).read_text() for eid, r in snapshot['target_records'].items()},
               'installed_sources': server_sources}
        result['before_snapshot'] = save('REPAIR_BEFORE_V1.json.gz', pre, True)
        spec = importlib.util.spec_from_file_location('lightsem_recovery_client', client_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.SERVER = str(adapter)  # Existing Client implementation and file remain unchanged.
        os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
        os.environ['LIGHTSEM_RECOVERY_WRITERS_QUIESCED'] = '1'
        client = module.Client()
        assert client.initialize() == {'name': 'lightsem-two-record-notes-recovery', 'version': '1'}
        receipt = client.call('restore_experiment_notes_snapshot', {'manifest_sha256': sha(manifest_raw)})
        result['native_receipt'] = save('MCP_REPAIR_RECEIPT_V1.json', receipt)
        assert receipt.get('ok') is True and len(receipt['records']) == 2, receipt
        client.close()
        assert client.p.returncode == 0
        client = None
        after_lightsem, after_foreign = file_map(VAULT), file_map(FOREIGN)
        allowed = {str(VAULT / 'experiments' / (eid + '.md')) for eid in IDS}
        differences = sorted(p for p in set(pre['lightsem']) | set(after_lightsem) if pre['lightsem'].get(p) != after_lightsem.get(p))
        assert set(differences) == allowed, differences
        assert pre['foreign'] == after_foreign, 'foreign vault changed'
        checks = []
        for eid, original in snapshot['target_records'].items():
            raw = Path(original['path']).read_bytes()
            old_raw = pre['target_text'][eid].encode()
            fm, _ = markdown_parts(raw)
            assert fm == markdown_parts(old_raw)[0], eid
            metadata = yaml.safe_load(fm.decode().split('---', 2)[1])
            assert {k: v for k, v in metadata.items() if k != 'interpretation'} == {
                k: v for k, v in original['metadata'].items() if k != 'interpretation'}, eid
            assert body_without_interpretation(raw) == original['body'].strip(), eid
            checks.append({'experiment_id': eid, 'whole_current_metadata_unchanged': True,
                           'original_noninterpretation_metadata_unchanged': True,
                           'original_body_restored': True})
        definition = json.loads((BASE / 'NATIVE_INTERPRETATION_DEFINITION_V1.json').read_text())
        for payload in definition['payloads']:
            raw = Path(snapshot['target_records'][payload['experiment_id']]['path']).read_bytes()
            fm, body = markdown_parts(raw)
            interpretation = yaml.safe_load(fm.decode().split('---', 2)[1])['interpretation']
            for key in ('lesson', 'failure_mode', 'transfers_to'):
                assert interpretation[key] == payload[key]
            assert payload['lesson'] in body.decode()
        for item in manifest['candidates']:
            assert sha(Path(item['live_path']).read_bytes()) == item['candidate']['sha256']
        assert all(sha(Path(p).read_bytes()) == h for p, h in server_sources.items())
        assert sha(client_path.read_bytes()) == result['source_sha256']['client']
        result.update(status='RESTORED_AND_INDEPENDENTLY_VERIFIED', changed_paths=differences,
                      protected_lightsem_files=len(after_lightsem) - 2,
                      protected_lightsem_markdown=sum(p.endswith('.md') for p in after_lightsem) - 2,
                      foreign_files_unchanged=len(after_foreign), all20_checks=checks,
                      all20_native_interpretation_payloads_match=True, installed_server_and_client_unchanged=True,
                      elapsed_seconds=time.monotonic() - started)
    except Exception as exc:
        result.update(status='FAILED_STOPPED', error=type(exc).__name__ + ': ' + str(exc), partial_failure_preserved=True)
        raise
    finally:
        if client:
            client.close()
        for row in paused:
            try:
                os.kill(row['pid'], signal.SIGCONT)
            except ProcessLookupError:
                pass
        os.close(writer)
        watchdog.wait(timeout=3)
        result['writer_resume_verified'] = all(
            (Path('/proc') / str(row['pid']) / 'stat').read_text().rsplit(')', 1)[1].split()[0] not in ('T', 't', 'Z') for row in paused)
        signal.alarm(0)
        save('REPAIR_AUDIT_V1.json', result)
        print(json.dumps({k: v for k, v in result.items() if k not in ('all20_checks', 'source_sha256')}))


if __name__ == '__main__':
    main()
