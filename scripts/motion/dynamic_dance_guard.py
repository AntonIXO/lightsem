"""Run frozen dynamic-dance phase queues once, with cumulative GPU accounting.

Queue schema 1: id, root, source, guard_sha256, jobs, optional limits. Each job
freezes id, command (argv), cwd, python, env, source_files and inputs (path: SHA),
imports (module: file), outputs, gpu, budget_group and timeout_seconds. New phases
may freeze new implementation snapshots; existing job identities are never reused.
Only verified completed jobs may be skipped. A partial job requires scientific
review and a new identity, not an automatic retry.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import shutil
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

GUARD = Path(__file__).resolve()
CAMPAIGN_ROOT = Path('/mnt/lightsem-data/experiments/lightsem/dynamic_dance_20261005_v1')
GPU_LOCKS = (Path('/tmp/lightsem-audio-understanding-gpu.lock'),
             Path('/mnt/lightsem-data/experiments/lightsem/stage11_20260908/gpu.lock'))
STAGE_SECONDS = {'features': 6 * 3600, 'training': 12 * 3600,
                 'evaluation': 4 * 3600, 'rendering': 2 * 3600}
STAGE_ALIASES = {'extraction': 'features', 'generation': 'evaluation'}
TOTAL_GPU_SECONDS = 24 * 3600
GIB = 2**30
DESKTOP = {'/usr/bin/gnome-shell', '/usr/bin/Xwayland', '/opt/Koala.Clash/koala-clash'}
PREFIX = 'LIGHTSEM_DYNAMIC_'
BAD_LOG = re.compile(r'CUDA out of memory|OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED|'
                     r'non[- ]finite (?:loss|gradient|output|value).*?(?:detected|abort)|'
                     r'\b(?:loss|grad_norm)\s*[=:]\s*[-+]?(?:nan|inf)\b', re.I)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def event(name, **fields):
    print(json.dumps({'event': name, **fields}, sort_keys=True), flush=True)


def budget_group(value):
    return STAGE_ALIASES.get(value, value)


def absolute(value):
    path = Path(value)
    if not path.is_absolute() or path != path.resolve():
        raise ValueError(f'Canonical absolute path required: {value}')
    return path


def safe_id(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_.-]+', value) or value in ('.', '..'):
        raise ValueError(f'Unsafe identifier: {value!r}')
    return value


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    with path.with_name(path.name + '.sha256').open('x') as stream:
        stream.write(sha(path) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def read_verified(path):
    path = Path(path)
    if sha(path) != path.with_name(path.name + '.sha256').read_text().strip():
        raise ValueError(f'Changed immutable record: {path}')
    return json.loads(path.read_text(), parse_constant=reject_nonfinite_json, parse_float=finite_json_float)


def reject_nonfinite_json(value):
    raise ValueError(f'Nonfinite JSON number: {value}')


def finite_json_float(value):
    number = float(value)
    if not math.isfinite(number):
        reject_nonfinite_json(value)
    return number


def verify_files(files, label):
    if not isinstance(files, dict):
        raise ValueError(f'{label} must map absolute paths to SHA256 values')
    for filename, expected in files.items():
        path = absolute(filename)
        if not isinstance(expected, str) or not re.fullmatch('[0-9a-f]{64}', expected):
            raise ValueError(f'Invalid SHA256: {filename}')
        if not path.is_file() or sha(path) != expected:
            raise ValueError(f'{label} drift: {path}')


def process_table():
    """pid -> (parent pid, process group, resident bytes, start clock ticks)."""
    result = {}
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit():
            continue
        try:
            fields = (directory / 'stat').read_text().rsplit(')', 1)[1].split()
            if fields[0] != 'Z':
                result[int(directory.name)] = (int(fields[1]), int(fields[2]),
                    int(fields[21]) * os.sysconf('SC_PAGE_SIZE'), int(fields[19]))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return result


def owned_processes(pid, table):
    owned = {pid}
    while True:
        extra = {p for p, row in table.items() if row[0] in owned or row[1] == pid}
        if extra.issubset(owned):
            return owned
        owned.update(extra)


def foreign_gpu(owned):
    xml = subprocess.check_output(['nvidia-smi', '-q', '-x'], text=True, timeout=15)
    processes = [{'pid': int(p.findtext('pid')), 'type': p.findtext('type', ''),
                  'name': p.findtext('process_name', '')}
                 for p in ET.fromstring(xml).findall('./gpu/processes/process_info')]
    return [p for p in processes if p['pid'] not in owned
            and not (p['type'] == 'G' and p['name'] in DESKTOP)]


@contextmanager
def lock(path, poll_seconds=2.0):
    """Wait for foreign ownership; never terminate the foreign process."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        announced = False
        while True:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if not announced:
                    event('waiting_for_lock', path=str(path))
                    announced = True
                time.sleep(poll_seconds)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def check_resources(root, limits):
    free = shutil.disk_usage(root).free
    available = next(int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines()
                     if line.startswith('MemAvailable:'))
    if free < limits['min_free_bytes']:
        raise RuntimeError(f'Minimum free space crossed: {free} bytes')
    if available < limits['min_available_memory_bytes']:
        raise RuntimeError(f'Minimum available memory crossed: {available} bytes')
    return {'free_bytes': free, 'available_memory_bytes': available}


def verify_runtime_imports(imports=None, sources=None):
    """Import the actual modules and verify resolved file identity and content."""
    imports = imports if imports is not None else json.loads(os.environ[PREFIX + 'IMPORTS_JSON'])
    sources = sources if sources is not None else json.loads(os.environ[PREFIX + 'SOURCES_JSON'])
    for name, filename in imports.items():
        module = importlib.import_module(name)
        actual = getattr(module, '__file__', None)
        if actual is None or Path(actual).resolve() != absolute(filename):
            raise ValueError(f'Actual import drift: {name}: {actual!r} != {filename!r}')
        verify_files({filename: sources[filename]}, 'Imported source')


def authenticate_guarded_job(expected_script=None):
    """Renderer entrypoint: authenticate a live guard, then skip nested GPU locks.

    Call only for --guarded, before opening a GPU. The guard PID must be a live
    ancestor with matching process birth ticks; the one-job random token and
    immutable receipt bind the invocation to its frozen queue and command.
    """
    pid = int(os.environ[PREFIX + 'GUARD_PID'])
    receipt_path = absolute(os.environ[PREFIX + 'RECEIPT'])
    if sha(receipt_path) != os.environ[PREFIX + 'RECEIPT_SHA256']:
        raise ValueError('Guard receipt digest differs')
    started = read_verified(receipt_path)
    table = process_table()
    current, ancestors = os.getpid(), set()
    while current in table and current not in ancestors:
        ancestors.add(current)
        current = table[current][0]
    if pid == os.getpid() or pid not in ancestors or pid not in table:
        raise ValueError('Guard is not a live parent process')
    if started['guard_pid'] != pid or table[pid][3] != started['guard_start_ticks']:
        raise ValueError('Guard process identity changed')
    token = os.environ[PREFIX + 'TOKEN']
    if not secrets.compare_digest(hashlib.sha256(token.encode()).hexdigest(), started['token_sha256']):
        raise ValueError('Guard authentication token differs')
    identity = started['identity']
    if not identity['gpu'] or identity['job_id'] != os.environ[PREFIX + 'JOB_ID']:
        raise ValueError('Guard receipt does not authorize this GPU job')
    if expected_script is not None and str(Path(expected_script).resolve()) not in identity['command']:
        raise ValueError('Renderer script is absent from frozen command')
    if expected_script is not None:
        script = str(Path(expected_script).resolve())
        index = identity['command'].index(script)
        if Path(sys.argv[0]).resolve() != Path(script) or sys.argv[1:] != identity['command'][index + 1:]:
            raise ValueError('Runtime script arguments differ from frozen command')
    if identity['guard_sha256'] != sha(GUARD):
        raise ValueError('Guard source changed after launch')
    if '--guarded' not in identity['command']:
        raise ValueError('Frozen renderer command must include --guarded')
    if receipt_path.parent.joinpath('completed.json').exists() or receipt_path.parent.joinpath('halted.json').exists():
        raise ValueError('Guard job has already finished')
    verify_files(identity['source_files'], 'Source')
    verify_runtime_imports(identity['imports'], identity['source_files'])
    return started


def validate_job(queue, job, check_hashes=True):
    safe_id(job['id'])
    if not isinstance(job.get('gpu'), bool) or budget_group(job.get('budget_group')) not in STAGE_SECONDS:
        raise ValueError('Explicit gpu boolean and valid budget_group required')
    timeout = job.get('timeout_seconds')
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('Positive finite timeout_seconds required')
    command = job.get('command')
    if not isinstance(command, list) or not command or not all(isinstance(x, str) for x in command):
        raise ValueError('Exact nonempty command argv required')
    if not Path(command[0]).is_absolute() or not Path(command[0]).is_file():
        raise ValueError('Command executable must be an existing absolute path')
    if not absolute(job['cwd']).is_dir() or not Path(job['python']).is_absolute() or not Path(job['python']).is_file():
        raise ValueError('Existing canonical cwd and absolute Python interpreter required')
    env = job.get('env', {})
    if not isinstance(env, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in env.items()):
        raise ValueError('Environment must map strings to strings')
    if any(k.startswith(PREFIX) for k in env):
        raise ValueError('Guard authentication environment cannot be supplied by a job')
    sources, imports = job.get('source_files', {}), job.get('imports', {})
    if str(GUARD) not in sources:
        raise ValueError('Each job must freeze the guard source')
    if not isinstance(imports, dict):
        raise ValueError('imports must map module names to absolute source files')
    for name, filename in imports.items():
        if not re.fullmatch(r'[A-Za-z_]\w*(\.[A-Za-z_]\w*)*', name) or filename not in sources:
            raise ValueError(f'Import identity absent from source_files: {name}')
        absolute(filename)
    if check_hashes:
        verify_files(sources, 'Source')
        verify_files(job.get('inputs', {}), 'Input')
    root = absolute(queue['root'])
    outputs = [absolute(p) for p in job.get('outputs', [])]
    if not outputs or len(set(outputs)) != len(outputs):
        raise ValueError('Unique nonempty expected outputs required')
    for i, path in enumerate(outputs):
        if path == root or not path.is_relative_to(root) or path.is_relative_to(root / 'records'):
            raise ValueError(f'Output outside campaign data space: {path}')
        if any(path.is_relative_to(other) or other.is_relative_to(path) for other in outputs[:i]):
            raise ValueError('Overlapping output paths')
        if any(Path(p).is_relative_to(path) for p in [*sources, *job.get('inputs', {})]):
            raise ValueError('Output overlaps a frozen source/input')
    return outputs


def output_snapshot(paths):
    result = {}
    for path in paths:
        if path.is_symlink() or not path.exists():
            raise ValueError(f'Missing/symlinked expected output: {path}')
        members = [path] if path.is_file() else sorted(path.rglob('*'))
        count = 0
        for item in members:
            if item.is_symlink():
                raise ValueError(f'Output contains symlink: {item}')
            if item.is_file():
                result[str(item.resolve())] = {'sha256': sha(item), 'bytes': item.stat().st_size}
                count += 1
            elif not item.is_dir():
                raise ValueError(f'Unexpected output type: {item}')
        if not count:
            raise ValueError(f'Empty expected output: {path}')
    return result


def check_finite_outputs(snapshot):
    """Check numeric array/JSON artifacts; strings such as 'NaN count' are fine."""
    for filename in snapshot:
        path = Path(filename)
        if path.suffix == '.json':
            json.loads(path.read_text(), parse_constant=reject_nonfinite_json, parse_float=finite_json_float)
        elif path.suffix in ('.npz', '.npy'):
            import numpy as np
            data = np.load(path, allow_pickle=False, mmap_mode='r' if path.suffix == '.npy' else None)
            try:
                arrays = (data[key] for key in data.files) if isinstance(data, np.lib.npyio.NpzFile) else (data,)
                for array in arrays:
                    if np.issubdtype(array.dtype, np.number):
                        flat = array.reshape(-1)
                        if any(not np.isfinite(flat[i:i + 1_000_000]).all() for i in range(0, flat.size, 1_000_000)):
                            raise ValueError(f'Nonfinite numeric output: {path}')
            finally:
                if hasattr(data, 'close'):
                    data.close()


def budget_usage(root):
    """Read every phase; unresolved GPU starts reserve their entire timeout."""
    usage = {key: 0.0 for key in STAGE_SECONDS}
    for folder in sorted((root / 'records/dynamic_jobs').glob('*')):
        started = read_verified(folder / 'started.json')
        identity = started['identity']
        if not identity['gpu']:
            continue
        finishes = [p for p in (folder / 'completed.json', folder / 'halted.json') if p.exists()]
        if len(finishes) > 1:
            raise ValueError(f'Conflicting job terminal records: {folder}')
        seconds = identity['timeout_seconds'] + 10
        if finishes:
            finish = read_verified(finishes[0])
            if finish['identity'] != identity:
                raise ValueError(f'Terminal job identity changed: {folder}')
            seconds = finish['gpu_elapsed_seconds']
        if not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0:
            raise ValueError(f'Invalid budget accounting: {folder}')
        usage[budget_group(identity['budget_group'])] += seconds
    return usage


def remaining_budget(usage, group):
    group = budget_group(group)
    return min(STAGE_SECONDS[group] - usage[group], TOTAL_GPU_SECONDS - sum(usage.values()))


def stop_owned(process):
    if process is None:
        return
    for sig, timeout in ((signal.SIGTERM, 3), (signal.SIGKILL, 3)):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            pass


def check_log(path):
    tail = ''
    with path.open() as stream:
        while block := stream.read(1024 * 1024):
            text = tail + block
            if BAD_LOG.search(text):
                raise RuntimeError('OOM/nonfinite failure found in job log')
            tail = text[-1024:]


def job_environment(queue, job):
    environment = {key: value for key, value in os.environ.items() if not key.startswith(PREFIX)}
    environment.update(job.get('env', {}))
    environment.pop('PYTHONHOME', None)
    environment['PYTHONPATH'] = job.get('env', {}).get('PYTHONPATH', queue['source'])
    environment.update(PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1')
    if not job['gpu']:
        environment['CUDA_VISIBLE_DEVICES'] = ''
    return environment


def probe_imports(job, environment, remaining):
    if not job['imports']:
        return
    probe_env = dict(environment, CUDA_VISIBLE_DEVICES='')
    probe_env[PREFIX + 'IMPORTS_JSON'] = json.dumps(job['imports'])
    probe_env[PREFIX + 'SOURCES_JSON'] = json.dumps(job['source_files'])
    code = ('import runpy; ns=runpy.run_path(' + repr(str(GUARD)) + '); ns["verify_runtime_imports"]()')
    subprocess.run([job['python'], '-c', code], cwd=job['cwd'], env=probe_env,
                   check=True, timeout=max(.001, min(60, remaining)), capture_output=True, text=True)


def run_job(queue, job, queue_sha, limits):
    outputs = validate_job(queue, job)
    root = absolute(queue['root'])
    folder = root / 'records/dynamic_jobs' / job['id']
    identity = {'job_id': job['id'], 'job_sha256': object_sha(job), 'queue_id': queue['id'],
                'queue_sha256': queue_sha, 'guard_sha256': sha(GUARD), **{key: job[key] for key in
                ('command', 'cwd', 'python', 'source_files', 'inputs', 'imports', 'gpu', 'budget_group', 'timeout_seconds')},
                'env': job.get('env', {}), 'outputs': [str(p) for p in outputs]}
    receipt_path = folder / 'completed.json'
    if receipt_path.exists():
        receipt, started = read_verified(receipt_path), read_verified(folder / 'started.json')
        manifest = read_verified(folder / 'manifest.json')
        if (receipt['identity'] != identity or started['identity'] != identity or receipt['status'] != 'completed'
                or manifest != output_snapshot(outputs) or receipt['manifest_sha256'] != sha(folder / 'manifest.json')
                or receipt['log_sha256'] != sha(folder / 'stdout.log')
                or receipt['started_sha256'] != sha(folder / 'started.json')):
            raise ValueError(f'Completed job identity/artifacts changed: {job["id"]}')
        event('skip_verified_completed', job=job['id'])
        return 0
    if folder.exists() or folder.is_symlink():
        raise FileExistsError(f'Incomplete prior job; preserve and review: {folder}')
    if any(path.exists() or path.is_symlink() for path in outputs):
        raise FileExistsError('Expected output already exists; refusing adoption or overwrite')
    with ExitStack() as stack:
        if job['gpu']:
            for gpu_path in GPU_LOCKS:
                stack.enter_context(lock(gpu_path, limits['poll_seconds']))
            announced = False
            while foreign_gpu({os.getpid()}):
                if not announced:
                    event('waiting_for_foreign_gpu', job=job['id'])
                    announced = True
                time.sleep(limits['poll_seconds'])
        return execute_claimed_job(queue, job, identity, outputs, folder, limits)


def execute_claimed_job(queue, job, identity, outputs, folder, limits):
    root = absolute(queue['root'])
    usage = budget_usage(root)
    if job['gpu'] and any(usage[group] > cap for group, cap in STAGE_SECONDS.items()):
        raise RuntimeError('A prior GPU stage exhausted its budget; review required')
    remaining = remaining_budget(usage, job['budget_group']) if job['gpu'] else job['timeout_seconds']
    # Reserve cleanup time within the hard campaign budget, including both import probes.
    if job['gpu'] and remaining < job['timeout_seconds'] + 10:
        raise RuntimeError(f'Insufficient GPU budget for frozen job timeout: {remaining:.3f}s remaining')
    resources = check_resources(root, limits)
    folder.mkdir(parents=True, exist_ok=False)
    token = secrets.token_hex(32)
    started = {'identity': identity, 'started_at': utc_now(), 'guard_pid': os.getpid(),
               'guard_start_ticks': process_table()[os.getpid()][3], 'token_sha256': hashlib.sha256(token.encode()).hexdigest(),
               'resource_before': resources, 'limits': limits, 'gpu_usage_before': usage}
    write_new(folder / 'started.json', started)
    environment = job_environment(queue, job)
    environment.update({PREFIX + 'GUARD_PID': str(os.getpid()), PREFIX + 'TOKEN': token,
                        PREFIX + 'RECEIPT': str(folder / 'started.json'),
                        PREFIX + 'RECEIPT_SHA256': sha(folder / 'started.json'),
                        PREFIX + 'JOB_ID': job['id'], PREFIX + 'IMPORTS_JSON': json.dumps(job['imports']),
                        PREFIX + 'SOURCES_JSON': json.dumps(job['source_files'])})
    process, peak, code = None, 0, None
    started_time = time.monotonic()
    deadline = started_time + min(job['timeout_seconds'], remaining - 10 if job['gpu'] else remaining)
    event('start_job', job=job['id'], gpu=job['gpu'], budget_group=job['budget_group'])
    # Wall timer also covers slow hashing/verification and nvidia-smi calls;
    # polling alone cannot impose a hard deadline over those operations.
    prior_alarm_handler = signal.getsignal(signal.SIGALRM)
    prior_alarm = signal.getitimer(signal.ITIMER_REAL)
    def deadline_alarm(_signum, _frame):
        raise TimeoutError('Frozen job timeout / GPU budget exhausted')
    signal.signal(signal.SIGALRM, deadline_alarm)
    signal.setitimer(signal.ITIMER_REAL, max(.001, deadline - time.monotonic()))
    try:
        probe_imports(job, environment, deadline - time.monotonic())
        with (folder / 'stdout.log').open('x') as log:
            process = subprocess.Popen(job['command'], cwd=job['cwd'], env=environment,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            with (folder / 'stdout.log').open() as log_reader:
                log_tail = ''
                while True:
                    table = process_table()
                    owned = owned_processes(process.pid, table)
                    peak = max(peak, sum(table[p][2] for p in owned if p in table))
                    if limits['max_rss_bytes'] is not None and peak > limits['max_rss_bytes']:
                        raise RuntimeError(f'Owned process RSS cap crossed: {peak} bytes')
                    check_resources(root, limits)
                    if time.monotonic() >= deadline:
                        raise TimeoutError('Frozen job timeout / GPU budget exhausted')
                    if job['gpu'] and foreign_gpu(owned | {os.getpid()}):
                        raise InterruptedError('Foreign GPU process appeared; stopped only this job')
                    log_tail = (log_tail + log_reader.read())[-65536:]
                    if BAD_LOG.search(log_tail):
                        raise RuntimeError('OOM/nonfinite failure found in job log')
                    code = process.poll()
                    if code is not None:
                        if (owned - {process.pid}) & set(table):
                            raise RuntimeError('Owned children outlived job leader')
                        break
                    time.sleep(min(limits['poll_seconds'], max(.001, deadline - time.monotonic())))
        if code != 0:
            raise subprocess.CalledProcessError(code, job['command'])
        check_log(folder / 'stdout.log')
        validate_job(queue, job)
        probe_imports(job, environment, deadline - time.monotonic())
        snapshot = output_snapshot(outputs)
        check_finite_outputs(snapshot)
        resources = check_resources(root, limits)
        if time.monotonic() >= deadline:
            raise TimeoutError('Verification crossed frozen job timeout / GPU budget')
        write_new(folder / 'manifest.json', snapshot)
        elapsed = time.monotonic() - started_time
        signal.setitimer(signal.ITIMER_REAL, 0)
        write_new(folder / 'completed.json', {'identity': identity, 'status': 'completed', 'returncode': 0,
                  'finished_at': utc_now(), 'elapsed_seconds': elapsed, 'gpu_elapsed_seconds': elapsed if job['gpu'] else 0,
                  'peak_group_rss_bytes': peak, 'resource_after': resources,
                  'manifest_sha256': sha(folder / 'manifest.json'), 'started_sha256': sha(folder / 'started.json'),
                  'log_sha256': sha(folder / 'stdout.log')})
        event('completed', job=job['id'], receipt_sha256=sha(folder / 'completed.json'))
        return 0
    except BaseException as exc:
        signal.setitimer(signal.ITIMER_REAL, 0)
        stop_owned(process)
        elapsed = time.monotonic() - started_time
        if (folder / 'completed.json.sha256').exists():
            raise
        write_new(folder / 'halted.json', {'identity': identity,
                  'status': 'interrupted' if isinstance(exc, (KeyboardInterrupt, SystemExit, InterruptedError)) else 'failed',
                  'error': f'{type(exc).__name__}: {exc}', 'returncode': code,
                  'finished_at': utc_now(), 'elapsed_seconds': elapsed, 'gpu_elapsed_seconds': elapsed if job['gpu'] else 0,
                  'peak_group_rss_bytes': peak, 'started_sha256': sha(folder / 'started.json')})
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, prior_alarm_handler)
        if prior_alarm[0]:
            signal.setitimer(signal.ITIMER_REAL, max(.001, prior_alarm[0] - (time.monotonic() - started_time)), prior_alarm[1])


def run_queue(path, expected_sha):
    path = absolute(path)
    if sha(path) != expected_sha:
        raise ValueError('Queue byte SHA256 differs from approved identity')
    queue = json.loads(path.read_text(), parse_constant=reject_nonfinite_json, parse_float=finite_json_float)
    if queue.get('schema_version') != 1 or queue.get('guard_sha256') != sha(GUARD):
        raise ValueError('Unsupported queue schema or changed guard')
    safe_id(queue['id'])
    if not absolute(queue['source']).is_dir():
        raise ValueError('Queue source snapshot directory does not exist')
    root = absolute(queue['root'])
    if root != CAMPAIGN_ROOT or not root.is_dir():
        raise ValueError('Queue must use the approved campaign root')
    jobs = queue.get('jobs')
    if not isinstance(jobs, list) or not jobs:
        raise ValueError('Nonempty fixed job sequence required')
    limits = {'min_free_bytes': 20 * GIB, 'min_available_memory_bytes': GIB,
              'max_rss_bytes': None, 'poll_seconds': 2.0}
    supplied = queue.get('limits', {})
    if set(supplied) - set(limits):
        raise ValueError('Unknown resource limit')
    limits.update(supplied)
    if any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v)
           for key, v in limits.items() if not (key == 'max_rss_bytes' and v is None)):
        raise ValueError('Resource limits must be finite numbers')
    if (limits['min_free_bytes'] < 20 * GIB or limits['min_available_memory_bytes'] < GIB
            or (limits['max_rss_bytes'] is not None and limits['max_rss_bytes'] <= 0)
            or not .05 <= limits['poll_seconds'] <= 10):
        raise ValueError('Resource limits loosen campaign constraints')
    with lock(root / 'records/dynamic_queues/campaign.lock', limits['poll_seconds']):
        queue_record = root / 'records/dynamic_queues' / (queue['id'] + '.json')
        frozen = {'queue_sha256': expected_sha, 'queue': queue}
        if queue_record.exists():
            if read_verified(queue_record) != frozen:
                raise ValueError('Queue phase identity already exists with different content')
        else:
            write_new(queue_record, frozen)
        all_outputs, ids = [], []
        for job in jobs:
            ids.append(job['id'])
            outputs = validate_job(queue, job, check_hashes=False)
            for output in outputs:
                if any(output.is_relative_to(other) or other.is_relative_to(output) for other in all_outputs):
                    raise ValueError('Jobs claim overlapping output paths')
            all_outputs.extend(outputs)
        if len(set(ids)) != len(ids):
            raise ValueError('Duplicate job IDs')
        # Reject output reuse across independently frozen implementation phases.
        for started_path in (root / 'records/dynamic_jobs').glob('*/started.json'):
            identity = read_verified(started_path)['identity']
            for job in jobs:
                if job['id'] == identity['job_id']:
                    continue
                for output in map(Path, job['outputs']):
                    if any(output.is_relative_to(Path(other)) or Path(other).is_relative_to(output)
                           for other in identity['outputs']):
                        raise ValueError('Job output overlaps a prior phase claim')
        for job in jobs:
            run_job(queue, job, expected_sha, limits)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', type=Path, required=True)
    parser.add_argument('--queue-sha256', required=True)
    args = parser.parse_args()
    def terminate(_signum, _frame):
        raise SystemExit(143)
    signal.signal(signal.SIGTERM, terminate)
    try:
        return run_queue(args.queue, args.queue_sha256)
    except Exception as exc:
        print(f'Dynamic dance guard halted: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
