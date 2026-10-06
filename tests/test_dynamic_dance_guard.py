"""CPU-only guard tests. Fake GPU inventory and temporary locks never use CUDA."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

GUARD_PATH = Path(__file__).resolve().parents[1] / 'scripts/motion/dynamic_dance_guard.py'
spec = importlib.util.spec_from_file_location('dynamic_dance_guard', GUARD_PATH)
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    root = tmp_path / 'campaign'
    root.mkdir()
    monkeypatch.setattr(guard, 'CAMPAIGN_ROOT', root)
    monkeypatch.setattr(guard, 'GPU_LOCKS', (tmp_path / 'gpu1.lock', tmp_path / 'gpu2.lock'))
    monkeypatch.setattr(guard, 'foreign_gpu', lambda owned: [])
    monkeypatch.setattr(guard, 'check_resources', lambda root, limits: {'free_bytes': 30 * guard.GIB})
    return root


def make_queue(root, name='phase1', code=None, gpu=False, **changes):
    script = root.parent / (name + '.py')
    output = root / name / 'value.json'
    script.write_text(code or ('from pathlib import Path\n'
                              f'p=Path({str(output)!r}); p.parent.mkdir(parents=True); p.write_text("{{\\\"value\\\": 1}}")\n'))
    job = {'id': name, 'command': [sys.executable, str(script)], 'cwd': str(root.parent),
           'python': sys.executable, 'env': {}, 'source_files': {str(GUARD_PATH): guard.sha(GUARD_PATH),
                                                              str(script): guard.sha(script)},
           'inputs': {}, 'imports': {}, 'outputs': [str(output)], 'gpu': gpu,
           'budget_group': 'evaluation', 'timeout_seconds': 10}
    job.update(changes)
    queue = {'schema_version': 1, 'id': name, 'root': str(root), 'source': str(root.parent),
             'guard_sha256': guard.sha(GUARD_PATH), 'limits': {'poll_seconds': .05}, 'jobs': [job]}
    path = root.parent / (name + '.json')
    path.write_text(json.dumps(queue))
    return path, queue, output


def run(path):
    return guard.run_queue(path, guard.sha(path))


def test_exact_hash_and_verified_completed_skip(campaign):
    path, queue, output = make_queue(campaign)
    with pytest.raises(ValueError, match='Queue byte SHA256'):
        guard.run_queue(path, '0' * 64)
    assert run(path) == 0
    original_mtime = output.stat().st_mtime_ns
    assert run(path) == 0
    assert output.stat().st_mtime_ns == original_mtime
    receipt = guard.read_verified(campaign / 'records/dynamic_jobs/phase1/completed.json')
    assert receipt['gpu_elapsed_seconds'] == 0
    assert receipt['manifest_sha256']
    output.write_text('{"value": 2}')
    with pytest.raises(ValueError, match='identity/artifacts changed'):
        run(path)


def test_source_and_input_drift_prevent_launch(campaign):
    source = campaign.parent / 'source.dat'
    source.write_text('initial')
    path, queue, _ = make_queue(campaign, inputs={str(source): guard.sha(source)})
    source.write_text('changed')
    with pytest.raises(ValueError, match='Input drift'):
        run(path)
    assert not (campaign / 'records/dynamic_jobs/phase1').exists()


def test_failed_job_is_preserved_and_never_replayed(campaign):
    path, _, _ = make_queue(campaign, code='raise SystemExit(7)\n')
    with pytest.raises(subprocess.CalledProcessError):
        run(path)
    halted = guard.read_verified(campaign / 'records/dynamic_jobs/phase1/halted.json')
    assert halted['status'] == 'failed'
    assert halted['returncode'] == 7
    with pytest.raises(FileExistsError, match='Incomplete prior job'):
        run(path)


def test_timeout_stops_owned_process_and_keeps_record(campaign):
    path, _, _ = make_queue(campaign, code='import time\ntime.sleep(30)\n', timeout_seconds=.15)
    with pytest.raises(TimeoutError):
        run(path)
    halted = guard.read_verified(campaign / 'records/dynamic_jobs/phase1/halted.json')
    assert 'timeout' in halted['error']
    assert halted['elapsed_seconds'] < 5


def test_source_mutation_after_launch_is_failure(campaign):
    path, queue, _ = make_queue(campaign, code='from pathlib import Path\nPath(__file__).write_text("changed")\n')
    with pytest.raises(ValueError, match='Source drift'):
        run(path)
    assert not (campaign / 'records/dynamic_jobs/phase1/completed.json').exists()


def test_actual_import_identity_is_checked(campaign):
    module = campaign.parent / 'frozen_module.py'
    module.write_text('VALUE=1\n')
    path, queue, _ = make_queue(campaign)
    job = queue['jobs'][0]
    job['imports'] = {'json': str(module)}
    job['source_files'][str(module)] = guard.sha(module)
    path.write_text(json.dumps(queue))
    with pytest.raises(subprocess.CalledProcessError):
        run(path)
    assert not (campaign / 'phase1').exists()


def test_nonfinite_numeric_artifact_fails(campaign):
    path, _, output = make_queue(campaign)
    script = campaign.parent / 'phase1.py'
    script.write_text('from pathlib import Path\n'
                      f'p=Path({str(output)!r}); p.parent.mkdir(parents=True); p.write_text("{{\\\"value\\\": NaN}}")\n')
    queue = json.loads(path.read_text())
    queue['jobs'][0]['source_files'][str(script)] = guard.sha(script)
    path.write_text(json.dumps(queue))
    with pytest.raises(ValueError, match='Nonfinite JSON'):
        run(path)
    assert output.exists()


def test_oom_log_failure_even_with_zero_exit(campaign):
    path, _, _ = make_queue(campaign, code='print("CUDA out of memory", flush=True)\n')
    with pytest.raises(RuntimeError, match='OOM/nonfinite'):
        run(path)


def test_output_exclusivity_across_phase_snapshots(campaign):
    first, _, output = make_queue(campaign)
    run(first)
    second, _, _ = make_queue(campaign, name='phase2', outputs=[str(output.parent)])
    with pytest.raises(ValueError, match='prior phase claim'):
        run(second)


def test_preexisting_output_is_not_adopted(campaign):
    path, _, output = make_queue(campaign)
    output.parent.mkdir()
    output.write_text('{}')
    with pytest.raises(FileExistsError, match='already exists'):
        run(path)


def test_gpu_budget_accumulates_across_phases_using_fake_gpu(campaign):
    first, _, _ = make_queue(campaign, gpu=True, budget_group='generation')
    run(first)
    before = guard.budget_usage(campaign)
    second, _, _ = make_queue(campaign, name='phase2', gpu=True, budget_group='evaluation')
    run(second)
    after = guard.budget_usage(campaign)
    assert after['evaluation'] > before['evaluation'] > 0
    assert not any(after[group] for group in ('features', 'training', 'rendering'))


def test_unresolved_gpu_start_reserves_full_timeout(campaign):
    folder = campaign / 'records/dynamic_jobs/interrupted'
    identity = {'gpu': True, 'budget_group': 'extraction', 'timeout_seconds': 900}
    guard.write_new(folder / 'started.json', {'identity': identity})
    assert guard.budget_usage(campaign)['features'] == 910


def test_stage_budget_prevents_launch(campaign, monkeypatch):
    path, _, _ = make_queue(campaign, gpu=True)
    monkeypatch.setattr(guard, 'budget_usage', lambda root: {'features': 0, 'training': 0,
                                                           'evaluation': 4 * 3600 - 5, 'rendering': 0})
    with pytest.raises(RuntimeError, match='Insufficient GPU budget'):
        run(path)
    assert not (campaign / 'records/dynamic_jobs/phase1').exists()


def test_authenticated_guarded_child_with_fake_gpu(campaign):
    output = campaign / 'phase1/value.json'
    code = (f'import runpy, json\nfrom pathlib import Path\n'
            f'ns=runpy.run_path({str(GUARD_PATH)!r})\n'
            'receipt=ns["authenticate_guarded_job"](__file__)\n'
            f'p=Path({str(output)!r}); p.parent.mkdir(parents=True); p.write_text(json.dumps(receipt))\n')
    path, queue, _ = make_queue(campaign, gpu=True, code=code)
    queue['jobs'][0]['command'].append('--guarded')
    path.write_text(json.dumps(queue))
    assert run(path) == 0
    assert json.loads(output.read_text())['guard_pid'] == os.getpid()


def test_forged_guard_environment_rejected(campaign):
    path, _, _ = make_queue(campaign, env={'LIGHTSEM_DYNAMIC_GUARD_PID': '1'})
    with pytest.raises(ValueError, match='authentication environment'):
        run(path)


def test_disk_floor_cannot_be_loosened(campaign):
    path, queue, _ = make_queue(campaign)
    queue['limits']['min_free_bytes'] = 1
    path.write_text(json.dumps(queue))
    with pytest.raises(ValueError, match='loosen campaign constraints'):
        run(path)


def test_json_overflow_is_nonfinite(campaign):
    output = campaign / 'overflow.json'
    output.write_text('{"value":1e999}')
    with pytest.raises(ValueError, match='Nonfinite JSON'):
        guard.check_finite_outputs({str(output): {}})


def test_lock_waits_without_killing_foreign_owner(campaign):
    lock_path, ready = campaign.parent / 'foreign.lock', campaign.parent / 'ready'
    code = ('import fcntl,time\nfrom pathlib import Path\n'
            f'f=open({str(lock_path)!r},"a"); fcntl.flock(f,fcntl.LOCK_EX)\n'
            f'Path({str(ready)!r}).touch()\ntime.sleep(.25)\n')
    process = subprocess.Popen([sys.executable, '-c', code])
    try:
        deadline = time.monotonic() + 2
        while not ready.exists():
            assert time.monotonic() < deadline
            time.sleep(.01)
        started = time.monotonic()
        with guard.lock(lock_path, .01):
            assert time.monotonic() - started > .1
        assert process.wait(timeout=2) == 0
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait()


def test_forged_child_token_is_rejected(campaign):
    code = (f'import os,runpy\nos.environ["LIGHTSEM_DYNAMIC_TOKEN"]="forged"\n'
            f'ns=runpy.run_path({str(GUARD_PATH)!r}); ns["authenticate_guarded_job"](__file__)\n')
    path, queue, _ = make_queue(campaign, gpu=True, code=code)
    queue['jobs'][0]['command'].append('--guarded')
    path.write_text(json.dumps(queue))
    with pytest.raises(subprocess.CalledProcessError):
        run(path)
    assert 'authentication token differs' in (campaign / 'records/dynamic_jobs/phase1/stdout.log').read_text()


def test_frozen_runtime_arguments_are_checked(campaign):
    code = (f'import sys,runpy\nsys.argv.append("--changed")\n'
            f'ns=runpy.run_path({str(GUARD_PATH)!r}); ns["authenticate_guarded_job"](__file__)\n')
    path, queue, _ = make_queue(campaign, gpu=True, code=code)
    queue['jobs'][0]['command'].append('--guarded')
    path.write_text(json.dumps(queue))
    with pytest.raises(subprocess.CalledProcessError):
        run(path)
    assert 'arguments differ' in (campaign / 'records/dynamic_jobs/phase1/stdout.log').read_text()
