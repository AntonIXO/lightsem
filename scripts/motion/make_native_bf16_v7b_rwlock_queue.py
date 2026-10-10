"""Freeze v7b GPU jobs against the writable-lock guard wrapper."""
from pathlib import Path
import hashlib
import json
import runpy

ROOT = Path('/root/lightsem/experiments/native_bf16_v7b')
GUARD = Path('/root/lightsem/scripts/motion/native_full_fit_guard_bf16_v7b_rwlock.py')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=('features', 'preflight', 'training'))
    p.add_argument('--queue', type=Path, required=True)
    args = p.parse_args()

    # Reuse the existing frozen queue recipe, then replace only the guard
    # identity and wrapper argument.  The stage inputs and command payload are
    # otherwise byte-for-byte identical to the failed-closed queue.
    old = runpy.run_path('/root/lightsem/scripts/motion/make_native_bf16_v7b_queue.py')
    tmp = args.queue.with_suffix('.recipe.json')
    old['main']
    # Reconstruct the queue directly to avoid invoking the old CLI.
    import sys
    old_argv = sys.argv
    try:
        sys.argv = ['make_native_bf16_v7b_queue.py', args.stage, '--queue', str(tmp)]
        old['main']()
    finally:
        sys.argv = old_argv
    queue = json.loads(tmp.read_text())
    tmp.unlink()
    queue['id'] = queue['id'].replace('-20261009', '-rwlock10-20261010')
    queue['guard_sha256'] = sha(GUARD)
    job = queue['jobs'][0]
    job['id'] = job['id'].replace('-20261009', '-rwlock10-20261010')
    # The generic guard reserves a ten-second verification margin inside the
    # stage budget.  A nominal 24-hour feature timeout therefore cannot fit
    # its own 24-hour budget; bind this new queue to 23 hours instead.
    if args.stage == 'features':
        job['timeout_seconds'] = 23 * 3600
        job['command'].extend([
            '--partial-root', str(ROOT),
        ])
        job['allow_existing_output'] = True
    elif args.stage in ('preflight', 'training'):
        job['timeout_seconds'] = 42000
        job['env']['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    job['command'] = [str(GUARD) if x == '/root/lightsem/scripts/motion/native_full_fit_guard_bf16_v7b.py' else x
                      for x in job['command']]
    job['command'] = [str(GUARD) if x == '/root/lightsem/scripts/motion/native_full_fit_guard_bf16_v7b.py' else x
                      for x in job['command']]
    source_files = job['source_files']
    old_guard = '/root/lightsem/scripts/motion/native_full_fit_guard_bf16_v7b.py'
    source_files.pop(old_guard, None)
    source_files[str(GUARD)] = sha(GUARD)
    dynamic_guard = '/root/lightsem/scripts/motion/dynamic_dance_guard.py'
    source_files[dynamic_guard] = sha(dynamic_guard)
    job['source_files'] = source_files
    args.queue.parent.mkdir(parents=True, exist_ok=True)
    args.queue.write_text(json.dumps(queue, indent=2, sort_keys=True) + '\n')
    args.queue.with_name(args.queue.name + '.sha256').write_text(sha(args.queue) + '\n')
    print(json.dumps({'queue': str(args.queue), 'sha256': sha(args.queue), 'stage': args.stage,
                      'job_id': job['id'], 'guard': str(GUARD)}))


if __name__ == '__main__':
    main()
