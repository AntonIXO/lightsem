#!/usr/bin/env python3
"""Render a frozen review inventory, with separate H3/H4 time accounting."""
from __future__ import annotations

import argparse
import copy
import json
import os
import resource
import shutil
import signal
import subprocess
import time
from pathlib import Path

import numpy as np

from lightsem.model.light_layers.visual import sha256, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', required=True, type=Path)
    args = parser.parse_args()
    freeze = json.loads(args.freeze.read_text())
    for path, digest in freeze['files'].items():
        if sha256(path) != digest:
            raise RuntimeError('Frozen delivery input changed: ' + path)
    manifest = copy.deepcopy(json.loads(Path(freeze['plan']).read_text()))
    root = Path(manifest['output_root'])
    root.mkdir(exist_ok=True)
    if (root / 'MANIFEST.json').exists() or (root / 'DELIVERY.json').exists():
        raise RuntimeError('Review output already published; no implicit resume or overwrite')
    if len({v['id'] for v in manifest['variants']}) != len(manifest['variants']):
        raise RuntimeError('Duplicate review identifier')
    elapsed = {stage: 0. for stage in freeze['stage_seconds_remaining']}
    started = time.monotonic()
    env = dict(os.environ, TORCH_ENV='/root/lightsem/.venv/torch-access', OPENBLAS_NUM_THREADS='2')

    def publish():
        temporary = root / 'MANIFEST.json.tmp'
        temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
        temporary.replace(root / 'MANIFEST.json')

    def resources():
        study = Path(freeze['study_root'])
        if any(shutil.disk_usage(p).free < 2**30 for p in (study, '/mnt')):
            raise RuntimeError('Study free-space floor')
        total = sum(p.stat().st_size for d in study.iterdir() if d.is_dir() and d.name != 'papers'
                    for p in d.rglob('*') if p.is_file())
        if total > freeze['max_study_bytes']:
            raise RuntimeError('Study artifact allowance')

    def run_phase(command, log, timeout):
        resources()
        active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True)
        if active.strip():
            raise RuntimeError('Another process occupies the GPU before delivery')
        child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        begin = time.monotonic()
        try:
            while child.poll() is None:
                if time.monotonic() - begin > timeout:
                    raise RuntimeError('Delivery phase timeout')
                resources()
                used = int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).strip().splitlines()[0]) * 2**20
                if used > 10 * 2**30:
                    raise RuntimeError('Delivery device memory allowance')
                # Includes the renderer and its owned NVENC process; other GPU
                # workloads cannot silently contaminate this fixed comparison.
                pids = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True)
                for pid in pids.splitlines():
                    try:
                        if pid.strip() and os.getpgid(int(pid)) != child.pid:
                            raise RuntimeError('Concurrent external GPU process')
                    except ProcessLookupError:
                        pass
                time.sleep(2)
            if child.returncode:
                raise subprocess.CalledProcessError(child.returncode, command)
            if resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * 1024 > 8 * 2**30:
                raise RuntimeError('Delivery child process RAM allowance')
        except BaseException:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            raise
        resources()

    publish()
    rendered = 0
    linked = 0
    for row in manifest['variants']:
        if row['status'] == 'verified_existing':
            if sha256(row['path']) != row['sha256']:
                raise RuntimeError('Reused media changed')
            continue
        if row['status'] != 'planned':
            raise RuntimeError('Unexpected candidate delivery state')
        if sha256(row['streams']) != row['streams_sha256']:
            raise RuntimeError('Candidate stream changed')
        if 'same_render_as' in row:
            original = next(v for v in manifest['variants'] if v['id'] == row['same_render_as'])
            if original['status'] != 'rendered_and_verified':
                raise RuntimeError('Identical-output representative is not verified')
            with np.load(original['streams'], allow_pickle=False) as a, np.load(row['streams'], allow_pickle=False) as b:
                if not np.array_equal(a['candidate'], b['candidate']) or not np.array_equal(a['time'], b['time']):
                    raise RuntimeError('Render reuse requires exact frame and clock equality')
            output = Path(row['path'])
            output.parent.mkdir(parents=True, exist_ok=False)
            os.link(original['path'], output)
            if sha256(output) != original['sha256']:
                raise RuntimeError('Identical-output media identity changed')
            row.update(status='rendered_and_verified', sha256=original['sha256'], bytes=output.stat().st_size,
                       audit=original['audit'], render_and_audit_seconds=0.,
                       render_reuse='Exact output; authenticated hard link to the verified representative. No second encoder invocation.')
            linked += 1
            publish()
            print('IDENTICAL_REVIEW_VIDEO_VERIFIED', row['id'], original['id'], flush=True)
            continue
        stage = row['budget_stage']
        remaining = freeze['stage_seconds_remaining'][stage] - elapsed[stage]
        if remaining <= 0:
            raise RuntimeError('No remaining ' + stage + ' delivery allowance')
        output = Path(row['path'])
        output.parent.mkdir(parents=True, exist_ok=False)
        begin = time.monotonic()
        for mode in ('render', 'audit'):
            remaining = freeze['stage_seconds_remaining'][stage] - elapsed[stage] - (time.monotonic() - begin)
            if remaining <= 0:
                raise RuntimeError('Delivery time allowance exhausted')
            command = ['bash', 'run_torch.sh', '-u', '-m', 'scripts.rendering.render_light_layers_review', mode,
                       '--streams', row['streams'], '--output', str(output), '--label', row['caption'],
                       '--audio', manifest['source_audio'], '--max-bytes', '50000000']
            with (output.parent / (mode + '.log')).open('xb') as log:
                run_phase(command, log, min(720., remaining))
        elapsed[stage] += time.monotonic() - begin
        audit_path = output.with_name(output.stem + '_audit.json')
        audit = json.loads(audit_path.read_text())
        render = json.loads(output.with_name(output.stem + '_render.json').read_text())
        if not (audit['decoded_frames'] == 8319 and audit['pts_uniform'] and audit['full_software_AV_decode']):
            raise RuntimeError('Incomplete media audit')
        if audit['pcm_sha256'] != manifest['expected_pcm_sha256']:
            raise RuntimeError('Delivered soundtrack differs')
        if render['video_sha256'] != sha256(output) or output.stat().st_size > 50_000_000:
            raise RuntimeError('Media identity or compact-size gate failed')
        if render['dmx_sha256'] != row['streams_sha256']:
            raise RuntimeError('Rendered the wrong stream')
        row.update(status='rendered_and_verified', sha256=sha256(output), bytes=output.stat().st_size,
                   audit=str(audit_path), render_and_audit_seconds=time.monotonic() - begin)
        rendered += 1
        manifest['delivery_stage_seconds'] = elapsed.copy()
        publish()
        print('REVIEW_VIDEO_VERIFIED', row['id'], row['letter'], elapsed, flush=True)
    for path, digest in freeze['files'].items():
        if sha256(path) != digest:
            raise RuntimeError('Frozen delivery input changed: ' + path)
    manifest.update(status='all_review_videos_rendered_and_verified', elapsed_s=time.monotonic() - started)
    publish()
    write_json(root / 'DELIVERY.json', {
        'status': manifest['status'], 'manifest_sha256': sha256(root / 'MANIFEST.json'),
        'freeze_sha256': sha256(args.freeze), 'new_encoded_videos': rendered,
        'new_named_video_files': rendered + linked, 'exact_duplicate_hard_links': linked,
        'existing_videos_reused': len(manifest['variants']) - rendered - linked,
        'review_variants': len(manifest['variants']), 'stage_seconds': elapsed,
        'same_audio_verified': True, 'baseline_promoted': False,
        'scope': 'Personal review of explicitly labeled learned, design and reference-oracle variants; no scientific gate waiver.',
    })


if __name__ == '__main__':
    main()
