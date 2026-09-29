#!/usr/bin/env python3
"""Verify the bound systemd envelope before a preparation-only native exec."""
import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path('/root/lightsem/outputs/legacy_struct_intensity_20260928_v1')
DRIVER = ROOT / 'prepare_struct_intensity_v5.py'
PLAN = Path('/root/lightsem/docs/research/2026-09-28-end-to-end/PROSPECTIVE_STRUCT_INTENSITY_PROTOCOL_CANDIDATE_V5.json')
AUTH = ROOT / 'PARENT_BOUND_PREPARATION_AUTHORIZATION_V5.json'
PROOF = ROOT / 'preparation_resource_envelope_v5.json'

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

def inspect_envelope(unit_name):
    require(unit_name.startswith('lightsem-') and unit_name.endswith('.service') and '/' not in unit_name, 'invalid unit')
    line = Path('/proc/self/cgroup').read_text().strip()
    require(line.startswith('0::/') and '\n' not in line, 'unrecognized cgroup interface')
    relative = line[3:]
    require(Path(relative).name == unit_name, 'wrong effective cgroup')
    cg = Path('/sys/fs/cgroup') / relative.lstrip('/')
    memory = (cg / 'memory.max').read_text().strip()
    swap = (cg / 'memory.swap.max').read_text().strip()
    require(memory == '4294967296' and swap == '0', 'wrong effective memory/swap limit')
    path = Path('/run/systemd/transient') / unit_name
    require(path.stat().st_size < 16384, 'oversize transient configuration')
    text = path.read_text()
    props = {}
    for line in text.splitlines():
        if line and not line.startswith('#') and '=' in line:
            key, value = line.split('=', 1)
            props[key] = value
    require(props.get('RuntimeMaxSec') in ('14min 50s', '14min50s', '890s'), 'wrong890s internal watchdog')
    require(props.get('KillMode') == 'control-group', 'wrong whole-service termination')
    require(props.get('KillSignal') in ('9', 'KILL', 'SIGKILL'), 'termination is not SIGKILL')
    for key in ('PrivateDevices', 'ProtectControlGroups', 'NoNewPrivileges'):
        require(props.get(key) == 'yes', 'missing isolation: ' + key)
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CUDA not hidden')
    require(os.environ.get('PYTHONDONTWRITEBYTECODE') == '1', 'bytecode not disabled')
    require(os.environ.get('TORCH_ENV') == '/root/lightsem/.venv/torch-access', 'wrong native Torch environment')
    require(not list(Path('/dev').glob('nvidia*')), 'GPU devices visible')
    return {'unit': unit_name, 'cgroup': relative, 'memory_max_bytes': int(memory),
            'memory_swap_max_bytes': int(swap), 'runtime_max_seconds': 890, 'declared_wall_ceiling_seconds': 900, 'dispatch_slack_seconds': 10,
            'transient_configuration': text, 'GPU_devices_visible': False}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--unit', required=True)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--authorization-sha256')
    args = parser.parse_args()
    proof = inspect_envelope(args.unit)
    proof['guard_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if args.check_only:
        require(args.authorization_sha256 is None, 'check-only cannot bind preparation')
        proof.update(status='ENVELOPE_CHECK_ONLY_NO_PREPARATION', payload_reads=0, model_imports=0)
        print(json.dumps(proof, sort_keys=True), flush=True)
        return
    require(args.unit == 'lightsem-struct-preparation-v5.service', 'unexpected preparation unit')
    require(ROOT.resolve() == ROOT and ROOT.is_dir(), 'noncanonical source directory')
    require(AUTH.resolve(strict=True) == AUTH and AUTH.stat().st_size < 2097152, 'invalid authorization path/size')
    data = AUTH.read_bytes()
    require(hashlib.sha256(data).hexdigest() == args.authorization_sha256, 'authorization hash differs')
    auth = json.loads(data)
    require(auth.get('scope') == 'PREPARATION_ONLY_NO_MODEL_OR_SCORE', 'not preparation-only')
    require(auth.get('preparation_external_limits_bound') == {'wall_seconds': 900, 'host_RAM_bytes': 4294967296, 'GPU_access': False}, 'authorization limits differ')
    require(auth.get('external_envelope_source_sha256') == proof['guard_sha256'], 'envelope source not authorized')
    # Bound source/control bytes, not scientific payloads. The entry run.sh and
    # this guard must additionally be hash-checked by root before admission.
    required_refs = [(DRIVER, auth['driver_ref']), (PLAN, auth['plan_ref'])]
    wrappers = [r for r in auth['source_refs'] if r['path'] == '/root/lightsem/run_torch.sh']
    require(len(wrappers) == 1, 'missing or duplicate native wrapper binding')
    required_refs.append((Path('/root/lightsem/run_torch.sh'), wrappers[0]))
    source_checks = []
    for expected_path, ref in required_refs:
        require(ref['path'] == str(expected_path) and expected_path.resolve(strict=True) == expected_path,
                'noncanonical or wrong bound source path')
        require(expected_path.stat().st_size == ref['size_bytes'] and ref['size_bytes'] <= 2097152,
                'bound source size differs')
        require(hashlib.sha256(expected_path.read_bytes()).hexdigest() == ref['sha256'],
                'bound source hash differs: ' + str(expected_path))
        source_checks.append(ref)
    proof['pre_exec_source_checks'] = source_checks
    for key in ('allow_model_construction', 'allow_model_forward', 'allow_scoring', 'allow_GPU', 'allow_TEST_access', 'allow_raw_manifest_access'):
        require(auth.get(key) is False, 'forbidden authority: ' + key)
    argv = ['/usr/bin/bash', '/root/lightsem/run_torch.sh', '-u', str(DRIVER), 'prepare', '--plan', str(PLAN), '--authorization', str(AUTH)]
    require(auth.get('argv') == argv[3:], 'preparation argv differs')
    proof.update(status='ENVELOPE_VERIFIED_BEFORE_PREPARATION_EXEC', authorization_sha256=args.authorization_sha256, native_argv=argv)
    payload = json.dumps(proof, sort_keys=True).encode() + b'\n'
    require(len(payload) <= 16384, 'oversize envelope proof')
    with PROOF.open('xb') as stream:
        stream.write(payload); stream.flush(); os.fsync(stream.fileno())
    # This is the same service/cgroup. The native Torch wrapper replaces this
    # process, so all preparation descendants retain the effective OS caps.
    os.execv('/usr/bin/bash', argv)

if __name__ == '__main__':
    main()
