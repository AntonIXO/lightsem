#!/usr/bin/env python3
"""CPU-only systemd envelope probe. No project/data/model imports or payload access."""
import argparse
import json
from pathlib import Path
import os
import signal

p = argparse.ArgumentParser()
p.add_argument('--unit', required=True)
p.add_argument('--expected-memory-bytes', type=int, required=True)
p.add_argument('--expected-runtime-text', required=True)
a = p.parse_args()
assert a.unit.startswith('lightsem-') and a.unit.endswith('.service') and '/' not in a.unit
line = Path('/proc/self/cgroup').read_text().strip()
assert line.startswith('0::/') and '\n' not in line
relative = line[3:]
assert Path(relative).name == a.unit, (relative, a.unit)
cg = Path('/sys/fs/cgroup') / relative.lstrip('/')
memory = (cg / 'memory.max').read_text().strip()
swap = (cg / 'memory.swap.max').read_text().strip()
assert memory == str(a.expected_memory_bytes) and swap == '0', (memory, swap)
unit_path = Path('/run/systemd/transient') / a.unit
assert unit_path.stat().st_size < 16384
unit = unit_path.read_text()
properties = {}
for item in unit.splitlines():
    if item and not item.startswith('#') and '=' in item:
        key, value = item.split('=', 1)
        properties[key] = value
assert properties.get('RuntimeMaxSec') == a.expected_runtime_text, properties
assert properties.get('KillMode') == 'control-group', properties
assert properties.get('KillSignal') in ('9', 'SIGKILL'), properties
assert properties.get('PrivateDevices') == 'yes', properties
assert properties.get('ProtectControlGroups') == 'yes', properties
assert properties.get('NoNewPrivileges') == 'yes', properties
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
assert os.environ.get('PYTHONDONTWRITEBYTECODE') == '1'
assert not list(Path('/dev').glob('nvidia*'))
print(json.dumps({'schema_version': 1, 'status': 'LIMITS_VERIFIED_AWAIT_EXPECTED_RUNTIME_KILL',
    'unit': a.unit, 'cgroup': relative, 'memory_max': memory, 'memory_swap_max': swap,
    'unit_configuration': unit, 'CUDA_VISIBLE_DEVICES': '', 'GPU_devices_visible': False,
    'model_imports': 0, 'payload_reads': 0}, sort_keys=True), flush=True)
# A blocking signal wait, not a poll or a CPU load. systemd must terminate this
# entire service at its two-second ceiling; normal exit is not expected.
signal.pause()
raise RuntimeError('Signal wait unexpectedly returned')
