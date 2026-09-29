"""Stop the existing extractor at a completed-record boundary (no GPU work)."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import selectors
import signal
import struct
import time
from datetime import datetime, timezone

p = argparse.ArgumentParser()
p.add_argument('--pid', type=int, required=True)
p.add_argument('--manifest', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--receipt', type=Path, required=True)
p.add_argument('--stop-after-recording-id')
a = p.parse_args()
if a.receipt.exists():
    raise RuntimeError('Do not overwrite a handoff receipt')
manifest_bytes = a.manifest.read_bytes()
manifest = json.loads(manifest_bytes)
cmdline = Path(f'/proc/{a.pid}/cmdline').read_bytes()
if b'run_light_layers_visual.py' not in cmdline or str(a.manifest).encode() not in cmdline:
    raise RuntimeError('PID does not match the authorized capture')
pgid = os.getpgid(a.pid)
if pgid == os.getpgrp():
    raise RuntimeError('Refuse to signal the controller process group')
libc = ctypes.CDLL(None, use_errno=True)
libc.inotify_init1.argtypes = [ctypes.c_int]
libc.inotify_add_watch.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
fd = libc.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC)
if fd < 0:
    raise OSError(ctypes.get_errno(), 'inotify_init1')
libc.pidfd_open.argtypes = [ctypes.c_int, ctypes.c_uint]
pidfd = libc.pidfd_open(a.pid, 0)
if pidfd < 0:
    raise OSError(ctypes.get_errno(), 'pidfd_open')
sel = selectors.DefaultSelector()
sel.register(fd, selectors.EVENT_READ, 'files')
sel.register(pidfd, selectors.EVENT_READ, 'process')
watched = {}
mask = 0x8 | 0x80 | 0x100  # close-write, moved-to, create

def watch(path):
    wd = libc.inotify_add_watch(fd, os.fsencode(path), mask)
    if wd < 0:
        raise OSError(ctypes.get_errno(), f'inotify watch {path}')
    watched[wd] = Path(path)

watch(a.output)
active = []
for row in manifest['inputs']:
    directory = a.output / row['recording_id']
    if directory.is_dir() and not (directory/'complete.json').exists() and not (directory/'invalid.json').exists():
        watch(directory)
        active.append(directory)

receipt = {'schema_version': 1, 'pid': a.pid, 'process_group': pgid,
           'manifest_path': str(a.manifest),
           'manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
           'reason': 'User authorized two simultaneous video workers; preserve completed work.',
           'started_at_utc': datetime.now(timezone.utc).isoformat()}

def save():
    tmp = a.receipt.with_suffix('.partial')
    tmp.write_text(json.dumps(receipt, indent=2))
    os.replace(tmp, a.receipt)

def complete(path):
    if path.name != 'complete.json' or not path.is_file():
        return False
    value = json.loads(path.read_text())
    return value.get('status') == 'complete' and value.get('recording_id') in {r['recording_id'] for r in manifest['inputs']}

def boundary_complete(path):
    return complete(path) and (a.stop_after_recording_id is None or path.parent.name == a.stop_after_recording_id)
if a.stop_after_recording_id is not None:
    if a.stop_after_recording_id not in {r['recording_id'] for r in manifest['inputs']}:
        raise RuntimeError('Requested boundary is outside this worker manifest')
    named = a.output/a.stop_after_recording_id/'complete.json'
    boundary = named if boundary_complete(named) else None
else:
    boundary = next((d/'complete.json' for d in active if boundary_complete(d/'complete.json')), None)
deadline = time.monotonic() + 240
while boundary is None:
    remaining = deadline-time.monotonic()
    if remaining <= 0:
        raise RuntimeError('No completed boundary within 240 seconds; original job left running')
    events = sel.select(remaining)
    for key, _ in events:
        if key.data == 'process':
            receipt.update(status='original_finished_before_handoff', finished_at_utc=datetime.now(timezone.utc).isoformat())
            save()
            raise SystemExit(0)
        payload = os.read(fd, 65536)
        pos = 0
        while pos < len(payload):
            wd, event_mask, cookie, length = struct.unpack_from('iIII', payload, pos)
            pos += 16
            name = os.fsdecode(payload[pos:pos+length].split(b'\0', 1)[0])
            pos += length
            path = watched[wd]/name
            if event_mask & 0x40000000 and path.is_dir():
                watch(path)
            elif event_mask & (0x8 | 0x80 | 0x100) and boundary_complete(path):
                boundary = path
                break
        if boundary is not None:
            break
# Freeze the producer before inventorying the completed boundary. Any already
# started next-record directory is preserved and will never be reused.
os.killpg(pgid, signal.SIGSTOP)
try:
    complete_ids = []
    incomplete_ids = []
    for row in manifest['inputs']:
        directory = a.output/row['recording_id']
        if complete(directory/'complete.json'):
            complete_ids.append(row['recording_id'])
        elif directory.exists():
            incomplete_ids.append(row['recording_id'])
    receipt.update(status='boundary_stop_requested', boundary_record=boundary.parent.name,
                   completed_recording_ids=complete_ids, preserved_incomplete_directories=incomplete_ids)
    save()
    os.killpg(pgid, signal.SIGINT)
finally:
    try:
        os.killpg(pgid, signal.SIGCONT)
    except ProcessLookupError:
        pass
# Event-driven wait for process exit, not repeated liveness polling.
sel.unregister(fd)
if not sel.select(30):
    raise RuntimeError('Extractor has not exited after boundary interrupt; inspect before launching workers')
receipt.update(status='stopped_at_record_boundary', finished_at_utc=datetime.now(timezone.utc).isoformat())
save()
print(json.dumps(receipt), flush=True)
sel.close()
os.close(pidfd)
os.close(fd)
