#!/usr/bin/python3 -B
"""Bounded CPU fixture check; preserves capped logs, source hashes and outcome."""
import hashlib
import json
import os
from pathlib import Path
import selectors
import subprocess
import time

HERE = Path(__file__).resolve().parent
UNIT = 'lightsem-notes-recovery-fixtures-20260929-v2'
LIMIT = 32768
command = ['systemd-run', '--quiet', '--wait', '--pipe', '--collect', '--unit=' + UNIT,
           '--property=MemoryMax=134217728', '--property=MemorySwapMax=0',
           '--property=PrivateDevices=yes', '--property=PrivateTmp=yes',
           '--property=RuntimeMaxSec=40', '--setenv=PYTHONDONTWRITEBYTECODE=1',
           '/usr/bin/python3', '-B', '-m', 'unittest', 'discover', '-s', str(HERE),
           '-p', 'test_restore_notes.py', '-v']
result = {'command': command, 'fixture_vaults_only': True, 'source_sha256': {
    p.name: hashlib.sha256(p.read_bytes()).hexdigest()
    for p in (HERE / 'restore_notes_mcp.py', HERE / 'test_restore_notes.py', Path(__file__))}}
started = time.monotonic()
with (HERE / 'FIXTURE_TEST_LOG_V2.txt').open('xb') as log:
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    sel = selectors.DefaultSelector()
    sel.register(process.stdout, selectors.EVENT_READ)
    count = 0
    stop_reason = None
    try:
        while sel.get_map():
            if time.monotonic() - started > 55:
                stop_reason = 'whole job watchdog'
                break
            for key, _ in sel.select(timeout=0.2):
                data = os.read(key.fd, 4096)
                if not data:
                    sel.unregister(key.fileobj)
                    continue
                allowed = min(len(data), LIMIT - count)
                log.write(data[:allowed])
                count += allowed
                if allowed < len(data):
                    stop_reason = 'combined log cap'
                    break
            if stop_reason:
                break
        if stop_reason:
            subprocess.run(['systemctl', 'kill', '--signal=KILL', UNIT], capture_output=True, timeout=5)
            process.kill()
        result.update(exit_code=process.wait(timeout=5), stop_reason=stop_reason,
                      elapsed_seconds=time.monotonic() - started, combined_log_bytes=count)
    finally:
        sel.close()
        if process.poll() is None:
            subprocess.run(['systemctl', 'kill', '--signal=KILL', UNIT], capture_output=True, timeout=5)
            process.kill()
            process.wait(timeout=5)
with (HERE / 'FIXTURE_TEST_RESULT_V2.json').open('x') as out:
    json.dump(result, out, indent=2)
print(json.dumps(result))
print((HERE / 'FIXTURE_TEST_LOG_V2.txt').read_text())
raise SystemExit(0 if result['exit_code'] == 0 and result['stop_reason'] is None else 1)
