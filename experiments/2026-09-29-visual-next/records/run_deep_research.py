"""Call the configured Perplexity research MCP with a900s client timeout.

The desktop transport timed out at300s. This uses the same configured server
and credentials without changing the gateway or writing credentials to disk.
"""
import hashlib
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import time
import tomllib

root = Path(__file__).resolve().parent
cfg = tomllib.loads(Path('/root/.codex/config.toml').read_text())['mcp_servers']['perplexity']
query = (root / 'PERPLEXITY_QUERY.txt').read_text()
env = dict(os.environ, **cfg.get('env', {}))
proc = subprocess.Popen([cfg['command'], *cfg.get('args', [])], stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                        bufsize=1, env=env, start_new_session=True)
selector = selectors.DefaultSelector()
selector.register(proc.stdout, selectors.EVENT_READ)
started = time.monotonic()

def send(obj):
    proc.stdin.write(json.dumps(obj) + '\n')
    proc.stdin.flush()

def receive(request_id, timeout):
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        ready = selector.select(min(30, max(0, until - time.monotonic())))
        if not ready:
            print('Research request pending; elapsed_s', round(time.monotonic()-started), flush=True)
            continue
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError('MCP server closed stdout')
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        if msg.get('id') == request_id:
            return msg
    raise TimeoutError('Direct MCP request exceeded timeout')

try:
    send({'jsonrpc':'2.0','id':1,'method':'initialize','params':{
        'protocolVersion':'2024-11-05','capabilities':{},'clientInfo':{'name':'lightsem-research','version':'1'}}})
    initialized = receive(1, 60)
    if 'error' in initialized:
        raise RuntimeError('MCP initialization failed')
    send({'jsonrpc':'2.0','method':'notifications/initialized'})
    print('Initialized configured Perplexity MCP; calling perplexity_research with900s timeout', flush=True)
    send({'jsonrpc':'2.0','id':2,'method':'tools/call','params':{
        'name':'perplexity_research','arguments':{'query':query,'sources':['web','scholar'],'language':'en-US'}}})
    response = receive(2, 900)
    receipt = {'tool':'perplexity_research','query_sha256':hashlib.sha256(query.encode()).hexdigest(),
               'query_characters':len(query),'timeout_s':900,'elapsed_s':time.monotonic()-started,
               'same_configured_server':True,'response':response}
    with (root/'PERPLEXITY_ATTEMPT_2.json').open('x') as f:
        json.dump(receipt, f, indent=2, ensure_ascii=False)
    result = response.get('result', {})
    report = '\n\n'.join(c['text'] for c in result.get('content', []) if c.get('type')=='text')
    with (root/'PERPLEXITY_REPORT_RAW.txt').open('x') as f:
        f.write(report)
    print('Research response saved; characters',len(report),'isError',result.get('isError',False),flush=True)
    if response.get('error') or result.get('isError'):
        raise RuntimeError('Research tool returned an error; see saved receipt')
finally:
    selector.close()
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
