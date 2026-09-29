#!/usr/bin/env python3
"""Show the RAW JSON-RPC responses so the write failure can be diagnosed."""
import json
import os
import subprocess

SERVER = "/root/vibescience-mcp/.venv/bin/vibescience-mcp"
env = dict(os.environ)
env["VIBESCIENCE_VAULT"] = "/root/lightsem-vault"
env["VIBESCIENCE_SERVER_NAME"] = "vibescience-lightsem"

lines = [
    json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                           "clientInfo": {"name": "p", "version": "1"}}}),
    json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
    json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                "params": {"name": "rule_out", "arguments": {
                    "problem_id": "audio-to-light-semantic-mapping-20260807",
                    "claim": "SMOKE TEST claim, delete me.",
                    "method": "smoke",
                    "evidence": "smoke"}}}),
]
p = subprocess.run([SERVER], input="\n".join(lines) + "\n",
                   capture_output=True, text=True, env=env, timeout=120)
print("--- RAW STDOUT ---")
print(p.stdout[:4000])
print("--- STDERR tail ---")
print(p.stderr[-1500:])
