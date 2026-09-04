#!/usr/bin/env bash
# Verify we can read Brave cookies from the desktop user's profile.
set -u
cd /root/lightsem
export PATH="/root/lightsem/.venv/bin:$PATH"

SRC="/home/kirill/.config/BraveSoftware/Brave-Browser"
WORK="/tmp/brave_prof"

echo "=== Brave running? ==="
pgrep -a -f "brave" | head -3 || echo "  (not running - no lock contention)"

echo
echo "=== snapshot profile (Brave locks the live DB) ==="
rm -rf "$WORK"
mkdir -p "$WORK/Default"
cp "$SRC/Default/Cookies" "$WORK/Default/Cookies" 2>&1 || echo "  cookies copy FAILED"
cp "$SRC/Local State" "$WORK/Local State" 2>&1 || echo "  local state copy FAILED"
chmod -R u+rw "$WORK"
ls -la "$WORK/Default/"

echo
echo "=== os_crypt shape ==="
python - <<'PY'
import json
try:
    d = json.load(open("/tmp/brave_prof/Local State"))
    osc = d.get("os_crypt", {})
    print("  os_crypt keys      :", list(osc.keys()))
    print("  encrypted_key      :", "encrypted_key" in osc)
except Exception as e:
    print("  ERR", type(e).__name__, e)
PY

echo
echo "=== raw cookie rows for youtube ==="
python - <<'PY'
import sqlite3
try:
    c = sqlite3.connect("file:/tmp/brave_prof/Default/Cookies?mode=ro", uri=True)
    n = c.execute("select count(*) from cookies where host_key like '%youtube%'").fetchone()[0]
    tot = c.execute("select count(*) from cookies").fetchone()[0]
    print(f"  total cookies      : {tot}")
    print(f"  youtube cookies    : {n}")
    for name, host in c.execute(
        "select name, host_key from cookies where host_key like '%youtube%' limit 8"
    ):
        print(f"    {host:28s} {name}")
except Exception as e:
    print("  ERR", type(e).__name__, e)
PY

echo
echo "=== yt-dlp cookie extraction (the real test) ==="
yt-dlp --cookies-from-browser "brave:$WORK" \
       --skip-download --print "%(title).60s" --no-warnings \
       "https://www.youtube.com/watch?v=BaW_jenozKc" 2>&1 | tail -5
