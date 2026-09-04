#!/usr/bin/env bash
# Why did the optiHealth vault's sqlite index match "lightsem|rpmc"?
# Either a real contaminated record, or a stale/incidental string. Find out.
cd /root/vibescience-vault
echo "=== markdown files mentioning lightsem/rpmc (should be ZERO) ==="
grep -rl -i "lightsem\|rpmc" --include="*.md" . 2>/dev/null || true
echo "(nothing above = no contaminated records)"

echo
echo "=== what rows in the index match? ==="
python3 - <<'PY'
import sqlite3
c = sqlite3.connect("/root/vibescience-vault/.index.sqlite")
tabs = [r[0] for r in c.execute(
    "select name from sqlite_master where type='table'")]
print("tables:", tabs)
hits = 0
for t in tabs:
    try:
        cols = [r[1] for r in c.execute(f"pragma table_info({t})")]
    except Exception:
        continue
    for col in cols:
        try:
            rows = c.execute(
                f"select \"{col}\" from \"{t}\" where lower(cast(\"{col}\" as text)) "
                f"like '%lightsem%' or lower(cast(\"{col}\" as text)) like '%rpmc%'"
            ).fetchall()
        except Exception:
            continue
        for r in rows[:5]:
            hits += 1
            print(f"  {t}.{col}: {str(r[0])[:160]}")
print("total matching rows:", hits)
PY
