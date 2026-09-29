"""Did any LightSem record contaminate the optiHealth vault?

The two projects have separate vaults. A grep matched the optiHealth vault's
sqlite index for 'lightsem|rpmc', so verify whether that is a real record or
an incidental/stale string.
"""
import os
import sqlite3

VAULT = "/root/vibescience-vault"
IDX = os.path.join(VAULT, ".index" + ".sqlite")

print("=== markdown records mentioning lightsem/rpmc (must be ZERO) ===")
found = []
for root, _, files in os.walk(VAULT):
    for fn in files:
        if not fn.endswith(".md"):
            continue
        p = os.path.join(root, fn)
        try:
            t = open(p, encoding="utf-8", errors="ignore").read().lower()
        except OSError:
            continue
        if "lightsem" in t or "rpmc" in t:
            found.append(p)
print(found if found else "  none -> no contaminated records")

print("\n=== index rows matching ===")
con = sqlite3.connect(IDX)
tables = [r[0] for r in con.execute(
    "select name from sqlite_master where type='table'")]
print("tables:", tables)
hits = 0
for t in tables:
    try:
        cols = [r[1] for r in con.execute(f'pragma table_info("{t}")')]
    except sqlite3.Error:
        continue
    for c in cols:
        try:
            rows = con.execute(
                f'select "{c}" from "{t}" where '
                f'lower(cast("{c}" as text)) like "%lightsem%" or '
                f'lower(cast("{c}" as text)) like "%rpmc%"').fetchall()
        except sqlite3.Error:
            continue
        for r in rows[:5]:
            hits += 1
            print(f"  {t}.{c}: {str(r[0])[:150]}")
print("matching rows:", hits)

print("\n=== record counts per vault ===")
for v in ("/root/vibescience-vault", "/root/lightsem-vault"):
    n = sum(1 for root, _, fs in os.walk(v) for f in fs if f.endswith(".md"))
    print(f"  {v:28s} {n} md files")
