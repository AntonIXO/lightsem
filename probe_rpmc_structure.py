"""Prove we can read RPMC-L2 structure over HTTP range requests, no download."""
import time

from lightsem.data.rpmc_remote import TOTAL, open_h5

t0 = time.time()
print(f"virtual file size: {TOTAL/1e9:.2f} GB")
f, raw = open_h5(verbose=True)
print(f"opened in {time.time()-t0:.1f}s  fetched={raw.bytes_fetched/1e6:.1f} MB")

keys = list(f.keys())
print(f"\ntop-level keys: {len(keys)}")
for k in keys[:5]:
    print("  ", k)

g = f[keys[0]]
print(f"\ngroups in '{keys[0][:24]}': {list(g.keys())}")
for grp in g.keys():
    print(f"  [{grp}]")
    for name, ds in g[grp].items():
        print(f"    {name:22s} {str(ds.shape):18s} {ds.dtype}")

print(f"\ntotal fetched: {raw.bytes_fetched/1e6:.1f} MB "
      f"in {raw.requests_made} requests  ({time.time()-t0:.1f}s)")
print(f"that is {100*raw.bytes_fetched/TOTAL:.4f}% of the dataset")
f.close()
