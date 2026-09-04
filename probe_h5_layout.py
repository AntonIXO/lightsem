"""What is the ACTUAL layout under music/? fetch_interp_feats.py assumed wrong.

Every song raised 'Field names only allowed for compound types', which h5py
raises when you index a DATASET with a string key. So `f[k]['music']` is not a
Group of named feature datasets as assumed -- it is itself a Dataset (probably
a single stacked array, or a compound/structured type). Print the truth.
"""
from __future__ import annotations

import h5py  # noqa: F401
import numpy as np

from lightsem.data.rpmc_remote import open_h5


def describe(obj, path, depth=0):
    pad = "  " * depth
    if isinstance(obj, h5py.Group):
        print(f"{pad}{path}/  GROUP  keys={list(obj.keys())[:12]}")
        for k in list(obj.keys())[:12]:
            describe(obj[k], k, depth + 1)
    else:
        print(f"{pad}{path}  DATASET shape={obj.shape} dtype={obj.dtype}")
        if obj.dtype.names:
            print(f"{pad}  compound fields: {obj.dtype.names}")
        for a, v in obj.attrs.items():
            print(f"{pad}  attr {a} = {v!r}")


def main():
    f, raw = open_h5(verbose=False)
    k = list(f.keys())[0]
    print(f"song key: {k}")
    print(f"song node type: {type(f[k]).__name__}  keys={list(f[k].keys())}\n")

    for sub in f[k].keys():
        describe(f[k][sub], sub)
        print()

    m = f[k]["music"]
    if isinstance(m, h5py.Dataset):
        print("music IS a dataset -- inspecting content")
        print(f"  shape {m.shape} dtype {m.dtype}")
        for a, v in m.attrs.items():
            print(f"  attr {a} = {v!r}")
        s = m[:5] if m.ndim == 1 else m[:3, :6]
        print(f"  head:\n{np.asarray(s)}")

    print(f"\nnetwork: {raw.bytes_fetched/1e6:.1f} MB")


if __name__ == "__main__":
    main()
