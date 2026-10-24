from __future__ import annotations
import pickle, pathlib, numpy as np
p=pathlib.Path('/mnt/lightsem-data/open_datasets/roskilde_2023/songs.pkl')
with p.open('rb') as f:
    obj=pickle.load(f)
print(type(obj), len(obj) if hasattr(obj,'__len__') else None)
item=obj[0] if isinstance(obj,(list,tuple)) else next(iter(obj.values()))
print('item_type',type(item))
if isinstance(item,dict):
 print('keys',list(item.keys()))
 for k,v in item.items():
  if isinstance(v,dict): print(k,'dict_keys',list(v.keys())[:20])
  elif isinstance(v,np.ndarray): print(k,'array',v.shape,v.dtype)
  else: print(k,type(v),str(v)[:200])
