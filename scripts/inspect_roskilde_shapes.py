from __future__ import annotations
import pickle, pathlib, numpy as np
p=pathlib.Path('/mnt/lightsem-data/open_datasets/roskilde_2023/songs.pkl')
with p.open('rb') as f: obj=pickle.load(f)
x=obj[0]
print('song',x['interpret'],x['song'])
for k,v in x['audio_features'].items():
 try: print('audio',k,np.asarray(v).shape,np.asarray(v).dtype)
 except Exception: print('audio',k,type(v))
print('lighting',x['lighting_features'].shape,x['lighting_features'].dtype)
print('lighting first',x['lighting_features'][0,:20])
