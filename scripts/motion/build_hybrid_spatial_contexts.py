#!/usr/bin/env python3
"""Build a TRAIN-statistics-normalized SpaIn+oBSRNN hybrid context.

Both method streams remain visible to the native end-to-end model. The fusion
is deterministic (per-method TRAIN z-score, timestamp interpolation, mean of
the two 64-D streams); native C learns the downstream combination. Showcase
contexts are transformed with TRAIN statistics only.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np

TRAIN_NATIVE_MANIFEST = Path('/mnt/lightsem-data/experiments/lightsem/native_bf16_v7b_stereo_v1/STEREO_NATIVE_MANIFEST.json')

def sha(p: Path) -> str: return hashlib.sha256(p.read_bytes()).hexdigest()
def load_manifest(p: Path) -> dict: return json.loads(p.read_text())
def load_ctx(row: dict) -> tuple[np.ndarray,np.ndarray]:
    with np.load(row['context_path'],allow_pickle=False) as z: return z['time30'].astype('f8'), z['fused30'].astype('f4')
def interp(t,x,target): return np.stack([np.interp(target,t,x[:,i]) for i in range(x.shape[1])],1).astype('f4')

def build(train_spain: Path, train_bsrnn: Path, show_spain: Path, show_bsrnn: Path, out: Path):
    if out.exists(): raise FileExistsError(out)
    sm,bm=load_manifest(train_spain),load_manifest(train_bsrnn)
    sr={r['id']:r for r in sm['rows']}; br={r['id']:r for r in bm['rows']}
    train_rows=[]; raw_s=[];raw_b=[]
    for key in sorted(sr):
        if key not in br: raise ValueError(f'missing hybrid row {key}')
        ts,xs=load_ctx(sr[key]); tb,xb=load_ctx(br[key]); target=ts[(ts>=max(ts[0],tb[0]))&(ts<=min(ts[-1],tb[-1]))]
        xs=interp(ts,xs,target); xb=interp(tb,xb,target); raw_s.append(xs);raw_b.append(xb);train_rows.append((key,target,xs,xb))
    s_all=np.concatenate(raw_s);b_all=np.concatenate(raw_b)
    smu,sstd=s_all.mean(0),np.maximum(s_all.std(0),1e-4);bmu,bstd=b_all.mean(0),np.maximum(b_all.std(0),1e-4)
    out.mkdir(parents=True)
    train_out=out/'train';train_out.mkdir()
    rows=[]
    for key,t,xs,xb in train_rows:
        hybrid=(.5*((xs-smu)/sstd)+.5*((xb-bmu)/bstd)).astype('f4');p=train_out/f'{key}.npz';np.savez_compressed(p,time30=t,fused30=hybrid);rows.append({'id':key,'native_key':key,'split':'train','context_path':str(p),'context_sha256':sha(p),'frames30':len(t),'feature_dim':64})
    train_payload={'schema_version':1,'status':'complete','method':'spain_plus_obsrnn_hybrid','rows':rows,'split':'train','test_access':False,'source_spain_manifest':str(train_spain),'source_spain_manifest_sha256':sha(train_spain),'source_bsrnn_manifest':str(train_bsrnn),'source_bsrnn_manifest_sha256':sha(train_bsrnn),'train_spain_mean':smu.tolist(),'train_spain_std':sstd.tolist(),'train_bsrnn_mean':bmu.tolist(),'train_bsrnn_std':bstd.tolist(),'scope':'TRAIN-only z-score fusion; no TEST/showcase statistics'}
    (out/'CONTEXT_MANIFEST.json').write_text(json.dumps(train_payload,indent=2,sort_keys=True)+'\n');(out/'CONTEXT_MANIFEST.json.sha256.json').write_text(json.dumps({'sha256':sha(out/'CONTEXT_MANIFEST.json')})+'\n')
    show_sm,show_bm=load_manifest(show_spain),load_manifest(show_bsrnn);show_sr={r['id']:r for r in show_sm['rows']};show_br={r['id']:r for r in show_bm['rows']};show_out=out/'showcase';show_out.mkdir()
    show_rows=[]
    for key in sorted(show_sr):
        ts,xs=load_ctx(show_sr[key]);tb,xb=load_ctx(show_br[key]);target=ts[(ts>=max(ts[0],tb[0]))&(ts<=min(ts[-1],tb[-1]))];hybrid=(.5*((interp(ts,xs,target)-smu)/sstd)+.5*((interp(tb,xb,target)-bmu)/bstd)).astype('f4');p=show_out/f'{key}.npz';np.savez_compressed(p,time30=target,fused30=hybrid);show_rows.append({'id':key,'split':'showcase','context_path':str(p),'context_sha256':sha(p),'frames30':len(target),'feature_dim':64})
    show_payload={'schema_version':1,'status':'complete','method':'spain_plus_obsrnn_hybrid','rows':show_rows,'split':'showcase','test_access':False,'train_manifest_sha256':sha(out/'CONTEXT_MANIFEST.json'),'scope':'showcase transform using frozen TRAIN statistics only'}
    (out/'SHOWCASE_CONTEXT_MANIFEST.json').write_text(json.dumps(show_payload,indent=2,sort_keys=True)+'\n');(out/'SHOWCASE_CONTEXT_MANIFEST.json.sha256.json').write_text(json.dumps({'sha256':sha(out/'SHOWCASE_CONTEXT_MANIFEST.json')})+'\n')

def main():
 p=argparse.ArgumentParser();p.add_argument('--train-spain',type=Path,required=True);p.add_argument('--train-bsrnn',type=Path,required=True);p.add_argument('--show-spain',type=Path,required=True);p.add_argument('--show-bsrnn',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();build(a.train_spain,a.train_bsrnn,a.show_spain,a.show_bsrnn,a.out);print(json.dumps({'status':'complete','out':str(a.out)}))
if __name__=='__main__':main()
