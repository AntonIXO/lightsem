#!/usr/bin/env python3
"""Reject raw native bodies whose motion is unsafe relative to a baseline.

This runs before mapper/interpolation/DMX/video.  It intentionally preserves
the failed body and records the ratios instead of repairing it cosmetically.
"""
from __future__ import annotations

import argparse, hashlib, json
from pathlib import Path
import numpy as np

from lightsem.eval.training_health import motion_metrics


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(candidate_root: Path, reference_root: Path, out: Path, *, speed_ratio: float,
          acceleration_ratio: float, jerk_ratio: float) -> dict:
    rows=[]
    for candidate_path in sorted(candidate_root.glob('*.npz')):
        song=candidate_path.stem; reference_path=reference_root/f'{song}.npz'
        with np.load(candidate_path,allow_pickle=False) as z: candidate=z['body'].astype(np.float32)
        with np.load(reference_path,allow_pickle=False) as z: reference=z['body'].astype(np.float32)
        cm=motion_metrics(candidate,fps=30); rm=motion_metrics(reference,fps=30)
        row={'song':song,'candidate':str(candidate_path),'candidate_sha256':sha(candidate_path),'reference':str(reference_path),'reference_sha256':sha(reference_path),'candidate_motion':cm,'reference_motion':rm,'ratios':{'speed_p95':cm['speed_p95']/max(rm['speed_p95'],1e-9),'acceleration_p95':cm['acceleration_raw']['p95']/max(rm['acceleration_raw']['p95'],1e-9),'jerk_p95':cm['jerk_raw']['p95']/max(rm['jerk_raw']['p95'],1e-9)}}
        row['pass']=bool(cm['finite'] and row['ratios']['speed_p95']<=speed_ratio and row['ratios']['acceleration_p95']<=acceleration_ratio and row['ratios']['jerk_p95']<=jerk_ratio)
        rows.append(row)
    result={'schema_version':1,'status':'complete' if all(r['pass'] for r in rows) else 'rejected_raw_safety','candidate_root':str(candidate_root),'reference_root':str(reference_root),'thresholds':{'speed_p95_ratio':speed_ratio,'acceleration_p95_ratio':acceleration_ratio,'jerk_p95_ratio':jerk_ratio},'rows':rows,'scope':'raw 30 Hz body only; before mapper/interpolation/DMX/video; no smoothing or retiming'}
    if out.exists(): raise FileExistsError(out)
    out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n'); out.with_name(out.name+'.sha256.json').write_text(json.dumps({'sha256':sha(out)},sort_keys=True)+'\n')
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--candidate-root',type=Path,required=True); p.add_argument('--reference-root',type=Path,required=True); p.add_argument('--out',type=Path,required=True); p.add_argument('--speed-ratio',type=float,default=1.5); p.add_argument('--acceleration-ratio',type=float,default=2.0); p.add_argument('--jerk-ratio',type=float,default=2.0); a=p.parse_args(argv); r=audit(a.candidate_root,a.reference_root,a.out,speed_ratio=a.speed_ratio,acceleration_ratio=a.acceleration_ratio,jerk_ratio=a.jerk_ratio); print(json.dumps({'status':r['status'],'rows':len(r['rows'])},sort_keys=True)); return 0
if __name__=='__main__': raise SystemExit(main())
