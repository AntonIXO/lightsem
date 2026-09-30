"""Frozen-cohort feature adapters; no source replacement or DEV fitting."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from .features import ARMS,temporal_arms

DIMENSIONS={'R0':96,'R1':1200,'Rrand':7344,'R2':7344,'R3':22032,'R3-shuffle':22032,'R3-reverse':22032}


def common_history_support(valid):
    v=np.asarray(valid)
    if v.ndim!=1 or v.dtype!=bool:raise ValueError('Explicit frame validity required')
    result=np.zeros(len(v),bool)
    if len(v)>=6:result[5:]=np.convolve(v.astype(int),np.ones(6,int),mode='valid')==6
    return result


def arm_features(row,shared_root,arm):
    if arm not in ARMS:raise ValueError('Unknown preregistered arm')
    path=Path(shared_root)/row['canonical_id']/'teacher_raw.npz'
    with np.load(path,allow_pickle=False) as z:
        valid=z['frame_valid'].copy();support=common_history_support(valid)
        if arm=='R0':
            with np.load(Path(row['capture_receipt']['path']).parent/'sam2.npz',allow_pickle=False) as old:
                x=old['rich96'].astype(np.float32)
                mass=x[:,:72].sum(1);color_valid=old['color_valid']
                if np.any(np.abs(mass[color_valid]-1)>.002):raise ValueError('Original histogram mass invalid')
                x[color_valid,:72]/=mass[color_valid,None]
        else:
            x=z['R1'].astype(np.float32)
            if arm!='R1':
                name='dino_random_seed17' if arm=='Rrand' else 'dino_pretrained'
                x=np.concatenate([x,z[name].astype(np.float32).reshape(len(x),-1)],axis=1)
                if arm.startswith('R3'):
                    arms,temporal_valid=temporal_arms(x,valid,row['canonical_id'])
                    if not np.array_equal(support,temporal_valid):raise ValueError('Temporal support drift')
                    x=arms[arm]
        if x.shape!=(len(valid),DIMENSIONS[arm]) or not np.isfinite(x).all():raise ValueError('Feature schema mismatch')
        return x,support


def train_factory(rows,shared_root,arm):
    if not rows or any(r['partition']!='train' for r in rows):raise ValueError('PCA fitting requires TRAIN-only rows')
    def iterate():
        for row in rows:
            x,valid=arm_features(row,shared_root,arm)
            if not valid.any():raise ValueError('Enrolled family has no common feature support')
            yield x[valid]
    return iterate
