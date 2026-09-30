"""Bounded deterministic PCA for streamed TRAIN feature matrices.

Exact covariance when dimension<=96; otherwise a fixed128-column covariance
range sketch with two extra power iterations. No outcomes select the rank/seed.
Input mean and projected global standardizer are fitted on TRAIN only by caller.
"""
from __future__ import annotations
import numpy as np


def _batches(factory,dimensions,batch_rows=256):
    for matrix in factory():
        x=np.asarray(matrix)
        if x.ndim!=2 or x.shape[1]!=dimensions or not np.isfinite(x).all():
            raise ValueError('Invalid streamed feature matrix')
        for begin in range(0,len(x),batch_rows):yield x[begin:begin+batch_rows]


def effective_rank(variances):
    v=np.asarray(variances,float)
    if v.ndim!=1 or not np.isfinite(v).all() or np.any(v<0):raise ValueError('Invalid variances')
    if v.sum()==0:return 0.
    p=v[v>0]/v.sum();return float(np.exp(-np.sum(p*np.log(p))))


def fit_pca(factory,dimensions,*,components=96,oversample=32,power_iterations=2,seed=20260929):
    if not 1<=components<=dimensions or oversample<0 or power_iterations<0:
        raise ValueError('Invalid fixed PCA dimensions')
    sums=np.zeros(dimensions,np.float64);squares=np.zeros_like(sums);count=0
    for x in _batches(factory,dimensions):
        x=x.astype(np.float64);sums+=x.sum(0);squares+=(x*x).sum(0);count+=len(x)
    if count<=components:raise ValueError('Not enough eligible TRAIN rows for PCA')
    mean=sums/count;trace=float(np.maximum(squares/count-mean*mean,0).sum())
    if trace<=0:raise ValueError('Constant input representation')
    rank=min(dimensions,components+oversample,count)
    if dimensions<=components:
        basis=np.eye(dimensions,dtype=np.float32)
    else:
        rng=np.random.default_rng(seed)
        basis=rng.standard_normal((dimensions,rank),dtype=np.float32)
        for _ in range(power_iterations+1):
            product=np.zeros((dimensions,rank),np.float64);seen=0
            for x in _batches(factory,dimensions):
                centered=(x.astype(np.float64)-mean).astype(np.float32)
                product+=centered.T@(centered@basis);seen+=len(x)
            if seen!=count:raise ValueError('Stream row count drift between PCA passes')
            basis=np.linalg.qr(product,mode='reduced')[0].astype(np.float32)
    covariance=np.zeros((rank,rank),np.float64);seen=0
    for x in _batches(factory,dimensions):
        y=((x.astype(np.float64)-mean).astype(np.float32)@basis).astype(np.float64)
        covariance+=y.T@y;seen+=len(x)
    if seen!=count:raise ValueError('Stream row count drift')
    values,vectors=np.linalg.eigh(covariance/count)
    order=np.argsort(values,kind='stable')[::-1]
    spectrum=np.maximum(values[order],0)
    matrix=(basis.astype(np.float64)@vectors[:,order[:components]]).T
    for row in matrix:
        if row[np.argmax(np.abs(row))]<0:row*=-1
    matrix=matrix.astype(np.float32)
    projected_sum=np.zeros(components);projected_sq=np.zeros(components);seen=0
    for x in _batches(factory,dimensions):
        y=((x.astype(np.float64)-mean)@matrix.T).astype(np.float64)
        projected_sum+=y.sum(0);projected_sq+=(y*y).sum(0);seen+=len(x)
    if seen!=count:raise ValueError('Stream row count drift')
    projected_mean=projected_sum/count
    variance=np.maximum(projected_sq/count-projected_mean**2,0)
    return {'input_mean':mean,'components':matrix,'projected_mean':projected_mean,
            'projected_std':np.maximum(np.sqrt(variance),1e-3),'projected_variance':variance,
            'range_spectrum':spectrum,'count':count,'input_dimension':dimensions,
            'trace_variance':trace,'retained_variance_fraction':float(variance.sum()/trace),
            'pre_normalization_effective_rank':effective_rank(variance),
            'seed':seed,'oversample':oversample,'power_iterations':power_iterations,
            'method':'exact_covariance' if dimensions<=components else 'streamed_fixed_covariance_range_sketch'}


def project(matrix,state,*,normalize=True):
    x=np.asarray(matrix)
    if x.ndim!=2 or x.shape[1]!=len(state['input_mean']) or not np.isfinite(x).all():
        raise ValueError('Projection input mismatch or nonfinite data')
    result=(x.astype(np.float64)-state['input_mean'])@state['components'].T
    if normalize:result=(result-state['projected_mean'])/state['projected_std']
    if not np.isfinite(result).all():raise ValueError('Nonfinite projected features')
    return result.astype(np.float32)
