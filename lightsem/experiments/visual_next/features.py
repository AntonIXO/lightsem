"""Explicit spatial summaries and fixed temporal-order controls for H3.

No teacher fitting, normalization or DEV access. Inputs and validity are supplied
by the bound extractor. A missing past frame invalidates temporal targets.
"""
from __future__ import annotations
import hashlib
import numpy as np
from lightsem.model.light_layers.targets import _rgb_to_hsv, _hs_histogram, MIN_SATURATION, MIN_VALUE

ARMS=('R0','R1','Rrand','R2','R3','R3-shuffle','R3-reverse')


def spatial_color_grid(rgb,retained):
    """Row-major 4x4 cells: HS72, meanS, chromatic occupancy, retained fraction."""
    image=np.asarray(rgb);mask=np.asarray(retained)
    if mask.dtype!=bool or image.ndim!=3 or image.shape[:2]!=mask.shape or min(mask.shape)<4:
        raise ValueError('RGB image and matching boolean mask, minimum4x4 required')
    if image.shape[-1]!=3:raise ValueError('RGB requires three channels')
    if image.dtype==np.uint8:image=image.astype(np.float64)/255
    elif image.dtype.kind=='f':image=image.astype(np.float64)
    else:raise ValueError('RGB must be uint8 or floating point')
    if not np.isfinite(image).all() or np.any((image<0)|(image>1)):
        raise ValueError('RGB outside finite unit interval')
    hue,saturation,value=_rgb_to_hsv(image)
    valid=mask&(saturation>=MIN_SATURATION)&(value>=MIN_VALUE)
    ys=np.linspace(0,image.shape[0],5,dtype=int);xs=np.linspace(0,image.shape[1],5,dtype=int)
    parts=[]
    for r in range(4):
        for c in range(4):
            region=np.s_[ys[r]:ys[r+1],xs[c]:xs[c+1]]
            m=mask[region];v=valid[region]
            hist=_hs_histogram(hue[region][v],saturation[region][v])
            # Mean saturation on retained chromatic pixels, reconstructed exactly
            # from bilinear HS saturation-bin deposits (centers s=0,.2,...,1).
            mean_s=float(hist@np.tile(np.arange(6)/5,12)) if v.any() else 0.
            parts.extend(hist)
            count=int(m.sum())
            parts.extend((mean_s,float(v.sum()/count) if count else 0.,float(count/m.size)))
    result=np.asarray(parts,dtype=np.float32)
    assert result.shape==(1200,)
    return result


def temporal_arms(r2,valid,canonical_id):
    """Current frame fixed; only previous five positions are reversed/permuted.

    Temporal vector is [current, current-minus-previous1, current-minus-previous5].
    All five previous positions must have valid support in every temporal arm.
    Per-family shuffle uses SHA256(canonical_id + '20260929') first64bits. One
    permutation is drawn per timeline position in fixed order, including invalid
    positions, so missingness cannot change later random draws.
    """
    a=np.asarray(r2,np.float32);v=np.asarray(valid)
    if a.ndim!=2 or not np.isfinite(a).all() or v.dtype!=bool or v.shape!=(len(a),):
        raise ValueError('Finite TxD features and explicit boolean validity required')
    seed=int.from_bytes(hashlib.sha256((canonical_id+'20260929').encode()).digest()[:8],'big')
    rng=np.random.default_rng(seed)
    result={k:np.zeros((len(a),a.shape[1]*3),np.float32) for k in ('R3','R3-shuffle','R3-reverse')}
    support=np.zeros(len(a),bool)
    for i in range(5,len(a)):
        past=np.arange(i-5,i)
        shuffled=rng.permutation(past)
        if not v[i-5:i+1].all():continue
        support[i]=True
        for key,order in (('R3',past),('R3-shuffle',shuffled),('R3-reverse',past[::-1])):
            result[key][i]=np.concatenate([a[i],a[i]-a[order[-1]],a[i]-a[order[0]]])
    return result,support
