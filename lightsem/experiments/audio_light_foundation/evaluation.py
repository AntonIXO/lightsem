import numpy as np
import torch


def effective_rank(x):
    a=np.asarray(x,dtype=np.float64).reshape(-1,x.shape[-1]); a=a-a.mean(0,keepdims=True)
    s=np.linalg.svd(a,compute_uv=False); p=(s*s)/max((s*s).sum(),1e-12)
    return float(np.exp(-(p[p>0]*np.log(p[p>0])).sum()))


def duplicate_count(x):
    a=np.asarray(x); return int(a.shape[-2]-len({np.ascontiguousarray(a[...,i,:]).tobytes() for i in range(a.shape[-2])}))


@torch.no_grad()
def metrics(model,batch,fixtures,chunk=32):
    a=batch['audio']; valid=batch['valid']; rate=float(model.config.local_rate_hz)
    ts=torch.arange(a.shape[1],device=a.device,dtype=torch.float32)[None].expand(a.shape[0],-1)/rate
    full=model(a,fixtures,timestamps=ts,valid=valid,duration_seconds=a.shape[1]/rate)
    state=None; pieces=[]
    for s in range(0,a.shape[1],chunk):
        e=min(a.shape[1],s+chunk); o=model(a[:,s:e],fixtures,timestamps=ts[:,s:e],valid=valid[:,s:e],state=state,duration_seconds=a.shape[1]/rate); state=o['state']; pieces.append(o)
    chunk_rgb=torch.cat([x['rgb'] for x in pieces],1); chunk_b=torch.cat([x['brightness'] for x in pieces],1)
    perm=torch.tensor([3,0,7,2,5,1,6,4],device=a.device); p=model(a,fixtures[:,perm],timestamps=ts,valid=valid,duration_seconds=a.shape[1]/rate)
    inv=torch.argsort(perm)
    # Counterfactual: reverse each sample in time, preserving its feature marginal
    # while destroying local audio/output alignment. This is not a batch reorder.
    shuffled=a.flip(1)
    sh=model(shuffled,fixtures,timestamps=ts,valid=valid,duration_seconds=a.shape[1]/rate)
    return {'chunk_rgb_max_abs':float((full['rgb']-chunk_rgb).abs().max()),'chunk_brightness_max_abs':float((full['brightness']-chunk_b).abs().max()),
            'permutation_brightness_max_abs':float((full['brightness']-p['brightness'][:,:,inv]).abs().max()),
            'audio_temporal_reverse_brightness_delta':float((full['brightness']-sh['brightness']).abs().mean()),
            'audio_temporal_reverse_aim_delta':float((full['aim']-sh['aim']).abs().mean()),
            'audio_temporal_reverse_rgb_delta':float((full['rgb']-sh['rgb']).abs().mean()),
            'rgb_effective_rank':effective_rank(full['rgb'].cpu().numpy()),'rgb_duplicate_stream_count':duplicate_count(full['rgb'].cpu().numpy()),
            'brightness_inactive_fraction':float((full['brightness']<.01).float().mean()),'event_probability_mean':float(full['event'].sigmoid().mean()),
            'aim_finite':bool(torch.isfinite(full['aim']).all()),'speed_max':float(full['speed'].max()),'acceleration_max':float(full['acceleration'].abs().max())}
