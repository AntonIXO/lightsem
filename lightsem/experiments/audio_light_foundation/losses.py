import torch
import torch.nn.functional as F


def masked_mean(x,mask):
    mask=mask.expand_as(x)
    return torch.where(mask,x,torch.zeros_like(x)).sum()/mask.sum().clamp_min(1)


def delta_loss(pred, target, valid):
    """Match adjacent-frame changes without inventing a temporal controller."""
    if pred.shape[1] < 2:
        return pred.new_zeros(())
    pair_valid = valid[:, 1:] & valid[:, :-1]
    return masked_mean((pred[:, 1:] - pred[:, :-1] - (target[:, 1:] - target[:, :-1])).square(),
                       pair_valid[..., None, None])


def teacher_objective(teacher, target, valid):
    """Supervise the privileged body→light teacher on the real light targets."""
    v=valid
    return {
        'teacher_rgb': masked_mean((teacher['rgb']-target['rgb']).abs(),v[...,None,None]),
        'teacher_brightness': masked_mean((teacher['brightness']-target['brightness']).abs(),v[...,None,None]),
        'teacher_event': masked_mean(F.binary_cross_entropy_with_logits(teacher['event'],target['event'],reduction='none'),v[...,None]),
        'teacher_aim': masked_mean((teacher['aim']-target['motion']).square(),v[...,None,None]),
        'teacher_speed': masked_mean((teacher['speed']-target['speed']).abs(),v[...,None,None]),
        'teacher_acceleration': masked_mean((teacher['acceleration']-target['acceleration']).abs(),v[...,None,None]),
    }


def distillation_objective(student, teacher, valid):
    """Transfer the privileged teacher's light controls into audio inference."""
    v=valid
    return {
        'distill_aim': masked_mean((student['aim']-teacher['aim'].detach()).square(),v[...,None,None]),
        'distill_speed': masked_mean((student['speed']-teacher['speed'].detach()).square(),v[...,None,None]),
        'distill_acceleration': masked_mean((student['acceleration']-teacher['acceleration'].detach()).square(),v[...,None,None]),
        'distill_rgb': masked_mean((student['rgb']-teacher['rgb'].detach()).square(),v[...,None,None]),
        'distill_brightness': masked_mean((student['brightness']-teacher['brightness'].detach()).square(),v[...,None,None]),
    }


def cross_song_motion_objective(pred, target, groups, valid):
    """Match song-specific, fixture-centered motion differences.

    Windows are paired by their ordinal position within each song.  Removing
    the per-frame fixture centroid keeps this from rewarding a global aim
    offset and forces the audio stream to retain song-dependent articulation.
    """
    by_song={}
    for i,g in enumerate(groups): by_song.setdefault(g,[]).append(i)
    songs=sorted(by_song); terms=[]
    for ai,a in enumerate(songs):
        for b in songs[ai+1:]:
            m=min(len(by_song[a]),len(by_song[b]))
            if not m: continue
            ia=torch.as_tensor(by_song[a][:m],device=pred['aim'].device)
            ib=torch.as_tensor(by_song[b][:m],device=pred['aim'].device)
            pa=pred['aim'][ia]; pb=pred['aim'][ib]
            ta=target['motion'][ia]; tb=target['motion'][ib]
            pa=pa-pa.mean(2,keepdim=True); pb=pb-pb.mean(2,keepdim=True)
            ta=ta-ta.mean(2,keepdim=True); tb=tb-tb.mean(2,keepdim=True)
            pair_valid=valid[ia] & valid[ib]
            terms.append(masked_mean(((pa-pb)-(ta-tb)).square(),pair_valid[...,None,None]))
    return pred['aim'].new_zeros(()) if not terms else torch.stack(terms).mean()


def losses(pred,target,valid,auxiliary,spread_loss=False):
    v=valid
    out={
      'rgb':masked_mean((pred['rgb']-target['rgb']).abs(),v[...,None,None]),
      'rgb_bce':masked_mean(F.binary_cross_entropy_with_logits(pred['rgb_logits'],target['rgb'],reduction='none'),v[...,None,None]),
      'brightness':masked_mean((pred['brightness']-target['brightness']).abs(),v[...,None,None]),
      'event':masked_mean(F.binary_cross_entropy_with_logits(pred['event'],target['event'],reduction='none'),v[...,None]),
      'rgb_delta':delta_loss(pred['rgb'],target['rgb'],v),
      'brightness_delta':delta_loss(pred['brightness'],target['brightness'],v),
    }
    if auxiliary:
        out.update({'aim_proxy':masked_mean((pred['aim']-target['motion']).square(),v[...,None,None]),
                    'aim_delta':delta_loss(pred['aim'],target['motion'],v),
                    'speed_proxy':masked_mean((pred['speed']-target['speed']).abs(),v[...,None,None]),
                    'acceleration_proxy':masked_mean((pred['acceleration']-target['acceleration']).abs(),v[...,None,None]),
                    'dance_body':masked_mean((pred['dance_body']-target['body']).square(),v[...,None])})
        # ``dance_teacher`` is a detached body embedding produced by the
        # privileged branch.  Comparing it to the audio hidden state is the
        # actual cross-modal distillation path; reconstructing the body alone
        # would train an autoencoder that never touches audio inference.
        if 'dance_teacher' in pred:
            out['dance_distill']=masked_mean((pred['audio_latent']-pred['dance_teacher']).square(),v[...,None])
        if spread_loss:
            pred_d=torch.cdist(pred['aim'],pred['aim']); target_d=torch.cdist(target['motion'],target['motion'])
            pair_mask=torch.triu(torch.ones((pred_d.shape[-1],pred_d.shape[-1]),device=pred_d.device,dtype=torch.bool),diagonal=1)
            pair_error=(pred_d-target_d).square()[...,pair_mask]
            out['aim_spread']=pair_error.mean()
        # Motion is the missing product capability in this pilot.  Give its
        # target and adjacent-frame change enough weight to overcome the
        # low-variance RGB/brightness median solution; the raw target-match
        # metrics remain visible so an overfit motion head cannot be hidden.
        out['total']=out['rgb']+out['rgb_bce']+2.5*out['brightness']+1.5*out['event']+.5*(out['rgb_delta']+out['brightness_delta'])+5.0*(out['aim_proxy']+out['aim_delta'])+.25*(out['speed_proxy']+out['acceleration_proxy'])+.1*out['dance_body']+.25*out.get('dance_distill',0.)+(0.5*out.get('aim_spread',0.))
    else:
        out['total']=out['rgb']+out['rgb_bce']+2.5*out['brightness']+1.5*out['event']+.5*(out['rgb_delta']+out['brightness_delta'])
    return out
