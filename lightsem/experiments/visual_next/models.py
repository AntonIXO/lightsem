"""Pinned offline model adapters. No automatic downloads or weight fallback."""
from __future__ import annotations
import gc
from pathlib import Path
import sys
import numpy as np

ROOT=Path('/root/lightsem/runs/visual_next_20260929')


def activate_vendor(name):
    for path in (ROOT/'runtime_deps',ROOT/'vendor'/name):
        if str(path) not in sys.path:sys.path.insert(0,str(path))


def load_transnet():
    import torch
    activate_vendor('transnetv2/inference-pytorch')
    from transnetv2_pytorch import TransNetV2
    model=TransNetV2()
    model.load_state_dict(torch.load(ROOT/'weights/transnetv2-pytorch-official.pth',
                                    map_location='cpu',weights_only=True),strict=True)
    return model.cuda().eval()


def transnet_predict(model, rgb_frames):
    """Released 100-frame/50-output convention, replicated endpoint padding.

    Input already on the caller's fixed source timeline, RGB uint8 27x48. The
    source-native timeline is required for real cut scoring; 10Hz smoke is only
    resource/shape evidence. Both single-frame and gradual-transition heads kept.
    """
    import torch
    a=np.asarray(rgb_frames)
    if a.dtype!=np.uint8 or a.ndim!=4 or a.shape[1:]!=(27,48,3) or not len(a):
        raise ValueError('Expected nonempty uint8 RGB frames at 48x27')
    end=25+50-(len(a)%50 if len(a)%50 else 50)
    padded=np.concatenate([np.repeat(a[:1],25,0),a,np.repeat(a[-1:],end,0)])
    out=[]
    with torch.inference_mode():
        for i in range(0,len(padded)-99,50):
            single,many=model(torch.from_numpy(padded[i:i+100][None]).cuda())
            out.append(np.stack([single.sigmoid()[0,25:75,0].cpu().numpy(),
                                many['many_hot'].sigmoid()[0,25:75,0].cpu().numpy()],axis=1))
    return np.concatenate(out)[:len(a)]


def load_cutie():
    import torch
    activate_vendor('cutie')
    from hydra import initialize_config_dir,compose
    from omegaconf import open_dict
    from cutie.model.cutie import CUTIE
    from cutie.model.utils import resnet
    with initialize_config_dir(version_base='1.3.2',config_dir=str(ROOT/'vendor/cutie/cutie/config')):
        cfg=compose(config_name='eval_config')
    with open_dict(cfg):
        cfg.mem_every=5;cfg.use_long_term=False;cfg.max_internal_size=480
    # Full checkpoint below supplies EVERY model parameter and buffer strictly.
    # Skip otherwise redundant ImageNet downloads during constructor initialization.
    old18,old50=resnet.resnet18,resnet.resnet50
    resnet.resnet18=lambda pretrained=True,**kw:old18(pretrained=False,**kw)
    resnet.resnet50=lambda pretrained=True,**kw:old50(pretrained=False,**kw)
    try:model=CUTIE(cfg)
    finally:resnet.resnet18,resnet.resnet50=old18,old50
    model.load_state_dict(torch.load(ROOT/'weights/cutie-base-mega.pth',map_location='cpu',
                                    weights_only=True),strict=True)
    return model.cuda().eval(),cfg


def cutie_predict(model,cfg,frames,anchors):
    """Sequential union-person propagation; caller freezes refresh/cut anchors."""
    import torch
    import torch.nn.functional as F
    from cutie.inference.inference_core import InferenceCore
    processor=InferenceCore(model,cfg=cfg)
    for i,rgb in enumerate(frames):
        h,w=rgb.shape[:2];shape=(round(h*480/min(h,w)),round(w*480/min(h,w)))
        image=torch.from_numpy(rgb.copy()).permute(2,0,1).float().cuda()/255
        image=F.interpolate(image[None],size=shape,mode='bilinear',align_corners=False)[0]
        mask=None
        if i in anchors:
            mask=torch.from_numpy(anchors[i].copy()).float().cuda()
            mask=F.interpolate(mask[None,None],size=shape,mode='nearest')[0,0].long()
        with torch.inference_mode():
            probability=processor.step(image,mask,objects=[1] if mask is not None else None)
            result=processor.output_prob_to_mask(probability)
            result=F.interpolate(result.float()[None,None],size=(h,w),mode='nearest')[0,0]
        yield result.bool().cpu().numpy()


def load_dino(*,pretrained=True):
    import torch
    activate_vendor('dinov2')
    from dinov2.hub.backbones import dinov2_vits14
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        model=dinov2_vits14(pretrained=False)
    if pretrained:
        model.load_state_dict(torch.load(ROOT/'weights/dinov2_vits14_pretrain.pth',
                                        map_location='cpu',weights_only=True),strict=True)
    return model.cuda().eval()


def dino_grid(model,frames):
    import torch
    import torch.nn.functional as F
    mean=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None]
    std=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
    with torch.inference_mode():
        for rgb in frames:
            image=torch.from_numpy(rgb.copy()).permute(2,0,1)[None].float().cuda()/255
            image=F.interpolate(image,size=(224,224),mode='bilinear',align_corners=False)
            patches=model.forward_features((image-mean)/std)['x_norm_patchtokens']
            # 16x16 patches -> row-major 4x4 cells, each with 4x4 patches.
            grid=patches.reshape(1,4,4,4,4,384).mean(dim=(2,4)).reshape(16,384)
            yield grid.cpu().numpy()


def release_cuda():
    import torch
    gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
