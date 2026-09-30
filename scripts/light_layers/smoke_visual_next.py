#!/usr/bin/env python3
"""Frozen 16-frame engineering smoke, first enrolled TRAIN source only."""
from __future__ import annotations
import argparse,json,resource,time
from pathlib import Path
import cv2
import numpy as np
import torch
from lightsem.experiments.visual_next import models
from lightsem.model.light_layers.visual import (sha256,write_json,write_arrays,iter_rgb_frames,
    load_pose,transform_boxes,SAM2FrameSegmenter,VisualConfig)
from lightsem.experiments.stage11.stage11_common import gpu_lock


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--freeze',required=True,type=Path)
    args=ap.parse_args();freeze=json.loads(args.freeze.read_text())
    for p,d in freeze['files'].items():assert sha256(p)==d, p
    out=Path(freeze['output']);out.mkdir(exist_ok=False)
    r=freeze['record'];capture=json.loads(Path(r['capture_receipt']['path']).read_text())
    with np.load(Path(r['capture_receipt']['path']).parent/'sam2.npz') as z:
        mapping={'frame_indices':z['frame_indices'][:16].copy()}
        pi=z['pose_indices'][:16].copy();valid=z['frame_valid'][:16].copy()
    frames=list(iter_rgb_frames(r['source_video']['path'],mapping,capture['geometry']['decode_wh']))
    assert len(frames)==16 and valid[0], 'First anchor must have registered prompt support'
    records=[];arrays={};torch.set_num_threads(2)
    with gpu_lock():
        torch.cuda.init()
        def begin():torch.cuda.reset_peak_memory_stats();return time.monotonic()
        def finish(name,started):
            torch.cuda.synchronize()
            row={'model':name,'seconds':time.monotonic()-started,
                 'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                 'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
                 'process_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
            assert row['peak_reserved_bytes']<=10*2**30 and row['process_peak_rss_bytes']<=8*2**30
            records.append(row);print('SMOKE',json.dumps(row),flush=True)
        started=begin();model=models.load_transnet()
        low=np.stack([cv2.resize(x,(48,27),interpolation=cv2.INTER_LINEAR) for x in frames])
        arrays['transnet_single_many']=models.transnet_predict(model,low)
        assert np.isfinite(arrays['transnet_single_many']).all()
        finish('transnetv2_official_converted',started);del model;models.release_cuda()
        started=begin();segmenter=SAM2FrameSegmenter(Path(freeze['sam_checkpoint']),VisualConfig())
        boxes,_,_=load_pose(capture['input'])
        registered=capture['input']['source_qualification']['registered_coach_indices']
        prompts=transform_boxes(boxes[pi[0],registered],capture['geometry']['original_wh'],
                                capture['geometry']['decode_wh'])
        anchor,_=segmenter(frames[0],prompts)
        assert anchor.any()
        arrays['sam_anchor']=anchor
        finish('same_sam2_union_anchor',started);del segmenter;models.release_cuda()
        started=begin();model,cfg=models.load_cutie()
        masks=np.stack(list(models.cutie_predict(model,cfg,frames,{0:anchor})))
        assert masks.shape==(16,*anchor.shape)
        arrays['cutie_masks_packed']=np.packbits(masks,axis=-1)
        finish('cutie_base_mega_480',started);del model,cfg;models.release_cuda()
        for pretrained in (True,False):
            started=begin();model=models.load_dino(pretrained=pretrained)
            key='dinov2_pretrained' if pretrained else 'dinov2_random_seed17'
            arrays[key]=np.stack(list(models.dino_grid(model,frames)))
            assert arrays[key].shape==(16,16,384) and np.isfinite(arrays[key]).all()
            finish(key,started);del model;models.release_cuda()
    write_arrays(out/'outputs.npz',arrays)
    for p,d in freeze['files'].items():assert sha256(p)==d,p
    write_json(out/'REPORT.json',{'status':'engineering_smoke_passed','frames':16,
        'canonical_id':r['canonical_id'],'role':r['role'],'torch':torch.__version__,
        'gpu':torch.cuda.get_device_name(),'precision':'float32_no_autocast',
        'freeze_sha256':sha256(args.freeze),'models':records,
        'output_sha256':sha256(out/'outputs.npz'),'annotation_scores':None,'test_access':False,
        'scope':'resource/shape feasibility only; first16 tenHz TRAIN frames; not cut accuracy or mask/color quality',
        'original_hypotheses_closed':False})


if __name__=='__main__':main()
