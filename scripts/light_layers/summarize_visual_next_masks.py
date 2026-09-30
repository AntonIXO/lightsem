#!/usr/bin/env python3
"""H2 automatic observer/consistency summaries; never label agreement accuracy."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from lightsem.experiments.visual_next.instruments import hs_distance,mask_scores
from lightsem.model.light_layers.targets import chromatic_event_targets
from lightsem.model.light_layers.visual import sha256,write_json,ui_mask


def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--freeze',required=True,type=Path)
    args=ap.parse_args();freeze=json.loads(args.freeze.read_text())
    for p,d in freeze['files'].items():assert sha256(p)==d,p
    rows=[]
    for row in freeze['families']:
        root=Path(freeze['shared_root'])/row['canonical_id'];reference=load(row['sam2']['path'])
        modes={'sam2':reference,'box':load(row['box']['path'])}
        modes.update({name:load(root/(name+'.npz')) for name in ('cutie','hold','lag_cutie')})
        samples=load(root/'mask_controls.npz');width,height=map(int,samples['geometry_wh']);ui=ui_mask((width,height))
        mode_rows={}
        for name,data in modes.items():
            valid_reference=reference['color_valid']&reference['frame_valid']
            distances=[hs_distance(reference['hs_histogram'][i],data['hs_histogram'][i]) for i in np.flatnonzero(valid_reference)]
            known=data['mask_activity_known'];novelty=SimpleNamespace(times_s=data['novelty_times_s'],scores=data['novelty_scores'],eligible=data['novelty_eligible'])
            events=chromatic_event_targets(novelty,threshold=freeze['TRAIN_color_threshold'])
            score={'mean_HS_distance_to_incumbent':float(np.mean(distances)) if distances else None,
                'reference_color_frames':int(valid_reference.sum()),
                'prediction_missing_on_reference_valid':int(np.count_nonzero(valid_reference&~data['color_valid'])),
                'mask_changed_pixel_fraction':float(data['mask_activity'][known].mean()) if known.any() else None,
                'mask_activity_known_boundaries':int(known.sum()),'observed_frames':int(data['frame_valid'].sum()),
                'color_valid_frames':int(data['color_valid'].sum()),'empty_retained_frames':int(np.count_nonzero(data['frame_valid']&(data['retained_pixels']==0))),
                'observer_event_count_own_support':len(events.event_times_s),'observer_available_bins':int(data['novelty_eligible'].sum()),
                'independent_background_error':None,'independent_foreground_IoU':None}
            if name in ('cutie','hold','lag_cutie'):
                agreements=[];missing=0
                for j in range(len(samples['sample_indices'])):
                    if not samples['sam2_present'][j]:continue
                    ref=np.unpackbits(samples['sam2_packed'][j],axis=-1)[:,:width].astype(bool)
                    if samples[name+'_present'][j]:pred=np.unpackbits(samples[name+'_packed'][j],axis=-1)[:,:width].astype(bool)
                    else:pred=np.zeros_like(ref);missing+=1
                    agreements.append(mask_scores(ref,pred,~(ref|ui)))
                ious=[a['foreground_iou'] for a in agreements if a['foreground_iou'] is not None]
                score.update(sampled_mask_IoU_to_incumbent=float(np.mean(ious)) if ious else None,
                    sampled_reference_masks=len(agreements),sampled_missing_predictions=missing)
            mode_rows[name]=score
        flow=samples['flow_agreement'];known=samples['flow_known']
        rows.append({'canonical_id':row['canonical_id'],'modes':mode_rows,
            'flow_pixel_agreement':float(flow[:,0][known[:,0]].mean()) if known[:,0].any() else None,
            'delayed_flow_pixel_agreement':float(flow[:,1][known[:,1]].mean()) if known[:,1].any() else None,
            'flow_warning':'Pixel agreement includes background true negatives; stability alone does not establish foreground accuracy.',
            'scope':'Agreement with statelessSAM2 and observer activity, not independent truth.'})
    summary={}
    for mode in ('sam2','box','cutie','hold','lag_cutie'):
        summary[mode]={}
        for metric in ('mean_HS_distance_to_incumbent','mask_changed_pixel_fraction','observer_event_count_own_support','color_valid_frames'):
            values=[row['modes'][mode][metric] for row in rows if row['modes'][mode][metric] is not None]
            summary[mode][metric]=float(np.mean(values)) if values else None
    for p,d in freeze['files'].items():assert sha256(p)==d,p
    write_json(Path(freeze['output']),{'status':'automatic_mask_observer_comparisons_complete','freeze_sha256':sha256(args.freeze),
        'families':len(rows),'rows':rows,'macro_family_summary':summary,'original_independent_error_reduction_gate':'unmeasured',
        'original_hypothesis_confirmed':False,'test_access':False,
        'interpretation':'Incumbent self-distance is zero by construction. Agreement/low motion can be achieved by a wrong static mask; held/lag controls are mandatory context, not proof of improvement. Downstream audio fits and visual demos provide a separate practical comparison.'})


if __name__=='__main__':main()
