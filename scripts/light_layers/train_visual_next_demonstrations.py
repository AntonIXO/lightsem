#!/usr/bin/env python3
"""Prospectively bound cut/mask/combined single-seed demonstration fits.

Uses the identical training loop/architecture/initialization and6000updates.
The frozen target recipe is different, so these are additional engineering fits,
not replacements for any of the original seven-by-three study cells.
"""
from __future__ import annotations
import copy,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from lightsem.model.light_layers.training import SourceStore,TrainingData
from lightsem.model.light_layers.targets import chromatic_novelty
from lightsem.experiments.visual_next.data import common_history_support
from lightsem.experiments.visual_next.projection import project
from lightsem.experiments.visual_next.instruments import edit_veto
from scripts.light_layers import train_visual_next as runner

ARMS=('cut_only','mask_only','combined')


def prepared_data(freeze,arm):
    if arm not in ARMS:raise ValueError('Unknown demonstration intervention')
    manifest=json.loads(Path(freeze['training_manifest']['path']).read_text());store=SourceStore(manifest)
    records=[store.visual(r['recording_id']) for r in manifest['records'] if r['partition']=='train']
    if len(records)!=91:raise RuntimeError('Original91TRAIN records required')
    transforms=copy.deepcopy(freeze['transforms']);transforms['aux']['rich']={'mean':[0.]*96,'std':[1.]*96,'counts':[0]*96}
    with np.load(Path(freeze['projected_root'])/'R0_PCA.npz') as z:r0_pca={k:z[k].copy() for k in z.files}
    for record in records:
        source_valid=record.frame_valid.copy();support=common_history_support(source_valid)
        if arm in ('mask_only','combined'):
            with np.load(Path(freeze['shared_root'])/record.canonical_id/'cutie.npz',allow_pickle=False) as z:
                hs=z['hs_histogram'].astype(np.float32);rich=z['rich96'].astype(np.float32)
                color_valid=z['color_valid']&source_valid
                available=z['frame_valid']&z['value_valid']&source_valid
                retained=z['retained_fraction'].astype(np.float32);chromatic=z['chromatic_fraction'].astype(np.float32)
                novelty_times=z['novelty_times_s'].copy()
                novelty_scores=z['novelty_scores'].astype(np.float64)
                cached_eligible=z['novelty_eligible'].copy()
            hs[~color_valid]=0;mass=hs.sum(1);hs[color_valid]/=mass[color_valid,None]
            rich[:,:72]=hs;rich[~source_valid]=0
            retained[~source_valid]=0;chromatic[~source_valid]=0
            record.hs=hs;record.color_valid=color_valid;record.retained=retained;record.chromatic=chromatic
            # Keep the same float16 score-storage boundary as the original
            # observer. Recompute eligibility on the original observed clock,
            # excluding filled pose-gap frames without perturbing valid scores.
            if not np.array_equal(novelty_times,record.source_times):raise RuntimeError('Cutie novelty clock drift')
            common=chromatic_novelty(record.source_times,hs,color_valid,
                retained_fraction=retained,chromatic_fraction=chromatic,cuts=record.cuts)
            record.stored_novelty=SimpleNamespace(times_s=novelty_times,scores=novelty_scores,
                                                  eligible=common.eligible&cached_eligible)
            if arm=='mask_only':
                aux_valid=support&available
                projected=project(rich[aux_valid],r0_pca)
                quantized=projected.astype(np.float16).astype(np.float32)
                if not np.isfinite(quantized).all():raise RuntimeError('Nonfinite mask-only auxiliary target')
                record.visual_next_aux_roundoff=float(np.max(np.abs(quantized-projected))) if len(projected) else 0.
                auxiliary=np.zeros((640,96),np.float32);auxiliary[aux_valid]=quantized
        if arm!='mask_only':
            key='R3' if arm=='combined' else 'R0'
            with np.load(Path(freeze['projected_root'])/(record.canonical_id+'.npz'),allow_pickle=False) as z:
                auxiliary=z[key].astype(np.float32);aux_valid=z['valid'].copy()
        record.rich=auxiliary;record.rich_valid=np.broadcast_to(aux_valid[:,None],auxiliary.shape).copy();record.validate()
    data=TrainingData(store,records,transforms,'I_VR_P0')
    if arm in ('cut_only','combined'):
        for record in records:
            ref=freeze['cut_traces'][record.canonical_id]
            with np.load(ref['path'],allow_pickle=False) as z:edits=z['edit_intervals_s'].copy()
            keep=edit_veto(record.source_times,edits,margin=.2)
            # Veto the positive color targets, never erase their valid time bins.
            data.targets[record.recording_id]['color'][~keep]=0
    return store,records,transforms,data


if __name__=='__main__':
    # Explicit adapter boundary, included in the experiment source freeze.
    runner.ARMS=ARMS
    runner.prepared_data=prepared_data
    runner.main()
