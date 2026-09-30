#!/usr/bin/env python3
"""CPU-only loader invariant check with explicitly artificial auxiliary targets."""
from __future__ import annotations
import json,tempfile,time
from pathlib import Path
import numpy as np
from lightsem.model.light_layers.visual import sha256,write_json
from lightsem.model.light_layers.training import SourceStore
from lightsem.experiments.visual_next.data import common_history_support
from scripts.light_layers.train_visual_next import prepared_data,main_target_digest


def main():
    p=Path('/root/lightsem/experiments/2026-09-29-visual-next/records/execution')
    cohort=json.loads((p/'SHARED_COHORT_V1.json').read_text())
    init=Path('/root/lightsem/runs/visual_next_20260929/initializations_v1/REPORT.json')
    invariants=json.loads(init.read_text());manifest=json.loads(Path(cohort['training_manifest']['path']).read_text())
    store=SourceStore(manifest);started=time.monotonic()
    with tempfile.TemporaryDirectory(prefix='loader_invariant_',dir='/root/lightsem/runs/visual_next_20260929') as directory:
        for row in manifest['records']:
            if row['partition']!='train':continue
            record=store.visual(row['recording_id']);valid=common_history_support(record.frame_valid)
            zero=np.zeros((640,96),np.float16);one=np.zeros_like(zero);one[valid]=1
            np.savez_compressed(Path(directory)/(row['canonical_id']+'.npz'),R0=zero,R3=one,valid=valid)
        fake={'training_manifest':cohort['training_manifest'],'projected_root':directory,'transforms':invariants['transforms']}
        _,records0,_,data0=prepared_data(fake,'R0');_,records3,_,data3=prepared_data(fake,'R3')
        d0,d3=main_target_digest(data0),main_target_digest(data3)
        assert d0==d3==invariants['main_targets_sha256']
        for rid in data0.targets:
            assert np.array_equal(data0.targets[rid]['rich_mask'],data3.targets[rid]['rich_mask'])
            assert np.all(data0.targets[rid]['rich']==0)
            assert np.all(data3.targets[rid]['rich'][data3.targets[rid]['rich_mask']]==1)
        samples=[(r.recording_id,0) for r in records0[:4]]
        a,b=data0.batch(samples,'cpu'),data3.batch(samples,'cpu')
        for key in a:
            if key!='visual_aux':assert np.array_equal(a[key].numpy(),b[key].numpy()),key
        assert a['audio40'].shape==(4,2560,94) and a['visual_aux'].shape==(4,640,96)
        assert not np.array_equal(a['visual_aux'].numpy(),b['visual_aux'].numpy())
    write_json(p/'TRAINING_DATA_INVARIANT_AUDIT_V1.json',{
        'status':'passed','source_main_targets_sha256':d0,'TRAIN_families':91,'batch_shape':[4,2560,94],
        'main_labels_and_audio_unchanged_when_auxiliary_changes':True,'auxiliary_masks_paired':True,
        'artificial_auxiliary_values':[0,1],'artificial_files_removed':True,'GPU_used':False,'model_fitting':False,
        'scope':'Engineering test of actual TRAIN loader paths; these artificial auxiliary targets are never study data.',
        'script_sha256':sha256(__file__),'initialization_report_sha256':sha256(init),'seconds':time.monotonic()-started})
    print('TRAIN_LOADER_INVARIANTS_PASSED',d0,flush=True)


if __name__=='__main__':main()
