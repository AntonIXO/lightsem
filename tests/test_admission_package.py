import json
from pathlib import Path

import numpy as np

from lightsem.experiments.native_dynamics.admission import group_balanced_schedule
from lightsem.experiments.native_dynamics.data import pose_observations


PACKAGE = Path('/mnt/lightsem-data/experiments/lightsem/native_admission_package_20261007_v2')


def test_group_balanced_sampler_has_no_duplicate_group_per_batch():
    entries = [{'canonical_id': f'g{i}', 'recording_id': f'r{i}', 'coach_index': 0} for i in range(5)]
    entries += [{'canonical_id': 'g0', 'recording_id': 'r0b', 'coach_index': 1}]
    a = group_balanced_schedule(entries, steps=4, batch_size=4, seed=7001)
    b = group_balanced_schedule(entries, steps=4, batch_size=4, seed=7001)
    assert a == b
    assert all(len({row['canonical_id'] for row in batch}) == len(batch) for batch in a)


def test_missing_observations_remain_masked(tmp_path):
    t = np.arange(600, dtype=float)/30
    k = np.zeros((600, 1, 17, 2), np.float32)
    k[..., 0] = 500; k[..., 1] = 300
    k[:, :, 11:13, 1] = 200
    c = np.ones((600, 1, 17), np.float32)
    box = np.tile(np.asarray([450, 0, 550, 400], np.float32), (600, 1, 1))
    k[300:305] = np.nan; c[300:305] = 0
    path = tmp_path/'pose.npz'
    np.savez(path, K=k, C=c, box=box, t=t, meta=json.dumps({'height': 1000, 'width': 1000}))
    out, _ = pose_observations(path, 20, coach_index=0)
    assert not out['mask'][300:305].any()
    assert np.all(out['pose'][300:305] == 0)


def test_package_protection_and_muq_receipts_are_metadata_only():
    strict = json.loads((PACKAGE/'strict_qualification_manifest.json').read_text())
    maximum = json.loads((PACKAGE/'max_masked_window_train_manifest.json').read_text())
    protected = json.loads((PACKAGE/'protected_val_test_manifest.json').read_text())
    leakage = json.loads((PACKAGE/'LEAKAGE_AUDIT.json').read_text())
    muq = json.loads((PACKAGE/'MUQ_ADMISSION_REPORT.json').read_text())
    assert strict['counts']['groups'] == 153
    assert maximum['counts']['groups'] == 290
    assert protected['protected_group_count'] == 104
    assert protected['test_payload_opened_this_run'] is False
    assert leakage['status'] == 'pass'
    assert not muq['strict_missing_recordings']
    assert muq['aggregate_matrix_rejected_for_reuse'] is True
    assert muq['model_receipt']['model_id'] == 'OpenMuQ/MuQ-large-msd-iter'
    assert muq['model_receipt']['revision'] == '0562a57814f6f8bbd9fdea0a25921a2fce1a841a'
    assert muq['model_receipt']['layers'] == [3, 6, 9, 12]
