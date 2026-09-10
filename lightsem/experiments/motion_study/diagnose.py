"""Post-result diagnostics, explicitly excluded from codec selection and training."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import ROOT
from .geometry import FPS, BODY_HEIGHT, joints_from_features
from .metrics import compare
from ..stage11_common import authenticated_json, read_authenticated, digest


def diagnose(root=ROOT):
    torch.set_num_threads(4)
    root = Path(root)
    manifest = read_authenticated(root/'data/manifest.json')
    norms = read_authenticated(root/'data/normalizers.json')
    entries = {e['key']: e for e in manifest['entries']}
    reports = {}
    for arm in ('frame24','temporal64','temporal128'):
        path = root/'codec/evaluation'/arm/'seed_0/final/report.json'
        if not path.exists():
            continue
        report = read_authenticated(path)
        rows = []
        for row in report['sequences']:
            entry = entries[row['key']]
            assert entry['partition'] == 'val'
            with np.load(entry['path']) as value:
                reference = value['features']
            assert digest(row['path']) == row['sha256']
            with np.load(row['path']) as value:
                prediction = value['features']
            with torch.inference_mode():
                truth, pred = [joints_from_features(torch.from_numpy(x)).numpy() for x in (reference,prediction)]
            roots = pred[:,0]-truth[:,0]
            tr, pr = truth-truth[:,:1], pred-pred[:,:1]
            relative_velocity = np.diff(pr-tr,axis=0)*FPS
            # Oracle diagnostic only: deliberately supplies true native root channels.
            # This is never a candidate model or an eligible reconstruction.
            oracle = prediction.copy()
            oracle[:,:3] = reference[:,:3]
            oracle_metrics = compare(reference,oracle,norms['velocity_rms'])
            quiet_starts = []
            quiet = []
            for start in range(0,len(reference)-60+1,60):
                local = reference[start:start+60]
                m = compare(local,prediction[start:start+60],norms['velocity_rms'])
                if m['kinetic_reference'] < .01:
                    quiet_starts.append(start)
                    quiet.append(float(np.sqrt(m['kinetic_prediction'])-np.sqrt(m['kinetic_reference'])))
            rows.append({'key':row['key'],'canonical_id':row['canonical_id'],
                         'mean_root_error_m':float(np.linalg.norm(roots,axis=-1).mean()),
                         'final_root_error_m':float(np.linalg.norm(roots[-1])),
                         'root_relative_joint_error_height':float(np.linalg.norm(pr-tr,axis=-1).mean()/BODY_HEIGHT),
                         'root_relative_velocity_nrmse':float(np.sqrt(np.square(relative_velocity).sum(-1).mean())/norms['velocity_rms']),
                         'oracle_root_joint_error_height':oracle_metrics['joint_error_height'],
                         'oracle_root_velocity_nrmse':oracle_metrics['velocity_nrmse'],
                         'quiet_2s_starts':quiet_starts,'quiet_2s_speed_excess_m_s':quiet})
        reports[arm] = {'evaluation_sha256':digest(path),'rows':rows,
                       'frozen_quiet_windows':sum(r['metrics']['kinetic_reference']<.01 for r in report['rows']),
                       'frozen_stationary_windows':sum(r['metrics']['travel_reference_m']<.1 for r in report['rows'])}
    result = {'purpose':'exploratory failure localization, not an altered gate or corrected candidate',
              'oracle_root_warning':'TRUE native root supplied for diagnosis only; cannot be used for music-only playback',
              'quiet_warning':'Additional two-second windows are post-result diagnostics, not preregistered selection evidence',
              'source_sha256':digest(__file__),'arms':reports,'test_access':False}
    destination = root/'codec'/('diagnosis_'+'_'.join(reports)+'.json')
    authenticated_json(destination,result)
    print(str(destination))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=ROOT)
    diagnose(parser.parse_args().root)
