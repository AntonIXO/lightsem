"""Non-learning structural and runtime preflight for the phase codec."""
import json
import time
from pathlib import Path

import numpy as np
import torch

from . import ROOT as PARENT
from .phase_registration import ROOT
from .phase_model import PhaseCodec,PhaseConfig
from ..stage11_common import authenticated_json,read_authenticated,digest,gpu_lock


def main():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifest = read_authenticated(PARENT/'data/manifest.json')
    norms = read_authenticated(PARENT/'data/normalizers.json')
    entry = next(e for e in manifest['entries'] if e['partition']=='train' and e['n_frames']>=481)
    if digest(entry['path'])!=entry['sha256']:
        raise ValueError('TRAIN source changed')
    with np.load(entry['path']) as value:
        features = value['features'][:481]
    rng = np.random.default_rng(1901)
    rows = []
    for arm in ('frame24','phase64','phase128','phase256'):
        cfg = PhaseConfig(arm)
        fit = {'dc_mean':np.zeros(132),'basis':[np.linalg.qr(rng.standard_normal((132,132)))[0] for _ in cfg.ranks]}
        model = PhaseCodec(cfg,norms,fit).cuda().eval()
        inp = torch.from_numpy(features[None]).cuda()
        batch = inp[:,:480].expand(32,-1,-1).contiguous()
        with torch.inference_mode():
            full = model(inp)
            chunk = model.forward_chunked(inp,32)
            delta = float((full-chunk).abs().max())
            root_delta = float((full[...,:3]-inp[...,:3]).abs().max())
            contact_delta = float((full[...,135:]-inp[...,135:]).abs().max())
            still = model(inp[:,:1].expand(-1,480,-1))
            static_delta = float((still-still[:,:1]).abs().max())
            assert torch.isfinite(full).all() and delta<5e-5 and root_delta==0 and contact_delta==0 and static_delta<2e-6
            for _ in range(10):
                model(batch)
            torch.cuda.synchronize()
            started = time.monotonic()
            for _ in range(100):
                model(batch)
            torch.cuda.synchronize()
        rows.append({'arm':arm,'seconds_per_batch32':(time.monotonic()-started)/100,
            'chunk_max_abs':delta,'root_max_abs':root_delta,'contacts_max_abs':contact_delta,
            'static_max_abs':static_delta,'latent':cfg.latent,'stride':cfg.stride})
    result = {'purpose':'structural/runtime preflight with random projections; no fitting, validation or quality selection',
        'real_train_key':entry['key'],'real_train_sha256':entry['sha256'],'rows':rows,
        'model_source_sha256':digest(Path(__file__).with_name('phase_model.py')),
        'source_sha256':digest(__file__),'test_source_sha256':digest('/root/lightsem/tests/test_motion_study.py'),
        'tests':'seven motion contract tests passed before preflight','gpu':torch.cuda.get_device_name()}
    authenticated_json(ROOT/'codec/engineering.json',result)
    print(json.dumps(result))


if __name__ == '__main__':
    with gpu_lock(ROOT):
        main()
