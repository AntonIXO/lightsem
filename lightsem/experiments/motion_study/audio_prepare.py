"""Versioned Base92 on native motion timestamps; TRAIN-only normalization."""
import json
from collections import Counter
from pathlib import Path

import numpy as np

from . import ROOT as SOURCE
from ..stage11_common import authenticated_json, read_authenticated, digest

ROOT = Path('/mnt/lightsem-data/experiments/lightsem/motion_native_audio_20260910')


def main():
    from ...audio.online import extract_bus
    from ...model.light_data import AUDIO_SCALARS
    out = ROOT/'data'
    if (out/'audio_manifest.json').exists():
        return read_authenticated(out/'audio_manifest.json')
    manifest = read_authenticated(SOURCE/'data/manifest.json')
    entries = []
    for entry in manifest['entries']:
        if not entry['audio_eligible']:
            continue
        if entry['partition'] not in ('train','val'):
            raise ValueError('No TEST audio extraction')
        if digest(entry['wav_path']) != entry['wav_sha256']:
            raise ValueError('Waveform changed')
        cache = out/'base92'/f'{entry["wav_sha256"]}.npz'
        cache.parent.mkdir(parents=True,exist_ok=True)
        if not cache.exists():
            bus = extract_bus(entry['wav_path'],fps_out=40,causal=True)
            # Match the established Base92 cached mel precision and exact order.
            mel = bus.mel.astype(np.float16).astype(np.float32)
            x = np.concatenate((mel.T,np.stack([bus.channels[k] for k in AUDIO_SCALARS],axis=1)),axis=1).astype(np.float32)
            if x.shape[1]!=92 or not np.isfinite(x).all():
                raise ValueError('Invalid Base92 features')
            np.savez_compressed(cache,x=x,times=bus.times)
        with np.load(cache) as data:
            # Fixed physical timestamps; no fitted phase, offset or time warp.
            times = np.arange(entry['n_frames'],dtype=np.float64)/30
            x = np.stack([np.interp(times,data['times'],data['x'][:,i]) for i in range(92)],axis=1).astype(np.float32)
        path = out/'aligned'/f'{entry["key"]}.npz'
        path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists():
            np.savez_compressed(path,x=x)
        else:
            with np.load(path) as old:
                if not np.array_equal(old['x'],x):
                    raise ValueError('Cached replay differs')
        entries.append({**entry,'audio_path':str(path),'audio_sha256':digest(path),
                        'base92_path':str(cache),'base92_sha256':digest(cache)})
        print(json.dumps({'key':entry['key'],'partition':entry['partition'],'frames':len(x)}),flush=True)
    train = [e for e in entries if e['partition']=='train']
    groups = Counter(e['canonical_id'] for e in train)
    mean,second = np.zeros(92),np.zeros(92)
    for entry in train:
        with np.load(entry['audio_path']) as data:
            x = data['x'].astype(np.float64)
        weight = 1/(len(groups)*groups[entry['canonical_id']])
        mean += weight*x.mean(0)
        second += weight*(x*x).mean(0)
    norms = {'mean':mean.tolist(),'std':np.sqrt(np.maximum(second-mean*mean,1e-4)).tolist(),
             'fit':'TRAIN only, equal canonical song / recording / native frame',
             'training_groups':len(groups),'training_records':len(train)}
    authenticated_json(out/'audio_normalizers.json',norms)
    result = {'entries':entries,'source_manifest_sha256':digest(SOURCE/'data/manifest.json'),
        'normalizers_sha256':digest(out/'audio_normalizers.json'),'features':['mel_'+str(i) for i in range(64)]+list(AUDIO_SCALARS),
        'extraction':'existing causal Base92 at40Hz, mel float16 cache precision, fixed interpolation to30Hz',
        'alignment':'waveform t=0 to source frame0; no fitted offset or warp; duration mismatches quarantined upstream',
        'test_access':False,'source_sha256':digest(__file__),
        'extractor_sha256':digest(Path(__file__).parents[2]/'audio/online.py')}
    authenticated_json(out/'audio_manifest.json',result)
    return result


if __name__=='__main__':
    main()
