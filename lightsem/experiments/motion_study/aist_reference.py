"""Separate official TRAIN-only source reference, never a FineDance training input."""
import io
import json
import pickle
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from . import ROOT
from ..stage11_common import authenticated_json, digest

EDGES = [(0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),
         (5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]


class ArrayReader(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in {('numpy.core.multiarray', '_reconstruct'),
                               ('numpy', 'ndarray'), ('numpy', 'dtype')}:
            return super().find_class(module, name)
        raise ValueError(f'Unexpected pickle global: {module}.{name}')


def main():
    source = Path('/mnt/lightsem-data/aist_reference_20260910')
    output = ROOT/'delivery/aist_reference'
    output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source/'splits.zip') as archive:
        train = set(archive.read('splits/crossmodal_train.txt').decode().splitlines())
    train -= set((source/'ignore_list.txt').read_text().splitlines())
    # One lexicographically first eligible recording in each of three distinct styles.
    selected = [next(k for k in sorted(train) if k.startswith(style))
                for style in ('gBR_', 'gHO_', 'gWA_')]
    protocol = {'selection': selected, 'selection_rule': 'first eligible TRAIN in BR, HO, WA',
                'fps': 60, 'field': 'keypoints3d_optim', 'units': 'original annotation units',
                'source_url': 'https://google.github.io/aistplusplus_dataset/download.html',
                'inputs': {n: digest(source/n) for n in ('splits.zip','keypoints3d.zip','ignore_list.txt')},
                'usage': 'separate source quality reference only; no training or EDM conditioning claim',
                'source_sha256': digest(__file__)}
    authenticated_json(output/'protocol.json', protocol)
    rows = []
    with zipfile.ZipFile(source/'keypoints3d.zip') as archive:
        for key in selected:
            joints = ArrayReader(io.BytesIO(archive.read(f'keypoints3d/{key}.pkl'))).load()['keypoints3d_optim']
            if joints.ndim != 3 or joints.shape[1:] != (17,3) or not np.isfinite(joints).all():
                raise ValueError('Invalid AIST reference geometry')
            np.savez_compressed(output/f'{key}.npz', joints=joints, fps=60)
            times = np.linspace(0, len(joints)-1, 9).astype(int)
            lengths = np.stack([np.linalg.norm(joints[:,a]-joints[:,b],axis=-1) for a,b in EDGES], axis=-1)
            fig, axes = plt.subplots(2,9,figsize=(18,5),facecolor='#101827')
            for row, theta in enumerate((0,np.pi/4)):
                horizontal = joints[:,:,0]*np.cos(theta)+joints[:,:,2]*np.sin(theta)
                limits = (horizontal.min()-10,horizontal.max()+10)
                ylim = (joints[:,:,1].min()-10,joints[:,:,1].max()+10)
                for col, t in enumerate(times):
                    ax = axes[row,col]
                    ax.set_facecolor('#101827')
                    for a,b in EDGES:
                        ax.plot(horizontal[t,[a,b]], joints[t,[a,b],1], color='#71ddc7', lw=2)
                    ax.scatter(horizontal[t],joints[t,:,1],s=5,color='white')
                    ax.set(xlim=limits,ylim=ylim,aspect='equal')
                    ax.axis('off')
                    ax.set_title(f'{t/60:.2f}s',color='white',fontsize=9)
            fig.suptitle(key+' | optimized 3D, 60 Hz | front / oblique | fixed whole-clip scale',color='white')
            fig.tight_layout()
            path = output/f'{key}.png'
            fig.savefig(path,dpi=120,facecolor=fig.get_facecolor())
            plt.close(fig)
            rows.append({'key':key,'frames':len(joints),'finite':True,
                         'bone_length_cv_median':float(np.median(lengths.std(0)/lengths.mean(0))),
                         'array_sha256':digest(output/f'{key}.npz'),'montage':str(path),'montage_sha256':digest(path)})
    authenticated_json(output/'report.json', {'protocol_sha256':digest(output/'protocol.json'),'rows':rows})
    print(json.dumps(rows))


if __name__ == '__main__':
    main()
