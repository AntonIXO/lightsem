"""Native-frame temporal transform and TRAIN-fitted linear rotation projections.

Root and contact channels are part of the latent, counted in its bitrate. Playback
must generate these channels too: the decoder has no access to source motion.
"""
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .geometry import FPS


@dataclass(frozen=True)
class PhaseConfig:
    arm: str = 'phase128'

    @property
    def ranks(self):
        return {'frame24':(24,), 'phase64':(32,16,12,4),
                'phase128':(64,32,20,12), 'phase256':(100,72,52,32)}[self.arm]

    @property
    def stride(self):
        return len(self.ranks)

    @property
    def latent(self):
        return sum(self.ranks)+7*self.stride


def temporal_basis(stride):
    # Orthonormal Haar basis. Every AC row sums to EXACT zero in FP32, so a
    # static source cannot acquire phase jitter through a learned AC bias.
    if stride == 1:
        return np.ones((1,1),dtype=np.float32)
    return np.asarray([[.5,.5,.5,.5],[.5,.5,-.5,-.5],
                       [2**-.5,-2**-.5,0,0],[0,0,2**-.5,-2**-.5]],dtype=np.float32)


class PhaseCodec(nn.Module):
    def __init__(self, config, normalizers, fit):
        super().__init__()
        self.config = config
        self.register_buffer('mean',torch.tensor(normalizers['mean'][3:135],dtype=torch.float32))
        self.register_buffer('std',torch.tensor(normalizers['std'][3:135],dtype=torch.float32))
        self.register_buffer('temporal',torch.from_numpy(temporal_basis(config.stride)))
        self.register_buffer('dc_mean',torch.tensor(fit['dc_mean'],dtype=torch.float32))
        for band,rank in enumerate(config.ranks):
            self.register_buffer(f'basis_{band}',torch.tensor(fit['basis'][band][:,:rank],dtype=torch.float32))

    def encode(self, features):
        # The small codec stays FP32 even when the downstream generator uses BF16.
        with torch.autocast(device_type=features.device.type,enabled=False):
            features = features.float()
            padding = (-features.shape[1]) % self.config.stride
            if padding:
                features = torch.cat((features,features[:,-1:].expand(-1,padding,-1)),dim=1)
            blocks = features.reshape(features.shape[0],-1,self.config.stride,139)
            rotation = (blocks[...,3:135]-self.mean)/self.std
            bands = torch.einsum('kp,bnpd->bnkd',self.temporal,rotation)
            encoded = []
            for band in range(self.config.stride):
                value = bands[:,:,band]-(self.dc_mean if band==0 else 0.)
                encoded.append(value @ getattr(self,f'basis_{band}'))
            side = torch.cat((blocks[...,:3],blocks[...,135:]),dim=-1).flatten(-2)
            return torch.cat((*encoded,side),dim=-1)

    def decode(self, latent):
        with torch.autocast(device_type=latent.device.type,enabled=False):
            latent = latent.float()
            bands = []
            cursor = 0
            for band,rank in enumerate(self.config.ranks):
                value = latent[...,cursor:cursor+rank] @ getattr(self,f'basis_{band}').T
                bands.append(value+(self.dc_mean if band==0 else 0.))
                cursor += rank
            rotation = torch.einsum('kp,bnkd->bnpd',self.temporal,torch.stack(bands,dim=-2))
            rotation = rotation*self.std+self.mean
            side = latent[...,cursor:].reshape(*latent.shape[:2],self.config.stride,7)
            features = torch.cat((side[...,:3],rotation,side[...,3:].clamp(0,1)),dim=-1)
            return features.flatten(1,2)

    def forward(self, features):
        return self.decode(self.encode(features))[:,:features.shape[1]]

    def forward_chunked(self, features, chunk_frames=960):
        if chunk_frames % self.config.stride:
            raise ValueError('Chunk must preserve the global temporal phase')
        return torch.cat([self(features[:,start:start+chunk_frames])
                          for start in range(0,features.shape[1],chunk_frames)],dim=1)

    def config_dict(self):
        return {'arm':self.config.arm,'ranks':list(self.config.ranks),'latent_channels':self.config.latent,
                'latent_fps':FPS/self.config.stride,'native_fps':FPS,
                'side_channels':7*self.config.stride,'root_and_contacts':'explicit generated latent channels; no decoder source input',
                'fit':'equal-canonical-song TRAIN second moments, exact eigenvectors; no gradient updates',
                'precision':'FP32 encode/decode, float64 covariance fit'}


def fit_moments(entries, normalizers):
    """Identical full TRAIN exposure across all ranks; no outcome-based fitting."""
    from collections import Counter
    from ..stage11_common import digest
    entries = [e for e in entries if e['partition']=='train']
    counts = Counter(e['canonical_id'] for e in entries)
    mean = np.asarray(normalizers['mean'][3:135],dtype=np.float64)
    std = np.asarray(normalizers['std'][3:135],dtype=np.float64)
    accum = {s:{'mean':np.zeros(132),'moments':np.zeros((s,132,132)),'weight':0.} for s in (1,4)}
    for entry in entries:
        if digest(entry['path']) != entry['sha256']:
            raise ValueError('Prepared TRAIN input changed')
        with np.load(entry['path']) as value:
            x = (value['features'][:,3:135].astype(np.float64)-mean)/std
        weight = 1/(len(counts)*counts[entry['canonical_id']])
        for stride in (1,4):
            # Include each real frame once; pad only the final partial block.
            padded = np.concatenate((x,np.repeat(x[-1:],(-len(x))%stride,axis=0)))
            blocks = padded.reshape(-1,stride,132)
            bands = np.einsum('kp,npd->nkd',temporal_basis(stride).astype(np.float64),blocks)
            acc = accum[stride]
            acc['mean'] += weight*bands[:,0].mean(0)
            for band in range(stride):
                acc['moments'][band] += weight*(bands[:,band].T@bands[:,band])/len(bands)
            acc['weight'] += weight
    result = {}
    for stride,acc in accum.items():
        dc = acc['mean']/acc['weight']
        bases,eigenvalues = [],[]
        for band in range(stride):
            moment = acc['moments'][band]/acc['weight']
            covariance = moment-np.outer(dc,dc) if band==0 else moment
            values,vectors = np.linalg.eigh(covariance)
            values,vectors = values[::-1].copy(),vectors[:,::-1].copy()
            indices = np.argmax(np.abs(vectors),axis=0)
            signs = np.sign(vectors[indices,np.arange(vectors.shape[1])])
            vectors *= np.where(signs==0,1.,signs)
            bases.append(vectors.astype(np.float32))
            eigenvalues.append(values.tolist())
        result[stride] = {'dc_mean':dc.astype(np.float32),'basis':bases,'eigenvalues':eigenvalues}
    return result
