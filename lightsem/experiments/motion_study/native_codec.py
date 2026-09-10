"""Uncompressed native motion reference, not a learned compression result.

The music generator must produce every channel. The decoder accepts only those
generated values; it has no source motion, contacts or trajectory input.
"""
import torch
from torch import nn


class NativeCodec(nn.Module):
    channels = 139
    fps = 30
    stride = 1

    def encode(self, features):
        if features.shape[-1]!=self.channels:
            raise ValueError('Expected root3 + 22x6D rotations + four contacts')
        return features.float()

    def decode(self, latent):
        if latent.shape[-1]!=self.channels:
            raise ValueError('Every native channel must be supplied by the generator')
        latent = latent.float()
        return torch.cat((latent[...,:135],latent[...,135:].clamp(0,1)),dim=-1)

    def forward(self, features):
        return self.decode(self.encode(features))

    def forward_chunked(self, features, chunk_frames=960):
        return torch.cat([self(features[:,i:i+chunk_frames]) for i in range(0,features.shape[1],chunk_frames)],dim=1)

    def config_dict(self):
        return {'arm':'native139','channels':139,'fps':30,'values_per_second':4170,
                'learned':False,'compression':False,'root_contacts':'generated channels, no decoder side input',
                'decode':'native 6D rotations to fixed-bone FK; integrate generated horizontal velocity once'}
