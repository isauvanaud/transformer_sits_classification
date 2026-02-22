''' Define the sublayers used in the transformer model
Source: https://github.com/jadore801120/attention-is-all-you-need-pytorch/
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingLayer(nn.Module):
    '''TODO: compute embeddings from raw pixel set data.
    '''

    def __init__(self,n_channels, n_pixels, d_model):
        super(EmbeddingLayer, self).__init__()
        self.conv = nn.Conv1d(in_channels=n_channels,out_channels=d_model,kernel_size=(n_pixels,), stride=(1,))

    def forward(self, x):
        print(x.shape)
        return self.conv(x)
    

class NDVI(nn.Module):
    '''TODO: compute NDVI time series from raw pixel set data.
    NDVI = (NIR - RED) / (NIR + RED)
    '''

    def __init__(self):
        super(NDVI, self).__init__()

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): batch_size x len_seq x n_channels x n_pixels data
        """
        raise NotImplementedError
    

class BI(nn.Module):
    '''TODO: compute BI time series from raw pixel set data.
    BI = ((SWIR1 + RED) - (NIR + BLUE)) / ((SWIR1 + RED) + (NIR + BLUE))
    '''
    def __init__(self):
        super(BI, self).__init__()

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): batch_size x len_seq x n_channels x n_pixels data
        """
        raise NotImplementedError
    

class SpectralIndicesLayer(nn.Module):
    '''TODO: compute features based on NDVI and BI time series from raw pixel set data.
    '''

    def __init__(self, d_model, blue=1, red=2, near_infrared=6, swir1=8, eps=1e-3):
        super(SpectralIndicesLayer, self).__init__()
        self.ndvi = NDVI(red, near_infrared, eps)
        self.bi = BI(blue, red, near_infrared, swir1, eps)
        self.mlp = nn.Linear(2 * d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): batch_size x len_seq x n_channels x n_pixels data
        """
        raise NotImplementedError
    
"""class PositionalEncoding(nn.Module):
    ''' Positional Encoding Layer.
    Source: https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py
    TODO: Update the positional encoding as described in "Satellite Image Time Series 
    Classification with Pixel-Set Encoders and Temporal Self-Attention, Garnot et al."
    '''
    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        # TODO: update forward function to return the positional embedding only.
        
        return self.pos_table[:, :x.size(0)].clone().detach()"""

class PositionalEncoding(nn.Module):
    """
    Positional Encoding for irregular temporal sampling
    (Garnot et al., 2020).

    For dimension i in [0, d_hid):
        p_i(t) = sin( DOY(t) / T^{2*floor(i/2)/d_hid} + (pi/2)*(i mod 2) )

    This version matches the API:
        __init__(d_hid, n_position=200)
        forward(x)
    """

    def __init__(self, d_hid, n_position=200, T=365.25):
        super(PositionalEncoding, self).__init__()
        self.d_hid = int(d_hid)
        self.n_position = int(n_position)  # kept for API compatibility
        self.T = float(T)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [L] containing DOY values

        Returns:
            Tensor of shape [1, L, d_hid]
        """
        if x.dim() != 1:
            raise ValueError("Input must be a 1D tensor of DOY values")

        L = x.size(0)
        device = x.device

        # dimension indices
        i = torch.arange(self.d_hid, device=device, dtype=torch.float32)

        # exponent term
        exponents = (2.0 * torch.floor(i / 2.0)) / float(self.d_hid)

        # T^(2i/d)
        T_pow = torch.pow(
            torch.tensor(self.T, device=device, dtype=torch.float32),
            exponents
        )

        # phase shift (0 for even dims, pi/2 for odd dims)
        phase = (np.pi / 2.0) * (i % 2.0)

        # reshape DOY
        doys = x.to(dtype=torch.float32).unsqueeze(1)  # [L, 1]

        # compute angles
        angles = doys / T_pow.unsqueeze(0) + phase.unsqueeze(0)

        pos = torch.sin(angles)  # [L, d_hid]

        return pos.unsqueeze(0)  # [1, L, d_hid]

class Temporal_Aggregator(nn.Module):
    ''' TODO: aggregate embeddings that are not masked.
    '''
    def __init__(self, mode='mean'):
        super(Temporal_Aggregator, self).__init__()
        self.mode = mode

    def forward(self, data, mask):
        if self.mode == 'mean':
            raise NotImplementedError
        elif self.mode == 'identity':
            out = data
        else:
            raise NotImplementedError
        return out