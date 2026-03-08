''' Define the sublayers used in the transformer model
Source: https://github.com/jadore801120/attention-is-all-you-need-pytorch/
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingLayer(nn.Module):
    '''compute embeddings from raw pixel set data.
    '''

    def __init__(self,n_channels, n_pixels, d_model):
        super(EmbeddingLayer, self).__init__()
        self.conv = nn.Conv1d(in_channels=n_channels,out_channels=d_model,kernel_size=(n_pixels,), stride=(1,))

    def forward(self, x):
        B, T, C, P = x.shape
        x = x.view(B*T, C, P)   # fusion batch et timesteps
        out = self.conv(x)       # Conv1d sur P avec C_in = 10
        out = out.view(B, T, -1)
        return out
    

class NDVI(nn.Module):
    '''compute NDVI time series from raw pixel set data.
    NDVI = (NIR - RED) / (NIR + RED)
    '''

    def __init__(self, red, near_infrared, eps):
        super(NDVI, self).__init__()
        # Define columns of interest
        self.red = red
        self.near_infrared = near_infrared
        self.eps = eps
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): batch_size x len_seq x n_channels x n_pixels data
        """
        NIR = x[:,:,self.near_infrared,:]
        RED = x[:,:,self.red,:]
        ndvi = (NIR - RED )/(NIR + RED + self.eps) #adding eps to avoid dividing by zero
        #print("shape ndvi :",ndvi.shape)
        return ndvi

        
    

class BI(nn.Module):
    '''compute BI time series from raw pixel set data.
    BI = ((SWIR1 + RED) - (NIR + BLUE)) / ((SWIR1 + RED) + (NIR + BLUE))
    '''
    def __init__(self,blue, red, near_infrared, swir1, eps):
        super(BI, self).__init__()
        # Define columns of interest
        self.blue = blue
        self.red = red
        self.near_infrared = near_infrared
        self.swir1 = swir1
        self.eps = eps

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): batch_size x len_seq x n_channels x n_pixels data
        """
        SWIR1 = x[:,:,self.swir1,:]
        RED = x[:,:,self.red,:]
        NIR = x[:,:,self.near_infrared,:]
        BLUE = x[:,:,self.blue,:]
        BI = ((SWIR1 + RED) - (NIR + BLUE)) / ((SWIR1 + RED) + (NIR + BLUE) + self.eps)
        #print("BI shape: ",BI.shape)
        return BI
    


class SpectralIndicesLayer(nn.Module):
    ''' compute features based on NDVI and BI time series from raw pixel set data.
'''

    def __init__(self, d_model, n_pixels, blue=1, red=2, near_infrared=6, swir1=8, eps=1e-3):
        super(SpectralIndicesLayer, self).__init__()
        self.d_model = d_model
        self.ndvi = NDVI(red, near_infrared, eps)
        self.bi = BI(blue, red, near_infrared, swir1, eps)
        self.embed_indices_ndvi = EmbeddingLayer(n_channels=1, n_pixels=n_pixels, d_model=d_model)
        self.embed_indices_bi = EmbeddingLayer(n_channels=1, n_pixels=n_pixels, d_model=d_model)
        self.mlp = nn.Linear(2 * d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x):
        embed_ndvi = self.embed_indices_ndvi(self.ndvi(x).unsqueeze(2))   # [B,T,P] -> [B,T,1,P]
        embed_bi   = self.embed_indices_bi(self.bi(x).unsqueeze(2))     # [B,T,P] -> [B,T,1,P]
        indices = torch.concat((embed_ndvi, embed_bi), dim=2)
        
        return self.layer_norm(self.mlp(indices))

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
    ''' aggregate embeddings that are not masked.
    '''
    def __init__(self, mode='mean'):
        super(Temporal_Aggregator, self).__init__()
        self.mode = mode

    def forward(self, data, mask):

        if self.mode == 'mean':
            print("data type: ",type(data))
            print("data shape: ", data.shape)
            print("mask type: ",type(mask))
            print("mask shape: ", mask.shape)   
            masked_data = data * mask
            sum_masked_data = masked_data.sum(dim=1)
            count_masked_data = mask.sum(dim=1)
            out = sum_masked_data / (count_masked_data + 1e-8)  # avoid division by zero
        elif self.mode == 'identity':
            out = data
        else:
            raise NotImplementedError
        return out