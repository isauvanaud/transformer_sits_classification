import torch
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode

import numpy as np
import re
from datetime import datetime
from typing import Union, Tuple, Optional, List


def dates2doys(dates: list[str]):
    '''Convert a list of date strings to a tensor of day-of-year (DOY).
    Handles common formats by stripping non-digit characters to parse YYYYMMDD.
    Returns a torch.LongTensor of DOYs, accounting for leap years.
    '''
    # date_strings: list of "YYYY-MM-DD"
    doys = [
        datetime.strptime(d, "%Y-%m-%d").timetuple().tm_yday
        for d in dates
    ]
    return torch.tensor(doys, dtype=torch.int32)


def pad_tensor(x: torch.Tensor, l: int, pad_value=0.):
    ''' Adds padding to a tensor.
    '''
    padlen = l - x.shape[0]
    pad = [0 for _ in range(2 * len(x.shape[1:]))] + [0, padlen]
    return F.pad(x, pad=pad, value=pad_value)


def fill_ts(ts: torch.Tensor, doys: torch.Tensor, full_doys: torch.Tensor):
    ''' Fill the gaps in a time series with NaN values.
    Args:
        ts: time series with missing data
        doys: days of year of the time series
        full_doys: complete list of days of year (including missing dates)
    '''
    full_length = len(full_doys)
    ts = pad_tensor(ts, full_length, pad_value=torch.nan)
    missing_doys = torch.tensor(list(
        set(full_doys.tolist()) - set(doys.tolist())
    ))
    missing_doys, _ = missing_doys.sort()
    doys = torch.cat((doys, missing_doys))
    doys, indices = doys.sort()
    indices = indices.view(-1, 1, 1).repeat(1, ts.shape[1], ts.shape[2])
    ts = torch.gather(ts, index=indices, dim=0)
    return ts


def get_params(model: torch.nn.Module):
    '''TODO: compute the number of trainable parameters of a model.
    '''
    raise NotImplementedError


def get_flops(model, inp: Union[torch.Tensor, Tuple], with_backward=False):
    '''Credit: https://alessiodevoto.github.io/Compute-Flops-with-Pytorch-built-in-flops-counter/
    '''
    istrain = model.training
    model.eval()
    
    inp = inp if isinstance(inp, torch.Tensor) else torch.randn(inp)

    flop_counter = FlopCounterMode(mods=model, display=False, depth=None)
    with flop_counter:
        if with_backward:
            model(inp).sum().backward()
        else:
            model(inp)
    total_flops =  flop_counter.get_total_flops()
    if istrain:
        model.train()
    return total_flops

def rgb_render(
    data: np.ndarray,
    clip: int = 2,
    bands: Optional[List[int]] = None,
    norm: bool = True,
    dmin: Optional[np.ndarray] = None,
    dmax: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Prepare data for visualization with matplot lib, taken (with minor modifications) from the sensorsio repo by Julien Michel
    Source: https://framagit.org/jmichel-otb/sensorsio/-/blob/master/src/sensorsio/utils.py?ref_type=heads
    License: Apache License, Version 2.0

    :param data: nd_array of shape [bands, w, h]
    :param clip: clip percentile (between 0 and 100). Ignored if norm is False
    :bands: List of bands to extract (len is 1 or 3 for RGB)
    :norm: If true, clip a percentile at each end

    :returns: a tuple of data ready for matplotlib, dmin, dmax
    """
    if bands is None:
        bands = [2, 1, 0]
    assert len(bands) == 1 or len(bands) == 3
    assert 0 <= clip <= 100

    # Extract bands from data
    data_ready = np.take(data, bands, axis=0)
    out_dmin = None
    out_dmax = None
    # If normalization is on
    if norm:
        # Rescale and clip data according to percentile
        if dmin is None:
            out_dmin = np.percentile(data_ready, clip, axis=(1, 2))
        else:
            out_dmin = dmin
        if dmax is None:
            out_dmax = np.percentile(data_ready, 100 - clip, axis=(1, 2))
        else:
            out_dmax = dmax
        data_ready = np.clip((data_ready.transpose(1, 2, 0) - out_dmin) / (out_dmax - out_dmin), 0, 1)

    else:
        data_ready.transpose(1, 2, 0)

    # Strip of one dimension if number of bands is 1
    if data_ready.shape[-1] == 1:
        data_ready = data_ready[:, :, 0]

    return data_ready, out_dmin, out_dmax