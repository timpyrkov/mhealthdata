#!/usr/bin/env python
# -*- coding: utf8 -*-

import numpy as np
from mhealthdata.utils import *


def extend_arrays(data):
    """
    Extend flattened arrays of "steps", "sleep", and "bpm" by 1 empty day
    
    Parameters
    ----------
    data : dict
        Dictionary of data

    Returns
    -------
    dict
        Dictionary of data with extended arrays

    """
    x = {}
    pad = np.zeros((1,1440))
    for key in data:
        x[key] = data[key]
        if key in ["steps", "sleep", "bpm"]:
            x_ = data[key].astype(float)
            if len(x_) > 0:
                x_ = np.vstack([pad, x_, pad])
            x[key] = x_.flatten()
    return x


def offset_overlap(x, y):
    """
    Calc overlap for offsets with 60-minute stride
    
    Parameters
    ----------
    x : ndarray
        1D-array of "steps" or "bpm"
    y : ndarray
        1D-array of "sleep" bout


    Returns
    -------
    ndarray
        1D-array of overlap counts [minutes]

    """
    x_ = (x > 0).astype(float)
    return np.convolve(x_, y, 'valid')[::60]


def offset_select(nsteps, npulse):
    """
    Select max overlap of "sleep" with "bpm" and min "steps" 
    
    Parameters
    ----------
    nsteps : ndarray
        1D-array of overalp counts for "sleep" and "steps"
    npulse : ndarray
        1D-array of overalp counts for "sleep" and "bpm"


    Returns
    -------
    ndarray
        1D-array of selected offsets [minutes]

    """
    offset = 720 - 60 * np.arange(24)
    mask = (npulse == npulse.max())
    offset = offset[mask]
    nsteps = nsteps[mask]
    offset = offset[nsteps < 5 * nsteps.min()]
    return offset

def offset_to_index(offset):
    """
    Cast offset to int and impute gaps by the most frequent value
    
    Parameters
    ----------
    offset : ndarray
        1D-array of offset  of dtype float (gaps filled with np.nan)


    Returns
    -------
    ndarray
        1D-array of offset of dtype int

    """
    offsets, counts = unique_sorted(offset)
    if len(offsets) > 0:
        offset[~np.isfinite(offset)] = offsets[0]
    else:
        offset = np.zeros_like(offset)
    offset = offset.astype(int)
    return offset
    
    
    
def fix_timezone(data):
    """
    Fix timezone offset for data imported from Fitbit
    - Assume "sleep" is in local time
    - Automatically detect day-wise "steps" and "bpm" offsets to match "sleep"
    
    Parameters
    ----------
    data : dict
        Dictionary of data


    Returns
    -------
    dict
        Dictionary of data with "steps" and "bpm" shifted to match local time

    """
    xdata = extend_arrays(data)
    intervals = find_nonzero_intervals(xdata["sleep"] > 0)
    nday = data["sleep"].shape[0] + 2
    offset = np.zeros((nday)) * np.nan
    for i0, i1 in intervals:
        duration = np.sum(xdata["sleep"][i0:i1] > 2)
        if duration >= 120:
            j0 = i0 - 720
            j1 = i1 + 720 - 1
            window = np.ones((i1 - i0))
            nsteps = offset_overlap(xdata["steps"][j0:j1], window)
            npulse = offset_overlap(xdata["bpm"][j0:j1], window)
            day = (i1 // 1440) - 1
            off = offset_select(nsteps, npulse)
            if len(off) == 1:
                offset[day] = offset_select(nsteps, npulse)[0]
    offset = offset_to_index(offset)
    for key in ["steps", "bpm"]:
        x = np.zeros((nday,1440))
        for j in range(1, nday - 1):
            i0 = 1440  * j - offset[j]
            i1 = i0 + 1440
            x[j] = xdata[key][i0:i1]
        xdata[key] = x[1:-1]
    xdata["sleep"] = data["sleep"]
    #xdata["offset"] = offset[1:-1]
    return xdata


