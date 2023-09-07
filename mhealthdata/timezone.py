#!/usr/bin/env python
# -*- coding: utf8 -*-

import numpy as np
from mhealthdata.utils import *


def _get_tzlist(dt):
    """
    Get array of possible timezone offsets

    Parameters
    ----------
    dt : int, default 60
        Stride to find best match timezone offset, [minutes]

    Returns
    -------
    ndarray
        1D array of possible timezone offsets

    """
    n = 1440 // dt
    tzlist = (np.arange(n) - n // 2) * dt
    return tzlist
    

def _get_slice(x, y, i, nday=3, backwards=True):
    """
    Get window slice

    Parameters
    ----------
    x : ndarray
        2D array of size N days x 1440 minutes (padded sleep > 0)
    y : ndarray
        2D array of size N days x 1440 minutes (padded steps + bpm, 0 replaced by np.nan)
    i : int
        Day index
    nday : int, default 3
        Window to find best match timezone offset, [days]
    backwards : bool, default True
        If True include previous day, otherwise include following days

    Returns
    -------
    ndarrays
        1D arrays of window slices

    """
    i0 = i - nday * backwards
    i1 = i0 + nday + 1
    return x[i0:i1].flatten(), y[i0:i1].flatten()


def _conv_slice(x, y, t):
    """
    Calculate convolution of window slices

    Parameters
    ----------
    x : ndarray
        2D array of size N days x 1440 minutes (padded sleep > 0)
    y : ndarray
        2D array of size N days x 1440 minutes (padded steps + bpm, 0 replaced by np.nan)
    t : int
        Offset [minutes]

    Returns
    -------
    float
        Convolution (the smaller - the better sleep-to-steps match)

    """
    mask = np.roll(x, -t) > 0
    return np.nanmean(y[mask]) * np.sum(mask)


def _calc_timezone(x, y, i, nday=3, dt=60):
    """
    Day-wise score for each timezone offset

    Parameters
    ----------
    x : ndarray
        2D array of size N days x 1440 minutes (padded sleep > 0)
    y : ndarray
        2D array of size N days x 1440 minutes (padded steps + bpm, 0 replaced by np.nan)
    i : int
        Day index
    nday : int, default 3
        Window to find best match timezone offset, [days]
    dt : int, default 60
        Stride to find best match timezone offset, [minutes]

    Returns
    -------
    ndarray
        1D array of score for each timezone offset

    """
    tzlist = _get_tzlist(dt)
    x_, y_ = _get_slice(x, y, i + nday, nday, backwards=False)
    score = np.array([_conv_slice(x_, y_, t) for t in tzlist])
    x_, y_ = _get_slice(x, y, i + nday, nday, backwards=True)
    score_ = np.array([_conv_slice(x_, y_, t) for t in tzlist])
    score = np.nanmin(np.stack([score, score_]), axis=0)
    score = score - np.nanmin(score)
    score[score > 0.1 * np.nanmax(score)] = np.nan
    mask = score == np.nanmin(score)
    t = tzlist[mask][0] if any(mask) else np.nan
    return score


def _baseline_timezone(tz, nday, dt):
    """
    Find baseline (i.e. long-term) timezone offsets

    Parameters
    ----------
    tz : ndarray
        1D array of length N days with timezone offset [minutes]
    nday : int, default 3
        Window to find best match timezone offset, [days]
    dt : int, default 60
        Stride to find best match timezone offset, [minutes]

    Returns
    -------
    ndarray
        1D array of length N days with baseline offset [minutes]

    """
    bz = np.zeros_like(tz) * np.nan
    tzlist = np.unique(tz[np.isfinite(tz)])
    for t in tzlist:
        idx = find_intervals(tz == t, tol = 4 * nday)
        for i0, i1 in idx:
            if i1 - i0 >= 4 * nday:
                bz[i0:i1] = t
    mask = np.isfinite(bz)
    if any(mask):
        idx = np.arange(len(bz))[mask]
        bz[:idx.min()] = bz[idx.min()]
        bz[idx.max():] = bz[idx.max()]
        idx = find_intervals(np.isnan(bz))
        if any(np.isfinite(bz)):
            for i0, i1 in idx:
                tx, _ = unique_sorted(tz[i0:i1])
                bx = np.zeros((2)) * np.nan
                bx[0] = bz[i0 - 1] if i0 > 0 else bz[i1]
                bx[1] = bz[i1] if i1 < len(bz) - 1 else bz[i0 - 1]
                dx = np.abs(bx - tx[0])
                if any(np.isfinite(dx)):
                    bz[i0:i1] = bx[np.nanargmin(dx)]
                else:
                    bz[i0:i1] = bx[np.isfinite(bx)][0]
    return bz


def _filter_timezone(tz, nday, dt):
    """
    Filter out too short or ambigouos timezone offsets

    Parameters
    ----------
    tz : ndarray
        1D array of length N days with timezone offset [minutes]
    nday : int, default 3
        Window to find best match timezone offset, [days]
    dt : int, default 60
        Stride to find best match timezone offset, [minutes]

    Returns
    -------
    ndarray
        1D array of length N days with timezone offset [minutes]

    """
    tz = tz.astype(float)
    if any(np.isfinite(tz)):
        bz = _baseline_timezone(tz, nday, dt)
        dz = tz - bz
        idx = find_intervals(dz != 0, tol=1)
        for i0, i1 in idx:
            n = max(np.sum(dz[i0:i1] >= 0), np.sum(dz[i0:i1] <= 0))
            if i1 - i0 <= nday or i1 - i0 > n:
                tz[i0:i1] = bz[i0:i1]
            else:
                tz[i0:i1] = unique_sorted(tz[i0:i1])[0][0]
    return tz


def _localize_activity(x, idx, tz):
    """
    Roll activity (steps or bpm) to local time zone

    Parameters
    ----------
    x : ndarray
        2D array of size N days x 1440 minutes
    idx : ndarray
        2D array of size N intervals x 2 indices (start, end)
    tz : ndarray
        1D array of length N intervals with timezone offset, [minutes]

    Returns
    -------
    ndarray
        2D array of size N days x 1440 minutes

    """
    x_ = np.zeros((x.size + 2 * 1440))
    for k, i in enumerate(idx):
        dt = tz[k]
        j = 1440 + i + dt
        x_[j[0]:j[1]] = x[i[0]:i[1]]
    x_ = x_[1440:-1440].reshape(-1,1440)
    return x_


def find_timezone_mismatch(data, nday=3, dt=60):
    """
    For Fitbit only: Identify timezone by mismatch of sleep, steps, and bpm
    - Assume "sleep" is in local time
    - Automatically detect day-wise "steps" & "bpm" offsets to match "sleep"
    
    Parameters
    ----------
    data : dict
        Dictionary of data, should have keys "sleep", "steps", "bpm", 
        each containing array of size N days x 1440 minutes
    nday : int, default 3
        Window to find best match timezone offset, [days]
    dt : int, default 60
        Stride to find best match timezone offset, [minutes]

    Returns
    -------
    ndarray
        1D array of length N days with timezone offset [minutes]

    """
    pad = np.zeros((nday,1440))
    x = np.vstack([pad, data["sleep"], pad])
    y = np.vstack([pad, data["steps"], pad]) + np.vstack([pad, data["bpm"], pad])
    x = (x > 0).astype(float)
    y[y == 0] = np.nan
    score = []
    for i in range(len(data["steps"])):
        score.append(_calc_timezone(x, y, i, nday, dt))
    score = np.stack(score)
    tzlist = _get_tzlist(dt)
    tz = np.zeros(len(score),) * np.nan
    for k, s in enumerate(score):
        if any(np.isfinite(s)):
            tz[k] = tzlist[np.nanargmin(s)]
    tz[0] = tz[1]
    tz[-1] = tz[-2]
    tz = _filter_timezone(tz, nday, dt)
    return tz

    
    
def fix_timezone_mismatch(data, tz=None):
    """
    Fix timezone offset for data imported from Fitbit
    - Assume "sleep" is in local time
    - Automatically detect day-wise "steps" and "bpm" offsets to match "sleep"
    
    Parameters
    ----------
    data : dict
        Dictionary of data, should have keys "sleep", "steps", "bpm"
        Each key should point to an array of size N days x 1440 minutes
    tz : ndarray or None, default None
        1D array of length N days with timezone offset [minutes]
        If None, timezone will be detected automatically

    Returns
    -------
    dict
        Dictionary of data with "steps" and "bpm" rolled to match "sleep"

    """
    tz = tz if tz is not None else find_timezone_mismatch(data)
    tz = np.repeat(tz, 1440)
    mask = np.diff(tz, prepend=tz[0]+1, append=tz[-1]+1) != 0
    idxs = find_intervals(data["sleep"].flatten())
    for k, (i0,i1) in enumerate(idxs):
        dt = unique_sorted(tz[i0:i1])[0]
        dt = int(dt[0]) if len(dt) > 0 else 0
        j0 = max(0, i0 - dt * (dt < 0))
        j1 = min(len(tz), i1 - dt * (dt > 0))
        dts = np.unique(tz[j0:j1])
        tz[j0:j1] = dt
    mask = np.diff(tz, prepend=tz[0]+1, append=tz[-1]+1) != 0
    idx = np.arange(len(tz) + 1)[mask]
    idx = np.stack([idx[:-1], idx[1:]]).T
    tz = np.array([np.unique(tz[i0:i1])[0] for i0, i1 in idx]).astype(int)
    data["steps"] = _localize_activity(data["steps"].flatten(), idx, tz)
    data["bpm"] = _localize_activity(data["bpm"].flatten(), idx, tz)
    return data


