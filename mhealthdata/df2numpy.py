#!/usr/bin/env python
# -*- coding: utf8 -*-

import numpy as np
import pandas as pd
from mhealthdata.utils import *


def _get_values(df, column):
    """
    Get record value arrays from loaded datframes.
    
    Parameters
    ----------
    df : DataFrame
        DataFrame of loaded health data for "steps", "sleep", "bpm", etc.
    column : str
        Values column name

    Returns
    -------
    ndarray
        1D array of record values, e.g. sleep stage, or stepcounts, etc.

    """
    values = df[column].values
    try:
        values = values.astype(float)
    except:
        s = sleep_stage_dict(mode="encode")
        s["None"] = 0
        values = values.astype(str)
        values = np.vectorize(s.get)(values)
        values = values.astype(float)
    return values


def _get_time(df, keys):
    """
    Get record timestamps from loaded datframes.
    
    Parameters
    ----------
    df : DataFrame
        DataFrames of loaded health data for "steps", "bpm", etc.
    keys : list
        List of keywords to seek for a column in a DataFrame

    Returns
    -------
    ndarray
        1D array of type datetime64

    """
    col = find_columns_by_key(df, keys)
    t = df[col[0]] if len(col) > 0 else None
    return t


def _get_idate_imin(t):
    """
    Get record ordinal days and minutes from loaded datframes.
    
    Parameters
    ----------
    t : ndarray
        1D array of datetime64

    Returns
    -------
    idate : ndarray
        1D array of ordinal days (January 1 of year 1 is day 1)
    imin : ndarray
        1D array of minutes since midnight

    """
    idate = t.apply(pd.Timestamp.toordinal).values.astype(int)
    imin = (60 * t.dt.hour + t.dt.minute).values.astype(int)
    return idate, imin


def _calc_duration(t0, t1):
    """
    Private method to calc record durations from start and end timestamps.
    
    Parameters
    ----------
    t0 : ndarray
        1D array of start datetime64
    t1 : ndarray
        1D array of end datetime64; if None, set t1 = t0 + 1 [min]

    Returns
    -------
    ndarray
        1D array record durations [minutes]

    """
    dt = np.ones((len(t0))).astype(int)
    if t1 is not None:
        dt = (t1 - t0).astype("timedelta64[s]").values
        dt = (dt / 60).astype(int)
        dt[dt < 1] = 1
    return dt


def to_1darray(df, column, tstart, tend=None, tz=None, idate=None, x=None, uint8=False):
    """
    Get value-per-day health data ("weight", "rhr", or "hrv") as 1D array.
    
    Parameters
    ----------
    df : DataFrame
        DataFrames of health data records - "steps", "bpm", etc.
    column : str
        Name of values column
    tstart : list
        List of columns to seek for start date/time
    tend : list or None, default None
        List of columns to seek for end date/time
    tz : list or None, default None
        List of columns to seek for date/time time zone
    idate : ndarray or None, default None
        1D array of continuous range of ordinal days
    x : ndarray or None, default None
        Initialized 1D array; if None, will be initialized with np.zeros()
    uint8 : bool, default False
        Flag to cast all health data to np.uint8 to save disk space

    Returns
    -------
    x : ndarray
        1D array of values of size (N days).
    idate : ndarray
        1D array of record ordinal days.

    """
    df = columns_to_datetime(df, tstart, tend, tz)
    val = _get_values(df, column)
    t0 = _get_time(df, tstart)
    iday, imin = _get_idate_imin(t0)
    idate = idate if idate is not None else to_range(iday)
    n = len(idate)
    x = np.zeros((n)) if x is None else x
    idx = iday - idate[0]
    mask = (idx >= 0) & (idx < n)
    idx = idx[mask]
    val = val[mask]
    for k in range(len(val)):
        i = idx[k]
        x[i] = val[k]
    if uint8:
        x = np.clip(x,0,255).astype(np.uint8)
    return x, idate


def to_2darray(df, column, tstart, tend=None, tz=None, dt=None, idate=None, x=None, uint8=False, mode="rate"):
    """
    Get value-per-minute health data ("steps", "sleep", or "bpm") as 2D array.
    
    Parameters
    ----------
    df : DataFrame
        DataFrames of health data records - "steps", "bpm", etc.
    column : str
        Name of values column
    tstart : list
        List of columns to seek for start date/time
    tend : list or None, default None
        List of columns to seek for end date/time
    tz : list or None, default None
        List of columns to seek for date/time time zone
    dt : str, ndarray, or None, default None
        Column name or 1D array of record durations [seconds].
    idate : ndarray or None, default None
        1D array of continuous range of ordinal days
    x : ndarray or None, default None
        Initialized 1D array; if None, will be initialized with np.zeros()
    uint8 : bool, default False
        Flag to cast all health data to np.uint8 to save disk space
    mode : {"rate", "count"}, default "rate"
        Way to treat values of records longer than 1 minute: 
        if "rate" - duplicate values, if "count" - split evenly between minutes

    Returns
    -------
    x : ndarray
        2D array of values of size (N days x 1440 minutes).
    idate : ndarray
        1D array of record ordinal days.

    """
    assert mode in ["rate", "count"]
    df = columns_to_datetime(df, tstart, tend, tz)
    val = _get_values(df, column)
    t0, t1 = [_get_time(df, k) for k in [tstart, tend]]
    iday, imin = _get_idate_imin(t0)
    dt = df[dt].values if isinstance(dt, str) else dt # seconds
    dt = dt / 60 if dt is not None else _calc_duration(t0, t1) # minutes
    dt = np.ceil(dt).astype(int)
    idate = idate if idate is not None else to_range(iday)
    n = 1440 * len(idate)
    x = np.zeros((n)) if x is None else x.flatten()
    idx = 1440 * (iday - idate[0]) + imin
    mask = (idx >= 0) & (idx < n)
    idx = idx[mask]
    val = val[mask]
    for k in range(len(val)):
        i = idx[k]
        j = i + dt[k]
        if mode == "count":
            x[i:j] = val[k] / dt[k]
        else:
            x[i:j] = val[k]
    x = x.reshape(-1,1440)
    if uint8:
        x = np.clip(x,0,255).astype(np.uint8)
    return x, idate


def combine_arrays(*args, labels=None, mode="valid"):
    """
    Convert arrays of e.g. steps, bpm, sleep into 
    the same length and combine in a dictionary
    
    Parameters
    ----------
    *args
        Tuples of (data, date) e.g. as output by to_2darray()
    labels : list or None, default None
        List of keyword labels for data
    mode : {"valid", "full"}, default "valid"
        If "valid" all arrys shrinked to min overlapping range, else expanded
        
    Returns
    -------
    dict
        Dictionary with numpy arrays of the same length
    
    """
    assert labels is None or len(labels) == len(args)
    assert mode in ["full", "valid"]
    data = {}
    if mode == "valid":
        t0 = max(a[1][0] for a in args)
        t1 = min(a[1][-1] for a in args) + 1
        data["idate"] = np.arange(t0,t1)
        for i in range(len(args)):
            label = f"x{i}" if labels is None else labels[i]
            mask = (args[i][1] >= t0) & (args[i][1] < t1)
            data[label] = args[i][0][mask]
    if mode == "full":
        t0 = min(a[1][0] for a in args)
        t1 = max(a[1][-1] for a in args) + 1
        data["idate"] = np.arange(t0,t1)
        for i in range(len(args)):
            x, t = args[i]
            label = f"x{i}" if labels is None else labels[i]
            val = np.zeros((t1-t0,1440), x.dtype)
            i0 = t[0] - t0
            i1 = i0 + len(t)
            val[i0:i1] = x
            data[label] = val
    return data
        



