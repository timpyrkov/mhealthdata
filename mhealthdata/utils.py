#!/usr/bin/env python
# -*- coding: utf8 -*-

import numpy as np
import pandas as pd
from tqdm import tqdm

def sleep_stage_dict(mode="decode"):
    """
    Get dictionary to encode/decode sleep stage
    
    Parameters
    ----------
    mode : {"decode", "encode"}, default "decode"
        If "decode", return dict num -> str
        If "encode", return dict str -> num

    Returns
    -------
    dict
        Dictionary for decoding/encoding sleep stages

    """
    assert mode in ["decode", "encode"]
    d = {
        "no_sleep": 0,
        "wake": 1,
        "awake": 1,
        "no_stage": 2,
        "rem": 3,
        "restless": 4,
        "light": 5,
        "asleep": 6,
        "deep": 7,
    }
    if mode == "decode":
        d = {val: key for key, val in d.items()}
    return d


def find_columns_by_key(df, keys):
    """
    Find all Dataframe column names containing any of the keys
    
    Parameters
    ----------
    df : Dataframe
        Dataframe
    keys : array_like
        List of keys of `str` type

    Returns
    -------
    list
        List of column names

    """
    col = [c for c in df.columns if any([k.lower() in c.lower() for k in keys])]
    return col


def columns_to_datetime(df, columns, tz_col=None):
    """
    Cast selected Dataframe columns to pandas.Timestamp

    Parameters
    ----------
    df : DataFrame
        Dataframe
    columns : array_like
        List of columns to convert into datetime
    tz_col : str, optional
        Time zone column

    Returns
    -------
    Dataframe
        Dataframe with selected columns converted to pandas.Timestamp

    """
    def localize(row):
        fmt = "%Y-%m-%d %H:%M:%S"
        t = row[col].tz_localize("UTC").tz_convert(row[tz_col])
        t = pd.to_datetime(t.strftime(fmt), format=fmt)
        return t

    for col in columns:
        if np.issubdtype(df[col].dtype, np.integer):
            df[col] = pd.to_datetime(df[col], unit="ms")
        else:
            sample = df[col].values[0]
            fmt = "%Y-%m-%d %H:%M:%S.%f" if "-" in sample else "%m/%d/%y %H:%M:%S"
            df[col] = pd.to_datetime(df[col], format=fmt)
        if tz_col is not None:
            df[col] = df.apply(localize, axis=1)            
    return df


def dates_to_ordinal(dates):
    """
    Get ordinal days (January 1 of year 1 is day 1)
    
    Parameters
    ----------
    dates : array_like
        List of dates of type `int` (ordinal) or `str`

    Returns
    -------
    ndarray
        Array of ordinal days

    """
    idates = np.array(dates)
    if np.issubdtype(idates.dtype, str):
        idates = pd.to_datetime(pd.Series(idates))
        idates = idates.apply(pd.Timestamp.toordinal).values
    return idates


def dates_to_weekday(dates):
    """
    Get days of the week 
    
    Parameters
    ----------
    dates : array_like
        List of dates of type `int` (ordinal) or `str`

    Returns
    -------
    ndarray
        Array of days of the week encoded as:
        (0 - Mon, 1 - Tue, 2 - Wed, 3 - Thu, 4 - Fri, 5 - Sat, 6 - Sun)

    """
    idates = dates_to_ordinal(dates)
    weekdays = (idates + 6) % 7
    return weekdays


def dates_to_range(dates, pad_full_week=True):
    """
    Get continuos range days 
    
    Parameters
    ----------
    dates : array_like
        List of dates of type `int` (ordinal) or `str`
    pad_full_week : bool, default True
        If True pads range to full weeks for easy reshape to (-1, 7)

    Returns
    -------
    ndarray
        Continuous range of ordinal days

    """
    idates = dates_to_ordinal(dates)
    i0 = idates.min()
    i1 = idates.max()
    if pad_full_week:
        i0 = i0 - dates_to_weekday(i0)
        i1 = i1 - dates_to_weekday(i1) + 6
    idate_range = np.arange(i0, i1 + 1)
    return idate_range



def find_nonzero_intervals(x):
    """
    Find continuous non-zero intervals in 1d-array
    
    Parameters
    ----------
    x : ndarray
        1D array of numeric values

    Returns
    -------
    list of tuples
        List tuples: (begin, end)-indices of non-zero intervals

    """
    x_ = (x > 0).astype(int)
    pad = np.zeros((1))
    x_ = np.concatenate([pad, x_, pad])
    x_ = np.diff(x_)
    idx = np.arange(len(x_))
    i0 = idx[x_ == 1].tolist()
    i1 = idx[x_ == -1].tolist()
    intervals = list(zip(i0, i1))
    return intervals


def unique_sorted(x):
    """
    Unique value sorted most frequent to most rare
    
    Parameters
    ----------
    x : ndarray
        1D array of numeric values

    Returns
    -------
    value : ndarray
        Unique values array
    count : ndarray
        Unique values counts

    """
    x_ = x if np.issubdtype(x.dtype, str) else x[np.isfinite(x)]
    value, count = np.unique(x_, return_counts=True)
    idx = np.argsort(count)[::-1]
    value = value[idx]
    count = count[idx]
    return value, count


