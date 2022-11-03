#!/usr/bin/env python
# -*- coding: utf8 -*-

import numpy as np
import pandas as pd
import itertools
import pylab as plt
from pytz import timezone
from datetime import datetime
import calendar
import re


def sleep_stage_dict(mode="decode"):
    """
    Get dictionary to encode/decode sleep stage.
    
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
        "unknown": 2,
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
    Find all DataFrame column names containing any of the keys.
    
    Parameters
    ----------
    df : DataFrame
        DataFrame
    keys : array_like
        List of keys of `str` type

    Returns
    -------
    list
        List of column names

    """
    keys = keys if keys is not None else []
    keys = keys if isinstance(keys, list) else [keys]
    col = [c for c in df.columns if any([k.lower() in c.lower() for k in keys])]
    return col


def find_intervals(x, tol=0, sort=False):
    """
    Find continuous positive intervals in 1d-array.
    
    Parameters
    ----------
    x : ndarray
        1D array of non-negative numeric values
    tol : int, default 0
        Gap duration tolerance
    sort : bool, default False
        If False - keep interval order by index in the array 
        If True - sort descending by interval duration

    Returns
    -------
    ndarray
        2D array - N intervals x 2 indices (start, end)

    """
    assert x.ndim == 1 and not any(x < 0)
    x_ = fill_gaps(x, tol)
    x_ = (x_ > 0).astype(int)
    pad = np.zeros((1))
    x_ = np.concatenate([pad, x_, pad])
    x_ = np.diff(x_)
    idx = np.arange(len(x_))
    i0 = idx[x_ == 1]
    i1 = idx[x_ == -1]
    idx = np.stack([i0,i1]).T
    if sort:
        duration = np.diff(idx, axis=-1).flatten()
        idx = idx[np.argsort(duration)[::-1]]
    return idx


def fill_gaps(x, gap=None, fill=1):
    """
    Fill zero gaps in 1d-array.
    
    Parameters
    ----------
    x : ndarray
        1D array of non-negative numeric values
    gap : int or None, default None
        Max gap duration (if None - fill all gaps)
    fill : float, default 1
        Value to fill zeros

    Returns
    -------
    ndarray
        1D array - arrray with all gaps <= gap filled

    """
    assert x.ndim == 1 and not any(x < 0)
    x_ = np.copy(x)
    if gap is None:
        x_[~(x>0)] = fill
    elif gap > 0:
        x__ = (np.nan_to_num(x) == 0).astype(int)
        idx = find_intervals(x__)
        duration = np.diff(idx, axis=-1).flatten()
        idx = idx[duration <= gap]
        for i0, i1 in idx:
            x_[i0:i1] = fill
    return x_


def unique_sorted(x, return_dict=False):
    """
    Unique values sorted descending.
    
    Parameters
    ----------
    x : ndarray
        1D array of numeric values
    return_dict : bool, default False
        If False - return value, count arrays
        If True - return dict value -> count

    Returns
    -------
    value : ndarray, optional
        Unique values array (if return_dict is False)
    count : ndarray, optional
        Unique values counts (if return_dict is False)
    dict : dict, optional
        Dictionary unique value -> count (if return_dict is True)

    """
    x_ = x
    if isinstance(x, list):
        if isinstance(x[0], list):
            x_ = list(itertools.chain(*x))
            x_ = np.array(x_)
        else:
            x_ = np.array(x)
    x_ = x_[np.isfinite(x_)] if np.issubdtype(x_.dtype, float) else x_
    value, count = np.unique(x_, return_counts=True)
    idx = np.argsort(count)[::-1]
    value = value[idx]
    count = count[idx]
    if return_dict:
        return dict(zip(value, count))
    return value, count


def columns_to_datetime(df, tstart, tend=None, tz=None):
    """
    Convert DataFrame date/time columns to pandas.Timestamp.

    Parameters
    ----------
    df : DataFrame
        DataFrame
    columns : array_like
        List of columns to convert into datetime
    tz_col : str, optional
        Time zone column

    Returns
    -------
    DataFrame
        DataFrame with selected columns converted to pandas.Timestamp

    """
    def localize(row):
        fmt = "%Y-%m-%d %H:%M:%S"
        t = row[col].tz_localize("UTC").tz_convert(row[tz_col])
        t = pd.to_datetime(t.strftime(fmt), format=fmt)
        return t

    columns = find_columns_by_key(df, tstart) + find_columns_by_key(df, tend)
    tz_col = find_columns_by_key(df, tz)
    tz_col = tz_col[0] if len(tz_col) > 0 else None
    # if tz_col is not None:
    #     print("TZ", df[tz_col])
    for col in columns:
        if np.issubdtype(df[col].dtype, np.datetime64):
            continue
        if np.issubdtype(df[col].dtype, np.integer):
            df[col] = pd.to_datetime(df[col], unit="ms").dt.tz_localize(None)
        else:
            sample = df[col].values[0]
            fmt = "%Y-%m-%d %H:%M:%S.%f" if "-" in sample else "%m/%d/%y %H:%M:%S"
            df[col] = pd.to_datetime(df[col], format=fmt).dt.tz_localize(None)
        if tz_col is not None:
            df[col] = df.apply(localize, axis=1)            
    return df


def from_ordinal(date, fmt="%Y-%m-%d"):
    """
    Convert ordinal day(s) to date(s), where 1 is January 1st of year 1.

    Parameters
    ----------
    date : array_like or int
        Day(s) of type `int` (ordinal)
    fmt : str, default "%Y-%m-%d"
        Output date format

    Returns
    -------
    ndarray or str
        Date(s)

    """

    if isinstance(date, (list, np.ndarray)):
        date = np.array([from_ordinal(d) for d in date])
    elif not isinstance(date, str):
        date = datetime.fromordinal(date).strftime(fmt)
    return date


def to_ordinal(date):
    """
    Convert date(s) to ordinal day(s), where January 1st of year 1 is 1.
    
    Parameters
    ----------
    date : array_like or str
        Date(s) of type `str`

    Returns
    -------
    ndarray or int
        Ordinal day(s)

    """
    if isinstance(date, (list, np.ndarray)):
        date = np.array(date)
        if np.issubdtype(date.dtype, str):
            try:
                date = pd.to_datetime(pd.Series(date))
                date = date.apply(pd.Timestamp.toordinal).values
            except:
                date = np.array([to_ordinal(d) for d in date])
    elif isinstance(date, str):
        date = datetime.strptime(date[:10], "%Y-%m-%d").toordinal()
    return date


def to_ordinal_day(date):
    """
    Convert date(s) to ordinal day(s), where January 1st of year 1 is 1.
    
    Parameters
    ----------
    date : array_like or str
        Date(s) of type `str`

    Returns
    -------
    ndarray or int
        Ordinal day(s)

    """
    return to_ordinal(date)


def to_ordinal_week(date):
    """
    Convert date(s) to ordinal week(s), where January 1st-7th of year 1 is 1.
    
    Parameters
    ----------
    date : array_like or str
        Date(s) of type `str`

    Returns
    -------
    ndarray or int
        Ordinal week(s)

    """
    return (to_ordinal(date) + 6) // 7


def to_ordinal_month(date):
    """
    Convert date(s) to ordinal month(s), where January of year 1 is 1.
    
    Parameters
    ----------
    date : array_like or str
        Date(s) of type `str`

    Returns
    -------
    ndarray or int
        Ordinal month(s)

    """
    if isinstance(date, (list, np.ndarray)):
        month = np.array([ordinal_month(d) for d in date])
    else:
        if isinstance(date, str):
            date = datetime.strptime(date[:10], "%Y-%m-%d")
        else:
            date = datetime.fromordinal(date)
        month = date.month + 12 * (date.year - 1)
    return month


def to_ordinal_year(date):
    """
    Convert date(s) to ordinal year(s).
    
    Parameters
    ----------
    date : array_like or str
        Date(s) of type `str`

    Returns
    -------
    ndarray or int
        Ordinal year(s)

    """
    if isinstance(date, (list, np.ndarray)):
        year = np.array([ordinal_year(d) for d in date])
    elif isinstance(date, str):
        year = datetime.strptime(date[:10], "%Y-%m-%d").year
    else:
        year = datetime.fromordinal(date).year
    return year


def to_year_month_day(date):
    """
    Convert date(s) to (arrays of) year, month, day.
    
    Parameters
    ----------
    date : array_like or str
        Date(s) of type `str`

    Returns
    -------
    year : ndarray or int
        Year(s)
    month : ndarray or int
        Month(s) of the year
    day : ndarray or int
        Day(s) of the month

    """
    if isinstance(date, (list, np.ndarray)):
        date = np.array([year_month_day(d) for d in date]).T
    else:
        if not isinstance(date, str):
            date = datetime.fromordinal(date)
            date = date.strftime("%Y-%m-%d")
        date = np.array(date.split("-")).astype(int)
    year, month, day = list(date)
    return year, month, day


def to_range(date, pad_to_full_week=True):
    """
    Enclose date(s) in a continuos range of ordinal dates.
    
    Parameters
    ----------
    date : array_like
        List of dates of type `int` (ordinal) or `str`
    pad_to_full_week : bool, default True
        If True - pad range to full weeks so that its length % 7 == 0

    Returns
    -------
    ndarray
        Continuous range of ordinal days

    """
    try:
        idate = to_ordinal(date)
        i0 = idate.min()
        i1 = idate.max()
        if pad_to_full_week:
            i0 = i0 - to_weekdayiso(i0) + 1
            i1 = i1 - to_weekdayiso(i1) + 7
        idate = np.arange(i0, i1 + 1)
    except ValueError:
        idate = np.array([])
    return idate


def to_weekdayiso(date):
    """
    Int day of the week, where Monday is 1 and Sunday is 7.
    
    Parameters
    ----------
    date : array_like or str or int
        Date(s)

    Returns
    -------
    ndarray or int
        Day(s) of the week

    """
    idate = to_ordinal(date)
    weekday = (idate + 6) % 7 + 1
    return weekday


def to_weekdayiso_name(date):
    """
    Name of day of the week, where 1 is Monday and 7 is Sunday.
    
    Parameters
    ----------
    date : array_like or str or int
        Date(s)

    Returns
    -------
    ndarray or str
        Day(s) of the week

    """
    d = list(calendar.day_name)
    d = dict(zip(np.arange(len(d)), np.array(d)))
    i = (to_ordinal(date) - 1) % 7
    w = np.vectorize(d.get)(i) if isinstance(i, np.ndarray) else d[i]
    return w


def to_weekdayiso_abbr(date):
    """
    Abbreviation of day of the week, where 1 is Mon and 7 is Sun.
    
    Parameters
    ----------
    date : array_like or str or int
        Date(s)

    Returns
    -------
    ndarray or str
        Day(s) of the week

    """
    d = list(calendar.day_abbr)
    d = dict(zip(np.arange(len(d)), np.array(d)))
    i = (to_ordinal(date) - 1) % 7
    w = np.vectorize(d.get)(i) if isinstance(i, np.ndarray) else d[i]
    return w


def to_month_name(date):
    """
    Name of month, where 1 is January and 12 is December.
    
    Parameters
    ----------
    date : array_like or str or int
        Date(s)

    Returns
    -------
    ndarray or str
        Month(s)

    """
    d = list(calendar.month_name)
    d = dict(zip(np.arange(len(d)), np.array(d)))
    i = (to_ordinal_month(date) - 1) % 12 + 1
    m = np.vectorize(d.get)(i) if isinstance(i, np.ndarray) else d[i]
    return m


def to_month_abbr(date):
    """
    Abbreviation of month, where 1 is Jan and 12 is Dec.
    
    Parameters
    ----------
    date : array_like or str or int
        Date(s)

    Returns
    -------
    ndarray or str
        Month(s)

    """
    d = list(calendar.month_abbr)
    d = dict(zip(np.arange(len(d)), np.array(d)))
    i = (to_ordinal_month(date) - 1) % 12 + 1
    m = np.vectorize(d.get)(i) if isinstance(i, np.ndarray) else d[i]
    return m


def timezone_txt_to_minutes(tz):
    """
    Timezone name to minutes relative to UTC


    Parameters
    ----------
    tz : str
        Timezone, e.g. "Europe/Madrid" or "UTC+0100"

    Returns
    -------
    int
        Minutes

    """
    if isinstance(tz, (list, np.ndarray)):
        dt = np.array([timezone_txt_to_minutes(t) for t in tz]).T
    else:
        fmt = "%Y-%m-%d %H:%M:%S"
        t0 = datetime.strptime("2000-01-01 00:00:00+0000", fmt + "%z")
        try:
            try:
                t1 = t0.astimezone(timezone(tz)).strftime(fmt)
            except:
                t1 = t0.astimezone(timezone("UTC")).strftime(fmt)
                t0 = "2000-01-01 00:00:00" + re.sub("[a-zA-ZÀ-ž]", "", tz)
                t0 = datetime.strptime(t0, fmt + "%z")
            t0 = t0.astimezone(timezone("UTC")).strftime(fmt)
            t1 = datetime.strptime(t1, fmt)
            t0 = datetime.strptime(t0, fmt)
            dt = (t1 - t0).total_seconds() // 60
        except:
            dt = np.nan
    return np.round(dt)


def xticks_hours(dt=1, mode="24H", ax=None, **kwargs):
    """
    Matplotlib xticks as hours, assuming xlim is (0,1440) [min/day].
    
    Parameters
    ----------
    dt : int, default 1
        stride, hours
    mode : {"24H", "12H"}, default "24H"
        Time format
    ax : matplotlib.pyplot.Axes object, default None
        Axes for plotting
    **kwargs
        Keyword arguments

    Returns
    -------
    ax : matplotlib.pyplot.Axes object
        Axes for plotting

    """
    if ax is None:
        ax = plt.gca()
    t = np.arange(25)[::dt]
    h = t.astype(str)
    if mode.upper() == "12H":
        h = [datetime.strptime(f"{t_}", "%H").strftime('%I %p').upper() for t_ in t % 24]
    ax.set_xticks(t * 60, h, **kwargs)
    return ax


def xticks_days(x, ax=None, **kwargs):
    """
    Matplotlib xticks as days, assuming xlim is (0,1440 x N days) [min].
    
    Parameters
    ----------
    x : array_like
        Data, to infer number of days
    ax : matplotlib.pyplot.Axes object, default None
        Axes for plotting
    **kwargs
        Keyword arguments
        
    Returns
    -------
    ax : matplotlib.pyplot.Axes object
        Axes for plotting

    """
    if ax is None:
        ax = plt.gca()
    n = len(x) // 1440
    t = np.arange(n + 1)
    ax.set_xticks(t * 1440, t, **kwargs)
    return ax


def xticks_dates(idate, mode="day", ax=None, **kwargs):
    """
    Matplotlib xticks as date(s).
    
    Parameters
    ----------
    idate : array_like
        Array or list of dates of type `int` (January 1 of year 1 is day 1)
    mode : {"day", "week", "fortnight", "month"}, default "day"
        Date spacing
    ax : matplotlib.pyplot.Axes object, default None
        Axes for plotting
    **kwargs
        Keyword arguments
        
    Returns
    -------
    ax : matplotlib.pyplot.Axes object
        Axes for plotting

    """
    assert mode in ["day", "week", "fortnight", "month", "year", 
                    "día", "semana", "quincena", "mes", "año"]
    if ax is None:
        ax = plt.gca()
    t = np.copy(idate)
    if mode in ["año", "year"]:
        _, month, day = to_year_month_day(t)
        t = t[(day == 1) & (month == 1)]
    if mode in ["mes", "month"]:
        day = to_year_month_day(t)[-1]
        t = t[day == 1]
    if mode in ["quincena", "fortnight"]:
        day = to_year_month_day(t)[-1]
        t = t[(day == 1) | (day == 15)]
    if mode in ["semana", "week"]:
        day = to_weekdayiso(t)
        t = t[day == 1]
    date = from_ordinal(t)
    if "rotation" not in kwargs:
        kwargs["rotation"] = 45
    if "ha" not in kwargs:
        kwargs["ha"] = "right"
    ax.set_xticks(t, date, **kwargs)
    return ax


