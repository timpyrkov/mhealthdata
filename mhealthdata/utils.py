#!/usr/bin/env python
# -*- coding: utf8 -*-

import numpy as np
import pandas as pd
import pylab as plt
import itertools
import datetime
import calendar
import pytz
import re


def sleep_stage_dict(mode="decode"):
    """
    Gets dictionary to encode/decode sleep stage.
    
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
    Finds all DataFrame column names containing any of the keys.
    
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


def find_intervals(x, tol=0, nmin=None, sort=False):
    """
    Finds continuous positive intervals in 1d-array.
    
    Parameters
    ----------
    x : ndarray
        1D array of non-negative numeric values
    tol : int, default 0
        Gap duration tolerance
    nmin : int or None, default None
        Minimal length of intervals
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
    if nmin is not None:
        duration = np.diff(idx, axis=-1).flatten()
        idx = idx[duration >= nmin]
    if sort:
        duration = np.diff(idx, axis=-1).flatten()
        idx = idx[np.argsort(duration)[::-1]]
    return idx


def fill_gaps(x, gap=None, fill=1):
    """
    Fills zero gaps in 1d-array.
    
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
    Sorts unique values in descending order.
    
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
    Converts DataFrame date/time columns to pandas.Timestamp.

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
            if len(sample) > 5 and sample[-2:] in ["AM", "PM"]:
                 fmt = f"{fmt[:8]} %I:%M:%S %p"
            try:
                df[col] = pd.to_datetime(df[col], format=fmt).dt.tz_localize(None)
            except:
                df[col] = pd.to_datetime(df[col]).dt.tz_localize(None)
        if tz_col is not None:
            df[col] = df.apply(localize, axis=1)            
    return df


def from_ordinal(date, fmt="%Y-%m-%d"):
    """
    Converts ordinal day(s) to date(s), where day 1 = Jan 1st, 1 AD.

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
        date = datetime.datetime.fromordinal(date).strftime(fmt)
    return date


def to_ordinal(date):
    """
    Converts date(s) to ordinal day(s), where day 1 = Jan 1st, 1 AD.
    
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
        date = datetime.datetime.strptime(date[:10], "%Y-%m-%d").toordinal()
    return date


def to_ordinal_day(date):
    """
    Converts date(s) to ordinal day(s), where day 1 = Jan 1st, 1 AD.
    
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
    Converts date(s) to ordinal week(s), where week 1 = Jan 1st-7th, 1 AD.
    
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
    Converts date(s) to ordinal month(s), where month 1 = Jan 1st-31st, 1 AD.
    
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
            date = datetime.datetime.strptime(date[:10], "%Y-%m-%d")
        else:
            date = datetime.datetime.fromordinal(date)
        month = date.month + 12 * (date.year - 1)
    return month


def to_ordinal_year(date):
    """
    Converts date(s) to ordinal year(s).
    
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
        year = datetime.datetime.strptime(date[:10], "%Y-%m-%d").year
    else:
        year = datetime.datetime.fromordinal(date).year
    return year


def to_year_month_day(date):
    """
    Converts date(s) to (arrays of) year, month, day.
    
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
        date = np.array([to_year_month_day(d) for d in date]).T
    else:
        if not isinstance(date, str):
            date = datetime.datetime.fromordinal(date)
            date = date.strftime("%Y-%m-%d")
        date = np.array(date.split("-")).astype(int)
    year, month, day = list(date)
    return year, month, day


def to_range(date, pad_to_full_week=True):
    """
    Encloses date(s) into a continuos range of ordinal dates.
    
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
    Converts date(s) to int day of the week (1 - Monday, 7 - Sunday).
    
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
    Converts date(s) to name(s) of day of the week (Monday - Sunday).
    
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
    Converts date(s) to name(s) of day of the week (Mon - Sun).
    
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
    Converts date(s) to name(s) of month (January - December).
    
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
    Converts date(s) to name(s) of month (Jan - Dec).
    
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
    Converts timezone name to minutes relative to UTC


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
        t0 = datetime.datetime.strptime("2000-01-01 00:00:00+0000", fmt + "%z")
        try:
            try:
                t1 = t0.astimezone(pytz.timezone(tz)).strftime(fmt)
            except:
                t1 = t0.astimezone(pytz.timezone("UTC")).strftime(fmt)
                t0 = "2000-01-01 00:00:00" + re.sub("[a-zA-ZÀ-ž]", "", tz)
                t0 = datetime.datetime.strptime(t0, fmt + "%z")
            t0 = t0.astimezone(pytz.timezone("UTC")).strftime(fmt)
            t1 = datetime.datetime.strptime(t1, fmt)
            t0 = datetime.datetime.strptime(t0, fmt)
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
        h = [datetime.datetime.strptime(f"{t_}", "%H").strftime('%I %p').upper() for t_ in t % 24]
    ax.set_xticks(t * 60, h, **kwargs)
    return ax


def xticks_days(x, ax=None, **kwargs):
    """
    Matplotlib xticks as days, assuming xlim is (0, 1440 x N days) [min].
    
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
        Array or list of dates of type `int` (day 1 = Jan 1st, 1 AD)
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


def smoother(x, window, pad=np.nan, epsilon=1e-10, roll=False):
    """
    Smooth data using running Hann window.

    Parameters
    ----------
    x : ndarray
        Array of equispaced time series data
    window : int,
        Window size
    pad : float, default np.nan
        Value to pad x if roll is False
    epsilon : float, default 1e-10
        Cutoff to account values as non-zeros
    roll : bool, default False
        If True, pad with rolled x, else pad with zeros

    Returns
    -------
    ndarray
        Smoothed time series data

    """
    assert window < x.size
    hann = np.hanning(window)
    x_ = x.flatten()
    n = int(window) // 2
    pad0 = x_[-n:] if roll else np.ones((n)) * pad
    pad1 = x_[:n] if roll else np.ones((n)) * pad
    x_ = np.concatenate([pad0, x_, pad1])
    if np.all(np.isfinite(x)):
        w = np.correlate(x_, hann, mode='same') / np.sum(hann)
        w = w[n:-n].reshape(x.shape)
    else:
        w = np.lib.stride_tricks.sliding_window_view(x_, window)
        s = np.nansum((np.isfinite(w) * hann), 1)
        w = np.nansum(w * hann, 1) / s
        w = w[:x.size].reshape(x.shape)
    w[np.abs(w) < epsilon] = 0
    return w


def window_avg_std(t, x, window=14, smooth=0):
    """
    Calculates running window average and std.

    Parameters
    ----------
    t : ndarray
        1D array of time indices of values
    x : ndarray
        1D array of values
    window : int, default 14
        Window size
    smooth : int, default 0
        Window size to smooth average and std


    Returns
    -------
    t_avg : ndarray
        1D array of time indices of avg / std values
    x_avg : ndarray
        1D array of running window average values
    x_std : ndarray
        1D array of running window std values

    """
    n = t[-1] - t[0] + 1 + 2 * window
    pad = np.zeros((window)) * np.nan
    x_ = np.concatenate([pad, x, pad])
    t_avg = np.arange(n) + t[0] - window
    x_avg = np.zeros((n)) * np.nan
    x_std = np.zeros((n)) * np.nan
    for i in range(n):
        x_avg[i] = np.nanmean(x_[i-window//2:i+window//2+1])
        x_std[i] = np.nanstd(x_[i-window//2:i+window//2+1])
    if smooth:
        x_avg = smoother(x_avg, window=smooth)[window//2:][:n-2*window]
        x_std = smoother(x_std, window=smooth)[window//2:][:n-2*window]
    t_avg = t_avg[window:][:n-2*window]
    x_avg = x_avg[window:][:n-2*window]
    x_std = x_std[window:][:n-2*window]
    return t_avg, x_avg, x_std


def _remove_nonlocal_peaks(x, idx, window):
    """
    Removes redundant and non-local (edge) peaks.

    Parameters
    ----------
    x : ndarray
        Array of equispaced time series data
    idx : ndarray
        Indices of peaks
    window : int,
        Window size

    Returns
    -------
    ndarray
        Indices of true local peaks

    """
    def is_max(val, arr):
        return val == np.nanmax(arr) and val > np.nanmin(arr)
    idxs = []
    x_ = x.flatten()
    n = len(x_)
    for i in np.unique(idx):
        i0 = max(0, i - window // 2)
        i1 = min(n, i + window // 2)
        if len(idxs) > 0 and idxs[-1] > i - window // 2:
            continue
        if (i > 0) & (i < n) & is_max(x_[i], x_[i0:i+1]) & is_max(x_[i], x_[i:i1]):
            idxs.append(i)
    idxs = np.array(idxs).astype(int)
    return idxs


def series_peaks(x, window, smooth=False):
    """
    Finds local peaks in array of time series.

    Notes
    -----
    - NaN, Inf values NOT allowed

    Parameters
    ----------
    x : ndarray
        Array of equispaced time series data
    window : int
        Window size
    smooth : bool, default False
        If True - apply Hann window averaging smooth

    Returns
    -------
    idx : ndarray
        1D array of peak coordinate(s)
    score : ndarray
        1D array of peak height(s)

    """
    s = smoother(x, window) if smooth else x
    w = np.lib.stride_tricks.sliding_window_view(s, window)[::window//2]
    m, n = w.shape
    idx = np.argmax(w, axis=1)
    idx = idx + np.arange(len(w)) * (window // 2)
    idx = idx[(idx >= 0) & (idx < len(x))]
    idx = _remove_nonlocal_peaks(s, idx, window)
    score = s[idx]
    return idx, score


def histogram_peaks(x, bins=100, smooth=False):
    """
    Finds local peaks in histogram of values.

    Parameters
    ----------
    x : ndarray
        Array of data values
    bins : int or ndarray, default 100
        Histogram bins or number of bins
    smooth : bool, default False
        If True - apply Hann window averaging smooth

    Returns
    -------
    idx : ndarray
        1D array of peak coordinate(s)
    score : ndarray
        1D array of peak height(s)

    """
    if isinstance(bins, int):
        x1 = np.nanmin(x)
        x2 = np.nanmax(x)
        bins = np.linspace(x1, x2, bins)
    dx = np.diff(bins)[0]
    x_std = np.nanstd(x)
    window = int(x_std / dx)
    s = np.histogram(x, bins)[0]
    idx, score = series_peaks(s, window, smooth)
    idx = bins[idx]
    return idx, score


def window_sigmoid(n, m=None):
    """
    Generates sigmoid window (-1 to 1) of length n and width m.

    Parameters
    ----------
    n : int
        Window length
    m : int or None, default None
        Window width, if None, m = int(max(1, 0.2 * n))

    Returns
    -------
    ndarray
        1D array of sigmoid window

    """
    n = int(n)
    m = int(max(1, 0.2*n)) if m is None else int(m)
    t = np.arange(n) - (n-1) / 2
    w = 1 - 2 / (1 + np.exp(t / m))
    return np.round(w,2)


def window_boxcar(n, m=None):
    """
    Generates boxcar window (-1 to 1) of length n and width m.

    Parameters
    ----------
    n : int
        Window length
    m : int or None, default None
        Window width, if None, m = int(max(1, 0.2 * n))

    Returns
    -------
    ndarray
        1D array of boxcar window

    """
    n = int(n)
    m = int(max(1, 0.2*n)) if m is None else int(m)
    l = (n - m + (n + m) % 2) // 2
    w = -1 * np.ones((n))
    w[l:-l] = 1
    return w


def calc_covariance(x, window):
    """
    Calculates covariance with running window.

    Parameters
    ----------
    x : ndarray
        1D array of data points
    window : ndarray
        1D window array

    Returns
    -------
    ndarray
        1D array of covariance

    """
    n = len(window)
    pad = np.zeros((n//2)) * np.nan
    x_ = np.concatenate([pad, x, pad])
    x_ = np.lib.stride_tricks.sliding_window_view(x_, n).T
    x_ = (x_ - np.nanmean(x_, 0))
    cov = np.nanmean(x_[window > 0], 0) 
    cov = cov - np.nanmean(x_[window < 0], 0)
    n0 = np.sum(np.isfinite(x_[window < 0]),0)
    n1 = np.sum(np.isfinite(x_[window > 0]),0)
    mask = (n0 >= np.sum(window < 0) * 2 / 3) & (n1 >= np.sum(window > 0) * 2 / 3)
    cov[~mask] = np.nan
    cov = cov[:x.size]
    return cov


def calc_interpolation(x):
    """
    Linearly interpolates data array.

    Parameters
    ----------
    x : ndarray
        1D array of data points

    Returns
    -------
    ndarray
        1D array of interpolated data

    """
    x_ = np.zeros_like(x)
    mask = np.isfinite(x)
    if np.any(mask):
        x0 = x[mask][0]
        x1 = x[mask][-1]
        t = np.arange(len(x))
        x_ = np.interp(t, t[mask], x[mask], x0, x1)
    return x_


def anomaly_detection(x, wlen=10, wtype="step", cutoff=None):
    """
    Detects anomalies using running step or boxcar window.

    Parameters
    ----------
    x : ndarray
        1D array of data points
    wlen : int, default 10
        Window length
    wtype : {"step", "box"}, default "box"
        Window type
    cutoff : float or None, default None
        Cuoff scale for std

    Returns
    -------
    ndarray
        1D array of anomaly indices

    """
    if x.size < wlen:
        idx = np.array([])
    else:
        wlen = int(wlen)
        wfunc = {"step": window_sigmoid, "box": window_boxcar}
        window = wfunc[wtype](wlen)
        cov = calc_covariance(x, window)
        cov = calc_interpolation(cov)
        idx = series_peaks(cov, int(wlen * 2 / 3))
        std = np.nanstd(cov)
        if cutoff is not None:
            idx = idx[cov[idx] >= std * cutoff]
    return idx


def calc_cadence(steps):
    """
    Calculates walking and running cadence (steps/min).

    Parameters
    ----------
    steps : ndarray
        Array of equispaced time series data

    Returns
    -------
    walk : float
        Walking cadence
    run : float
        Running cadence

    """
    # histogram of steps
    bins = np.linspace(0,255,256)
    hist = np.histogram(steps, bins)[0]
    hist[0] = 0
    hist = np.log(hist+1)
    # log-linear fit in the scale-invariant range 20-60 steps/min
    x, y = bins[:-1], hist
    mask = (x >= 20) & (x <= 60)
    p = np.polyfit(x[mask], y[mask], 1)
    pred = np.polyval(p, x)
    pred[pred < 0] = 0
    # subtract scale-invariant trend to contrast peaks
    diff = hist - pred
    diff[diff < 0] = 0
    # find local maxima
    idx = series_peaks(diff, window=40)
    walk, run = np.nan, np.nan
    if len(idx):
        walk = idx[np.argmin(np.abs(idx - 110))]
        run = idx[np.argmin(np.abs(idx - 160))]
        run = run if run > walk else np.nan
        walk = walk if walk > 60 else np.nan
    return walk, run


def impute_bpm(bpm, tol=15):
    """
    Impputes short bpm gaps (defaukt 15 min) by linear interpolation.
    Some devices output bpm every 5 or 10 min. This function imputes 
    short gaps to make them compatible with bpm output every 1 min.

    Parameters
    ----------
    bpm : ndarray
        Array of equispaced time series data (N days x 1440 min)
    tol : int, default 15
        Max length of imputted intervals [minutes]

    Returns
    -------
    ndarray
        Imputed bpm of the same shape as the input bpm

    """

    # flatten to 1D and make placeholder for imputted bpm
    x = bpm.flatten()
    ibpm = np.copy(x)
    # Collect intervals id that should not be interpolated (longer then tol)
    idx = find_intervals(x <= 0, tol=0, nmin=tol)
    # Mask non-zero bpm as reference points for interpolation
    mask = x > 0
    # make interpolation, then turn too long gaps back to zero (using idx)
    if np.any(mask):
        t = np.arange(len(x))
        ibpm = np.interp(t, t[mask], x[mask])
        for i0, i1, in idx:
            ibpm[i0:i1] = 0
    # reshape flattened 1D array back to the shape of the input array
    ibpm = ibpm.reshape(bpm.shape)
    return ibpm


def defragment_sleep(sleep, tol=60):
    """
    Defragments sleep intervals. Some samples have fragmented sleep records.
    This function imputes short gaps (defalut 60 min) to merge a series of 
    fragmented sleep bouts into a countinous sleep interval.

    Parameters
    ----------
    sleep : ndarray
        Array of equispaced time series data (N days x 1440 min)
    tol : int, default 60
        Max length of imputted intervals [minutes]

    Returns
    -------
    ndarray
        Defragmented sleep of the same shape as the input sleep

    """

    # flatten to 1D and make placeholder for imputted sleep
    x = sleep.flatten()
    dsleep = np.zeros_like(x)
    # Collect continuous sleep intervals indices and impute with ones
    idxs = find_intervals(x > 0, tol=tol)
    for i0, i1 in idxs:
        dsleep[i0:i1] = 1
    # reshape flattened 1D array back to the shape of the input array
    dsleep = dsleep.reshape(sleep.shape)
    return dsleep


import types
__all__ = [name for name, thing in globals().items()
          if not (name.startswith('_') or isinstance(thing, types.ModuleType))]
del types

