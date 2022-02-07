#!/usr/bin/env python
# -*- coding: utf8 -*-

from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import xml.etree.ElementTree as ET
import json
import glob
from mhealthdata.utils import *
from mhealthdata.offset import fix_timezone



class DataLoader():
    """
    This is the class from which all loaders inherit.

    DataLoader subclasses make numpy arrays from basic health sensor data, 
    all at local time zone loaded from different mobile health apps.

    Basic health data are:
    - per minute values of "steps", "sleep", and "bpm"
    - per day values of "weight", "rhr", and "hrv"
    - per user values of "dob", "sex", and "height"

    Data can be accessed:
    - as pandas dataframe in self.df dict attribute 
    - as numpy arrays using get_device_data() or save_device_npz() methods

    Methods are device-centric and can be used to get data for any specific device
    connected to your health app aggregater, e.g. iPhone, or Apple Watch, or Fitbit wristband

    Note
    ----
    Other data can be found in data export (e.g. VO2Max), but not processed by DataLoader subclasses.
    In case those data needed, the self.category dict attribute should be modified.

    Attributes
    ----------
    df : dict
        Dictionary of pandas Dataframes of loaded health data for "steps", "bpm", etc.
    categories : dict
        Dictionary of health data categories. Keys used to find files. Attributes: 
        "name" - to rename, "column" - to seek value column in corresponding Dataframe.
    userdata : dict
        Dictionary of "Date-of-birth", "Biological sex", and "Height".
        Other data like country of residence, or phone number, etc. are ignored.
    start_keys : list
        List of keywords to seek for start timestamp column in a Dataframe.
    end_keys : list
        List of keywords to seek for end timestamp column in a Dataframe.
    tz_keys : list
        List of keywords to seek for timezone offset column in a Dataframe.
    dev_col : list
        List of keywords to seek for device id column in a Dataframe.
    path : list
        Path to unzipped local folder containing health app data.

    """

    def __init__(self, path):
        """
        Initialize with loading data from an export folder path.

        Parameters
        ----------
        path : str
            Path to unzipped local folder containing health app data

        """
        self.df = {}
        self.categories = {}
        self.userdata = {}
        self.start_keys = ["start_time", "startTime", "startDate", 
                           "dateTime", "day_time"]
        self.end_keys = ["end_time", "endTime", "endDate"]
        self.tz_keys = ["time_offset"]
        self.dev_col = []
        self.path = path


    @property
    def devices_dict(self):
        """
        Get dictionary of loaded devices identifiers.

        Returns
        -------
        dict
            Dictionary of loaded devices identifiers.
        
        """
        return {"all": ["all"]}


    @property
    def devices(self):
        """
        Get list of loaded devices.

        Returns
        -------
        list
            List of loaded devices.
        
        """
        dev = self.devices_dict
        return list(dev.keys())


    @property
    def dataframes(self):
        """
        Get list of loaded dataframes.
        
        Returns
        -------
        list
            List of loaded dataframes.

        """
        return list(self.df.keys())


    @property
    def all_categories(self):
        """
        Get list of all data categories found (not all loaded).
        
        Returns
        -------
        list
            List of all data categories found in provided path.

        """
        categories = []
        return categories


    @staticmethod
    def _special_cases(df, category):
        """
        Private method to process special cases during loading of data.
        
        Parameters
        ----------
        df : Dataframe
            Dataframes of loaded health data for "steps", "bpm", etc.
        category : str
            Key used to find health data files.

        Returns
        -------
        Dataframe
            Dataframe with applied health app-specific fixes.

        """
        return df


    @staticmethod
    def _get_values(df, category):
        """
        Private method to get record value arrays from loaded datframes.
        
        Parameters
        ----------
        df : Dataframe
            Dataframes of loaded health data for "steps", "bpm", etc.
        category : str
            Key used to find health data files.

        Returns
        -------
        ndarray
            1D array of record values, e.g. sleep stage, or stepcounts, etc.

        """
        values = df.values
        if "sleep" in category.lower():
            s = sleep_stage_dict(mode="encode")
            values = np.vectorize(s.get)(values)
        values = values.astype(float)
        return values

    @staticmethod
    def _get_time(df, keys):
        """
        Private method to get record timestamps from loaded datframes.
        
        Parameters
        ----------
        df : Dataframe
            Dataframes of loaded health data for "steps", "bpm", etc.
        keys : list
            List of keywords to seek for a column in a Dataframe.

        Returns
        -------
        ndarray
            1D array of datetime64.

        """
        col = find_columns_by_key(df, keys)
        t = df[col[0]] if len(col) > 0 else None
        return t

    @staticmethod
    def _get_idate_imin(t):
        """
        Private method to get record ordinal days and minutes from loaded datframes.
        
        Parameters
        ----------
        t : ndarray
            1D array of datetime64.

        Returns
        -------
        idate : ndarray
            1D array of ordinal days (January 1 of year 1 is day 1).
        imin : ndarray
            1D array of minutes since midnight

        """
        idate = t.apply(pd.Timestamp.toordinal).values
        imin = (60 * t.dt.hour + t.dt.minute).values
        return idate, imin

    @staticmethod
    def _get_duration(df, duration=None):
        """
        Private method to get record durations [minutes] from loaded datframes.
        
        Parameters
        ----------
        df : Dataframe
            Dataframes of loaded health data for "steps", "bpm", etc.
        duration : ndarray or None, default None
            Initialized record durations [minutes] 1D array.
            If None, will be initialized with np.ones().

        Returns
        -------
        ndarray
            1D array record durations [minutes]

        """
        if duration is None:
            duration = np.ones((df.shape[0])).astype(int)
        if "binning_period" in df.columns:
            duration = df["binning_period"].values.astype(int)
        elif "seconds" in df.columns:
            duration = (df["seconds"].values + 1) / 60.0
            duration = np.round(duration).astype(int)
        duration[duration < 1] = 1
        return duration

    @staticmethod
    def _calc_duration(t0, t1, duration=None):
        """
        Private method to calc record durations from start and end timestamps.
        
        Parameters
        ----------
        t0 : ndarray
            1D array of start datetime64.
        t0 : ndarray
            1D array of end datetime64.
            If None, the output will be np.ones() or duration (if not None).
        duration : ndarray or None, default None
            Initialized record durations [minutes] 1D array.
            If None, will be initialized with np.ones().

        Returns
        -------
        ndarray
            1D array record durations [minutes].

        """
        if duration is None:
            duration = np.ones((len(t0))).astype(int)
        if t1 is not None:
            duration = (t1 - t0).astype("timedelta64[m]").values.astype(int)
        duration[duration < 1] = 1
        return duration


    @staticmethod
    def _values_to_1darray(x, date_range, value, idate):
        """
        Private method to combine records into 1D array.
        Used for per day values like "weight", "rhr", or "hrv".
        
        Parameters
        ----------
        x : ndarray or None
            1D array of values (e.g. np.zeros()) or None.
            If None, will be initialize with np.zeros() based on date_range length.
        date_range : ndarray
            1D array of continuous range of ordinal days.
        value : ndarray
            1D array of record values, e.g. sleep stage, or stepcounts, etc.
        idate : ndarray
            1D array of record ordinal days.

        Returns
        -------
        ndarray
            1D array of values of size (N days).

        """
        n = len(date_range)
        x = np.zeros((n)) if x is None else x
        idx = idate - date_range[0]
        mask = (idx >= 0) & (idx < n)
        idx = idx[mask]
        val = value[mask]
        for k in range(len(val)):
            i = idx[k]
            x[i] = val[k]
        return x


    @staticmethod
    def _values_to_2darray(x, date_range, value, idate, imin, dt, mode="rate"):
        """
        Private method to combine records into 2D array.
        Used for per minute values like "steps", "sleep", or "bpm".
        
        Parameters
        ----------
        x : ndarray or None
            1D array of values (e.g. np.zeros()) or None.
            If None, will be initialize with np.zeros() based on date_range length.
        date_range : ndarray
            1D array of continuous range of ordinal days.
        value : ndarray
            1D array of record values, e.g. sleep stage, or stepcounts, etc.
        idate : ndarray
            1D array of record ordinal days.
        imin : ndarray
            1D array of record minutes since midnight.
        dt : ndarray
            1D array of record durations [minutes].
        mode : {"rate", "count"}, default "rate"
            If "rate", the original value shall be assigned to all points.
            If "count", the evenly distributed value shall be assigned to all record minues.

        Returns
        -------
        ndarray
            2D array of values of size (N days x 1440 minutes).

        """
        assert mode in ["rate", "count"]
        n = 1440 * len(date_range)
        x = np.zeros((n)) if x is None else x.flatten()
        idx = 1440 * (idate - date_range[0]) + imin
        mask = (idx >= 0) & (idx < n)
        idx = idx[mask]
        val = value[mask]
        for k in range(len(val)):
            i = idx[k]
            j = i + dt[k]
            if mode == "count":
                x[i:j] = val[k] / dt[k]
            else:
                x[i:j] = val[k]
        x = x.reshape(-1,1440)
        return x

    @staticmethod
    def _get_device_slice(df, uuids, dev_col):
        """
        Private method to get Dataframe slice for specified device.
        
        Parameters
        ----------
        df : Dataframe
            Dataframes of loaded health data for "steps", "bpm", etc.
        uuids : list
            List of device identifiers.
        dev_col : list
            List of keywords to seek for device id column in a Dataframe.

        Returns
        -------
        Dataframe
            Dataframe slice matching mask of specified device identifiers.

        """
        mask = np.zeros((df.shape[0])).astype(bool)
        deviceuuid = find_columns_by_key(df, dev_col)
        if len(deviceuuid) > 0:
            deviceuuid = df[deviceuuid[0]].values.astype(str)
            for uuid in uuids:
                mask[deviceuuid == uuid] = True
        return df[mask]


    def _parse_userdata(self):
        """
        Private method to retrieve "Date-of-birth", "Biological sex", and "Height".
        Other data like country of residence, or phone number, etc. are ignored.

        Returns
        -------
        list
            List of loaded userdata keys.

        """
        return list(self.userdata.keys())


    def _parse_timestamps(self, df):
        """
        Private method to parse columns of loaded Dataframe to datetime64.
        Time columns are found based on class attributes self.start_keys and self.end_keys.
        
        Parameters
        ----------
        df : Dataframe
            Dataframes of loaded health data for "steps", "bpm", etc.

        Returns
        -------
        Dataframe
            Dataframe with time columns converted to datetime64.

        """
        tkeys = self.start_keys + self.end_keys
        tcol = find_columns_by_key(df, tkeys)
        tz_col = find_columns_by_key(df, self.tz_keys)
        tz_col = tz_col[0] if len(tz_col) > 0 else None
        df = columns_to_datetime(df, tcol, tz_col)
        return df

        
    def _parse_dataframe(self, category, device):
        """
        Private method to extract record data from a (device-sliced) Dataframe.
        
        Parameters
        ----------
        category : str
            Key used to find health data files.
        device : str
            Device name (sould match any one of self.devices).

        Returns
        -------
        value : ndarray
            1D array of record values, e.g. sleep stage, or stepcounts, etc.
        idate : ndarray
            1D array of record ordinal days.
        imin : ndarray
            1D array of record minutes since midnight.
        duration : ndarray
            1D array of record durations [minutes].

        """
        value = idate = imin = duration = np.array([])
        if category in self.df and device in self.devices:
            df = self.df[category]
            if category not in ["weight"] and device not in ["all"]:
                uuids = self.devices_dict[device]
                df = self._get_device_slice(df, uuids, self.dev_col)
            if df.shape[0] > 0:
                column = self.categories[category]["column"]
                value = self._get_values(df[column], category)
                t0, t1 = [self._get_time(df, k) for k in [self.start_keys, self.end_keys]]
                idate, imin = self._get_idate_imin(t0)
                duration = self._get_duration(df)
                duration = self._calc_duration(t0, t1, duration)
        return value, idate, imin, duration


    def _parse_arrays(self, data, dname, date_range, value, idate, imin, dt, trunc=True):
        """
        Private method to combine records into 1D or 2D array.
        Ou for per minute values like "steps", "sleep", or "bpm".
        
        Parameters
        ----------
        data : dict
            Dictionary of ountput ndarrays.
        dname : str
            Renamed data type. For example "steps" for either "pedometer_day_summary" 
            or "HKQuantityTypeIdentifierStepCount", etc.
        date_range : ndarray
            1D array of continuous range of ordinal days.
        value : ndarray
            1D array of record values, e.g. sleep stage, or stepcounts, etc.
        idate : ndarray
            1D array of record ordinal days.
        imin : ndarray
            1D array of record minutes since midnight.
        dt : ndarray
            1D array of record durations [minutes].
        trunc : bool, default True
            If True, truncate outliers to np.uint8 range (0 - 255)

        Returns
        -------
        ndarray
            1D array of size (N days) for "weight", "rhr", or "hrv".
            2D array of size (N days x 1440 minutes) "steps", "sleep", or "bpm".

        """
        x = data[dname] if dname in data else None
        if len(date_range) > 0:
            if dname in ["weight", "rhr"]:
                x = self._values_to_1darray(x, date_range, value, idate) 
            else:
                mode = "count" if dname == "steps" else "rate"
                x = self._values_to_2darray(x, date_range, value, idate, imin, dt, mode)
            if trunc:
                x = np.clip(0,255,x)
        return x


    def get_device_data(self, device="all", date_range=None, trunc=True):
        """
        Get dictionary of per day and per minute ndarrays.
        The method is device-centric and can output data for specified device.
        
        Parameters
        ----------
        device : str, default "all"
            Device name (sould match any one of self.devices).
        date_range : ndarray or None, default None
            1D array of continuous range of ordinal days.
            If None, automatically get from on min and max dates of "steps".
        trunc : bool, default True
            If True, truncate outliers to np.uint8 range (0 - 255)

        Returns
        -------
        dict
            Dictionary of ountput ndarrays.

        """
        data = {}
        if device not in self.devices:
            raise KeyError(f"Wrong device '{device}'. Use 'devices' property to get valid devices.")
        for category in self.categories:
            dname = self.categories[category]["name"]
            value, idate, imin, dt = self._parse_dataframe(category, device)
            try:
                date_range = idate if date_range is None else date_range
                date_range = dates_to_range(date_range)
            except ValueError:
                date_range = np.array([])
            x = self._parse_arrays(data, dname, date_range, value, idate, imin, dt, trunc)
            x = np.array([]) if x is None else x
            data[dname] = x
        data["idate"] = date_range
        return data
    
    
    def save_device_npz(self, output_file, device="all", date_range=None, uint8=False, trunc=True):
        """
        Save dictionary of per day and per minute ndarrays to npz.
        The method is device-centric and can output data for specified device.
        
        Parameters
        ----------
        output_file : str
            Path to output .npz file.
        device : str, default "all"
            Device name (sould match any one of self.devices).
        date_range : ndarray or None, default None
            1D array of continuous range of ordinal days.
            If None, automatically get from on min and max dates of "steps".
        uint8 : bool, default False
            Flag to cast all health data to np.uint8 to save disk space.
        trunc : bool, default True
            If True, truncate outliers to np.uint8 range (0 - 255)

        Returns
        -------
        bool
            True if data ndarrays length not zero, else False.

        """

        data = self.get_device_data(device, date_range, trunc)
        if uint8:
            if not trunc:
                raise  ValueError("When 'uint8' == True, should be 'trunc' = True. Wrong value: 'trunc' = False")
            for d in data:
                data[d] = data[d].astype(np.uint8) if d != "idate" else data[d]
        if data:
            np.savez_compressed(output_file, **data)
        return bool(len(data["idate"]))







class FitbitLoader(DataLoader):
    """
    
    Note
    ----
    One may note that Fitbit exported
    
        - ``sleep`` in local time
        - ``steps`` and ``bpm`` timestamps in UTC 

    FitbitLoader

        - Attempts to infer time zone offset from data mismatch
        - Converts all timestamps to local time, see ``self._fix_timezone()``

    Example
    -------
	Assume we have data export ``MyFitbitData.zip`` downloaded to folder \
    ``/Users/username/Downloads/wearable_data/`` and unzipped into a subfolder \
    ``/Users/username/Downloads/wearable_data/User/``.

    >>> import mhealthdata
    >>> path = '/Users/username/Downloads/wearable_data/User/'
    >>> wdata = mhealthdata.FitbitLoader(path)
    
    """

    def __init__(self, path):
        super().__init__(path)
        self.categories = {
            "steps": {
                "name": "steps", "column": "value"},
            "sleep": {
                "name": "sleep", "column": "level"},
            "heart_rate": {
                "name": "bpm", "column": "value.bpm"},
            "resting_heart_rate": {
                "name": "rhr", "column": "value.value"},
            "weight": {
                "name": "weight", "column": "weight"},
        }
        self.load_data()


    @property
    def all_categories(self):
        fnames = glob.glob(self.path + "/*/*.csv")
        fnames += glob.glob(self.path + "/*/*.json")
        categories = []
        for fname in fnames:
            category = Path(fname).stem.split("-")[0]
            if category not in categories:
                categories.append(category)
        return categories


    @staticmethod
    def _special_cases(df, category):
        if category == "weight":
            df["dateTime"] = df["date"] + " " + df["time"]
            df["weight"] = 0.4536 * df["weight"] # pounds to kg
        return df


    def _parse_userdata(self):
        try:
            fname = glob.glob(self.path + "/*/Profile.csv")[0]
        except IndexError as e:
            return None
        df= pd.read_csv(fname)
        df = self.df["Profile"] = df
        self.userdata["Date of birth"] = df["date_of_birth"].values.astype(str)[0]
        self.userdata["Biological sex"] = df["gender"].values.astype(str)[0]
        self.userdata["Height"] = df["height"].values.astype(str)[0]
        return list(self.userdata.keys())


    def get_device_data(self, device="all", date_range=None, trunc=True):
        data = super().get_device_data(device, date_range, trunc)
        if "sleep" in data and len(data["sleep"]) > 0:
            data = fix_timezone(data)
        return data
    

    def load_sleep(self):
        """
        Load sleep data from .json files.
        Path to seek files is taken from self.path attribute.
        
        Returns
        -------
        Dataframe
            Dataframe of raw data loaded from .csv.

        """
        fnamelist = glob.glob(self.path + "/*/" + "sleep" + "-*")
        if len(fnamelist) == 0:
            return None
        df = []
        for fname in tqdm(fnamelist):
            for json_data in json.load(open(fname)):
                df.append(pd.json_normalize(json_data["levels"]["data"]))
                if "shortData" in json_data["levels"]:
                    df.append(pd.json_normalize(json_data["levels"]["shortData"]))
        df = pd.concat(df, ignore_index=True)
        return df


    def load_nonsleep(self, category):
        """
        Load non-sleep data from .json files.
        Path to seek files is taken from self.path attribute.
        
        Parameters
        ----------
        category : str
            Key used to find health data files.

        Returns
        -------
        Dataframe
            Dataframe of raw data loaded from .csv.

        """
        fnamelist = glob.glob(self.path + "/*/" + category + "-*")
        if len(fnamelist) == 0:
            return None
        df = [pd.json_normalize(json.load(open(fname))) for fname in tqdm(fnamelist)]
        df = pd.concat(df, ignore_index=True) if len(df) > 0 else pd.DataFrame()
        return df


    def load_data(self):
        """
        Load data from .csv and .json files.
        Cycling over category from self.categories attribute.
        Path to seek files is taken from self.path attribute.
        
        Returns
        -------
        list
            List of loiaded Dataframes.

        """
        self.df = {}
        for category in list(self.categories.keys()):
            df = self.load_sleep() if category == "sleep" else self.load_nonsleep(category)
            if df is not None:
                df = self._special_cases(df, category)
                df = self._parse_timestamps(df)
                self.df[category] = df
            elif "step" in category or "pedometer" in category:
                raise FileNotFoundError(f"Wrong 'path'. Cannot find files for '{category}'.")
        self._parse_userdata()
        return self.dataframes









class ShealthLoader(DataLoader):
    """
    
    Note
    ----
    One may note that Samsung Health exported
    
        - ``.json`` (``step`` binning data) timestamps in local time
        - ``.csv`` (``sleep``, ``bpm``, ``weight``) in UTC with \
            additional time zone column ``time_offset``

    ShealthLoader

        - Converts all timestamps to local time, see ``utils.columns_to_datetime()``

    Example
    -------
	Assume we have data export downloaded to folder  \
    ``/Users/username/Downloads/wearable_data/Samsung Health/`` \
    which contains a subfolder ``samsunghealth_<username>_<date-time>``.


 	>>> import mhealthdata
	>>> path = '/Users/username/Downloads/wearable_data/Samsung Health/'
	>>> wdata = mhealthdata.ShealthLoader(path)
    
    """

    def __init__(self, path):
        super().__init__(path)
        self.start_keys = ["start_time"]
        self.dev_col = ["deviceuuid"]
        self.categories = {
            "pedometer_day_summary": {
                "name": "steps", "column": "mStepCount"},
            "step_daily_trend": {
                "name": "steps", "column": "count"},
            "heart_rate": {
                "name": "bpm", "column": "com.samsung.health.heart_rate.heart_rate"},
            "sleep": {
                "name": "sleep", "column": "stage"},
            "sleep_stage": {
                "name": "sleep", "column": "stage"},
            "weight": {
                "name": "weight", "column": "weight"},
        }
        self.load_data()



    @property
    def devices_dict(self):
        def clean_username(s):
            s = " ".join(w for w in s.split() if "'" not in w)
            s = " ".join(w for w in s.split() if "(" not in w)
            return s
        dev = {}
        if "device_profile" in self.df:
            col = ["deviceuuid", "device_group", "name", "model", "fixed_name"]
            df = self.df["device_profile"][col].astype(str)
            dev_group = df["device_group"].values.astype(int).clip(360001)
            dev_uuid = df["deviceuuid"].values.astype(str)
            dev_name = df["fixed_name"].values.astype(str)
            dev_name = np.where(dev_name == "nan", df["model"], dev_name)
            dev_name = np.where(dev_group == 360003, df["name"], dev_name)
            dev_name = [clean_username(name) for name in dev_name]
            dev_name = np.array(dev_name, dtype=str)
            dev["all"] = dev_uuid
            if 360001 in dev_group:
                dev["mobile"] = dev_uuid[dev_group == 360001]
            if 360003 in dev_group:
                dev["wearable"] = dev_uuid[dev_group == 360003]
            for name in np.unique(dev_name):
                dev[name] = dev_uuid[dev_name == name]
        return dev

    
    @property
    def all_categories(self):
        fnames = glob.glob(self.path + "/*/*.csv")
        categories = []
        for fname in fnames:
            category = fname.split(".")
            if len(category) > 3:
                categories.append(category)
        return categories


    @staticmethod
    def _special_cases(df, category):
        if "sleep" in category:
            if "stage" in df.columns:
                s = {40001: "awake", 40002: "light", 40003: "deep", 40004: "rem"}
                df["stage"] = np.vectorize(s.get)(df["stage"].values)
            else:
                df["stage"] = np.array(["no_stage"] * df.shape[0])
        return df
            

    def _parse_userdata(self):
        if "user_profile" in self.df:
            d = self.df["user_profile"]
            d["text_value"].fillna(d["float_value"], inplace=True)
            d.set_index("key", inplace=True)
            d = d[["text_value"]].T.to_dict("list")
            self.userdata["Date of birth"] = d["birth_date"][0]
            self.userdata["Biological sex"] = d["gender"][0]
            self.userdata["Height"] = d["height"][0]
        return list(self.userdata.keys())


    def _binning_dict(self, category, idx):
        """
        Private method to get device-id and date dictionaries for daily .json.
        
        Parameters
        ----------
        category : str
            Key used to find health data files.
        idx : str
            Column name containing binning data file names.

        Returns
        -------
        dev : dict
            Dictionary of device id for binning data file names.
        dat : dict
            Dictionary of dates for binning data file names.

        """
        df = self.load_csv(category)
        df.set_index(idx, inplace=True)
        dat = df["day_time"].to_dict()
        dev = df["deviceuuid"].to_dict()
        return dev, dat


    def load_csv(self, category):
        """
        Load data from .csv file.
        Path to seek files is taken from self.path attribute.
        
        Parameters
        ----------
        category : str
            Key used to find health data files.

        Returns
        -------
        Dataframe
            Dataframe of raw data loaded from .csv.

        """
        try:
            fname = glob.glob(self.path + "/*/*." + category + ".*.csv")[0]
        except IndexError as e:
            return None
        df = pd.read_csv(fname, skiprows=1, index_col=False)
        return df


    def load_jsons(self, category, idx="binning_data"):
        """
        Load data from .json files.
        Path to seek files is taken from self.path attribute.
        
        Parameters
        ----------
        category : str
            Key used to find health data files.
        idx : str, default "binning_data"
            Column name containing binning data file names.

        Returns
        -------
        Dataframe
            Dataframe of raw data loaded from .csv.

        """
        dev, dat = self._binning_dict(category, idx)
        fnamelist = glob.glob(self.path + '/*/jsons/*' + category + '*/*/*.' + idx + '.json')
        if len(fnamelist) == 0:
            return None
        df = []
        for fname in tqdm(fnamelist):
            df_ = pd.json_normalize(json.load(open(fname)))
            nrec, ncol = df_.shape
            if ncol > 0:
                f = fname.split("/")[-1]
                df_["deviceuuid"] = np.array([dev[f]] * nrec)
                df_["start_time"] = dat[f] + 600000 * df_.index
                df_["binning_period"] = 10 * np.ones((nrec)).astype(int)
                df.append(df_)
        df = pd.concat(df, ignore_index=True)
        return df


    def load_data(self):
        """
        Load data from .csv and .json files.
        Cycling over category from self.categories attribute.
        Path to seek files is taken from self.path attribute.
        
        Returns
        -------
        list
            List of loiaded Dataframes.

        """
        self.df = {}
        for category in list(self.categories.keys()) + ["device_profile", "user_profile"]:
            df = self.load_csv(category)
            if "binning_data" in df.columns:
                df = self.load_jsons(category)
            if df is not None:
                df = self._special_cases(df, category)
                df = self._parse_timestamps(df)
                self.df[category] = df
            elif "step" in category or "pedometer" in category:
                raise FileNotFoundError(f"Wrong 'path'. Cannot find files for '{category}'.")
        self._parse_userdata()
        return self.dataframes


    






class HealthkitLoader(DataLoader):
    """

    Example
    -------
	Assume we have data export ``export.zip`` downloaded to folder  \
    ``/Users/username/Downloads/wearable_data/`` and unzipped into a \
    subfolder ``/Users/username/Downloads/wearable_data/apple_health_export/``.


 	>>> import mhealthdata
	>>> path = '/Users/username/Downloads/wearable_data/apple_health_export/'
	>>> wdata = mhealthdata.HealthkitLoader(path)
    
    """

    
    def __init__(self, path):
        super().__init__(path)
        self.dev_col = ["sourceName"]
        self.categories = {
            "HKQuantityTypeIdentifierStepCount": {
                "name": "steps", "column": "value"},
            "HKCategoryTypeIdentifierSleepAnalysis": {
                "name": "sleep", "column": "value"},
            "HKQuantityTypeIdentifierHeartRate": {
                "name": "bpm", "column": "value"},
            "HKQuantityTypeIdentifierRestingHeartRate": {
                "name": "rhr", "column": "value"},
            "HKQuantityTypeIdentifierHeartRateVariabilitySDNN": {
                "name": "hrv", "column": "value"},
            "HKQuantityTypeIdentifierBodyMass": {
                "name": "weight", "column": "value"},
        }
        self.load_data()


    @property
    def devices_dict(self):
        dev = {"all": ["all"]}
        for category in self.df:
            df = self.df[category]
            if "sourceName" in df.columns:
                for d in np.unique(df["sourceName"].values.astype(str)):
                    dev[d] = [d]
        return dev


    @property
    def all_categories(self):
        categories = []
        for tag in ["Record", "Workout"]:
            df = self.df[tag]
            col = find_columns_by_key(df, ["type"])
            if len(col) > 0:
                categories.append(df[col[0]].values.astype(str))
        categories = np.concatenate(categories)
        categories = np.unique(categories).tolist()
        return categories


    @staticmethod
    def _special_cases(df, category):
        def clean_username(s):
            if "iphone" in s.lower():
                s = "iPhone"
            elif "apple" in s.lower() and "watch" in s.lower():
                s = "Apple Watch"
            return s
        if "sourceName" in df.columns:
            df["sourceName"] = df["sourceName"].apply(clean_username)
        if category == "HKCategoryTypeIdentifierSleepAnalysis":
            s = {"HKCategoryValueSleepAnalysisAwake": "awake", 
                 "HKCategoryValueSleepAnalysisInBed": "no_stage", 
                 "HKCategoryValueSleepAnalysisAsleep": "asleep"}
            df["value"] = np.vectorize(s.get)(df["value"].values)
        return df

    def _parse_userdata(self, data):
        if "Me" in data:
            dob = data["Me"]["HKCharacteristicTypeIdentifierDateOfBirth"]
            sex = data["Me"]["HKCharacteristicTypeIdentifierBiologicalSex"]
            self.userdata["Date of birth"] = dob.values.astype(str)[0]
            self.userdata["Biological sex"] = sex.values.astype(str)[0]
        if "Record" in data:
            mask = data["Record"]["type"] == "HKQuantityTypeIdentifierHeight"
            height = data["Record"][mask]
            height = height["value"] + " " + height["unit"]
            self.userdata["Height"] = unique_sorted(height.values.astype(str))[0][0]
        return list(self.userdata.keys())


    def _parse_xml(self, root):
        """
        Private method to parse records from loaded .xml
        Path to seek files is taken from self.path attribute.
        
        Parameters
        ----------
        root
            Root element attribute for .xml tree

        Returns
        -------
        dict
            Dictionary of Dataframes.

        """
        data = {}
        for tag in ["Record", "Workout", "Me"]:
            records = []
            for child in tqdm(root):
                if child.tag == tag:
                    for node in list(child):
                        if node.tag == "MetadataEntry":
                            if node.attrib["key"] == "HKTimeZone":
                                child.attrib["HKTimeZone"] = node.attrib["value"]
                    records.append(child.attrib)
            df = pd.DataFrame(records)
            df = self._parse_timestamps(df)
            data[tag] = df
        return data


    def load_data(self):
        """
        Load data from .csv and .json files.
        Cycling over category from self.categories attribute.
        Path to seek files is taken from self.path attribute.
        
        Returns
        -------
        list
            List of loiaded Dataframes.

        """
        try:
            fname = glob.glob(self.path + "/[eE][xX][pP][oO][rR][tT].[xX][mM][lL]")[0]
        except IndexError as e:
            raise FileNotFoundError(f"Wrong 'path'. Cannot find file 'export.xml'.")
        tree = ET.parse(fname)
        root = tree.getroot()
        data = self._parse_xml(root)
        for tag in data:
            self.df[tag] = data[tag]
        self._parse_userdata(data)
        for category in self.categories:
            mask = data["Record"]["type"] == category
            df = data["Record"][mask].copy()
            df = self._special_cases(df, category)
            self.df[category] = df
        return self.dataframes
        



