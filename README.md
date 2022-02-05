[![Python Versions](https://img.shields.io/pypi/pyversions/mhealthdata?style=plastic)](https://pypi.org/project/mhealthdata/)
[![PyPI](https://img.shields.io/pypi/v/mhealthdata?style=plastic)](https://pypi.org/project/mhealthdata/)
[![License](https://img.shields.io/pypi/l/mhealthdata?style=plastic)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://readthedocs.org/projects/mhealthdata/badge/?version=latest)](https://mhealthdata.readthedocs.io/en/latest/?badge=latest)

# mHealthData
## Wearable health data to NumPy
#
# Features
- Read health data export of `Fitbit`, `Samsung Health`, and `Apple Healthkit`
- Read all `.xml`, `.csv`, `.json` to pandas Dataframe
- Fix time zone inconsistency and convert to local time
- Device-centric approach - output numpy arrays for selected wearable devices
- Resample to per-minute or per-day numpy arrays:
    - (N days x 1440 minutes) for `steps`, `sleep`, and `bpm`
    - (N days) for `weight`, `rhr`, and `hrv`

# Installation
```
pip install mhealthdata
```

# Quick start
Assume we have `Fitbit` data export `.zip` downloaded to folder `/Users/username/Downloads/wearable_data/` and unzipped into a subfolder `/Users/username/Downloads/wearable_data/User/` with lots of `.xml`, `.csv`, `.json` and sub-folders inside.

Load data:
```
import mhealthdata

path = '/Users/username/Downloads/wearable_data/User/'
wdata = mhealthdata.FitbitLoader(path)
```
- Use `SHealthLoader()` for loading Samsung Health export 
- Use `HealthkitLoader()` for loading Apple Health export

Show loaded dataframes aswell as `steps` records dataframe:
```
print(wdata.dataframes)
print(wdata.df['steps'].head())
```

# Data Analysis and Visualization
Convert data to numpy arrays:
```
data = wdata.get_device_data()
```
- By default `mhealthdata` truncates `steps/min`, `heart rate/`, and `weight` in [kg] to physically meaningful range `0 - 255`.
- See valid device list: `wdata.devices` 
- Get numpy arrays for specific device e.g. `data = wdata.get_device_data('iPhone')`

Date range:
```
from datetime import datetime

idates = data['idate'] # ordinal days (January 1st of year 1 - is day 1)
dates = [datetime.fromordinal(d) for d in idates]
print(f'Date range {dates[0]} - {dates[-1]}')
```

Plot one day of data: 
```
import pylab as plt

i = 9 # Let us plot day 10 (numbering starts with zero)
plt.figure(facecolor='white')
plt.title(f'Date {dates[i]}')
for dname in ['steps', 'sleep', 'bpm']:
    plt.plot(data[dname][i], label=dname)
plt.legend()
plt.show()
```
- Zero values indicate missing data (also not walking and not sleeping for `steps` and `sleep`)
- By default `mhealthdata` pads data with zeros to match full weeks (Monday through Sunday), so some days at the beginning and at the end may be empty

Data correlations:
```
from scipy.stats import pearsonr

x = data['rhr']
y = data['weight']

# IMPORTANT: zero values indicate missing data and should be disregarded
mask = (x > 0) & (y > 0)
r, p = pearsonr(x[mask], y[mask])
print(f'Correlation {r:.2f}, P-value {p:.2g}')
```
- Missing data are a certaing problem in wearable data analysis
- A study [Pyrkov T.V. et al., Nat Comms 12, 2765 (2021)](https://www.nature.com/articles/s41467-021-23014-1) shows high consistency of recovery rates in quite different biological signals - physical activity measured by consumer wearable devices and laboratory blood cell counts. The typical recovery time of 1-2 weeks. The finding suggests it may be safe to use averaging windows or impute data gaps of several day length (though both affect noise and correlation and therefore should be used with caution).



# Documentation

[https://mhealthdata.readthedocs.io](https://mhealthdata.readthedocs.io)