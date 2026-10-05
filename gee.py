"""Implementation of the wildfire gas analysis module."""
from __future__ import annotations
from .english import tr as _best
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from .config import GASES, AnalysisSettings

def launch(project: Optional[str]=None):
    import ee
    try:
        ee.Initialize(project=project) if project else ee.Initialize()
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project) if project else ee.Initialize()
    return ee

def ee_geometry(geom):
    import ee
    return ee.Geometry(geom.__geo_interface__)

def _daily_series(col, geom, start, n_day, reducer, scale, value_fn):
    import ee
    start_ee = ee.Date(pd.Timestamp(start).strftime('%Y-%m-%d'))

    def day(d):
        t0 = start_ee.advance(d, 'day')
        c = col.filterDate(t0, t0.advance(1, 'day'))
        value = ee.Algorithms.If(c.size().gt(0), value_fn(c).reduceRegion(reducer=reducer, geometry=geom, scale=scale, bestEffort=True, maxPixels=1000000000.0), ee.Dictionary({}))
        return ee.Feature(None, ee.Dictionary(value).set('t', t0.millis()))
    fc = ee.FeatureCollection(ee.List.sequence(0, n_day - 1).map(day))
    record = [f['properties'] for f in fc.getInfo()['features']]
    df = pd.DataFrame(record)
    df['date'] = pd.to_datetime(df.pop('t'), unit='ms').dt.normalize()
    return df.set_index('date').sort_index()

def s5p_series(geom_ee, start, end, settings: AnalysisSettings, gases=None) -> pd.DataFrame:
    """Implementation of s5p_series."""
    import ee
    gases = gases or list(GASES)
    n = int((pd.Timestamp(end) - pd.Timestamp(start)).days) + 1
    red = ee.Reducer.mean().combine(ee.Reducer.max(), sharedInputs=True)
    output = {}
    for g in gases:
        gi = GASES[g]
        column = gi.gee_offl if settings.gee_product.upper() == 'OFFL' else gi.gee_nrti
        col = ee.ImageCollection(column).select(gi.band).filterBounds(geom_ee)
        try:
            df = _daily_series(col, geom_ee, start, n, red, settings.gee_scale, lambda c: c.mean())
            key = f'{gi.band}_{settings.gee_reduction}'
            output[g] = df[key] if key in df else pd.Series(np.nan, index=df.index)
            print(_best(f'  {g}: {output[g].notna().sum()} valid days'))
        except Exception as e:
            print(_best(f'  {g}: data could not be retrieved ({e})'))
    return pd.DataFrame(output).apply(pd.to_numeric, errors='coerce')

def firms_daily(geom_ee, search_start, search_end, buffer_m: float=1000) -> pd.Series:
    """Implementation of firms_daily."""
    import ee
    area = geom_ee.buffer(buffer_m)
    a = pd.Timestamp(search_start).strftime('%Y-%m-%d')
    b = (pd.Timestamp(search_end) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    col = ee.ImageCollection('FIRMS').filterDate(a, b).filterBounds(area).select('T21')

    def number(img):
        n = img.gt(0).selfMask().reduceRegion(ee.Reducer.count(), area, 1000, bestEffort=True, maxPixels=1000000000.0).get('T21')
        return ee.Feature(None, {'t': img.date().millis(), 'n': n})
    record = [f['properties'] for f in ee.FeatureCollection(col.map(number)).getInfo()['features']]
    if not record:
        return pd.Series(dtype=float, name='firms_pixel')
    df = pd.DataFrame(record)
    df['date'] = pd.to_datetime(df['t'], unit='ms').dt.normalize()
    s = pd.to_numeric(df['n'], errors='coerce').fillna(0).groupby(df['date']).sum()
    return s[s > 0].rename('firms_pixel')

def mcd64_burn_days(geom_ee, search_start, search_end) -> pd.Series:
    """Implementation of mcd64_burn_days."""
    import ee
    a = pd.Timestamp(search_start).strftime('%Y-%m-%d')
    b = (pd.Timestamp(search_end) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    col = ee.ImageCollection('MODIS/061/MCD64A1').filterDate(a, b).select('BurnDate')

    def hist(img):
        bd = img.updateMask(img.gt(0))
        h = bd.reduceRegion(ee.Reducer.frequencyHistogram(), geom_ee, 500, bestEffort=True, maxPixels=1000000000.0).get('BurnDate')
        return ee.Feature(None, {'year': img.date().get('year'), 'h': h})
    record = [f['properties'] for f in ee.FeatureCollection(col.map(hist)).getInfo()['features']]
    counter = {}
    for k in record:
        for doy, n in (k.get('h') or {}).items():
            t = pd.Timestamp(int(k['year']), 1, 1) + pd.Timedelta(days=int(float(doy)) - 1)
            counter[t] = counter.get(t, 0) + n
    return pd.Series(counter, name='mcd64_pixel', dtype=float).sort_index()

def era5_weather(geom_ee, start, end) -> pd.DataFrame:
    """Implementation of era5_weather."""
    import ee
    n = int((pd.Timestamp(end) - pd.Timestamp(start)).days) + 1
    bands = ['temperature_2m', 'dewpoint_temperature_2m', 'u_component_of_wind_10m', 'v_component_of_wind_10m']
    col = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR').select(bands)
    df = _daily_series(col, geom_ee, start, n, ee.Reducer.mean(), 11132, lambda c: c.mean())
    if not set(bands) <= set(df.columns):
        return pd.DataFrame()
    T = df['temperature_2m'] - 273.15
    Td = df['dewpoint_temperature_2m'] - 273.15
    rh = 100 * np.exp(17.625 * Td / (243.04 + Td)) / np.exp(17.625 * T / (243.04 + T))
    u, v = (df['u_component_of_wind_10m'], df['v_component_of_wind_10m'])
    return pd.DataFrame({'temperature': T, 'humidity': rh.clip(0, 100), 'wind_speed': np.hypot(u, v), 'wind_direction': (np.rad2deg(np.arctan2(-u, -v)) + 360) % 360})

def peak_map_download(geom_ee, date, gas, settings: AnalysisSettings, target: Path, buffer_m: float=5000) -> Optional[Path]:
    """Implementation of peak_map_download."""
    import ee
    import requests
    gi = GASES[gas]
    column = gi.gee_offl if settings.gee_product.upper() == 'OFFL' else gi.gee_nrti
    t0 = ee.Date(pd.Timestamp(date).strftime('%Y-%m-%d'))
    img = ee.ImageCollection(column).select(gi.band).filterDate(t0, t0.advance(1, 'day')).mean()
    region = geom_ee.buffer(buffer_m).bounds()
    try:
        url = img.clip(region).getDownloadURL({'scale': settings.gee_scale, 'region': region, 'format': 'GEO_TIFF', 'crs': 'EPSG:4326'})
        r = requests.get(url, timeout=180)
        r.raise_for_status()
        target.write_bytes(r.content)
        return target
    except Exception as e:
        print(_best(f'  {gas} map could not be downloaded: {e}'))
        return None
