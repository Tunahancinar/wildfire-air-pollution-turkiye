"""Implementation of the wildfire gas analysis module."""
from __future__ import annotations
from .english import tr as _best
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from .config import GAS_ALIASES, GASES

def _norm(s: str) -> str:
    s = str(s)
    s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode().lower()
    s = re.sub('\\(.*?\\)', '', s)
    s = re.sub('[^a-z0-9_]', '', s)
    return s

def _table_read(path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in ('.xlsx', '.xls'):
        return pd.read_excel(path)
    with open(path, 'r', encoding='utf-8-sig', errors='ignore') as f:
        first = f.readline()
    sep = ';' if first.count(';') > first.count(',') else ','
    dec = ',' if sep == ';' else '.'
    return pd.read_csv(path, sep=sep, decimal=dec, encoding='utf-8-sig')

def _date_column(df: pd.DataFrame) -> str:
    for c in df.columns:
        if _norm(c) in ('date', 'date', 'time', 'time', 'datetime', 't', 'day', 'day'):
            return c
    for c in df.columns:
        if 'date' in _norm(c) or 'date' in _norm(c):
            return c
    return df.columns[0]

def _date_convert(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_datetime(series, unit='ms')
    return pd.to_datetime(series, dayfirst=True, errors='coerce')

def gas_columns_find(df: pd.DataFrame) -> Dict[str, str]:
    matching = {}
    normalizers = {c: _norm(c) for c in df.columns}
    for gas, names in GAS_ALIASES.items():
        names_n = [_norm(a) for a in names] + [_norm(a) for a in GASES[gas].station_names]
        for c, n in normalizers.items():
            if n in names_n:
                matching[gas] = c
                break
    return matching

def satellite_series_read(path, fire_name: Optional[str]=None) -> pd.DataFrame:
    """Implementation of satellite_series_read."""
    df = _table_read(path)
    ys = next((c for c in df.columns if _norm(c) in ('fire', 'fire_name', 'fire', 'name', 'name')), None)
    if ys is not None and fire_name is not None:
        df = df[df[ys].astype(str) == str(fire_name)]
    tc = _date_column(df)
    gs = gas_columns_find(df)
    if not gs:
        raise ValueError(_best(f'{path}: No gas column found (expected CO, NO2, CH2O/HCHO, SO2 or O3)'))
    out = pd.DataFrame({'date': _date_convert(df[tc]).dt.normalize()})
    for gas, c in gs.items():
        out[gas] = pd.to_numeric(df[c], errors='coerce')
    return out.groupby('date').mean().sort_index()

def read_station(path, daily: str='Mean') -> pd.DataFrame:
    """Implementation of read_station."""
    df = _table_read(path)
    tc = _date_column(df)
    gs = gas_columns_find(df)
    if not gs:
        raise ValueError(_best(f'{path}: No gas column found in the station file'))
    out = pd.DataFrame({'time': _date_convert(df[tc])})
    for gas, c in gs.items():
        v = df[c]
        if v.dtype == object:
            v = v.astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False) if v.astype(str).str.contains(',').any() else v
        out[gas] = pd.to_numeric(v, errors='coerce')
    out['date'] = out['time'].dt.normalize()
    out = out.drop(columns='time')
    g = out.groupby('date')
    return (g.max() if daily == 'max' else g.mean()).sort_index()
_MET = {'temperature': ['temperature', 'temp', 't2m', 'air_temperature', 'airtemp'], 'humidity': ['humidity', 'rh', 'relativehumidity'], 'wind_speed': ['wind_speed', 'windspeed', 'ws'], 'wind_direction': ['wind_direction', 'winddirection', 'wd']}

def weather_read(path) -> pd.DataFrame:
    df = _table_read(path)
    tc = _date_column(df)
    out = pd.DataFrame({'time': _date_convert(df[tc])})
    for k, names in _MET.items():
        name_value = [_norm(a) for a in names]
        for c in df.columns:
            if _norm(c) in name_value:
                out[k] = pd.to_numeric(df[c], errors='coerce')
                break
    return out
_ENV = {'name': ['fire', 'fire_name', 'name', 'name', 'name', 'fire', 'fire_name', 'location', 'location', 'district'], 'start': ['start', 'start_date', 'start', 'start_date', 'start', 'date', 'ignition'], 'end': ['end', 'end_date', 'end', 'end_date', 'extinguish', 'control'], 'latitude': ['latitude', 'lat', 'latitude', 'y'], 'longitude': ['longitude', 'lon', 'lng', 'longitude', 'x'], 'area_ha': ['area', 'area_ha', 'area', 'area_ha', 'hectare']}

def _find(cols, names):
    name_value = [_norm(a) for a in names]
    for c in cols:
        if _norm(c) in name_value:
            return c
    return None

def fire_inventory_read(path, buffer_km: float=5.0, epsg=None, name_column=None):
    """Implementation of fire_inventory_read."""
    import geopandas as gpd
    path = Path(path)
    if path.suffix.lower() in ('.shp', '.geojson', '.json', '.gpkg', '.kml', '.zip'):
        gdf = gpd.read_file(path)
    else:
        df = _table_read(path)
        la, lo = (_find(df.columns, _ENV['latitude']), _find(df.columns, _ENV['longitude']))
        if la is None or lo is None:
            raise ValueError(_best('CSV/XLSX inventories require latitude/longitude columns'))
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lo], df[la]), crs='EPSG:4326')
    if epsg:
        gdf = gdf.set_crs(epsg=int(epsg), allow_override=True)
    if gdf.crs is None:
        x0, y0, x1, y1 = gdf.total_bounds
        if max(abs(x0), abs(x1)) > 180 or max(abs(y0), abs(y1)) > 90:
            raise ValueError(_best(f'{path}: No coordinate reference system (.prj); coordinates are not degrees ({x0:.0f}, {y0:.0f}). Specify --epsg (e.g. TM30 ITRF96: 5254, UTM 36N WGS84: 32636, UTM 35N: 32635, ED50 UTM 36N: 23036).'))
        gdf = gdf.set_crs('EPSG:4326')
    gdf = gdf.to_crs('EPSG:4326')
    print(_best(f"Inventory: {len(gdf)} geometries; columns: {', '.join(map(str, gdf.columns.drop('geometry')))}"))
    out = gpd.GeoDataFrame(geometry=gdf.geometry.copy(), crs='EPSG:4326')
    c = name_column if name_column and name_column in gdf.columns else _find(gdf.columns, _ENV['name'])
    out['name'] = gdf[c].astype(str).values if c else [f'fire_{i + 1}' for i in range(len(gdf))]
    if c and out['name'].duplicated().any() and (not out.geometry.geom_type.isin(['Point', 'MultiPoint']).all()):
        out['name'] = out['name'] + '_' + out.groupby('name').cumcount().add(1).astype(str)
    for k in ('start', 'end'):
        c = _find(gdf.columns, _ENV[k])
        out[k] = _date_convert(gdf[c]).dt.normalize().values if c else pd.NaT
    c = _find(gdf.columns, _ENV['area_ha'])
    out['area_ha'] = pd.to_numeric(gdf[c], errors='coerce').values if c else np.nan
    point = out.geometry.geom_type.isin(['Point', 'MultiPoint'])
    if point.any():
        utm = out.estimate_utm_crs()
        buffer = out.loc[point].to_crs(utm).buffer(buffer_km * 1000).to_crs('EPSG:4326')
        out.loc[point, 'geometry'] = buffer.values
    out = out.dissolve(by='name', aggfunc={'start': 'min', 'end': 'max', 'area_ha': 'sum'}).reset_index()
    out['area_ha_geometry'] = out.to_crs(out.estimate_utm_crs()).area.values / 10000.0
    return out
