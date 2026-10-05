"""Implementation of the wildfire gas analysis module."""
from __future__ import annotations
from .english import tr as _best
import json
import math
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
BASE = 'https://sim.csb.gov.tr'
PAGE = f'{BASE}/STN/STN_Report/StationDataDownloadNew'
DEFAULT = f'{BASE}/STN/STN_Report/StationDataDownloadNewDefaults'
DATA = f'{BASE}/STN/STN_Report/StationDataDownloadNewData'
AGENT = 'YanginGazAnaliz/1.0 (akademik arastirma; Duzce Universitesi Orman Fakultesi)'
HOURLY, DAILY = (8, 16)
PARAMETER = {'CO': 'CO', 'NO2': 'NO2', 'SO2': 'SO2', 'O3': 'O3'}
CACHE = Path(__file__).resolve().parent.parent / '.wildfire_gas'

class StationServiceError(RuntimeError):
    pass

class StationClient:

    def __init__(self, wait_sn: float=1.5, time_timeout: int=120):
        import requests
        self.s = requests.Session()
        self.s.headers.update({'User-Agent': AGENT})
        self.wait = wait_sn
        self.time_timeout = time_timeout
        self._key = None
        self._last = 0.0

    def _wait(self):
        d = self.wait - (time.monotonic() - self._last)
        if d > 0:
            time.sleep(d)
        self._last = time.monotonic()

    def _validation(self) -> str:
        if self._key is None:
            self._wait()
            try:
                html = self.s.get(PAGE, timeout=self.time_timeout).text
            except Exception as e:
                raise StationServiceError(f"sim.csb.gov.tr'ye connection failed: {e}") from e
            m = re.search('name="__RequestVerificationToken"[^>]*value="([^"]+)"', html)
            if not m:
                raise StationServiceError('Verification token not found on SIM (site structure may have changed)')
            self._key = m.group(1)
        return self._key

    def _post(self, url, data, strict=True):
        self._wait()
        try:
            r = self.s.post(url, data=[('__RequestVerificationToken', self._validation()), *data], timeout=self.time_timeout, headers={'X-Requested-With': 'XMLHttpRequest', 'Referer': PAGE})
            r.raise_for_status()
            body = r.json()
        except StationServiceError:
            raise
        except Exception as e:
            raise StationServiceError(f'SIM request failed: {e}') from e
        if not body.get('Result'):
            if strict:
                raise StationServiceError(f"SIM returned an error: {body.get('Message')!r}")
            return None
        return body.get('Object')

    def stations(self) -> pd.DataFrame:
        obj = self._post(DEFAULT, [('StationType', '1')])
        provinces = {c['Id']: c['Name'] for c in obj.get('CityId', [])}
        area = {str(a['Id']): a['Name'] for a in obj.get('AreaType', [])}
        source = {str(s['Id']): s['Name'] for s in obj.get('SourceType', [])}
        row = []
        for s in obj.get('StationIds', []):
            m = re.match('POINT \\(([-\\d.]+) ([-\\d.]+)\\)', s.get('Location') or '')
            if not m:
                continue
            row.append({'id': s['id'], 'name': (s.get('Name') or '').strip(), 'province': provinces.get(s.get('CityId'), ''), 'longitude': float(m.group(1)), 'latitude': float(m.group(2)), 'area_type': area.get(str(s.get('AreaType')), ''), 'source_type': source.get(str(s.get('SourceType')), '')})
        if not row:
            raise StationServiceError('SIM returned an empty station list')
        return pd.DataFrame(row)

    def measurements(self, station_id: str, start: datetime, end: datetime, parameters: List[str], period: int=HOURLY) -> pd.DataFrame:
        """Implementation of measurements."""

        def request(params):
            data = [('StationType', '1'), ('DataPeriods', str(period)), ('StartDateTime', pd.Timestamp(start).strftime('%d.%m.%Y %H:%M')), ('EndDateTime', pd.Timestamp(end).strftime('%d.%m.%Y %H:%M')), ('StationIds', station_id)] + [('Parameters', p) for p in params]
            obj = self._post(DATA, data, strict=False)
            return (obj or {}).get('Data') or []
        rows = request(parameters)
        if not rows and len(parameters) > 1:
            part = []
            for p in parameters:
                r = request([p])
                if r:
                    part.append(_frame(r, [p]))
            if not part:
                return pd.DataFrame(columns=parameters)
            return pd.concat(part, axis=1)
        return _frame(rows, parameters)

def _count(v):
    if isinstance(v, str):
        v = v.replace(',', '.')
    return pd.to_numeric(v, errors='coerce')

def _frame(rows, parameters) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    time = next((c for c in df.columns if c.lower() in ('readtime', 'time', 'date')), None)
    if time is None:
        return pd.DataFrame(columns=parameters)
    df.index = pd.to_datetime(df[time])
    out = pd.DataFrame(index=df.index)
    for p in parameters:
        column = next((c for c in df.columns if c.upper().replace('.', '') == p.upper()), None)
        out[p] = df[column].map(_count) if column else np.nan
    out = out[~out.index.duplicated()].sort_index()
    out[out <= 0] = np.nan
    return out

def distance_km(lon1, lat1, lon2, lat2):
    r = 6371.0
    p1, p2 = (np.radians(lat1), np.radians(lat2))
    dp, dl = (p2 - p1, np.radians(np.asarray(lon2) - lon1))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))

def station_list(client: Optional[StationClient]=None, refresh_day: int=30) -> pd.DataFrame:
    """Implementation of station_list."""
    CACHE.mkdir(parents=True, exist_ok=True)
    file = CACHE / 'sim_stations.csv'
    if file.exists() and time.time() - file.stat().st_mtime < refresh_day * 86400:
        return pd.read_csv(file)
    df = (client or StationClient()).stations()
    df.to_csv(file, index=False)
    return df

def nearest_candidates(stations: pd.DataFrame, lon: float, lat: float, n: int=6, max_km: float=150) -> pd.DataFrame:
    d = stations.copy()
    d['distance_km'] = distance_km(lon, lat, d['longitude'].values, d['latitude'].values)
    return d[d.distance_km <= max_km].sort_values('distance_km').head(n).reset_index(drop=True)

def _measurement_cached(client, sid, start, end, params):
    CACHE.mkdir(parents=True, exist_ok=True)
    file = CACHE / 'measurement' / f'{sid}_{pd.Timestamp(start):%Y%m%d}_{pd.Timestamp(end):%Y%m%d}.csv'
    if file.exists():
        cached = pd.read_csv(file, index_col=0, parse_dates=True)
        if set(params).issubset(cached.columns):
            return cached
    df = client.measurements(sid, start, end, params)
    file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(file)
    return df

def station_candidates(items, lon, lat, max_km=150, selection='Wind-aligned', weather=None, fire_start=None, fire_end=None, sector=45):
    """Implementation of station_candidates."""
    if selection not in ('Nearest', 'Wind-aligned') or not 0 < sector <= 90:
        raise ValueError(_best('Invalid station selection or wind sector'))
    from pyproj import Geod
    d = nearest_candidates(items, lon, lat, len(items), max_km)
    d['bearing'] = Geod(ellps='WGS84').inv(np.full(len(d), lon), np.full(len(d), lat), d.longitude.values, d.latitude.values)[0] % 360 if len(d) else []
    d['wind_alignment'] = np.nan
    d['wind_day'] = 0
    d['wind_suitable'] = False
    if weather is not None and {'wind_direction', 'wind_speed'}.issubset(weather.columns):
        w = weather[['wind_direction', 'wind_speed']].replace([np.inf, -np.inf], np.nan).dropna()
        if fire_start is not None and fire_end is not None:
            w = w.loc[pd.Timestamp(fire_start):pd.Timestamp(fire_end)]
        w = w[(w.wind_speed >= 0.5) & w.wind_direction.between(0, 360)]
        w = w[~w.index.normalize().duplicated()]
        d['wind_day'] = len(w)
        if len(w) >= 3:
            to = (w.wind_direction.to_numpy() + 180) % 360
            for i in d.index:
                difference = np.abs((to - d.loc[i, 'bearing'] + 180) % 360 - 180)
                score = np.average(difference <= sector, weights=w.wind_speed)
                d.loc[i, 'wind_alignment'] = float(score)
                d.loc[i, 'wind_suitable'] = bool(score >= 0.5)
    d['priority'] = (~d.wind_suitable).astype(int) if selection == 'Wind-aligned' else 0
    return d.sort_values(['priority', 'distance_km']).reset_index(drop=True)

def fire_for_station(lon: float, lat: float, start, end, gases=('CO', 'NO2', 'SO2', 'O3'), n_candidate: int=6, max_km: float=150, min_coverage: float=0.5, daily: str='Mean', client: Optional[StationClient]=None, log=print, selection='Wind-aligned', weather=None, fire_start=None, fire_end=None, sector=45, satellite=None):
    """Implementation of fire_for_station."""
    list_stats = client or StationClient()
    items = station_list(list_stats)
    candidate = station_candidates(items, lon, lat, max_km, selection, weather, fire_start, fire_end, sector)
    candidate = pd.concat([candidate.head(n_candidate), candidate.nsmallest(n_candidate, 'distance_km')]).drop_duplicates('id')
    if candidate.empty:
        raise StationServiceError(f'{max_km:g} no station within the radius (km)')
    start, end = (pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(hours=23))
    expected = (end - start).total_seconds() / 3600 + 1
    params = [PARAMETER[g] for g in gases if g in PARAMETER]
    remaining = set((g for g in gases if g in PARAMETER))
    daily_series, matching, hourly = ({}, [], {})
    for r in candidate.itertuples():
        if not remaining:
            break
        log(f'  station_service: {r.name} ({r.distance_km:.1f} km) downloading...')
        try:
            df = _measurement_cached(list_stats, r.id, start, end, params)
        except StationServiceError as e:
            log(f'    skipped: {e}')
            continue
        if df.empty or not isinstance(df.index, pd.DatetimeIndex):
            log('    skipped: no valid time-indexed observations')
            for g in sorted(remaining):
                matching.append({'Gas': g, 'Station': r.name, 'Distance (km)': round(r.distance_km, 1), 'Selected': 'No (no measurements)', 'Method': selection})
            continue
        df = df.loc[start:end].replace([np.inf, -np.inf], np.nan)
        df = df[~df.index.duplicated()].mask(df < 0)
        hourly[r.name] = df
        for g in sorted(remaining):
            p = PARAMETER[g]
            if p not in df:
                continue
            fire_start = pd.Timestamp(fire_start) if fire_start is not None else start.normalize()
            fire_end = (pd.Timestamp(fire_end) if fire_end is not None else end.normalize()) + pd.Timedelta(hours=23)
            fire_series = df.loc[fire_start:fire_end, p]
            coverage = fire_series.notna().sum() / ((fire_end - fire_start).total_seconds() / 3600 + 1)
            daily = df[p].resample('D').max() if daily == 'max' else df[p].resample('D').mean()
            daily = daily.where(df[p].resample('D').count() >= 12)
            pair = daily.loc[fire_start:fire_end].dropna()
            if satellite is not None and g in satellite:
                pair = pd.concat([pair.rename('station'), satellite[g].rename('satellite')], axis=1).dropna()
            combined = len(pair)
            details = {'Station ID': r.id, 'Station bearing (°)': round(r.bearing, 1), 'Wind alignment (%)': round(r.wind_alignment * 100, 1), 'Wind days': r.wind_day, 'Overlapping days': combined, 'Method': selection, 'Selection reason': ('Wind sector matched; nearest eligible candidate' if r.wind_suitable else 'No eligible wind-aligned station; nearest eligible candidate') if selection == 'Wind-aligned' else 'Nearest eligible candidate'}
            if coverage < min_coverage or combined < 3:
                matching.append({'Gas': g, 'Station': r.name, 'Province': r.province, 'Distance (km)': round(r.distance_km, 1), 'Coverage (%)': round(coverage * 100, 1), **details, 'Selected': 'No (insufficient fire-period coverage or overlap)'})
                continue
            daily_series[g] = daily
            matching.append({'Gas': g, 'Station': r.name, 'Province': r.province, 'Distance (km)': round(r.distance_km, 1), 'Coverage (%)': round(coverage * 100, 1), 'Area type': r.area_type, 'Source type': r.source_type, **details, 'Selected': 'Yes'})
            remaining.discard(g)
    for g in sorted(remaining):
        matching.append({'Gas': g, 'Station': '-', 'Selected': f'{max_km:g} no evaluated station meets coverage/overlap requirements within the radius (km)'})
    gdf = pd.DataFrame(daily_series)
    if not gdf.empty:
        gdf.index = gdf.index.normalize()
        gdf.index.name = 'date'
    return (gdf, pd.DataFrame(matching), hourly)

def connection_test():
    """Implementation of connection_test."""
    print(_best('sim.csb.gov.tr checking connection...'))
    list_stats = StationClient()
    items = station_list(list_stats, refresh_day=0)
    print(_best(f"  {len(items)} stations retrieved. Example: {', '.join(items.name.head(5))}"))
    anchor = items[items.province.str.contains('Ankara', case=False, na=False)]
    r = (anchor if len(anchor) else items).iloc[0]
    last = pd.Timestamp.now().normalize() - pd.Timedelta(days=3)
    df = list_stats.measurements(r['id'], last - pd.Timedelta(days=2), last, list(PARAMETER.values()))
    print(_best(f"  {r['name']}: {len(df)} hourly records; populated columns: {[c for c in df.columns if df[c].notna().any()]}"))
    print(_best(df.head(5).to_string()))
    print(_best('OK — automatic station data are available.'))
