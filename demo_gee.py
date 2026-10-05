"""Implementation of the wildfire gas analysis module."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .demo import _PATTERN, _FACTOR
_REF = pd.Timestamp('2025-07-28')

def _synthetic_daily(start_day: pd.Timestamp, days: pd.DatetimeIndex, seed: int):
    """Implementation of _synthetic_daily."""
    rng = np.random.default_rng(seed)
    pattern_start = start_day - pd.Timedelta(days=3)
    out = {}
    for g, d in _PATTERN.items():
        d = np.array(d, float)
        baseline = d[:4].mean()
        v = []
        for t in days:
            i = (t - pattern_start).days
            v.append(d[i] if 0 <= i < len(d) else baseline * (1 + rng.normal(0, 0.12)))
        out[g] = np.array(v) * (1 + rng.normal(0, 0.03, len(v)))
    return pd.DataFrame(out, index=days)

def demo_fires():
    import geopandas as gpd
    from shapely.geometry import Polygon
    record = [('Manavgat', 'Antalya', 31.42, 36.93, 0.22, 0.13, '2025-07-28', '2025-08-06', 41250, 0.81, 212), ('Milas', 'Mugla', 27.72, 37.27, 0.14, 0.09, '2025-07-31', '2025-08-09', 12870, 0.74, 96), ('Seferihisar', 'Izmir', 26.86, 38.12, 0.08, 0.06, '2025-06-29', '2025-07-03', 3120, 0.58, 41), ('Aladag', 'Adana', 35.37, 37.52, 0.07, 0.05, '2025-07-17', '2025-07-21', 2230, 0.88, 23), ('Egirdir', 'Isparta', 30.78, 37.93, 0.05, 0.04, '2025-08-11', '2025-08-14', 980, 0.66, 12), ('Canakkale Center', 'Canakkale', 26.47, 40.07, 0.06, 0.04, '2025-08-14', '2025-08-16', 1450, 0.52, 18), ('Kastamonu Tosya', 'Kastamonu', 34.03, 41.05, 0.03, 0.03, '2025-07-26', '2025-07-27', 210, 0.93, 4)]
    geoms, row = ([], [])
    rng = np.random.default_rng(3)
    for i, (district, province, x, y, dx, dy, b, s, ha, sample, fp) in enumerate(record, 1):
        angle = np.linspace(0, 2 * np.pi, 14, endpoint=False)
        r = 1 + rng.normal(0, 0.18, len(angle))
        geoms.append(Polygon([(x + dx * ri * np.cos(a), y + dy * ri * np.sin(a)) for a, ri in zip(angle, r)]))
        row.append({'id': i, 'name': f'{district}_{pd.Timestamp(b):%Y%m%d}', 'province': province, 'district': district, 'area_ha': ha, 'forest_fraction': sample, 'mcd_start': pd.Timestamp(b), 'mcd_end': pd.Timestamp(s), 'firms_pixel': fp, 'source': 'DEMO'})
    return gpd.GeoDataFrame(row, geometry=geoms, crs='EPSG:4326')

def demo_station(start, end):
    """Implementation of demo_station."""
    rng = np.random.default_rng(11)
    days = pd.date_range(pd.Timestamp(start) - pd.Timedelta(days=15), pd.Timestamp(end) + pd.Timedelta(days=15))
    gd = _synthetic_daily(pd.Timestamp(start), days, 5)
    hour = pd.date_range(days[0], days[-1] + pd.Timedelta(hours=23), freq='h')
    t = {'Date': hour.strftime('%d.%m.%Y %H:%M')}
    for g in ('CO', 'NO2', 'SO2', 'O3'):
        daily = gd[g].reindex(hour.normalize()).values
        error = pd.Series(1 + rng.normal(0, 0.15, len(days)), index=days).reindex(hour.normalize()).values
        t[f'{g} ( μg/m3 )'] = np.round(daily * 0.85 * (error + 0.15 * np.sin((hour.hour - 8) / 24 * 2 * np.pi)) * (1 + rng.normal(0, 0.1, len(hour))), 2)
    return pd.DataFrame(t)

def mock_gee_setup(fires):
    """Implementation of mock_gee_setup."""
    from . import gee, turkey
    center = {r.name: r for r in fires.itertuples()}

    def fire_find(geom):
        c = geom.centroid
        return min(center.values(), key=lambda r: r.geometry.centroid.distance(c))
    gee.launch = lambda p=None: None
    gee.ee_geometry = lambda g: g

    def firms(geom, a, b, buffer_m=1000):
        r = fire_find(geom)
        day = pd.date_range(r.mcd_start, r.mcd_end)
        w = np.exp(-np.arange(len(day)) / 2.0)
        n = np.maximum(1, np.round(r.firms_pixel * w / w.sum()))
        s = pd.Series(n, index=day)
        early = r.mcd_start - pd.Timedelta(days=70)
        if a <= early:
            s = pd.concat([pd.Series([1.0, 2.0], index=[early, early + pd.Timedelta(days=1)]), s])
        return s[(s.index >= a) & (s.index <= b)].rename('firms_pixel')

    def mcd(geom, a, b):
        r = fire_find(geom)
        day = pd.date_range(r.mcd_start, r.mcd_start + pd.Timedelta(days=2))
        return pd.Series([r.area_ha / 21.5 * p for p in (0.3, 0.5, 0.2)], index=day)

    def s5p(geom, a, b, settings, gases=None):
        r = fire_find(geom)
        days = pd.date_range(a, b)
        ug = _synthetic_daily(r.mcd_start, days, int(r.id))
        df = pd.DataFrame({g: ug[g] / _FACTOR[g] for g in ug})
        df.iloc[[5, len(df) - 6]] = np.nan
        return df[[g for g in gases or df.columns if g in df]]

    def era5(geom, a, b):
        r = fire_find(geom)
        rng = np.random.default_rng(int(r.id))
        days = pd.date_range(a, b)
        fire = (days >= r.mcd_start) & (days <= r.mcd_end)
        return pd.DataFrame({'temperature': 29 + rng.normal(0, 1.2, len(days)) + 2 * fire, 'humidity': np.clip(48 - 14 * fire + rng.normal(0, 4, len(days)), 10, 100), 'wind_speed': np.abs(4 + 3 * fire + rng.normal(0, 0.8, len(days))), 'wind_direction': (np.where(fire, 225, 300) + rng.normal(0, 20, len(days))) % 360}, index=days)
    from . import station_service
    from .data_reading import read_station
    import tempfile
    from pathlib import Path

    def demo_list_stats_list(*a, **k):
        row = []
        for r in fires.itertuples():
            c = r.geometry.centroid
            for j, (dx, dy, type) in enumerate(((0.25, 0.12, 'Urban'), (-0.4, 0.3, 'Suburban'), (0.7, -0.2, 'Urban'))):
                row.append({'id': f'demo-{r.id}-{j}', 'name': f'DEMO {r.district} {j + 1}', 'province': r.province, 'longitude': c.x + dx, 'latitude': c.y + dy, 'area_type': type, 'source_type': 'Traffic' if j == 0 else 'Arka plan'})
        return pd.DataFrame(row)

    def demo_fire_list_stats(lon, lat, start, end, gases=('CO', 'NO2', 'SO2', 'O3'), max_km=150, daily='Mean', **k):
        from shapely.geometry import Point
        r = fire_find(Point(lon, lat))
        path = Path(tempfile.gettempdir()) / f'demo_list_stats_{r.id}.csv'
        demo_station(r.mcd_start, r.mcd_end).to_csv(path, index=False)
        g = read_station(path, daily)
        candidate = station_service.station_candidates(demo_list_stats_list(), lon, lat, max_km, k.get('selection', 'Wind-aligned'), k.get('weather'), k.get('fire_start'), k.get('fire_end'), k.get('sector', 45))
        if candidate.empty:
            raise ValueError('No demo station within the search radius')
        chosen = candidate.iloc[0]
        matches = [{'Gas': gz, 'Station': chosen.name, 'Province': r.province, 'Distance (km)': round(chosen.distance_km, 1), 'Coverage (%)': 96.0, 'Wind alignment (%)': round(chosen.wind_alignment * 100, 1), 'Selection reason': 'DEMO: distance and wind ranking; synthetic observations', 'Area type': 'Urban', 'Source type': 'Traffic', 'Selected': 'Yes'} for gz in gases]
        return (g[[x for x in gases if x in g]], pd.DataFrame(matches), {})
    station_service.station_list = demo_list_stats_list
    station_service.fire_for_station = demo_fire_list_stats
    gee.firms_daily, gee.mcd64_burn_days = (firms, mcd)
    gee.s5p_series, gee.era5_weather = (s5p, era5)
    gee.peak_map_download = lambda *a, **k: None
    turkey.turkey_fires = lambda *a, **k: fires.copy()
