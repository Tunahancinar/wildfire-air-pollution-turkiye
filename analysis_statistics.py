"""Implementation of the wildfire gas analysis module."""
from __future__ import annotations
from typing import Dict, Optional
import numpy as np
import pandas as pd
DAY = pd.Timedelta(days=1)

def make_periods(start, end, before_day: int, after_day: int):
    """Implementation of make_periods."""
    return {'Before': (start - before_day * DAY, start - DAY), 'During fire': (start, end), 'After': (end + DAY, end + after_day * DAY)}

def _slice(df, a, b):
    idx = df.index.normalize() if isinstance(df.index, pd.DatetimeIndex) else df.index
    return df[(idx >= a) & (idx <= b)]

def conversion_factors(satellite: pd.DataFrame, station: Optional[pd.DataFrame], window=None, manual: Optional[Dict[str, float]]=None, method: str='station'):
    """Implementation of conversion_factors."""
    factor, description = ({}, {})
    for gas, value in (manual or {}).items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f'{gas}: Conversion factor must be finite and positive.')
    for gas in satellite.columns:
        if method == 'Manual' and manual and (gas in manual):
            factor[gas], description[gas] = (float(manual[gas]), 'Manual factor')
            continue
        if method == 'station' and station is not None and (gas in station.columns):
            match = pd.concat([satellite[gas].rename('u'), station[gas].rename('s')], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
            match = match[(match.u >= 0) & (match.s >= 0)]
            if window is not None:
                match = _slice(match, *window)
            u, s = (match.u, match.s)
            if len(match) >= 3 and u.max() > 0 and (s.max() > 0):
                factor[gas] = float(s.max() / u.max())
                description[gas] = f'station maximum {s.max():.2f} μg/m3 / satellite maximum {u.max():.3e} mol/m2; matched {len(match)} days, fire period'
                continue
        if manual and gas in manual:
            factor[gas], description[gas] = (float(manual[gas]), 'Manual factor (no station data)')
    return (factor, description)

def convert(satellite: pd.DataFrame, factor: Dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame({g: satellite[g] * factor[g] for g in satellite.columns if g in factor})

def period_statistics(df: pd.DataFrame, period_ranges) -> pd.DataFrame:
    rows = []
    for dad, (a, b) in period_ranges.items():
        part = _slice(df, a, b)
        for c in df.columns:
            v = part[c].dropna()
            rows.append({'Period': dad, 'Variable': c, 'Start': a.date(), 'End': b.date(), 'n': len(v), 'Maximum': v.max() if len(v) else np.nan, 'Maximum date': v.idxmax().date() if len(v) else None, 'Minimum': v.min() if len(v) else np.nan, 'Mean': v.mean() if len(v) else np.nan, 'Standard deviation': v.std(ddof=1) if len(v) > 1 else np.nan})
    return pd.DataFrame(rows)

def period_difference(df: pd.DataFrame, period_ranges) -> pd.DataFrame:
    """Implementation of period_difference."""
    from scipy.stats import mannwhitneyu
    out = []
    for c in df.columns:
        average = {}
        ser = {}
        for dad, (a, b) in period_ranges.items():
            ser[dad] = _slice(df, a, b)[c].dropna()
            average[dad] = ser[dad].mean()
        p = np.nan
        if len(ser['Before']) >= 3 and len(ser['During fire']) >= 3:
            p = mannwhitneyu(ser['During fire'], ser['Before'], alternative='two-sided').pvalue
        before = average['Before']
        out.append({'Gas': c, 'Before n': len(ser['Before']), 'During n': len(ser['During fire']), 'After n': len(ser['After']), 'Before mean': before, 'During mean': average['During fire'], 'After mean': average['After'], 'During/before (%)': (average['During fire'] - before) / abs(before) * 100 if before else np.nan, 'Mann-Whitney p': p, 'Interpretation': 'Exploratory test; independence of daily observations has not been verified.' if np.isfinite(p) else 'At least 3 observations per period are required.'})
    return pd.DataFrame(out)

def performance(P, O) -> Dict[str, float]:
    """Implementation of performance."""
    P, O = (np.asarray(P, float), np.asarray(O, float))
    m = np.isfinite(P) & np.isfinite(O)
    P, O = (P[m], O[m])
    n = len(P)
    if n < 3:
        return {'n': n, 'Status': 'Insufficient overlap (at least 3 days required)'}
    Pb, Ob = (P.mean(), O.mean())
    r = np.corrcoef(P, O)[0, 1] if P.std() > 0 and O.std() > 0 else np.nan
    mae = np.mean(np.abs(P - O))
    rmse = np.sqrt(np.mean((P - O) ** 2))
    sse = np.sum((P - O) ** 2)
    sst = np.sum((O - Ob) ** 2)
    ia_den = np.sum((np.abs(P - Ob) + np.abs(O - Ob)) ** 2)
    ia = 1 - sse / ia_den if ia_den > 0 else np.nan
    return {'n': n, 'Pearson_r': r, 'MAE': mae, 'nMAE': mae / Ob if Ob else np.nan, 'RMSE': rmse, 'RRMSE_%': rmse / np.sqrt(np.sum(O ** 2)) * 100 if np.sum(O ** 2) > 0 else np.nan, 'NRMSE_mean_%': rmse / abs(Ob) * 100 if Ob else np.nan, 'IA': ia, 'Status': 'Constant station series; Pearson r is undefined' if sst == 0 else 'Same data used for conversion; not independent validation'}

def performance_table(predicted: pd.DataFrame, station: pd.DataFrame, window=None) -> pd.DataFrame:
    row = []
    for g in predicted.columns:
        if g not in station.columns:
            continue
        j = pd.concat([predicted[g].rename('P'), station[g].rename('O')], axis=1)
        if window is not None:
            j = _slice(j, *window)
        d = performance(j['P'], j['O'])
        d['Gas'] = g
        row.append(d)
    return pd.DataFrame(row).set_index('Gas') if row else pd.DataFrame()

def weather_daily(met: pd.DataFrame) -> pd.DataFrame:
    m = met.copy()
    m['date'] = m['time'].dt.normalize()
    agg = {}
    for c in ('temperature', 'humidity', 'wind_speed'):
        if c in m:
            agg[c] = 'mean'
    g = m.groupby('date').agg(agg) if agg else pd.DataFrame(index=m['date'].unique())
    if 'wind_direction' in m:
        w = m.dropna(subset=['wind_direction'])
        ws = w['wind_speed'] if 'wind_speed' in w else 1.0
        u = -ws * np.sin(np.deg2rad(w['wind_direction']))
        v = -ws * np.cos(np.deg2rad(w['wind_direction']))
        uv = pd.DataFrame({'u': u, 'v': v, 'date': w['date']}).groupby('date').mean()
        g['wind_direction'] = (np.rad2deg(np.arctan2(-uv['u'], -uv['v'])) + 360) % 360
    return g

def direction_name(degrees) -> str:
    if degrees is None or pd.isna(degrees):
        return '-'
    names = ['K', 'KKD', 'KD', 'DKD', 'D', 'DGD', 'GD', 'GGD', 'G', 'GGB', 'GB', 'BGB', 'B', 'BKB', 'KB', 'KKB']
    return names[int(degrees % 360 / 22.5 + 0.5) % 16]
