"""Implementation of the wildfire gas analysis module."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional
import numpy as np
import pandas as pd
from .config import AnalysisSettings
DAY = pd.Timedelta(days=1)

@dataclass
class ChangeResult:
    gas: str
    direction: str
    baseline_median: float
    baseline_scale: float
    baseline_n: int
    baseline_source: str
    threshold_date: Optional[pd.Timestamp]
    cusum_date: Optional[pd.Timestamp]
    pelt_date: Optional[pd.Timestamp]
    increase_onset: Optional[pd.Timestamp]
    agreeing_method: int
    confidence: str
    delay_day: Optional[int]
    peak_date: Optional[pd.Timestamp]
    peak_value: float
    peak_z: float
    increase_percent: float
    elevated_end: Optional[pd.Timestamp]
    duration_day: Optional[int]
    excess_load: float

    def dictionary(self):
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, pd.Timestamp):
                d[k] = v.date()
        return d

def _baseline_statistics(x: pd.Series, start, settings: AnalysisSettings):
    source = ''
    if start is not None and (not pd.isna(start)):
        b = x[(x.index >= start - settings.before_day * DAY) & (x.index < start)].dropna()
        source = f'pre-fire {settings.before_day} day'
    else:
        b = x.iloc[:settings.before_day].dropna()
        source = f'first observations of the series: {settings.before_day} days'
    if len(b) < settings.min_baseline_observation:
        v = x.dropna()
        b = v[v <= v.median()]
        source = 'lower 50% of the series (insufficient baseline data)'
    med = float(np.median(b))
    mad = float(np.median(np.abs(b - med))) * 1.4826
    sd = float(np.std(b, ddof=1)) if len(b) > 1 else 0.0
    scale = max(mad, 0.5 * sd, 0.01 * abs(med), 1e-15)
    return (med, scale, len(b), source)

def _threshold_method(z: pd.Series, settings: AnalysisSettings):
    v = z.dropna()
    for i, (t, zi) in enumerate(v.items()):
        if zi >= settings.z_strong:
            return t
        if zi >= settings.z_threshold:
            next = v.iloc[i:i + settings.consecutive_day]
            if len(next) == settings.consecutive_day and (next >= settings.z_threshold).all():
                return t
    return None

def _cusum(z: pd.Series, settings: AnalysisSettings):
    v = z.dropna()
    s, last_zero = (0.0, None)
    for t, zi in v.items():
        s_new = max(0.0, s + zi - settings.cusum_drift)
        if s == 0.0 and s_new > 0:
            last_zero = t
        s = s_new
        if s > settings.cusum_h:
            return last_zero if last_zero is not None else t
    return None

def _pelt(z_full: pd.Series, search_start, settings: AnalysisSettings):
    try:
        import ruptures as rpt
    except ImportError:
        return _binary_split(z_full, search_start)
    y = z_full.interpolate(limit_direction='both').to_numpy()
    n = len(y)
    if n < 6:
        return None
    penalty = settings.pelt_penalty if settings.pelt_penalty is not None else 3.0 * np.log(n)
    bk = rpt.Pelt(model='l2', min_size=1, jump=1).fit(y).predict(pen=penalty)
    boundary = [0] + bk
    for a, b, c in zip(boundary[:-2], boundary[1:-1], boundary[2:]):
        t = z_full.index[b]
        if t >= search_start and y[b:c].mean() - y[a:b].mean() > 1.0:
            return t
    return None

def _binary_split(z_full: pd.Series, search_start):
    """Implementation of _binary_split."""
    y = z_full.interpolate(limit_direction='both').to_numpy()
    best_best, t_best = (0, None)
    for k in range(2, len(y) - 1):
        if z_full.index[k] < search_start:
            continue
        difference = y[k:].mean() - y[:k].mean()
        cost = ((y[:k] - y[:k].mean()) ** 2).sum() + ((y[k:] - y[k:].mean()) ** 2).sum()
        score = difference / (np.sqrt(cost / len(y)) + 1e-09)
        if difference > 1.0 and score > best_best:
            best_best, t_best = (score, z_full.index[k])
    return t_best

def detect_change_onset(series: pd.Series, gas: str, direction: str, start=None, end=None, settings: Optional[AnalysisSettings]=None):
    """Implementation of detect_change_onset."""
    settings = settings or AnalysisSettings()
    x = series.asfreq('D')
    med, scale, n_b, source = _baseline_statistics(x, start, settings)
    z = (x - med) / scale
    if start is not None and (not pd.isna(start)):
        search_start = start - settings.search_before_day * DAY
    else:
        search_start = x.index[min(settings.before_day, len(x) - 1)]
    search_end = end + settings.after_day * DAY if end is not None and (not pd.isna(end)) else x.index[-1]
    used_direction = 'Increase'
    if direction == 'two_sided':
        w = z[(z.index >= search_start) & (z.index <= search_end)]
        if w.dropna().size and abs(w.min()) > abs(w.max()):
            z = -z
            used_direction = 'Decrease'
    zp = z[(z.index >= search_start) & (z.index <= search_end)]
    t1 = _threshold_method(zp, settings)
    t2 = _cusum(zp, settings)
    t3 = _pelt(z[z.index <= search_end], search_start, settings)
    dates = sorted((t for t in (t1, t2, t3) if t is not None))
    if dates:
        consensus = dates[(len(dates) - 1) // 2]
        agreement = sum((abs((t - consensus).days) <= 1 for t in dates))
        confidence = {3: 'High', 2: 'Medium'}.get(agreement, 'Low')
    else:
        consensus, agreement, confidence = (None, 0, 'No detection')
    peak_t, peak_v, peak_z = (None, np.nan, np.nan)
    if zp.dropna().size:
        peak_t = zp.idxmax()
        peak_z = float(zp.max())
        peak_v = float(x.loc[peak_t])
    elevated_end, duration, excess = (None, None, np.nan)
    if consensus is not None and peak_t is not None:
        if peak_t < consensus:
            peak_t2 = z[(z.index >= consensus) & (z.index <= search_end)].idxmax()
            peak_t, peak_z, peak_v = (peak_t2, float(z.loc[peak_t2]), float(x.loc[peak_t2]))
        after = z[z.index > peak_t].dropna()
        decline = after[after < 1.0]
        elevated_end = decline.index[0] - DAY if len(decline) else after.index[-1] if len(after) else peak_t
        duration = int((elevated_end - consensus).days) + 1
        window = x[(x.index >= consensus) & (x.index <= elevated_end)].dropna()
        sign = -1 if used_direction == 'Decrease' else 1
        excess = float(np.clip(sign * (window - med), 0, None).sum())
    increase = (peak_v - med) / abs(med) * 100 if med and (not np.isnan(peak_v)) else np.nan
    delay = int((consensus - start).days) if consensus is not None and start is not None and (not pd.isna(start)) else None
    return (ChangeResult(gas, used_direction, med, scale, n_b, source, t1, t2, t3, consensus, agreement, confidence, delay, peak_t, peak_v, peak_z, increase, elevated_end, duration, excess), z)

def fire_signal(results) -> Optional[pd.Timestamp]:
    """Implementation of fire_signal."""
    co = [s for s in results if s.gas == 'CO' and s.increase_onset is not None]
    if co and co[0].confidence in ('High', 'Medium'):
        return co[0].increase_onset
    t = sorted((s.increase_onset for s in results if s.increase_onset is not None and s.direction == 'Increase'))
    for ti in t:
        if sum((abs((tj - ti).days) <= 1 for tj in t)) >= 2:
            return ti
    return co[0].increase_onset if co else t[0] if t else None
