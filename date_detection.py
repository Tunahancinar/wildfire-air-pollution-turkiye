"""Implementation of the wildfire gas analysis module."""
from __future__ import annotations
from .english import tr as _best
import pandas as pd
DAY = pd.Timedelta(days=1)

def episodes_split(series: pd.Series, gap_day: int=3) -> pd.DataFrame:
    """Implementation of episodes_split."""
    s = series[series > 0].sort_index()
    if s.empty:
        return pd.DataFrame(columns=['episode', 'start', 'end', 'duration_day', 'active_day', 'pixels', 'peak_day'])
    group = (s.index.to_series().diff() > gap_day * DAY).cumsum()
    row = []
    for i, (_, p) in enumerate(s.groupby(group.values), 1):
        row.append({'episode': i, 'start': p.index.min(), 'end': p.index.max(), 'duration_day': (p.index.max() - p.index.min()).days + 1, 'active_day': len(p), 'pixels': float(p.sum()), 'peak_day': p.idxmax()})
    return pd.DataFrame(row)

def main_episode(section: pd.DataFrame):
    if section is None or section.empty:
        return None
    return section.sort_values(['pixels', 'start'], ascending=[False, True]).iloc[0]

def fire_date_find(geom_ee, search_start, search_end, gap_day: int=3, mcd64: bool=True):
    """Implementation of fire_date_find."""
    from . import gee
    notes = []
    firms = pd.Series(dtype=float)
    try:
        firms = gee.firms_daily(geom_ee, search_start, search_end)
    except Exception as e:
        notes.append(f'FIRMS could not be read: {e}')
    fb = episodes_split(firms, gap_day)
    fb.insert(0, 'source', 'FIRMS')
    mcd = pd.Series(dtype=float)
    mb = pd.DataFrame()
    if mcd64:
        try:
            mcd = gee.mcd64_burn_days(geom_ee, search_start, search_end)
        except Exception as e:
            notes.append(f'MCD64A1 could not be read: {e}')
        mb = episodes_split(mcd, gap_day + 2)
        mb.insert(0, 'source', 'MCD64A1')
    f_main, m_main = (main_episode(fb), main_episode(mb))
    start = end = None
    source = 'not found'
    if f_main is not None:
        start, end, source = (f_main['start'], f_main['end'], 'FIRMS')
        if m_main is not None and abs((m_main['start'] - start).days) > 3:
            notes.append(f"FIRMS ({start:%d.%m.%Y}) and MCD64A1 ({m_main['start']:%d.%m.%Y}) start dates differ by more than 3 days; check these dates.")
        if m_main is not None and m_main['start'] < start and ((start - m_main['start']).days <= 3):
            start = m_main['start']
            source = 'FIRMS + MCD64A1 (earliest)'
    elif m_main is not None:
        start, end, source = (m_main['start'], m_main['end'], 'MCD64A1')
        notes.append('No active fires in FIRMS; the date was obtained from MCD64A1 burned-area data.')
    if len(fb) > 1:
        notes.append(f'Within the polygon: {len(fb)} separate FIRMS burn episodes; the largest was selected as the main fire.')
    if start is not None:
        previous = fb[(fb['end'] < start) & (fb['end'] >= start - 12 * DAY)] if not fb.empty else fb
        if len(previous):
            notes.append('Other burning occurred within 12 days before the main fire; the baseline may be affected.')
    episodes = pd.concat([d for d in (fb, mb) if not d.empty], ignore_index=True) if not fb.empty or not mb.empty else pd.DataFrame()
    return {'start': start, 'end': end, 'source': source, 'episodes': episodes, 'firms': firms, 'mcd64': mcd, 'notes': notes}
