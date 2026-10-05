"""Implementation of the wildfire gas analysis module."""
from __future__ import annotations
from .english import tr as _best
import json
import re
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from . import chart, analysis_statistics as list_stats
from .config import GASES, AnalysisSettings
from .detection import detect_change_onset, fire_signal
DAY = pd.Timedelta(days=1)

def _file_name(s: str) -> str:
    s = re.sub('[^\\w\\-]+', '_', str(s), flags=re.UNICODE).strip('_')
    return s or 'fire'

def _t(x):
    return None if x is None or pd.isna(x) else pd.Timestamp(x).normalize()

def _fmt(t):
    return '-' if t is None or pd.isna(t) else pd.Timestamp(t).strftime('%d.%m.%Y')

def fire_analysis(name: str, start=None, end=None, geometry=None, satellite: Optional[pd.DataFrame]=None, station: Optional[pd.DataFrame]=None, weather: Optional[pd.DataFrame]=None, settings: Optional[AnalysisSettings]=None, output='output', gee_use=False, search_start=None, search_end=None, gases=None, date_source='user', firms_daily=None):
    """Implementation of fire_analysis."""
    settings = settings or AnalysisSettings()
    gases = gases or list(GASES)
    start, end = (_t(start), _t(end))
    directory = Path(output) / _file_name(name)
    (directory / 'charts').mkdir(parents=True, exist_ok=True)
    notes = []
    geom_ee = None
    met_daily = None
    firms_series = firms_daily
    date_episodes = None
    if gee_use:
        from . import gee
        ee = gee.launch(settings.gee_project)
        geom_ee = gee.ee_geometry(geometry.geometry.iloc[0])
        if (start is None or end is None) and settings.firms_use:
            from .date_detection import fire_date_find
            a = start - 3 * DAY if start is not None else _t(search_start)
            b = start + 45 * DAY if start is not None else _t(search_end)
            if a is None or b is None:
                raise ValueError(_best(f'{name}: No date: a search period is required (e.g. --year 2025).'))
            print(_best(f'[{name}] Detecting fire dates: FIRMS + MCD64A1 ({_fmt(a)}–{_fmt(b)})'))
            tb = fire_date_find(geom_ee, a, b)
            notes += tb['notes']
            firms_series = tb['firms']
            date_episodes = tb['episodes']
            if tb['start'] is not None:
                if start is None:
                    start, date_source = (_t(tb['start']), tb['source'])
                if end is None:
                    end = _t(tb['end'])
                    notes.append(f"End date ({_fmt(end)}) {tb['source']} taken from the last burn date.")
            print(_best(f'[{name}]   → {_fmt(start)} – {_fmt(end)} ({date_source})'))
        if start is None:
            raise ValueError(_best(f'{name}: No start date: neither FIRMS nor MCD64A1 detected burning in this polygon (check the search period and coordinate reference system).'))
        if end is None:
            end = start + 7 * DAY
            notes.append('The end date was set to start + 7 days because it was unknown.')
        pb, ps = (start - (settings.before_day + 2) * DAY, end + (settings.after_day + 2) * DAY)
        print(_best(f'[{name}] Sentinel-5P {settings.gee_product} fetching daily time series ({_fmt(pb)}–{_fmt(ps)})'))
        satellite = gee.s5p_series(geom_ee, pb, ps, settings, gases)
        if weather is None:
            try:
                print(_best(f'[{name}] fetching ERA5-Land meteorology'))
                met_daily = gee.era5_weather(geom_ee, pb, ps)
            except Exception as e:
                notes.append(f'ERA5-Land could not be retrieved: {e}')
        satellite.to_csv(directory / 's5p_raw_mol_m2.csv', encoding='utf-8-sig')
    if satellite is None or satellite.empty:
        raise ValueError(_best(f'{name}: no satellite data'))
    satellite = satellite[[g for g in gases if g in satellite.columns]].sort_index().asfreq('D')
    satellite.to_csv(directory / 's5p_raw_mol_m2.csv', encoding='utf-8-sig')
    satellite = satellite.replace([np.inf, -np.inf], np.nan).copy()
    if 'SO2' in satellite:
        negative = int((satellite.SO2 < 0).sum())
        satellite['SO2'] = satellite.SO2.clip(lower=0)
        (directory / 'data_processing.json').write_text(json.dumps({'SO2_negative_count': negative, 'SO2_analysis_lower_boundary': 0, 'raw_file': 's5p_raw_mol_m2.csv'}), encoding='utf-8')
    results, zscores = ({}, {})
    for g in satellite.columns:
        if satellite[g].notna().sum() < 5:
            notes.append(f'{g}: insufficient satellite observations; skipped.')
            continue
        r, z = detect_change_onset(satellite[g], g, GASES[g].direction, start, end, settings)
        results[g], zscores[g] = (r, z)
    signal = fire_signal(list(results.values()))
    if start is None:
        start, date_source = (signal, 'gas signal (automatic)')
        if start is not None:
            for g in list(results):
                results[g], zscores[g] = detect_change_onset(satellite[g], g, GASES[g].direction, start, end, settings)
            notes.append('The start date was inferred from the gas signal because no date was provided.')
    if end is None and start is not None:
        endings = [r.elevated_end for r in results.values() if r.elevated_end is not None and r.direction == 'Increase']
        end = max(endings) if endings else start + 7 * DAY
        notes.append(f'The end date was inferred from gas concentrations returning to baseline ({_fmt(end)}).')
    period = list_stats.make_periods(start, end, settings.before_day, settings.after_day) if start is not None else None
    if period:
        satellite = satellite.loc[period['Before'][0]:period['After'][1]]
    if weather is not None:
        met_daily = list_stats.weather_daily(weather)
    station_matching = None
    if station is None and settings.station_automatic and (geometry is not None) and (start is not None):
        from . import station_service
        g4326 = geometry.to_crs(4326).geometry
        m = (g4326.union_all() if hasattr(g4326, 'union_all') else g4326.unary_union).centroid
        print(_best(f'[{name}] station_service: searching for nearest stations'))
        try:
            g_list_stats = [g for g in satellite.columns if g in station_service.PARAMETER]
            daily, station_matching, _ = station_service.fire_for_station(m.x, m.y, start - (settings.before_day + 2) * DAY, end + (settings.after_day + 2) * DAY, gases=g_list_stats, max_km=settings.station_max_km, daily=settings.station_daily, selection=settings.station_selection, weather=met_daily, fire_start=start, fire_end=end, sector=settings.wind_sector, satellite=satellite)
            if not daily.empty:
                station = daily
                station.to_csv(directory / 'station_daily_ug_m3.csv', encoding='utf-8-sig', float_format='%.2f')
                select = station_matching[station_matching['Selected'] == 'Yes']
                notes.append('Ground-station data retrieved automatically from station_service: ' + '; '.join((f"{r['Gas']} → {r['Station']} ({r['Distance (km)']} km)" for _, r in select.iterrows())))
                for _, r in select.iterrows():
                    notes.append(f"{r['Gas']} station selection: {r.get('Selection reason', 'demo')}; wind alignment %{r.get('Wind alignment (%)', float('nan')):.1f}.")
            else:
                notes.append('station_service: no data found at nearby stations for this period.')
        except Exception as e:
            notes.append(f'station_service data could not be retrieved ({e}); results may remain unconverted.')
            print(_best(f'[{name}]   station_service error: {e}'))
    if station is not None:
        station = station.replace([np.inf, -np.inf], np.nan).copy()
        station = station.mask(station < 0)
    factor, factor_description = ({}, {})
    if settings.conversion != 'none':
        factor, factor_description = list_stats.conversion_factors(satellite, station, window=(start, end) if start is not None else None, manual=settings.manual_factor, method=settings.conversion)
    ugm3 = list_stats.convert(satellite, factor) if factor else pd.DataFrame(index=satellite.index)
    if factor:
        ugm3.to_csv(directory / 'time_series_ug_m3.csv', encoding='utf-8-sig', float_format='%.4f')
    satellite.to_csv(directory / 'time_series_mol_m2.csv', encoding='utf-8-sig', float_format='%.6e')
    missing = [g for g in satellite.columns if g not in factor]
    if settings.conversion != 'none' and missing:
        notes.append('Gases without μg/m3 conversion (station data or factor unavailable): ' + ', '.join(missing))
    analysis_data = ugm3 if settings.conversion != 'none' else satellite
    analysis_unit = 'μg/m3' if settings.conversion != 'none' else 'mol/m2'
    if analysis_data.empty or not analysis_data.notna().any().any():
        raise ValueError(_best('No valid converted data are available for analysis. Check the station data or a positive conversion factor.'))
    results, zscores = ({}, {})
    for g in analysis_data:
        if analysis_data[g].notna().sum() < 5:
            notes.append(f'{g}: for final analysis: at least 5 valid days are required.')
            continue
        results[g], zscores[g] = detect_change_onset(analysis_data[g], g, GASES[g].direction, start, end, settings)
    signal = fire_signal(list(results.values()))
    notes.append(f'Final change detection, peak/baseline values and period comparisons use {analysis_unit} time series.')
    for g, r in results.items():
        if end is not None and r.increase_onset is not None and (r.increase_onset > end):
            notes.append(f'{g}: Detected after the fire ended; this cannot be interpreted as a fire-caused increase.')
        if r.baseline_median == 0:
            notes.append(f'{g}: Percentage change is undefined because the baseline median is zero.')
    notes.append('Conversion by a positive constant alone does not change correlation or rank-based p-values.')
    notes.append('Station comparisons use the conversion data and are not independent validation. The Mann–Whitney test is exploratory; independence of daily observations has not been verified.')
    notes.append('Wind-based selection uses daily local wind and distance; topography, transport time and chemical transformation are not modeled.')
    if weather is not None:
        met_daily = list_stats.weather_daily(weather)
    if met_daily is not None and period:
        met_daily = met_daily.loc[period['Before'][0]:period['After'][1]]
    if met_daily is not None and (not met_daily.empty):
        met_daily.to_csv(directory / 'weather_daily.csv', encoding='utf-8-sig', float_format='%.2f')
    tables = {}
    detection_df = pd.DataFrame([r.dictionary() for r in results.values()])
    if not detection_df.empty:
        for g, r in results.items():
            f = factor.get(g)
            i = detection_df.index[detection_df.gas == g][0]
            detection_df.loc[i, 'unit'] = analysis_unit
            detection_df.loc[i, 'peak_ug_m3'] = r.peak_value if f else np.nan
            detection_df.loc[i, 'baseline_ug_m3'] = r.baseline_median if f else np.nan
            if met_daily is not None and r.peak_date in getattr(met_daily, 'index', []):
                m = met_daily.loc[r.peak_date]
                detection_df.loc[i, 'peak_gunu_wind_speed_km_sa'] = m.get('wind_speed', np.nan) * 3.6
                detection_df.loc[i, 'peak_gunu_wind_direction'] = list_stats.direction_name(m.get('wind_direction'))
    tables['Change detection'] = detection_df
    if station_matching is not None and (not station_matching.empty):
        tables['Station matching'] = station_matching
    if date_episodes is not None and (not date_episodes.empty):
        tables['Burn episodes'] = date_episodes
    if firms_series is not None and (not firms_series.empty):
        tables['Daily FIRMS'] = firms_series.rename_axis('date').reset_index()
    if period:
        tables['Period statistics (mol m-2)'] = list_stats.period_statistics(satellite, period)
        tables['Period comparison'] = list_stats.period_difference(analysis_data, period)
        tables['Period comparison']['Unit'] = analysis_unit
        if factor:
            tables['Period statistics (ug m-3)'] = list_stats.period_statistics(ugm3, period)
        if station is not None:
            tables['Station period statistics'] = list_stats.period_statistics(station[[c for c in station.columns if c in GASES]], period)
        if met_daily is not None and (not met_daily.empty):
            met_report = met_daily.copy()
            if 'wind_speed' in met_report:
                met_report['wind_speed'] *= 3.6
                met_report = met_report.rename(columns={'wind_speed': 'wind_speed_km_sa'})
            tables['Weather period statistics'] = list_stats.period_statistics(met_report, period)
    if factor:
        tables['Conversion factors'] = pd.DataFrame({'Gas': list(factor), 'Factor (μg m-3 / mol m-2)': list(factor.values()), 'Description': [factor_description[g] for g in factor]})
    if factor and station is not None:
        window = (period['Before'][0], period['After'][1]) if period else None
        perf = list_stats.performance_table(ugm3, station, window)
        if not perf.empty:
            tables['Performance'] = perf.reset_index()
    gdir = directory / 'charts'
    for g in satellite.columns:
        if g not in results:
            continue
        f = factor.get(g)
        series = ugm3[g] if f else satellite[g]
        unit = 'μg/m3' if f else 'mol/m2'
        chart.gas_time_series(series, results[g], period, unit, f'{name} — {GASES[g].name} time series', gdir / f'{g}_time_series.png', factor=1.0, gas=g)
    if factor:
        chart.multiple_panel(ugm3, results, period, 'μg/m3', f'{name} — all gases', gdir / 'all_gases_ug_m3.png', firms=firms_series)
    chart.multiple_panel(satellite, results, period, 'mol/m2', f'{name} — all gases (Sentinel-5P)', gdir / 'all_gases_mol_m2.png', firms=firms_series)
    chart.detection_chart(zscores, results, settings, start, gdir / 'increase_onset_tespiti.png', f'{name} — change detection (robust z-score)')
    if met_daily is not None and (not met_daily.empty):
        chart.weather_chart(met_daily, period, gdir / 'weather.png', f'{name} — meteorological conditions')
    if gee_use and settings.map_download and (geometry is not None):
        from . import gee
        hdir = directory / 'maps'
        hdir.mkdir(exist_ok=True)
        for g, r in results.items():
            if r.peak_date is None:
                continue
            tif = gee.peak_map_download(geom_ee, r.peak_date, g, settings, hdir / f'{g}_{r.peak_date:%Y%m%d}.tif')
            if tif:
                f = factor.get(g)
                wind = {}
                if met_daily is not None and r.peak_date in met_daily.index:
                    wind = met_daily.loc[r.peak_date].to_dict()
                chart.map(tif, geometry, f or 1.0, 'μg/m3' if f else 'mol/m2', f"{name} — {GASES[g].name} {('minimum' if r.direction == 'Decrease' else 'peak')} day ({r.peak_date:%d.%m.%Y})", gdir / f'{g}_map.png', gas=g, wind_direction=wind.get('wind_direction'), wind_speed=wind.get('wind_speed'))
    text = _interpretation(name, start, end, date_source, signal, results, factor, met_daily, tables.get('Performance'), notes)
    (directory / 'summary.txt').write_text(text, encoding='utf-8')
    with pd.ExcelWriter(directory / f'{_file_name(name)}_report.xlsx', engine='openpyxl') as xw:
        pd.DataFrame({'Summary': text.split('\n')}).to_excel(xw, sheet_name='Summary', index=False)
        for name_s, tb in tables.items():
            if tb is not None and (not tb.empty):
                tb.to_excel(xw, sheet_name=name_s[:31], index=False)
        satellite.reset_index().to_excel(xw, sheet_name='Series mol m-2', index=False)
        if factor:
            ugm3.reset_index().to_excel(xw, sheet_name='Series ug m-3', index=False)
        for ws in xw.book.worksheets:
            for column in ws.columns:
                w = max((len(str(c.value)) if c.value is not None else 0 for c in column[:50]))
                ws.column_dimensions[column[0].column_letter].width = min(max(10, w + 2), 60)
            ws.freeze_panes = 'A2'
    settings_record = {k: v if not isinstance(v, dict) else dict(v) for k, v in vars(settings).items()}
    (directory / 'settings.json').write_text(json.dumps(settings_record, ensure_ascii=False, indent=2), encoding='utf-8')
    print(_best(f'[{name}] completed → {directory}'))
    summary = {'Fire': name, 'Start': _fmt(start), 'End': _fmt(end), 'Date source': date_source, 'Combined gas signal': _fmt(signal)}
    for g, r in results.items():
        summary[f'{g} change onset'] = _fmt(r.increase_onset)
        summary[f'{g} confidence'] = r.confidence
        summary[f'{g} peak date'] = _fmt(r.peak_date)
        f = factor.get(g)
        summary[f'{g} peak'] = r.peak_value
        summary[f'{g} unit'] = 'μg/m3' if f else 'mol/m2'
        summary[f'{g} change (%)'] = r.increase_percent
    (directory / 'result.json').write_text(json.dumps(summary, ensure_ascii=False, default=str), encoding='utf-8')
    (directory / 'version.txt').write_text('2026-09-29-wind', encoding='utf-8')
    grid = str(geometry['grid_version'].iloc[0]) if geometry is not None and 'grid_version' in geometry else 'legacy'
    (directory / 'geometry_version.txt').write_text(grid, encoding='utf-8')
    return summary

def _suffix(n):
    return {1: '1', 2: '2', 3: '3'}.get(n, str(n))

def _interpretation(name, start, end, source, signal, results, factor, met, perf, notes) -> str:
    s = [f'{name} WILDFIRE — AUTOMATED GAS EMISSION ANALYSIS', f'Fire period: {_fmt(start)} – {_fmt(end)} (date source: {source})', f'Combined fire gas signal (CO preferred; otherwise concurrent increases in ≥2 gases): {_fmt(signal)}', '']
    for g, r in results.items():
        f = factor.get(g)
        unit = 'μg/m3' if f else 'mol/m2'
        peak = r.peak_value
        gas = g.replace('2', '2').replace('3', '3')
        if r.increase_onset is None:
            s.append(f'• {gas}: No event met the change threshold in the search period.')
            continue
        direction = 'increase' if r.direction == 'Increase' else 'decrease'
        late = ''
        if r.delay_day is not None:
            late = f' (days relative to fire start: {r.delay_day} days after)' if r.delay_day > 0 else ' (fire start date)' if r.delay_day == 0 else f' (days relative to recorded start: {-r.delay_day} days before)'
        sentence = f"• {gas}: {direction} {_fmt(r.increase_onset)} onset date{late}; out of 3 methods: {_suffix(r.agreeing_method)} agree (confidence: {r.confidence}). {('Maximum' if r.direction == 'Increase' else 'Minimum')} value {_fmt(r.peak_date)} on {peak:.4g} {unit}" + (f' change from baseline (%):{r.increase_percent:+.1f} change' if np.isfinite(r.increase_percent) else '; percentage change is undefined for a zero baseline')
        if met is not None and r.peak_date in getattr(met, 'index', []):
            m = met.loc[r.peak_date]
            if 'wind_speed' in m and (not pd.isna(m['wind_speed'])):
                sentence += f"; prevailing wind on that day: {list_stats.direction_name(m.get('wind_direction'))} from {m['wind_speed'] * 3.6:.1f} km/h"
        if r.duration_day:
            sentence += f'. Duration of substantial baseline departure: {r.duration_day} days'
        s.append(sentence + '.')
        who = GASES[g].who_threshold
        if f and who and (peak > who):
            s.append(f'   {gas} for {who:g} μg/m3 threshold exceeded.')
    if perf is not None and (not perf.empty):
        s.append('')
        s.append('Model performance (converted satellite vs. ground station):')
        for _, p in perf.iterrows():
            if 'MAE' in p and (not pd.isna(p.get('MAE'))):
                s.append(f"   {p['Gas']}: MAE={p['MAE']:.2f} μg/m3, RRMSE=%{p['RRMSE_%']:.2f}, IA={p['IA']:.2f} (n={int(p['n'])})")
    if notes:
        s.append('')
        s.append('Notes:')
        s += [f'   - {n}' for n in notes]
    return '\n'.join(s)

def batch_analysis(inventory, output='output', settings=None, gee_use=True, satellite_file=None, station_dictionary=None, search_start=None, search_end=None, gases=None):
    """Implementation of batch_analysis."""
    from .data_reading import satellite_series_read
    settings = settings or AnalysisSettings()
    row = []
    for i in range(len(inventory)):
        r = inventory.iloc[[i]]
        name = r['name'].iloc[0]
        try:
            satellite = None if gee_use else satellite_series_read(satellite_file, name)
            list_stats_df = (station_dictionary or {}).get(name, (station_dictionary or {}).get('*'))
            row.append(fire_analysis(name, r['start'].iloc[0], r['end'].iloc[0], r, satellite, list_stats_df, None, settings, output, gee_use, search_start, search_end, gases))
        except Exception as e:
            print(_best(f'[{name}] ERROR: {e}'))
            row.append({'Fire': name, 'Error': str(e)})
    df = pd.DataFrame(row)
    Path(output).mkdir(parents=True, exist_ok=True)
    df.to_excel(Path(output) / 'ALL_FIRES_SUMMARY.xlsx', index=False)
    return df

def dates_find_batch(inventory, search_start, search_end, settings=None, output='output'):
    """Implementation of dates_find_batch."""
    from . import gee
    from .date_detection import fire_date_find
    settings = settings or AnalysisSettings()
    gee.launch(settings.gee_project)
    summary, section = ([], [])
    for i in range(len(inventory)):
        r = inventory.iloc[i]
        print(_best(f"[{i + 1}/{len(inventory)}] {r['name']}"))
        try:
            tb = fire_date_find(gee.ee_geometry(r.geometry), search_start, search_end)
            summary.append({'name': r['name'], 'start': _fmt(tb['start']), 'end': _fmt(tb['end']), 'source': tb['source'], 'area_ha': round(float(r.get('area_ha_geometry', np.nan)), 1), 'firms_toplam_pixel': float(tb['firms'].sum()) if len(tb['firms']) else 0, 'episode_sayisi': int((tb['episodes']['source'] == 'FIRMS').sum()) if not tb['episodes'].empty else 0, 'notes': ' | '.join(tb['notes'])})
            if not tb['episodes'].empty:
                section.append(tb['episodes'].assign(name=r['name']))
            print(_best(f"   → {summary[-1]['start']} – {summary[-1]['end']} ({tb['source']})"))
        except Exception as e:
            print(_best(f'   ERROR: {e}'))
            summary.append({'name': r['name'], 'notes': f'ERROR: {e}'})
    Path(output).mkdir(parents=True, exist_ok=True)
    file = Path(output) / 'fire_dates.xlsx'
    with pd.ExcelWriter(file, engine='openpyxl') as xw:
        pd.DataFrame(summary).to_excel(xw, sheet_name='Dates', index=False)
        if section:
            pd.concat(section, ignore_index=True).to_excel(xw, sheet_name='Burn episodes', index=False)
    env2 = inventory.drop(columns=['start', 'end'], errors='ignore').merge(pd.DataFrame(summary)[['name', 'start', 'end']], on='name', how='left')
    env2.to_file(Path(output) / 'dated_fire_inventory.gpkg', driver='GPKG')
    print(_best(f'Dates saved → {file}'))
    return pd.DataFrame(summary)
