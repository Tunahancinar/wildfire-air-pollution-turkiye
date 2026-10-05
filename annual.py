"""Implementation of the wildfire gas analysis module."""
import hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .comparison import maxima, ready, label
from .config import GASES
from .analysis_statistics import direction_name

def year_data(cases, year, scope):
    selected = {}
    for name, case in cases.items():
        if case['periods']['During fire'][0].year != year:
            continue
        item = case.copy()
        item['gas'] = case['gas'].loc[case['gas'].index.year == year].sort_index() if not case['gas'].empty else case['gas']
        selected[name] = item
    if not selected:
        return pd.DataFrame()
    table = maxima(selected, scope)
    for col in ['wind_speed', 'wind_direction', 'humidity', 'temperature']:
        table[col] = np.nan
    for i, row in table.iterrows():
        met = selected[row.fire]['met']
        if pd.isna(row.date) or row.date not in met.index:
            continue
        day = met.loc[row.date]
        if isinstance(day, pd.DataFrame):
            raise ValueError(f'{row.fire}: multiple weather records exist for the same day')
        for col in ['wind_speed', 'wind_direction', 'humidity', 'temperature']:
            table.loc[i, col] = day.get(col, np.nan)
    return table

def count(value, digits=1):
    return f'{value:.{digits}f}' if pd.notna(value) else 'No data'

def annual_charts(root, cases, year, scope):
    table = year_data(cases, year, scope)
    if table.empty:
        return []
    names = list(dict.fromkeys(table.fire))
    paths = []
    for gas in GASES:
        allrows = table[table.gas.eq(gas)]
        best = allrows.maximum.max()
        for start in range(0, len(names), 8):
            selected = names[start:start + 8]
            rows = allrows.set_index('fire').reindex(selected)
            key = hashlib.sha256(('year-best-v1' + str(year) + gas + scope + str(best) + ''.join((n + cases[n]['fingerprint'] for n in selected))).encode()).hexdigest()

            def draw():
                fig = plt.figure(figsize=(14, 9), layout='constrained')
                gs = fig.add_gridspec(2, 1, height_ratios=[1.5, 1.1])
                ax = fig.add_subplot(gs[0])
                detail = fig.add_subplot(gs[1])
                detail.axis('off')
                fig.suptitle(f'{year} · {GASES[gas].name} maximum concentrations\n{scope} · Converted μg/m3 · Page {start // 8 + 1}', fontsize=15, fontweight='bold')
                vals = rows.maximum.to_numpy()
                positions = np.arange(len(rows))
                colors = ['#dc652d' if pd.notna(v) and v == best else '#287fb5' for v in vals]
                ax.bar(positions, vals, color=colors, width=0.58)
                ax.set_xticks(positions, [f"{label(n)}\nRecord: {n.rsplit('_', 1)[-1]}" for n in selected], fontsize=8)
                ax.set_ylabel('Maximum concentration (μg/m3)')
                top = float(np.nanmax(vals)) if np.isfinite(vals).any() else 0
                ax.set_ylim(0, (top if top > 0 else 1) * 1.35)
                ax.grid(axis='x', visible=False)
                for i, row in enumerate(rows.itertuples()):
                    label = f'{row.maximum:.4g}\n{row.date:%d.%m.%Y}' if pd.notna(row.maximum) else 'No data'
                    ax.annotate(label, (i, row.maximum if pd.notna(row.maximum) else 0), xytext=(0, 7), textcoords='offset points', ha='center', fontsize=9, fontweight='bold')
                cells = []
                for name, row in rows.iterrows():
                    full = dict(zip(['K', 'KKD', 'KD', 'DKD', 'D', 'DGD', 'GD', 'GGD', 'G', 'GGB', 'GB', 'BGB', 'B', 'BKB', 'KB', 'KKB'], ['North', 'North-northeast', 'Northeast', 'East-northeast', 'East', 'East-southeast', 'Southeast', 'South-southeast', 'South', 'South-southwest', 'Southwest', 'West-southwest', 'West', 'West-northwest', 'Northwest', 'North-northwest']))
                    direction = f'{full[direction_name(row.wind_direction)]} · {row.wind_direction % 360:.0f}°' if pd.notna(row.wind_direction) else 'No data'
                    if pd.notna(row.wind_speed) and row.wind_speed < 1.8:
                        direction = 'Calm / direction undefined'
                    cells.append([name, row.date.strftime('%d.%m.%Y') if pd.notna(row.date) else 'No data', count(row.wind_speed), direction, count(row.humidity), count(row.temperature)])
                detail.set_title('Meteorological conditions on each gas peak date', loc='left', fontsize=11, fontweight='bold', pad=10)
                tbl = detail.table(cellText=cells, colLabels=['Fire record', 'Peak date', 'Wind speed\n(km/h)', 'Wind origin', 'Relative humidity\n(%)', 'Temperature\n(°C)'], colWidths=[0.24, 0.15, 0.13, 0.23, 0.12, 0.13], bbox=[0, 0.16, 1, 0.82], cellLoc='center')
                tbl.auto_set_font_size(False)
                tbl.set_fontsize(9)
                for (r, c), cell in tbl.get_celld().items():
                    cell.set_edgecolor('#cbd5e1')
                    cell.set_linewidth(0.6)
                    cell.set_facecolor('#183b51' if r == 0 else '#edf4f8' if r % 2 else 'white')
                    cell.set_text_props(color='white' if r == 0 else '#111111', weight='bold' if r == 0 or c >= 2 else 'normal')
                detail.text(0, 0.06, 'Orange: highest value for the selected year and period. Weather values are daily records, not conditions at the measurement instant.', fontsize=9, transform=detail.transAxes)
                detail.text(0, 0, 'Tied maxima use the first day. Missing weather is not replaced by another day. Year refers to fire start year; gas observations are restricted to that year.', fontsize=8, transform=detail.transAxes)
                return fig
            paths.append(ready(root, key, draw, f'{year}_{gas}_maximum_ve_weather_{start // 8 + 1}.png'))
    return paths
