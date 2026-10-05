"""Implementation of the wildfire gas analysis module."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from .config import GASES
from .analysis_statistics import performance_table
from .comparison import ready, selection_label, label
METRICS = [('MAE', 'MAE (μg/m3)'), ('RMSE', 'RMSE (μg/m3)'), ('nMAE', 'Normalized MAE'), ('NRMSE_mean_%', 'NRMSE (% of station mean)'), ('RRMSE_%', 'RRMSE (%)'), ('Pearson_r', 'Pearson r'), ('IA', 'Index of agreement (IA)')]

def performance_data(root, cases):
    tables = []
    notes = []
    digest = hashlib.sha256(b'performance-extrema-v1')
    for name, case in sorted(cases.items()):
        folder = Path(root) / name
        report = folder / f'{name}_report.xlsx'
        station = folder / 'station_daily_ug_m3.csv'
        digest.update((name + case['fingerprint']).encode())
        a, b = (case['periods']['Before'][0], case['periods']['After'][1])
        year = case.get('selected_year', case['periods']['During fire'][0].year)
        try:
            for file in (report, station):
                if file.exists():
                    digest.update(file.read_bytes())
            table = pd.DataFrame()
            if a.year == year and b.year == year:
                with pd.ExcelFile(report) as book:
                    if 'Performance' in book.sheet_names:
                        table = pd.read_excel(book, sheet_name='Performance')
            if table.empty and station.exists():
                observed = pd.read_csv(station, index_col=0, parse_dates=True)
                observed.index = pd.to_datetime(observed.index).normalize()
                observed = observed.loc[observed.index.year == year].apply(pd.to_numeric, errors='coerce')
                observed = observed.replace([np.inf, -np.inf], np.nan).mask(observed < 0)
                table = performance_table(case['gas'], observed, (max(a, pd.Timestamp(year, 1, 1)), min(b, pd.Timestamp(year, 12, 31)))).reset_index()
            if table.empty or 'Gas' not in table or 'n' not in table:
                notes.append(f'{name}: no paired station performance data')
                continue
            table = table[table.Gas.isin(GASES)].copy()
            table['n'] = pd.to_numeric(table['n'], errors='coerce')
            table = table[table.n.ge(3)]
            for metric, _ in METRICS:
                table[metric] = pd.to_numeric(table.get(metric, pd.Series(index=table.index, dtype=float)), errors='coerce').replace([np.inf, -np.inf], np.nan)
            table['fire'] = name
            table['label'] = f"{label(name)} ({case['periods']['During fire'][0]:%d.%m.%Y})"
            tables.append(table)
        except (OSError, ValueError, KeyError) as exc:
            notes.append(f'{name}: {exc}')
    columns = ['Gas', 'n', 'fire', 'label'] + [m for m, _ in METRICS]
    return (pd.concat(tables, ignore_index=True) if tables else pd.DataFrame(columns=columns), notes, digest.hexdigest())

def metric_extrema(table, gas, metric):
    data = table[table.Gas.eq(gas)].dropna(subset=[metric]).sort_values('fire')
    if data.empty:
        return (data, data, data)
    return (data, data[data[metric].eq(data[metric].min())], data[data[metric].eq(data[metric].max())])

def performance_chart(cache, cases, table, fingerprint):
    year = selection_label(cases)
    key = hashlib.sha256(('performance-chart-v1' + year + fingerprint).encode()).hexdigest()

    def draw():
        fig, axes = plt.subplots(4, 2, figsize=(16, 18))
        fig.subplots_adjust(top=0.91, bottom=0.05, hspace=0.35, wspace=0.22)
        fig.suptitle(f'Performance: converted satellite vs. ground station\nTurkey — {year} · lowest and highest values across analysed wildfires', fontsize=17, fontweight='bold', y=0.98)
        fig.legend(handles=[Line2D([], [], marker='o', color='#168a87', ls='', label='Lowest value'), Line2D([], [], marker='o', color='#d65a28', ls='', label='Highest value'), Line2D([], [], marker='o', color='#9ba9b7', ls='', label='Other fire values')], loc='upper center', bbox_to_anchor=(0.5, 0.946), ncol=3, frameon=False)
        for ax, (metric, title) in zip(axes.flat, METRICS):
            ax.set_title(title, loc='left', fontsize=12, fontweight='bold')
            valid = pd.to_numeric(table[metric], errors='coerce').dropna()
            lo = min(float(valid.min()), 0) if len(valid) else 0
            hi = float(valid.max()) if len(valid) else 1
            span = max(hi - lo, abs(hi) * 0.1, 1e-06)
            ax.set_xlim(lo - span * 0.06, hi + span * 0.1)
            for i, gas in enumerate(GASES):
                data, low, high = metric_extrema(table, gas, metric)
                if data.empty:
                    ax.text(0.03, i, 'No paired data', transform=ax.get_yaxis_transform(), va='center', fontsize=9, color='#687787')
                    continue
                vmin, vmax = (low.iloc[0][metric], high.iloc[0][metric])
                ax.plot([vmin, vmax], [i, i], color='#8795a5', lw=1.5)
                ax.scatter(data[metric], np.full(len(data), i), s=19, color='#9ba9b7', zorder=2)
                ax.scatter([vmin], [i], s=48, color='#168a87', zorder=3)
                ax.scatter([vmax], [i], s=48, facecolors='none', edgecolors='#d65a28', lw=2, zorder=4)
                for rows, value, offset, prefix, color in [(low, vmin, -0.25, 'Min', '#126d6b'), (high, vmax, 0.25, 'Max', '#a93e16')]:
                    names = '; '.join((f'{r.label}, n={int(r.n)}' for r in rows.itertuples()))
                    ax.text(0.02, i + offset, f'{prefix}: {value:.4g} — {names}', transform=ax.get_yaxis_transform(), fontsize=7.7, fontweight='bold', color=color, va='center')
            ax.set_yticks(range(len(GASES)), [GASES[g].name for g in GASES], fontsize=10)
            ax.set_ylim(len(GASES) - 0.5, -0.5)
            ax.grid(axis='y', visible=False)
        ax = axes.flat[-1]
        ax.axis('off')
        count = table.fire.nunique() if not table.empty else 0
        ax.text(0.02, 0.98, f'Interpretation · {count}/{len(cases)} fires with eligible station data\n\nMAE, RMSE, nMAE, NRMSE and RRMSE: lower is better.\nPearson r: closer to +1 indicates stronger positive association.\nIA: closer to 1 indicates greater agreement.\n\nn = paired daily observations; at least 3 are required.\nLabels identify all tied extrema. Missing metrics are excluded.\nEach metric is compared within the same gas and year.\nPeriod: full before/during/after analysis window within the year.\n\nStation data also informed the conversion: this is agreement\non the available data, not independent validation.\nExtrema are descriptive, not statistical significance tests.\n\nnMAE = MAE / station mean.\nNRMSE = RMSE / |station mean| × 100.\nRRMSE = RMSE / √Σ(station2) × 100; depends on sample size.', va='top', fontsize=9, linespacing=1.5)
        return fig
    return ready(cache, key, draw, f'Turkey_{year}_performance_extrema.png')
