"""Implementation of the wildfire gas analysis module."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .config import GASES
from .analysis_statistics import make_periods
from .chart import _save
PERIODS = ['Before', 'During fire', 'After']
COLORS = ['#526b89', '#d65a28', '#228b88']

def read(root):
    cases = {}
    errors = []
    for file in sorted(Path(root).glob('*/result.json')):
        folder = file.parent
        try:
            if (folder / 'version.txt').read_text(encoding='utf-8').strip() != '2026-09-29-wind':
                continue
            summary = json.loads(file.read_text(encoding='utf-8'))
            settings = json.loads((folder / 'settings.json').read_text(encoding='utf-8'))
            start = pd.to_datetime(summary['Start'], format='%d.%m.%Y', errors='coerce')
            end = pd.to_datetime(summary['End'], format='%d.%m.%Y', errors='coerce')
            if pd.isna(start) or pd.isna(end) or end < start:
                raise ValueError('No valid start/end dates')
            periods = make_periods(start, end, settings['before_day'], settings['after_day'])
            report = folder / f'{folder.name}_report.xlsx'
            with pd.ExcelFile(report) as book:
                gas = pd.read_excel(book, sheet_name='Series ug m-3', index_col=0) if 'Series ug m-3' in book.sheet_names else pd.DataFrame()
            if not gas.empty:
                gas.index = pd.to_datetime(gas.index).normalize()
                gas = gas[[g for g in GASES if g in gas]].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan).clip(lower=0)
            metfile = folder / 'weather_daily.csv'
            met = pd.read_csv(metfile, index_col=0, parse_dates=True) if metfile.exists() else pd.DataFrame()
            if not met.empty:
                met.index = pd.to_datetime(met.index).normalize()
                met = met.apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
                if 'wind_speed' in met:
                    met['wind_speed'] *= 3.6
            digest = hashlib.sha256(b'comparison-v1')
            for p in [file, folder / 'settings.json', report, metfile]:
                if p.exists():
                    digest.update(p.read_bytes())
            cases[folder.name] = dict(gas=gas, met=met, periods=periods, summary=summary, fingerprint=digest.hexdigest())
        except (OSError, ValueError, KeyError) as exc:
            errors.append(f'{folder.name}: {exc}')
    return (cases, errors)

def slice(df, a, b):
    return df.loc[(df.index >= a) & (df.index <= b)] if not df.empty else df

def statistics(case, source):
    rows = []
    for period, (a, b) in case['periods'].items():
        data = slice(case[source], a, b)
        for var in data:
            s = data[var].dropna()
            rows.append(dict(period=period, variable=var, n=len(s), minimum=s.min(), maximum=s.max(), mean=s.mean(), std=s.std(ddof=1)))
    return pd.DataFrame(rows, columns=['period', 'variable', 'n', 'minimum', 'Maximum', 'Mean', 'std'])

def maxima(cases, scope):
    rows = []
    for name, c in cases.items():
        a, b = c['periods']['During fire'] if scope == 'During fire' else (c['periods']['Before'][0], c['periods']['After'][1])
        data = slice(c['gas'], a, b)
        for gas in GASES:
            s = data[gas].dropna() if gas in data else pd.Series(dtype=float)
            rows.append(dict(fire=name, gas=gas, maximum=s.max(), date=s.idxmax() if len(s) else pd.NaT, n=len(s)))
    return pd.DataFrame(rows)

def label(name):
    return name.rsplit('_', 1)[0].replace('_', ' ')

def ready(root, key, draw, filename):
    folder = Path(root) / '_comparison' / key
    path = folder / filename
    if not path.exists() or not (folder / '600dpi' / filename).exists():
        folder.mkdir(parents=True, exist_ok=True)
        fig = draw()
        try:
            _save(fig, path)
        finally:
            plt.close(fig)
    return path

def maximum_charts(root, cases, scope):
    table = maxima(cases, scope)
    names = list(cases)
    paths = []
    for page in range(0, len(names), 10):
        selected = names[page:page + 10]
        key = hashlib.sha256(('max-best-v1' + scope + ''.join((n + cases[n]['fingerprint'] for n in selected))).encode()).hexdigest()

        def draw():
            fig, axs = plt.subplots(3, 2, figsize=(14, 12), layout='constrained')
            fig.suptitle(f'Maximum gas concentrations across wildfires · {scope}\nConverted concentration (μg/m3) · Page {page // 10 + 1}', fontsize=15, fontweight='bold')
            for ax, gas in zip(axs.flat, GASES):
                rows = table[table.gas.eq(gas)].set_index('fire').reindex(selected)
                y = np.arange(len(rows))
                vals = rows.maximum.to_numpy()
                ax.barh(y, vals, color='#2478b5', height=0.62)
                labels = []
                for name in selected:
                    a, b = cases[name]['periods']['During fire']
                    labels.append(f'{label(name)}\n{a:%d.%m.%Y}–{b:%d.%m.%Y}')
                ax.set_yticks(y, labels, fontsize=7)
                ax.set_ylim(len(rows) - 0.5, -0.5)
                finite = vals[np.isfinite(vals)]
                ax.set_xlim(0, max(float(finite.max()) if len(finite) else 0, 1e-06) * 1.48)
                ax.set_title(GASES[gas].name, loc='left', fontsize=12, fontweight='bold')
                ax.set_xlabel('μg/m3')
                ax.grid(axis='y', visible=False)
                for i, row in enumerate(rows.itertuples()):
                    label = f'{row.maximum:.4g} · {row.date:%d.%m.%Y}' if pd.notna(row.maximum) else 'No data'
                    ax.annotate(label, (row.maximum if pd.notna(row.maximum) else 0, i), xytext=(4, 0), textcoords='offset points', va='center', fontsize=7, fontweight='bold')
            axs.flat[-1].axis('off')
            axs.flat[-1].text(0.03, 0.8, 'Bar: highest daily value within the selected period\nDate beside bar: day of the maximum\nRow dates: fire start and end\n\nMissing observations are not treated as zero.\nGases use separate axes and the same unit.', va='top', fontsize=11, linespacing=1.7)
            return fig
        paths.append(ready(root, key, draw, f'fire_maksimumlari_page_{page // 10 + 1}.png'))
    return paths

def measurement_charts(root, name, case):
    paths = []
    for source, title in [('gas', 'Gas measurement statistics'), ('met', 'Meteorological statistics')]:
        variables = [(g, GASES[g].name, 'μg/m3') for g in GASES] if source == 'gas' else [('temperature', 'Temperature', '°C'), ('humidity', 'Relative humidity', '%'), ('wind_speed', 'Wind speed', 'km/h')]
        if case[source].empty:
            continue
        stats = statistics(case, source)
        key = hashlib.sha256(('stats-best-v1' + name + case['fingerprint'] + source).encode()).hexdigest()

        def draw():
            fig, axs = plt.subplots(2 if source == 'met' else 3, 2, figsize=(14, 9 if source == 'met' else 12), layout='constrained')
            a, b = case['periods']['During fire']
            fig.suptitle(f'{title} — {label(name)}\nFire: {a:%d.%m.%Y}–{b:%d.%m.%Y}', fontsize=15, fontweight='bold')
            for ax, (var, label, unit) in zip(axs.flat, variables):
                ax.set_title(f'{label} ({unit})', loc='left', fontweight='bold')
                labels = []
                for i, period in enumerate(PERIODS):
                    a, b = case['periods'][period]
                    rows = stats[(stats.variable == var) & (stats.period == period)]
                    row = rows.iloc[0] if len(rows) else None
                    n = int(row.n) if row is not None else 0
                    labels.append(f'{period}\n{a:%d.%m.%Y}–{b:%d.%m.%Y}\nn={n}')
                    if not n:
                        ax.text(i, 0.5, 'No data', transform=ax.get_xaxis_transform(), ha='center', fontsize=8)
                        continue
                    ax.vlines(i, row.minimum, row.maximum, color=COLORS[i], lw=2)
                    ax.plot([i, i], [row.minimum, row.maximum], ls='', marker='_', ms=16, color=COLORS[i])
                    if pd.notna(row['std']):
                        lower = min(row['std'], row.mean) if source == 'gas' else row['std']
                        ax.errorbar(i + 0.1, row.mean, yerr=[[lower], [row['std']]], fmt='none', capsize=4, color=COLORS[i], lw=5, alpha=0.45)
                    ax.plot(i + 0.1, row.mean, 'o', color=COLORS[i], ms=7)
                    sd = f"{row['std']:.3g}" if pd.notna(row['std']) else '—'
                    ax.text(i, 0.97, f'Max: {row.maximum:.3g}\nMin: {row.minimum:.3g}\nMean: {row.mean:.3g}\nSD: {sd}', transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=8, fontweight='bold')
                ax.set_xticks(range(3), labels, fontsize=7)
                ax.set_xlim(-0.5, 2.5)
                ax.margins(y=0.55)
                if source == 'gas':
                    part = stats[stats.variable.eq(var)]
                    top = max(part.maximum.max(), (part.mean + part['std'].fillna(0)).max()) if len(part) else 1
                    ax.set_ylim(0, (top if pd.notna(top) and top > 0 else 1) * 1.5)
                ax.set_ylabel(unit)
                ax.grid(axis='x', visible=False)
            ax = axs.flat[-1]
            ax.axis('off')
            note = '\nFor gases, the lower end of the SD band is clipped at zero.' if source == 'gas' else ''
            ax.text(0.03, 0.88, 'Thin line: minimum–maximum range\nDot: arithmetic mean\nThick band: mean ± 1 standard deviation (SD)\nn: valid daily observations\n\nSD is undefined when n < 2.\nThe SD band is not an observation or a confidence interval.' + note, va='top', fontsize=10, linespacing=1.7)
            return fig
        paths.append(ready(root, key, draw, f'{name}_{source}_measurement_statistics.png'))
    return paths

def statistics_archive_path(root, cases, dpi):
    if dpi not in (300, 600):
        raise ValueError('DPI must be 300 or 600')
    key = hashlib.sha256(('all-stats-best-v1' + str(dpi) + ''.join((name + cases[name]['fingerprint'] for name in sorted(cases)))).encode()).hexdigest()
    return Path(root) / '_comparison' / key / f'Turkey_{selection_label(cases)}_wildfire_measurement_statistics_{dpi}dpi.zip'

def all_statistics_archive(root, cases, dpi, progress=None):
    """Implementation of all_statistics_archive."""
    import zipfile
    from .english import filename
    target = statistics_archive_path(root, cases, dpi)
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix('.tmp')
    lines = ['Turkey wildfire measurement statistics — ' + selection_label(cases), 'Coverage: completed analyses available in this program, not a complete national inventory.', 'Each fire uses its own before, during and after periods. Missing observations are not zeros.', f'Resolution: {dpi} DPI', '', 'Included analyses:']
    try:
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_STORED) as archive:
            for index, name in enumerate(sorted(cases)):
                paths = measurement_charts(root, name, cases[name])
                for path in paths:
                    source = path if dpi == 300 else path.parent / '600dpi' / path.name
                    archive.write(source, name + '/' + filename(path.stem) + f'_{dpi}dpi.png')
                a, b = cases[name]['periods']['During fire']
                lines.append(f'{name}: {a:%d.%m.%Y}–{b:%d.%m.%Y}; {len(paths)} charts')
                if progress:
                    progress((index + 1) / len(cases))
            archive.writestr('README.txt', '\n'.join(lines))
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target

def mean_fire_statistics(cases):
    """Implementation of mean_fire_statistics."""
    rows = []
    for source, variables in [('gas', list(GASES)), ('met', ['temperature', 'humidity', 'wind_speed'])]:
        for variable in variables:
            for period in PERIODS:
                means = []
                for case in cases.values():
                    data = slice(case[source], *case['periods'][period])
                    if variable in data:
                        values = pd.to_numeric(data[variable], errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
                        if len(values):
                            means.append(values.mean())
                rows.append(dict(source=source, variable=variable, period=period, mean=float(np.mean(means)) if means else np.nan, fires=len(means), minimum=min(means) if means else np.nan, maximum=max(means) if means else np.nan, sd=float(np.std(means, ddof=1)) if len(means) > 1 else np.nan))
    return pd.DataFrame(rows)

def mean_fire_chart(root, cases):
    stats = mean_fire_statistics(cases)
    key = hashlib.sha256(('mean-fires-stats-best-v3' + ''.join((name + cases[name]['fingerprint'] for name in sorted(cases)))).encode()).hexdigest()

    def draw():
        fig, axes = plt.subplots(4, 2, figsize=(14, 14))
        fig.subplots_adjust(top=0.9, bottom=0.12, hspace=0.7, wspace=0.25)
        years = sorted({case['periods']['During fire'][0].year for case in cases.values()})
        fig.suptitle(f'Gas and meteorological statistics across fires\nTurkey · {len(cases)} analysed fires · Years: ' + ', '.join(map(str, years)), fontsize=17, fontweight='bold', y=0.975)
        variables = [('gas', gas, GASES[gas].name, 'μg/m3') for gas in GASES]
        variables += [('met', 'temperature', 'Temperature', '°C'), ('met', 'humidity', 'Relative humidity', '%'), ('met', 'wind_speed', 'Wind speed', 'km/h')]
        for ax, (source, var, title, unit) in zip(axes.flat, variables):
            part = stats[(stats.source == source) & (stats.variable == var)].set_index('period').reindex(PERIODS)
            values = part['mean'].to_numpy()
            for i, row in enumerate(part.itertuples()):
                if not np.isfinite(row.mean):
                    continue
                ax.vlines(i, row.minimum, row.maximum, color=COLORS[i], lw=1.8)
                ax.plot([i, i], [row.minimum, row.maximum], marker='_', ls='', color=COLORS[i], ms=13)
                if np.isfinite(row.sd):
                    lower = min(row.sd, row.mean) if var != 'temperature' else row.sd
                    ax.errorbar(i + 0.1, row.mean, yerr=[[lower], [row.sd]], fmt='o', color=COLORS[i], lw=5, alpha=0.55, capsize=4)
                ax.plot(i + 0.1, row.mean, 'o', color=COLORS[i])
                sd = f'{row.sd:.3g}' if np.isfinite(row.sd) else '—'
                ax.text(i, 0.97, f'Max: {row.maximum:.3g}\nMin: {row.minimum:.3g}\nMean: {row.mean:.3g}\nSD: {sd}', transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=9, fontweight='bold')
            ax.set_xticks(range(3), [f'{label}\nn={int(n)} fires' for label, n in zip(['Before fire', 'During fire', 'After fire'], part.fires)], fontsize=9)
            ax.set_xlim(-0.35, 2.35)
            ax.margins(y=0.65)
            if var != 'temperature':
                top = max(part.maximum.max(), (part['mean'] + part.sd.fillna(0)).max())
                top = top if np.isfinite(top) and top > 0 else 1
                ax.set_ylim(0, max(top * 1.75, 1e-06))
            if var == 'temperature':
                low = min(part.minimum.min(), (part['mean'] - part.sd.fillna(0)).min())
                high = max(part.maximum.max(), (part['mean'] + part.sd.fillna(0)).max())
                if np.isfinite(low) and np.isfinite(high):
                    span = max(high - low, 1)
                    ax.set_ylim(low - span * 0.15, high + span * 1.05)
            ax.set_title(f'{title} ({unit})', loc='left', fontweight='bold', fontsize=12)
            ax.set_ylabel(unit)
            ax.grid(axis='x', visible=False)
        fig.text(0.06, 0.035, 'Min, max, mean and sample SD describe individual fire-period means; each fire has equal weight. n: contributing fires.\nThin line: min–max; dot: mean; thick band: mean ± SD (lower bound clipped at zero except temperature). SD requires n ≥ 2.\nCoverage: available completed analyses, not a complete national wildfire inventory.', fontsize=10, linespacing=1.5)
        return fig
    return ready(root, key, draw, f'Turkey_{selection_label(cases)}_all_fires_mean_gas_and_weather.png')

def selected_year_cases(cases, year):
    """Implementation of selected_year_cases."""
    selected = {}
    for name, case in cases.items():
        if case['periods']['During fire'][0].year != int(year):
            continue
        item = case.copy()
        for source in ('gas', 'met'):
            data = case[source]
            item[source] = data.loc[data.index.year == int(year)].copy() if not data.empty else data.copy()
        item['selected_year'] = int(year)
        item['fingerprint'] = hashlib.sha256((case['fingerprint'] + f':calendar-year-v1:{year}').encode()).hexdigest()
        selected[name] = item
    return selected

def selection_label(cases):
    years = sorted({case['periods']['During fire'][0].year for case in cases.values()})
    return '-'.join(map(str, years))
