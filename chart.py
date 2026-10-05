"""Implementation of the wildfire gas analysis module."""
from __future__ import annotations
from pathlib import Path
from .english import install_plot_language
install_plot_language()
import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .config import GASES
BLUE = '#2a78d6'
ORANGE = '#eb6834'
RED = '#d03b3b'
PURPLE = '#4a3aa7'
GREEN = '#1baf7a'
INK = '#0b0b0b'
SECONDARY = '#52514e'
FADED = '#898781'
PERIOD_COLOR = {'Before': '#f0efec', 'During fire': '#fde6dc', 'After': '#e6eef8'}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.edgecolor': FADED, 'axes.labelcolor': SECONDARY, 'xtick.color': SECONDARY, 'ytick.color': SECONDARY, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': '#e2e8f0', 'grid.linewidth': 0.6, 'figure.dpi': 110, 'savefig.dpi': 300, 'axes.facecolor': '#f8fafc', 'figure.facecolor': 'white', 'axes.titleweight': 'bold', 'axes.axisbelow': True, 'lines.solid_capstyle': 'round'})

def _save(fig, file):
    """Implementation of _save."""
    for ax in fig.axes:
        for text in [*ax.get_xticklabels(), *ax.get_yticklabels(), ax.xaxis.get_offset_text(), ax.yaxis.get_offset_text()]:
            text.set_fontweight('bold')
            text.set_color(INK)
    file = Path(file)
    fig.savefig(file, dpi=300, bbox_inches='tight')
    high = file.parent / '600dpi' / file.name
    high.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(high, dpi=600, bbox_inches='tight')

def _period_shading(ax, period, label=True):
    for name, (a, b) in period.items():
        ax.axvspan(a - pd.Timedelta(hours=12), b + pd.Timedelta(hours=12), color=PERIOD_COLOR[name], zorder=0, lw=0)
        if label:
            ax.text(a + (b - a) / 2, 1.01, name, transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=8, color=SECONDARY)

def _date_axis(ax):
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=6, interval_multiples=False))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m.%Y'))
    ax.set_xlabel('Date (day.month.year)', fontsize=9)
    ax.tick_params(axis='x', labelsize=8, labelrotation=20)
    ax.margins(x=0.01)

def gas_time_series(series, result, period, unit, title, file: Path, factor=1.0, gas=None):
    """Implementation of gas_time_series."""
    fig, ax = plt.subplots(figsize=(10, 4.2))
    if period:
        _period_shading(ax, period)
    s = series.asfreq('D')
    s = s.clip(lower=0)
    ax.plot(s.dropna().index, s.dropna().values, color=BLUE, lw=2, marker='o', ms=4, zorder=3)
    if gas == 'SO2':
        zeros = s[s.eq(0)]
        ax.scatter(zeros.index, zeros.values, s=32, color=BLUE, edgecolors='white', linewidths=0.6, zorder=5, clip_on=False)
    if result is not None:
        med = result.baseline_median * factor
        med = max(0, med)
        ax.axhline(med, color=FADED, lw=1, ls='--', zorder=2)
        ax.text(1.0, med, 'baseline median ', transform=ax.get_yaxis_transform(), color=SECONDARY, fontsize=7, va='bottom', ha='right')
        if result.increase_onset is not None:
            ax.axvline(result.increase_onset, color=ORANGE, lw=1.5, ls='--', zorder=2)
            ax.text(result.increase_onset, 0.5, f" {('Increase' if result.direction == 'Increase' else 'Decrease')} onset\n {result.increase_onset:%d.%m.%Y}", transform=ax.get_xaxis_transform(), va='center', fontsize=8, color=INK, bbox=dict(fc='white', ec='none', alpha=0.8, pad=1))
        if result.peak_date is not None and (not np.isnan(result.peak_value)):
            tv = result.peak_value * factor
            tv = max(0, tv)
            ax.plot([result.peak_date], [tv], 'o', ms=9, mfc='none', mec=RED, mew=2, zorder=4)
            ax.annotate(f"{('Peak' if tv >= med else 'Minimum')}: {tv:.4g}", (result.peak_date, tv), xytext=(9, -3), textcoords='offset points', fontsize=8, color=INK)
    ax.set_title(title, loc='left', fontsize=10, color=INK, pad=16)
    ax.set_ylabel(unit)
    ax.set_ylim(bottom=0)
    _date_axis(ax)
    fig.tight_layout()
    _save(fig, file)
    plt.close(fig)

def multiple_panel(df, results, period, unit, title, file: Path, factors=None, firms=None):
    gases = [g for g in df.columns if df[g].notna().any()]
    if not gases:
        return
    f = None
    if firms is not None and len(firms):
        f = firms[(firms.index >= df.index.min()) & (firms.index <= df.index.max())]
        f = f if len(f) else None
    n = len(gases) + (1 if f is not None else 0)
    fig, axs = plt.subplots(n, 1, figsize=(8, 1.9 * n + 0.6), sharex=True)
    axs = np.atleast_1d(axs)
    if f is not None:
        ax0 = axs[0]
        if period:
            _period_shading(ax0, period)
        ax0.bar(f.index, f.values, width=0.7, color=RED)
        ax0.set_ylabel('FIRMS active\nfire pixels', fontsize=8)
        ax0.yaxis.set_major_locator(plt.MaxNLocator(4, integer=True))
        axs_gas = axs[1:]
    else:
        axs_gas = axs
    first = f is None
    for i, (ax, g) in enumerate(zip(axs_gas, gases)):
        if period:
            _period_shading(ax, period, label=i == 0 and first)
        s = df[g].asfreq('D')
        s = s.clip(lower=0)
        ax.plot(s.dropna().index, s.dropna().values, color=BLUE, lw=1.8, marker='o', ms=3)
        if g == 'SO2':
            zeros = s[s.eq(0)]
            ax.scatter(zeros.index, zeros.values, s=25, color=BLUE, edgecolors='white', linewidths=0.5, zorder=5, clip_on=False)
        r = results.get(g)
        if r is not None and r.increase_onset is not None:
            ax.axvline(r.increase_onset, color=ORANGE, lw=1.3, ls='--')
        ax.set_ylabel(f'{GASES[g].name}\n({unit})' if g in GASES else g, fontsize=8)
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        ax.set_ylim(bottom=0)
    _date_axis(axs[-1])
    axs[0].set_title(title, loc='left', fontsize=10, color=INK, pad=16)
    axs[-1].plot([], [], color=ORANGE, ls='--', label='Detected change onset')
    axs[-1].legend(loc='upper right', fontsize=7, frameon=False)
    fig.tight_layout()
    _save(fig, file)
    plt.close(fig)

def detection_chart(zscores, results, settings, start, file: Path, title: str):
    """Implementation of detection_chart."""
    gases = [g for g in zscores if zscores[g].notna().any()]
    if not gases:
        return
    fig, axs = plt.subplots(len(gases), 1, figsize=(8, 1.7 * len(gases) + 0.8), sharex=True)
    axs = np.atleast_1d(axs)
    for ax, g in zip(axs, gases):
        z = zscores[g]
        r = results[g]
        ax.bar(z.index, z.values, width=0.7, color=[ORANGE if v >= settings.z_threshold else '#b9c9dd' for v in z.fillna(0).values])
        ax.axhline(settings.z_threshold, color=SECONDARY, lw=1, ls='--')
        ax.axhline(0, color=FADED, lw=0.8)
        if start is not None and (not pd.isna(start)):
            ax.axvline(start, color=INK, lw=1, ls=':')
        for t, m, color, name in ((r.threshold_date, 'v', PURPLE, 'Threshold'), (r.cusum_date, 's', GREEN, 'CUSUM'), (r.pelt_date, 'D', RED, 'PELT')):
            if t is not None:
                ax.plot([t], [ax.get_ylim()[1] * 0.9], m, color=color, ms=7, label=name)
        direction = ' (decrease direction)' if r.direction == 'Decrease' else ''
        ax.set_ylabel(f'{(GASES[g].name if g in GASES else g)}\nz{direction}', fontsize=8)
    h = [plt.Line2D([], [], marker=m, color=c, ls='', ms=7) for m, c in (('v', PURPLE), ('s', GREEN), ('D', RED))]
    h += [plt.Line2D([], [], color=SECONDARY, ls='--'), plt.Line2D([], [], color=INK, ls=':')]
    axs[0].legend(h, ['Threshold method', 'CUSUM', 'PELT', f'z = {settings.z_threshold}', 'Recorded start'], ncol=5, fontsize=7, frameon=False, loc='lower left', bbox_to_anchor=(0, 1.02))
    axs[0].set_title(title, loc='left', fontsize=10, color=INK, pad=26)
    _date_axis(axs[-1])
    fig.tight_layout()
    _save(fig, file)
    plt.close(fig)

def weather_chart(met, period, file: Path, title: str):
    met = met.copy()
    if 'wind_speed' in met:
        met['wind_speed'] = met['wind_speed'] * 3.6
    column = [(c, e) for c, e in (('temperature', 'Temperature (°C)'), ('humidity', 'Relative humidity (%)'), ('wind_speed', 'Wind speed (km/h)')) if c in met and met[c].notna().any()]
    if not column:
        return
    fig, axs = plt.subplots(len(column), 1, figsize=(10, 2.1 * len(column) + 1.0), sharex=True)
    axs = np.atleast_1d(axs)
    for i, (ax, (c, e)) in enumerate(zip(axs, column)):
        if period:
            _period_shading(ax, period, label=i == 0)
        ax.set_ylabel(e, fontsize=8)
        ax.plot(met[c].dropna().index, met[c].dropna(), color=BLUE, lw=1.8, marker='o', ms=3, label='Wind speed (km/h)' if c == 'wind_speed' else None)
        if c == 'wind_speed':
            upper = max(float(met[c].max()), 1.0)
            ax.set_ylim(-0.18 * upper, 1.35 * upper)
            ax.yaxis.set_major_locator(plt.MaxNLocator(5))
            ax.set_yticks([v for v in ax.get_yticks() if 0 <= v <= 1.35 * upper])
            ax.axhline(0, color=FADED, lw=0.7)
            if 'wind_direction' in met:
                direction = np.deg2rad(met.wind_direction.where(met[c] >= 1.8).values)
                ax.quiver(met.index, met[c], -np.sin(direction), -np.cos(direction), color=SECONDARY, angles='uv', pivot='mid', width=0.003, scale=42, zorder=4)
                ax.plot([], [], color=SECONDARY, marker='$→$', ls='', ms=12, label='Downwind direction')
            ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.01), ncol=2, fontsize=8, frameon=False)
    axs[0].set_title(title, loc='left', fontsize=10, color=INK, pad=16)
    _date_axis(axs[-1])
    fig.text(0.08, 0.025, 'ARROW LEGEND — downwind: ↑ North · ↗ Northeast · → East · ↘ Southeast\n↓ South · ↙ Southwest · ← West · ↖ Northwest. Wind origin is opposite the arrow. Length does not encode speed; directions below 1.8 km/h are omitted.', fontsize=8, color=SECONDARY)
    fig.tight_layout(rect=(0, 0.095, 1, 1))
    _save(fig, file)
    plt.close(fig)

def map(tif: Path, polygon_gdf, factor, unit, title, file: Path, gas=None, wind_direction=None, wind_speed=None, panel=None):
    try:
        import rasterio
    except ImportError:
        print('Rasterio is unavailable; the map could not be generated.')
        return
    with rasterio.open(tif) as src:
        from rasterio.warp import calculate_default_transform, reproject, Resampling
        from rasterio.transform import array_bounds
        if src.crs is None:
            raise ValueError('A raster coordinate reference system is required for the scale bar.')
        transform, width, height = calculate_default_transform(src.crs, 'EPSG:4326', src.width, src.height, *src.bounds)
        a = np.full((height, width), np.nan)
        reproject(src.read(1).astype(float), a, src_transform=src.transform, src_crs=src.crs, src_nodata=src.nodata, dst_transform=transform, dst_crs='EPSG:4326', dst_nodata=np.nan, resampling=Resampling.nearest)
        left, bottom, right, top = array_bounds(height, width, transform)
    a = a * factor
    a = np.maximum(a, 0)
    from .map_style import STUDY, north_arrow, coordinates, scale, study_legend
    import matplotlib.patheffects as pe
    from matplotlib.lines import Line2D
    from pyproj import Geod
    fig = panel if panel is not None else plt.figure(figsize=(10, 7.6))
    ax = fig.add_axes([0.09, 0.19, 0.54, 0.63 if panel is not None else 0.7])
    vmax = max(float(np.nanmax(a)), 1e-12) if np.isfinite(a).any() else 1.0
    im = ax.imshow(a, extent=(left, right, bottom, top), cmap='YlOrRd', origin='upper', interpolation='nearest', vmin=0, vmax=vmax)
    polygon_gdf.to_crs(4326).boundary.plot(ax=ax, color=STUDY, lw=1.6, path_effects=[pe.withStroke(linewidth=2.5, foreground='black')])
    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top)
    ax.set_aspect(1 / np.cos(np.deg2rad((bottom + top) / 2)))
    ax.set_xlabel('Longitude (°)')
    ax.set_ylabel('Latitude (°)')
    ax.ticklabel_format(useOffset=False)
    coordinates(ax, fontsize=8, nx=4, ny=5)
    north_arrow(fig, [0.89, 0.74, 0.045, 0.13])
    fig.suptitle(title, x=0.09, ha='left', fontsize=12, fontweight='bold', y=0.96)
    cax = fig.add_axes([0.74, 0.58, 0.023, 0.28])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(unit, fontsize=10)
    fig.text(0.73, 0.9, 'LEGEND', fontsize=11, fontweight='bold', color=INK)
    fig.legend(handles=[study_legend()], loc='center left', bbox_to_anchor=(0.71, 0.51), frameon=True, fancybox=False, edgecolor='black', facecolor='white', fontsize=9)
    wind = fig.add_axes([0.73, 0.23, 0.18, 0.22])
    wind.set(xlim=(-1.5, 1.5), ylim=(-1.5, 1.5), aspect='equal')
    wind.axis('off')
    for x, y, label in [(0, 1.25, 'North'), (1.18, 0, 'D'), (0, -1.2, 'G'), (-1.18, 0, 'B')]:
        wind.text(x, y, label, ha='center', va='center', fontsize=9)
    wind.plot([0, 0], [-0.8, 0.8], color='#cbd5e1', lw=1)
    wind.plot([-0.8, 0.8], [0, 0], color='#cbd5e1', lw=1)
    if wind_direction is not None and np.isfinite(wind_direction) and (wind_speed is not None) and np.isfinite(wind_speed) and (wind_speed > 0):
        angle = np.deg2rad(wind_direction)
        wind.annotate('', xy=(-np.sin(angle), -np.cos(angle)), xytext=(0, 0), arrowprops=dict(arrowstyle='-|>', color=BLUE, lw=2.5))
        from .analysis_statistics import direction_name
        fig.text(0.73, 0.19, f'Wind: {wind_speed * 3.6:.1f} km/h\nWind origin: {direction_name(wind_direction)} ({wind_direction % 360:.0f}°)', fontsize=9)
    else:
        fig.text(0.73, 0.18, 'Wind: Not estimable\nor calm conditions', fontsize=9)
    lat = (bottom + top) / 2
    fig.canvas.draw()
    scale(fig, ax, 0.07)
    fig.text(0.09, 0.022, f'Scale calculated at the central latitude ({lat:.2f}°).', fontsize=8, color=SECONDARY)
    if panel is None:
        _save(fig, file)
        plt.close(fig)
