"""Implementation of the wildfire gas analysis module."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .config import GASES, AnalysisSettings
from .comparison import maxima, ready, label, selection_label

def winners(cases, scope):
    data = maxima(cases, scope).dropna(subset=['Maximum'])
    if data.empty:
        return data
    return data.sort_values(['Maximum', 'date', 'fire'], ascending=[False, True, True]).drop_duplicates('gas').set_index('gas').reindex(GASES).dropna(subset=['Maximum']).reset_index()

def maxima_chart(root, cases, scope):
    table = winners(cases, scope)
    key = hashlib.sha256(('winners-v1' + scope + ''.join((n + c['fingerprint'] for n, c in sorted(cases.items())))).encode()).hexdigest()

    def draw():
        fig, axes = plt.subplots(3, 2, figsize=(14, 11), layout='constrained')
        fig.suptitle(f'Wildfires with the highest gas concentrations in Turkey — {selection_label(cases)}\n' + ('During fire' if scope == 'During fire' else 'Full analysis period'), fontsize=17, fontweight='bold')
        for ax, gas in zip(axes.flat, GASES):
            row = table[table.gas.eq(gas)]
            ax.set_title(GASES[gas].name, fontweight='bold', loc='left')
            if row.empty:
                ax.text(0.5, 0.5, 'No converted data', ha='center')
                ax.axis('off')
                continue
            row = row.iloc[0]
            ax.barh([0], [row.maximum], height=0.4, color='#d65a28')
            ax.set_xlim(0, max(row.maximum * 1.4, 1e-06))
            ax.set_ylim(-0.6, 0.6)
            ax.set_yticks([])
            ax.set_xlabel('μg/m3')
            ax.text(0.02, 0.93, f'{label(row.fire)} · {row.date:%d.%m.%Y}', transform=ax.transAxes, fontweight='bold')
            ax.annotate(f'{row.maximum:.4g}', (row.maximum, 0), xytext=(6, 0), textcoords='offset points', va='center', fontweight='bold')
        axes.flat[-1].axis('off')
        axes.flat[-1].text(0.02, 0.9, f'{len(cases)} completed analyses screened.\nEach panel identifies the fire with the highest\ndaily converted concentration for one gas.\n\nAxes have different ranges; units are μg/m3.\nTies: earliest date, then case ID.\nMissing observations are excluded.', va='top', fontsize=11, linespacing=1.6)
        return fig
    return ready(root, key, draw, f'Turkey_{selection_label(cases)}_gas_maxima_winning_fires.png')

def map_inputs(data_root, cache_root, cases, scope, fetch=False):
    import geopandas as gpd
    from dataclasses import fields
    from . import gee
    inputs = []
    errors = []
    table = winners(cases, scope)
    inventories = []
    for file in Path(data_root).glob('fires_*.geojson'):
        inventories.append((file, gpd.read_file(file)))
    for row in table.itertuples():
        folder = Path(data_root) / row.fire
        try:
            version = (folder / 'geometry_version.txt').read_text().strip() if (folder / 'geometry_version.txt').exists() else 'legacy'
            candidates = [(p, g[g.name.eq(row.fire)]) for p, g in inventories if 'name' in g and g.name.eq(row.fire).any()]
            candidates.sort(key=lambda item: ('native_v2' in item[0].name) != (version == 'native_v2'))
            if not candidates:
                raise ValueError('Fire geometry unavailable')
            geo = candidates[0][1]
            factors = pd.read_excel(folder / f'{row.fire}_report.xlsx', sheet_name='Conversion factors').set_index('Gas')['Factor (μg m-3 / mol m-2)']
            factor = float(factors[row.gas])
            if not np.isfinite(factor) or factor <= 0:
                raise ValueError('Valid conversion factor unavailable')
            name = f'{row.gas}_{row.date:%Y%m%d}.tif'
            tif = folder / 'maps' / name
            if not tif.exists():
                tif = Path(cache_root) / '_maximum_rasters' / row.fire / name
                if not tif.exists() and fetch:
                    import ee
                    settings = json.loads((folder / 'settings.json').read_text(encoding='utf-8'))
                    settings = AnalysisSettings(**{k: v for k, v in settings.items() if k in {f.name for f in fields(AnalysisSettings)}})
                    ee.Initialize(project=settings.gee_project)
                    tif.parent.mkdir(parents=True, exist_ok=True)
                    result = gee.peak_map_download(gee.ee_geometry(geo.to_crs(4326).geometry.union_all()), row.date, row.gas, settings, tif)
                    if result is None:
                        raise ValueError('Exact-day raster download failed')
                if not tif.exists():
                    raise ValueError('Exact-day raster not downloaded')
            inputs.append(dict(gas=row.gas, name=row.fire, date=row.date, maximum=row.maximum, tif=tif, geo=geo, factor=factor))
        except Exception as exc:
            errors.append(f'{row.gas} · {row.fire} · {row.date:%d.%m.%Y}: {exc}')
    return (inputs, errors)

def maxima_maps(root, cases, scope, inputs):
    from .chart import map
    signature = 'atlas-v2' + scope + ''.join((i['tif'].name + ''.join(i['geo'].geometry.to_wkb(hex=True)) + hashlib.sha256(i['tif'].read_bytes()).hexdigest() + str(i['factor']) + cases[i['name']]['fingerprint'] for i in inputs))
    key = hashlib.sha256(signature.encode()).hexdigest()

    def draw():
        fig = plt.figure(figsize=(18, 21))
        panels = fig.subfigures(3, 2, hspace=0.06, wspace=0.02)
        lookup = {i['gas']: i for i in inputs}
        for panel, gas in zip(panels.flat, GASES):
            item = lookup.get(gas)
            if item is None:
                panel.text(0.5, 0.5, f'{GASES[gas].name}\nExact-day map unavailable', ha='center')
                continue
            met = cases[item['name']]['met']
            wind = met.loc[item['date']] if item['date'] in met.index else {}
            speed = wind.get('wind_speed', np.nan)
            title = f"{label(item['name'])} — {GASES[gas].name}\n{item['date']:%d.%m.%Y} · Daily concentration: {item['maximum']:.4g} μg/m3"
            map(item['tif'], item['geo'], item['factor'], 'μg/m3', title, None, gas=gas, wind_direction=wind.get('wind_direction'), wind_speed=speed / 3.6, panel=panel)
        panels.flat[-1].text(0.12, 0.78, f'Gas maxima — wildfire map atlas · {selection_label(cases)}', fontsize=17, fontweight='bold')
        panels.flat[-1].text(0.12, 0.68, ('During fire' if scope == 'During fire' else 'Full analysis period') + '\n\nEach map shows its gas on the exact maximum day.\nYellow boundary: wildfire study area.\nArrow: direction of air movement.\nWind label: direction the wind comes from.\n\nColour scales differ between gases.\nRaster cells show the spatial distribution; the\nreported daily value uses the analysis aggregation.\n\nCoverage: available completed fire analyses.', fontsize=12, va='top', linespacing=1.6)
        return fig
    return ready(root, key, draw, f'Turkey_{selection_label(cases)}_maximum_gas_fire_maps.png')
