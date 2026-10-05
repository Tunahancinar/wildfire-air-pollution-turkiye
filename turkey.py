"""Implementation of the wildfire gas analysis module."""
from __future__ import annotations
from .english import tr as _best
import json
from pathlib import Path
import pandas as pd

def _turkey(ee):
    return ee.FeatureCollection('FAO/GAUL/2015/level0').filter(ee.Filter.eq('ADM0_NAME', 'Turkey')).geometry()

def turkey_fires(year: int, min_ha: float=100, min_forest: float=0.3, shrubland_forest: bool=True, source: str='MCD64A1', project=None):
    """Implementation of turkey_fires."""
    import geopandas as gpd
    from . import gee
    ee = gee.launch(project)
    tr = _turkey(ee)
    a, b = (f'{year}-01-01', f'{year + 1}-01-01')
    if source.upper() == 'MCD64A1':
        col = ee.ImageCollection('MODIS/061/MCD64A1').filterDate(a, b).select('BurnDate')
        bd = col.map(lambda i: i.updateMask(i.gt(0)))
        first, last = (bd.min().rename('first'), bd.max().rename('last'))
        mask = first.gt(0).rename('yanik')
        scale = 463.3127
    else:
        col = ee.ImageCollection('FIRMS').filterDate(a, b).select('T21')
        day = col.map(lambda i: i.gt(0).selfMask().multiply(ee.Number(i.date().getRelative('day', 'year')).add(1)).toInt16())
        first, last = (day.min().rename('first'), day.max().rename('last'))
        mask = first.gt(0).rename('yanik')
        scale = 926.6254
    native = col.first().projection().getInfo()
    vector = mask.addBands(first).addBands(last).reduceToVectors(geometry=tr, crs=native['crs'], crsTransform=native['transform'], geometryType='polygon', eightConnected=True, labelProperty='yanik', reducer=ee.Reducer.min().combine(ee.Reducer.max(), '', False), maxPixels=10000000000.0, bestEffort=False, tileScale=4)
    vector = vector.map(lambda f: f.set('area_ha', f.geometry().area(1).divide(10000.0)))
    vector = vector.filter(ee.Filter.gte('area_ha', min_ha))
    wc = ee.ImageCollection('ESA/WorldCover/v200').first().select('Map')
    forest = wc.eq(10).Or(wc.eq(20)) if shrubland_forest else wc.eq(10)
    firms = ee.ImageCollection('FIRMS').filterDate(a, b).select('T21').max().gt(0).selfMask()
    province = ee.FeatureCollection('FAO/GAUL/2015/level1').filterBounds(tr)
    district = ee.FeatureCollection('FAO/GAUL/2015/level2').filterBounds(tr)

    def enrich(f):
        g = f.geometry()
        ratio = forest.reduceRegion(ee.Reducer.mean(), g, 100, maxPixels=1000000000.0, bestEffort=True).get('Map')
        fp = firms.reduceRegion(ee.Reducer.count(), g.buffer(500), 1000, maxPixels=1000000000.0, bestEffort=True).get('T21')
        m = g.centroid(100)
        province_name = ee.Algorithms.If(province.filterBounds(m).size().gt(0), province.filterBounds(m).first().get('ADM1_NAME'), '')
        district_name = ee.Algorithms.If(district.filterBounds(m).size().gt(0), district.filterBounds(m).first().get('ADM2_NAME'), '')
        return f.set({'forest_fraction': ratio, 'firms_pixel': fp, 'province': province_name, 'district': district_name})
    fc = vector.map(enrich).filter(ee.Filter.gte('forest_fraction', min_forest))
    geo = fc.getInfo()
    if not geo['features']:
        return gpd.GeoDataFrame(columns=['id', 'name', 'geometry'], geometry='geometry', crs='EPSG:4326')
    gdf = gpd.GeoDataFrame.from_features(geo['features'], crs='EPSG:4326')
    return _format(gdf, year, source)

def _day(year, doy):
    if doy is None or pd.isna(doy) or doy <= 0:
        return pd.NaT
    return pd.Timestamp(int(year), 1, 1) + pd.Timedelta(days=int(doy) - 1)

def _format(gdf, year, source):

    def select(*names):
        for a in names:
            if a in gdf:
                return gdf[a]
        return [None] * len(gdf)
    first = select('min', 'first_min', 'first')
    last = select('max', 'last_max', 'last')
    gdf['mcd_start'] = [_day(year, v) for v in first]
    gdf['mcd_end'] = [_day(year, v) for v in last]
    gdf['source'] = source
    gdf['grid_version'] = 'native_v2'
    for c in ('province', 'district'):
        gdf[c] = gdf.get(c, '').fillna('').astype(str)
    gdf = gdf.sort_values('area_ha', ascending=False).reset_index(drop=True)
    gdf['id'] = range(1, len(gdf) + 1)
    gdf['name'] = [f"{r.district or r.province or 'Fire'}_{r.mcd_start:%Y%m%d}" if pd.notna(r.mcd_start) else f"{r.district or r.province or 'Fire'}_{r.id}" for r in gdf.itertuples()]
    keep = ['id', 'name', 'province', 'district', 'area_ha', 'forest_fraction', 'mcd_start', 'mcd_end', 'firms_pixel', 'source', 'grid_version', 'geometry']
    return gdf[[c for c in keep if c in gdf.columns]]

def save(gdf, path: Path):
    g = gdf.copy()
    for c in ('mcd_start', 'mcd_end'):
        g[c] = g[c].dt.strftime('%Y-%m-%d')
    Path(path).write_text(g.to_json(), encoding='utf-8')

def load(path: Path):
    import geopandas as gpd
    g = gpd.read_file(path)
    for c in ('mcd_start', 'mcd_end'):
        g[c] = pd.to_datetime(g[c])
    return g
