"""Implementation of the wildfire gas analysis module."""
from pathlib import Path
import io
import json
import tempfile
import zipfile
import pandas as pd
FIELDS = {'id': 'id', 'name': 'name', 'province': 'province', 'district': 'district', 'area_ha': 'area_ha', 'forest_fraction': 'forest_pct', 'mcd_start': 'first_burn', 'mcd_end': 'last_burn', 'firms_pixel': 'firms_px', 'source': 'source', 'grid_version': 'grid_ver'}

def shapefile_zip(gdf):
    if gdf.empty:
        raise ValueError('No fire areas to export.')
    if gdf.crs is None:
        raise ValueError('Fire geometries have no coordinate reference system.')
    if gdf.geometry.isna().any() or gdf.geometry.is_empty.any():
        raise ValueError('Empty fire geometries cannot be exported.')
    if not gdf.geom_type.isin(['Polygon', 'MultiPolygon']).all():
        raise ValueError('Fire areas must be polygons.')
    fields = {c: FIELDS[c] for c in FIELDS if c in gdf.columns}
    out = gdf[list(fields) + [gdf.geometry.name]].copy().to_crs(4326)
    for c in ('mcd_start', 'mcd_end'):
        if c in out:
            out[c] = pd.to_datetime(out[c]).dt.strftime('%Y-%m-%d').where(out[c].notna(), None)
    out = out.rename(columns=fields)
    buf = io.BytesIO()
    with tempfile.TemporaryDirectory(prefix='fire_shp_') as folder:
        path = Path(folder) / 'fire_areas.shp'
        out.to_file(path, driver='ESRI Shapefile', encoding='UTF-8', index=False)
        with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as archive:
            for f in sorted(Path(folder).glob('fire_areas.*')):
                archive.write(f, f.name)
            archive.writestr('field_names.json', json.dumps({v: v for v in fields.values()}, ensure_ascii=False, indent=2))
            archive.writestr('README.txt', 'Fire polygons — EPSG:4326 (WGS 84), UTF-8.\nExtract all files into one folder and open the .shp file.\narea_ha: area in hectares; forest_pct: forest fraction (0–1).\nDates: YYYY-MM-DD. Geometry and areas match the analysis data.\n')
    return buf.getvalue()
