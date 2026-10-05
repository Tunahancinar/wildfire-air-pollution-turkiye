"""Implementation of the wildfire gas analysis module."""
from __future__ import annotations
import io
import sys
import tempfile
import zipfile
from pathlib import Path
import pandas as pd
import streamlit as st
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from wildfire_gas.english import EnglishUI, tr, workbook_bytes, english_zip, summary_text
st = EnglishUI(st)
from wildfire_gas.map_style import STUDY, LARGEST, BOUNDARY
from wildfire_gas import analysis_statistics as list_stats
from wildfire_gas.config import AnalysisSettings, GASES
DAY = pd.Timedelta(days=1)
st.set_page_config(page_title='Wildfire Gas Analysis', page_icon='🔥', layout='wide')
st.markdown('\n<style>\n .block-container {padding-top: 1.4rem; padding-bottom: 1rem;}\n div[data-testid="stMetricValue"] {font-size: 1.5rem;}\n .small {color:#6b6a66; font-size:0.85rem;}\n [data-testid="stAppViewContainer"] {background:linear-gradient(180deg,#eef4f8 0%,#ffffff 480px);}\n [data-testid="stSidebar"] {background:#f1f5f9;}\n [data-testid="stMetric"] {background:white;border:1px solid #dce5ed;border-radius:14px;padding:16px;}\n h1,h2,h3 {color:#18354b;letter-spacing:-0.025em;}\n .stButton>button {border-radius:9px;}\n [data-testid="stImage"] {background:white;border:1px solid #e2e8f0;border-radius:12px;padding:8px;}\n</style>', unsafe_allow_html=True)
ss = st.session_state
ss.setdefault('fires', None)
ss.setdefault('selected', None)
ss.setdefault('result', {})
with st.sidebar:
    st.header('1 · Identify wildfires')
    demo = st.toggle('Demo mode (no GEE required)', value=False, help='Explore the interface using synthetic data; results are not real observations.')
    project = st.text_input('GEE project ID', value='', disabled=demo)
    year = st.number_input('Year', 2019, 2026, 2025, step=1)
    min_ha = st.number_input('Minimum area (ha)', 10, 100000, 100, step=50)
    min_forest = st.slider('Minimum forest fraction', 0.0, 1.0, 0.3, 0.05, help='Tree cover (including shrubland if selected) within the burned area — ESA WorldCover')
    shrubland = st.checkbox('Include shrubland as forest', value=True)
    source = st.selectbox('Burned-area source', ['MCD64A1', 'FIRMS'], help='MCD64A1: MODIS burned area (500 m). FIRMS: active-fire clusters (1 km), available when MCD64A1 has not yet been released; areas are approximate.')
    load = st.button('🔎  Load wildfires', type='primary', width='stretch')
    st.header('2 · Analysis settings')
    station_selection = st.selectbox('Station selection method', ['Wind-aligned', 'Nearest'], format_func={'Wind-aligned': 'Wind direction + nearest suitable station', 'Nearest': 'Nearest suitable station'}.get, help='Wind mode selects the nearest suitable station along the transport direction. It falls back to distance when suitable data are unavailable. Selection is performed separately for each gas.')
    with st.expander('Details', expanded=False):
        before = st.number_input('Pre-fire baseline period (days)', 3, 30, 10)
        after = st.number_input('Post-fire period (days)', 3, 30, 8)
        z = st.number_input('z-score threshold', 1.5, 6.0, 2.5, 0.1)
        product = st.selectbox('Sentinel-5P product', ['OFFL', 'NRTI'])
        download = st.selectbox('Spatial aggregation', ['mean', 'max'], format_func={'mean': 'Mean', 'max': 'Maximum'}.get)
        station_km = st.number_input('Station search radius (km)', 10, 400, 150, 10)
        wind_sector = st.slider('Wind sector (±degrees)', 15, 90, 45, 5, help='At least 3 valid fire days and wind speed ≥1.8 km/h are required. Direction must agree for at least 50% of speed-weighted days. This is not a dispersion model.')
        ch2o = st.number_input('CH2O conversion factor (μg m−3 / mol m−2)', 0.0, 10000000.0, 16000.0, 1000.0, help='Enter manually because stations do not measure CH2O; 0 = no conversion.')
    output = st.text_input('Output directory', 'output')
if demo:
    from wildfire_gas.demo_gee import demo_fires, mock_gee_setup
    if not ss.get('_demo_installed'):
        ss._demo_gdf = demo_fires()
        mock_gee_setup(ss._demo_gdf)
        ss._demo_installed = True
        ss.fires, ss.selected, ss.result = (None, None, {})
elif ss.get('_demo_installed'):
    import importlib
    from wildfire_gas import gee as _g, turkey as _t, station_service as _h
    importlib.reload(_g)
    importlib.reload(_t)
    importlib.reload(_h)
    ss.pop('list_stats_list', None)
    ss._demo_installed = False
    ss.fires, ss.selected, ss.result = (None, None, {})
st.title('Wildfire Gas Analysis')
from wildfire_gas.batch_tests import save as tests_save, NOTE as test_notu
st.header(f'Change detection and statistical comparisons — {year}')
test_root = Path(output)
test_csv = test_root / 'ALL_FIRES_DETECTION_AND_MANN_WHITNEY.csv'
test_xlsx = test_root / 'ALL_FIRES_DETECTION_AND_MANN_WHITNEY.xlsx'
if st.button('Recalculate saved fire results', key='batch_test_refresh') or not test_csv.exists():
    with st.spinner('Computing test results from saved daily observations...'):
        tests_save(test_root)
if test_csv.exists():
    test_table = pd.read_csv(test_csv)
    test_table = test_table[test_table['Year'] == int(year)].copy()
    st.caption(f"Selected year: {year} · {test_table['Fire'].nunique()} analysed fires · {len(test_table)} gas results. † marks saved satellite column series used when station conversion is unavailable.")
    from wildfire_gas.test_charts import charts as test_charts_generate
    visual_paths = [test_root / 'test_charts' / str(year) / f'{name}.png' for name in ('start', 'confidence', 'mann_whitney')]
    if not test_table.empty and any((not p.exists() or p.stat().st_mtime < test_csv.stat().st_mtime for p in visual_paths)):
        test_charts_generate(test_table, test_root, year)
    visual_tabs = st.tabs(['CUSUM / PELT onset', 'Confidence classes', 'Mann–Whitney results'])
    for visual_tab, visual_path in zip(visual_tabs, visual_paths):
        with visual_tab:
            if not visual_path.exists():
                st.info('No saved fire analyses for this year. Run an analysis to generate results.')
                continue
            st.image(visual_path.read_bytes(), width='stretch')
            st.download_button('Download chart (PNG)', visual_path.read_bytes(), file_name=visual_path.name, mime='image/png', key='test_chart_' + visual_path.stem)
            st.download_button('Download chart (PDF)', visual_path.with_suffix('.pdf').read_bytes(), file_name=visual_path.with_suffix('.pdf').name, mime='application/pdf', key='test_pdf_' + visual_path.stem)
    with st.expander('Numeric results table'):
        st.dataframe(test_table, hide_index=True, width='stretch')
    st.caption('Confidence is method agreement, not a statistical probability. No onset is assigned when no method detects a change. Mann–Whitney uses all available observations; periods with fewer than 3 days are marked *. Tests are exploratory, with unadjusted p values.')
    if test_xlsx.exists():
        st.download_button('All fire test results (Excel)', test_xlsx.read_bytes(), file_name=test_xlsx.name, mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', key='batch_test_excel')
    st.download_button('All fire test results (CSV)', test_csv.read_bytes(), file_name=test_csv.name, mime='text/csv', key='batch_test_csv')
saved_names = sorted((p.parent.name for p in Path(output).glob('*/result.json')))
with st.sidebar:
    if saved_names and (not demo):
        st.divider()
        saved_name = st.selectbox('Saved analyses', saved_names)
        saved_open = st.button('Open saved result', width='stretch')
    else:
        saved_open = False
requested = st.query_params.get('result') if ss.fires is None and (not demo) else None
if saved_open or requested in saved_names:
    wanted = saved_name if saved_open else requested
    from wildfire_gas import turkey
    for cache in Path(output).glob('fires_*.geojson'):
        cached = turkey.load(cache)
        matched = cached[cached.name == wanted]
        if not matched.empty:
            ss.fires, ss.selected = (cached, int(matched.iloc[0].id))
            break
if load:
    if not demo and (not project):
        st.sidebar.error('A GEE project ID is required (or enable demo mode).')
    else:
        cache = Path(output) / f'fires_{year}_{source}_{min_ha}ha_{int(min_forest * 100)}_native_v2.geojson'
        from wildfire_gas import turkey
        with st.spinner('Scanning burned areas across Turkey (1–3 minutes on the first run)...'):
            try:
                if cache.exists() and (not demo):
                    gdf = turkey.load(cache)
                else:
                    gdf = turkey.turkey_fires(int(year), float(min_ha), float(min_forest), shrubland, source, project or None)
                    if not demo:
                        cache.parent.mkdir(parents=True, exist_ok=True)
                        turkey.save(gdf, cache)
                ss.fires, ss.selected = (gdf, None)
            except Exception as e:
                st.error(f'Could not load wildfires: {e}')
gdf = ss.fires
if gdf is None:
    st.info('Select a year and area threshold in the sidebar, then click **Load wildfires**. Try **Demo mode** if you do not have a GEE account.')
    st.stop()
if gdf.empty:
    st.warning('No wildfires match these criteria. Reduce the area or forest threshold. MCD64A1 may not yet cover the final months of the year; try FIRMS.')
    st.stop()
m1, m2, m3, m4 = st.columns(4)
m1.metric('Number of wildfires', f'{len(gdf)}')
m2.metric('Total burned area', f'{gdf.area_ha.sum():,.0f} ha')
m3.metric('Largest', f'{gdf.area_ha.max():,.0f} ha', gdf.iloc[gdf.area_ha.argmax()]['district'])
m4.metric('Period', f'{gdf.mcd_start.min():%d.%m} – {gdf.mcd_end.max():%d.%m.%Y}')

@st.cache_data(show_spinner=False, max_entries=3)
def _location_generate(geojson, period, satellite, dpi=120, map=None):
    import geopandas as gpd
    from wildfire_gas.location_map import location_maps
    areas = gpd.read_file(io.StringIO(geojson))
    return location_maps(areas, period=period, satellite=satellite, dpi=dpi, map=map, formats=('png',) if dpi == 120 else ('png', 'pdf'))
with st.expander('Turkey and wildfire location maps', expanded=True):
    _style = st.radio('Map style', ['Satellite imagery', 'Outline map'], horizontal=True)
    _satellite = _style == 'Satellite imagery'
    _geo = gdf[[c for c in ('name', 'province', 'district', 'area_ha', 'source', 'geometry') if c in gdf]].to_json()
    _period = f'{gdf.mcd_start.min():%d.%m.%Y} – {gdf.mcd_end.max():%d.%m.%Y}'
    if _satellite:
        st.caption('Satellite imagery provides spatial context; it is not imagery from the fire date.')
    try:
        with st.spinner('Preparing location maps...'):
            _location = _location_generate(_geo, _period, _satellite)
        _tabs = st.tabs(['1 · Turkey location map', '2 · Wildfire locations'])
        for _tab, _name in zip(_tabs, ('turkey_location', 'fire_locations')):
            with _tab:
                st.image(_location[_name]['png'], width='stretch')
                from wildfire_gas.location_map import MAP_DIMENSIONS
                _w, _h = MAP_DIMENSIONS[_name]
                _dpi = st.selectbox('Download resolution', [300, 600], format_func=lambda v, _w=_w, _h=_h: f'{v} DPI · {round(_w * v)} × {round(_h * v)} pixels', key=_name + '_dpi')
                st.caption('DPI controls print resolution, not the spatial resolution of satellite imagery. Text and boundaries remain vector objects in PDF exports.')
                import hashlib
                _export_id = hashlib.sha256(('english-turkey-v2' + _geo + _period + str(_satellite) + str(_dpi) + _name).encode()).hexdigest()
                if st.button('Prepare download files', key=_name + '_prepare'):
                    with st.spinner(f'{_dpi} DPI files are being prepared...'):
                        _ready = _location_generate(_geo, _period, _satellite, _dpi, _name)[_name]
                        ss[_name + '_export'] = (_export_id, _ready)
                _saved = ss.get(_name + '_export')
                if _saved and _saved[0] == _export_id:
                    _png, _pdf = st.columns(2)
                    _png.download_button(f'Download PNG · {_dpi} DPI', _saved[1]['png'], file_name=f'{_name}_{_dpi}dpi.png', mime='image/png', key=_name + '_png')
                    _pdf.download_button(f'Download PDF · {_dpi} DPI', _saved[1]['pdf'], file_name=f'{_name}_{_dpi}dpi.pdf', mime='application/pdf', key=_name + '_pdf')
    except Exception as exc:
        st.error(f'Could not prepare location maps: {exc}')
with st.expander('Download fire polygons (SHP)'):
    from wildfire_gas.export import shapefile_zip
    st.caption('Download mapped fire boundaries as SHP. Extract all files from the ZIP together. Coordinate reference system: WGS 84.')
    try:
        st.download_button('Download all listed fires (.shp ZIP)', shapefile_zip(gdf), file_name='fire_areas_shp.zip', mime='application/zip', key='shp_all')
        selected_export = gdf[gdf.id == ss.selected] if ss.selected is not None else gdf.iloc[:0]
        if not selected_export.empty:
            from wildfire_gas.analysis import _file_name as _shp_name
            st.download_button('Download selected fire (.shp ZIP)', shapefile_zip(selected_export), file_name=_shp_name(str(selected_export.iloc[0].name)) + '_shp.zip', mime='application/zip', key='shp_selected')
    except Exception as exc:
        st.error(f'Could not prepare SHP export: {exc}')
import folium
from streamlit_folium import st_folium

def nearest_stations(row):
    """Implementation of nearest_stations."""
    from wildfire_gas import station_service
    if ss.get('list_stats_list') is None:
        try:
            ss.list_stats_list = station_service.station_list()
            ss.list_stats_error = None
        except Exception as e:
            ss.list_stats_list, ss.list_stats_error = (None, str(e))
            return None
    c = row.geometry.centroid
    return station_service.nearest_candidates(ss.list_stats_list, c.x, c.y, 6, float(station_km))
left, right = st.columns([3, 2], gap='medium')
with left:
    selected_id = ss.selected
    select_row = gdf[gdf.id == selected_id].iloc[0] if selected_id is not None and (gdf.id == selected_id).any() else None
    map = folium.Map(location=[39.0, 35.2], zoom_start=6, tiles=None, control_scale=False)
    if select_row is not None:
        x0, y0, x1, y1 = select_row.geometry.bounds
        map.fit_bounds([[y0, x0], [y1, x1]], padding=(30, 30))
    else:
        x0, y0, x1, y1 = gdf.total_bounds
        map.fit_bounds([[min(y0, 36.0), min(x0, 26.0)], [max(y1, 42.0), max(x1, 44.8)]])
    boundary = Path(__file__).with_name('turkey_boundary.geojson')
    if boundary.exists():
        folium.GeoJson(str(boundary), name='Turkey boundary', interactive=False, style_function=lambda f: {'color': BOUNDARY, 'weight': 1.3, 'fill': False}).add_to(map)
    folium.TileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', attr='Esri World Imagery', name='Satellite imagery').add_to(map)
    folium.TileLayer('OpenStreetMap', name='Street basemap', show=False).add_to(map)
    g2 = gdf.copy()
    g2['date'] = g2.mcd_start.dt.strftime('%d.%m.%Y')
    g2['area'] = g2.area_ha.map(lambda v: f'{v:,.0f} ha')
    g2 = g2[['id', 'name', 'province', 'district', 'area', 'date', 'geometry']]
    folium.GeoJson(g2, name='Burned areas', style_function=lambda f: {'color': STUDY, 'weight': 3 if f['properties']['id'] == selected_id else 1.5, 'fillColor': STUDY, 'fillOpacity': 0.12}, tooltip=folium.GeoJsonTooltip(['district', 'province', 'area', 'date'], ['District', 'Province', 'Area', 'Start'])).add_to(map)
    largest_id = int(gdf.loc[gdf.area_ha.idxmax(), 'id'])
    for r in gdf.itertuples():
        c = r.geometry.representative_point()
        folium.CircleMarker([c.y, c.x], radius=9 if r.id == largest_id else 5, color=BOUNDARY, weight=3 if r.id == selected_id else 1, fill=True, fill_color=LARGEST if r.id == largest_id else STUDY, fill_opacity=1, tooltip=f'{r.district} ({r.province}) · {r.area_ha:,.0f} ha · {r.mcd_start:%d.%m.%Y}').add_to(map)
    nearest = nearest_stations(select_row) if select_row is not None else None
    if nearest is not None:
        for r in nearest.itertuples():
            folium.Marker([r.latitude, r.longitude], tooltip=f'Station: {r.name} · {r.distance_km:.1f} km', icon=folium.DivIcon(icon_size=(14, 14), icon_anchor=(7, 7), html='<div style="width:12px;height:12px;background:#4a3aa7;border:2px solid white;transform:rotate(45deg);box-shadow:0 0 2px #0006"></div>')).add_to(map)
        if len(nearest):
            x0, y0, x1, y1 = select_row.geometry.bounds
            map.fit_bounds([[min(y0, nearest.latitude.min()), min(x0, nearest.longitude.min())], [max(y1, nearest.latitude.max()), max(x1, nearest.longitude.max())]], padding=(20, 20))
    from wildfire_gas.web_map_style import combined_checks
    combined_checks(map)
    folium.LayerControl(collapsed=True).add_to(map)
    exit_code = st_folium(map, height=520, use_container_width=True, key=f'map_{selected_id}', returned_objects=['last_object_clicked'])
    st.markdown('<div style="border:1px solid #111;padding:8px;background:white;color:black;font-size:13px"><span style="display:inline-block;width:14px;height:10px;background:#ffff00;border:1px solid black"></span> Study fire boundary &nbsp; <span style="display:inline-block;width:14px;height:10px;background:white;border:1px solid black"></span> Turkey boundary &nbsp; <span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#00d4f0;border:1px solid black"></span> Largest wildfire &nbsp; ◆ Air-quality station · Thick outline: selected fire</div>', unsafe_allow_html=True)
    click = (exit_code or {}).get('last_object_clicked')
    if click:
        from shapely.geometry import Point
        p = Point(click['lng'], click['lat'])
        ic = gdf[gdf.contains(p)]
        new = int(ic.iloc[0].id) if len(ic) else int(gdf.loc[gdf.geometry.centroid.distance(p).idxmin(), 'id'])
        if new != ss.selected:
            ss.selected = new
            st.rerun()
    table = gdf.drop(columns='geometry').copy()
    table['mcd_start'] = table.mcd_start.dt.strftime('%d.%m.%Y')
    table['mcd_end'] = table.mcd_end.dt.strftime('%d.%m.%Y')
    table['forest_fraction'] = (table.forest_fraction * 100).round(0)
    table = table.rename(columns={'district': 'District', 'province': 'Province', 'area_ha': 'Area (ha)', 'forest_fraction': 'Forest cover (%)', 'mcd_start': 'First burn', 'mcd_end': 'Last burn', 'firms_pixel': 'FIRMS pixels'})
    select = st.dataframe(table[['id', 'District', 'Province', 'Area (ha)', 'Forest cover (%)', 'First burn', 'Last burn', 'FIRMS pixels']], hide_index=True, width='stretch', height=210, on_select='rerun', selection_mode='single-row', column_config={'Area (ha)': st.column_config.NumberColumn(format='%.0f')})
    if select and select.selection.rows:
        new = int(table.iloc[select.selection.rows[0]]['id'])
        if new != ss.selected:
            ss.selected = new
            st.rerun()
with right:
    if select_row is None:
        st.subheader('Select a wildfire')
        st.write('Click a fire on the map or select a table row.')
        st.stop()
    st.subheader(f'{select_row.district} · {select_row.province}')
    a1, a2 = st.columns(2)
    a1.metric('Burned area', f'{select_row.area_ha:,.0f} ha')
    a2.metric('Forest fraction', f'%{select_row.forest_fraction * 100:.0f}')
    st.caption(f'MODIS burn dates: {select_row.mcd_start:%d.%m.%Y} – {select_row.mcd_end:%d.%m.%Y} · FIRMS pixels: {int(select_row.firms_pixel or 0)}. Start and end dates are determined during analysis using daily FIRMS data.')
    st.markdown('**Ground-station data** (for μg/m3 conversion and validation)')
    source_list_stats = st.radio('Station data source', ['Automatic (national station service)', 'Upload file', 'Do not use'], horizontal=True, label_visibility='collapsed')
    station_df, automatic = (None, False)
    if source_list_stats.startswith('Otomatik'):
        automatic = True
        if nearest is not None and len(nearest):
            st.caption('Purple diamonds on the map. For each gas, the nearest station with at least 50% coverage during the fire is selected.')
            st.dataframe(nearest[['name', 'province', 'distance_km', 'area_type', 'source_type']].rename(columns={'name': 'Station', 'province': 'Province', 'distance_km': 'km', 'area_type': 'Area', 'source_type': 'Source'}), hide_index=True, width='stretch', height=160, column_config={'km': st.column_config.NumberColumn(format='%.1f')})
        elif nearest is not None:
            st.warning(f'{station_km} no station within this radius (km); increase the radius.')
        else:
            st.warning(f"Could not connect to station_service: {ss.get('list_stats_error')}. Analysis can continue; without station data, results remain in mol/m2. You can also upload a file.")
    elif source_list_stats == 'Upload file':
        list_stats_start = select_row.mcd_start - (before + 5) * DAY
        list_stats_end = select_row.mcd_end + (after + 5) * DAY
        st.markdown(f'<span class="small">havaizleme.gov.tr → Download data: nearest station, <b>Hourly</b>, <b>{list_stats_start:%d.%m.%Y} – {list_stats_end:%d.%m.%Y}</b>, CO · NO2 · SO2 · O3</span>', unsafe_allow_html=True)
        load = st.file_uploader('Station file (Excel/CSV)', type=['xlsx', 'xls', 'csv'], label_visibility='collapsed')
        if load is not None:
            from wildfire_gas.data_reading import read_station
            with tempfile.NamedTemporaryFile(suffix=Path(load.name).suffix, delete=False) as tf:
                tf.write(load.getvalue())
            try:
                station_df = read_station(tf.name)
                st.success(f"Station: {', '.join(station_df.columns)} · {station_df.index.min():%d.%m.%Y} – {station_df.index.max():%d.%m.%Y}")
            except Exception as e:
                st.error(f'Could not read station file: {e}')
    else:
        st.caption('Without station data, results are reported in mol/m2 (Equation 1 cannot be applied).')
    work = st.button('▶  Detect dates and run analysis', type='primary', width='stretch')
if work:
    from wildfire_gas.analysis import _file_name, fire_analysis
    settings = AnalysisSettings(before_day=int(before), after_day=int(after), z_threshold=float(z), gee_product=product, gee_reduction=download, gee_project=project or None, manual_factor={'CH2O': float(ch2o)} if ch2o > 0 else {}, station_automatic=automatic, station_max_km=float(station_km), station_selection=station_selection, wind_sector=float(wind_sector))
    row = gdf[gdf.id == select_row.id]
    box = st.status(f'{select_row.name} analysis in progress...', expanded=True)
    record = box.empty()
    buffer = []

    class _Stream(io.TextIOBase):

        def write(self, s):
            buffer.append(s)
            record.code(''.join(buffer)[-3000:], language=None)
            return len(s)
    previous = sys.stdout
    sys.stdout = _Stream()
    try:
        summary = fire_analysis(select_row.name, None, None, row, None, station_df, None, settings, output, True, select_row.mcd_start - 20 * DAY, select_row.mcd_end + 20 * DAY)
        ss.result[select_row.id] = {'summary': summary, 'directory': Path(output) / _file_name(select_row.name)}
        box.update(label='Analysis complete', state='complete', expanded=False)
    except Exception as e:
        box.update(label=f'Error: {e}', state='error')
    finally:
        sys.stdout = previous
from wildfire_gas.analysis import _file_name
import json
record_directory = Path(output) / _file_name(select_row.name)
record_json = record_directory / 'result.json'
if record_json.exists():
    ss.result[select_row.id] = {'summary': json.loads(record_json.read_text(encoding='utf-8')), 'directory': record_directory}
last = ss.result.get(select_row.id)
if not last:
    st.stop()
directory = last['directory']
summary = last['summary']
if select_row.get('grid_version') == 'native_v2':
    geometry_version = directory / 'geometry_version.txt'
    if not geometry_version.exists() or geometry_version.read_text(encoding='utf-8') != 'native_v2':
        st.info('This fire boundary uses the new pixel grid. Run the analysis to calculate results for this boundary.')
        st.stop()
if not (directory / 'version.txt').exists() or (directory / 'version.txt').read_text(encoding='utf-8') != '2026-09-29-wind':
    st.warning('This result uses an older calculation version. Rerun the analysis for updated charts and statistics.')
    st.stop()
report = directory / f'{directory.name}_report.xlsx'
sheets = pd.read_excel(report, sheet_name=None) if report.exists() else {}
for _name, _df in sheets.items():
    for _c in _df.columns:
        if pd.api.types.is_datetime64_any_dtype(_df[_c]):
            _df[_c] = _df[_c].dt.strftime('%d.%m.%Y')
st.divider()
st.header(f'Results — {select_row.district} ({select_row.province})')
if (directory / 'settings.json').exists():
    saved_settings = json.loads((directory / 'settings.json').read_text(encoding='utf-8'))
    saved_mode = saved_settings.get('station_selection', 'Nearest')
    st.caption('Station selection method for this result: ' + ('Wind direction + nearest suitable station' if saved_mode == 'Wind-aligned' else 'Nearest suitable station') + '. Gas-specific selections and fallback details appear in the Station matching table.')
k = st.columns(5)
k[0].metric('Start', summary['Start'], summary['Date source'], delta_color='off')
k[1].metric('End', summary['End'])
k[2].metric('Combined gas signal', summary['Combined gas signal'])
if 'CO peak' in summary:
    k[3].metric('CO peak', f"{summary['CO peak']:.4g} {summary['CO unit']}", summary['CO peak date'], delta_color='off')
    k[4].metric('CO change', f"%{summary['CO change (%)']:+.0f}", f"confidence: {summary['CO confidence']}", delta_color='off')

def _chart_show(file):
    st.image(file.read_bytes(), width='stretch')
    key = str(file.resolve())
    dpi = st.selectbox('PNG resolution', [300, 600], format_func=lambda v: f'{v} DPI', key='chart_dpi_' + key)
    download = file if dpi == 300 else file.parent / '600dpi' / file.name
    if download.exists():
        st.download_button(f'Download PNG · {dpi} DPI', download.read_bytes(), file_name=f'{file.stem}_{dpi}dpi.png', mime='image/png', key='chart_png_' + key)
    else:
        st.info('Rerun the analysis to generate the high-resolution version of this chart.')
t1, t2, t3, t4, t5 = st.tabs(['Summary and charts', 'Change onset', 'Period statistics', 'Performance', 'Downloads'])
gr = directory / 'charts'
with t1:
    st.text(summary_text(directory))
    for f in ('all_gases_ug_m3.png', 'all_gases_mol_m2.png'):
        if (gr / f).exists():
            _chart_show(gr / f)
            break
    for i, g in enumerate(GASES):
        if (gr / f'{g}_time_series.png').exists():
            _chart_show(gr / f'{g}_time_series.png')
    for f in sorted(gr.glob('*_map.png')):
        _chart_show(f)
    if (gr / 'weather.png').exists():
        _chart_show(gr / 'weather.png')
        st.caption('Arrows point downwind: ↓ south, ↙ southwest, → east. Wind origin is the opposite direction. Arrow length does not encode speed; use the km/h axis.')
with t2:
    if 'Change detection' in sheets:
        st.dataframe(sheets['Change detection'], hide_index=True, width='stretch')
    if (gr / 'increase_onset_tespiti.png').exists():
        _chart_show(gr / 'increase_onset_tespiti.png')
        st.caption('The z-score is not a concentration; it can be negative below the baseline.')
    if 'Burn episodes' in sheets:
        st.markdown('**FIRMS / MCD64A1 burn episodes**')
        st.dataframe(sheets['Burn episodes'], hide_index=True, width='stretch')
with t3:
    for name in ('Station matching', 'Period statistics (ug m-3)', 'Period statistics (mol m-2)', 'Period comparison', 'Station period statistics', 'Weather period statistics', 'Conversion factors'):
        if name in sheets:
            st.markdown(f'**{name}**')
            st.dataframe(sheets[name], hide_index=True, width='stretch')
with t4:
    if 'Performance' in sheets:
        st.dataframe(sheets['Performance'].round(3), hide_index=True, width='stretch')
        st.caption('Pearson r is signed correlation. MAE and RMSE: μg/m3. NRMSE: RMSE/observed mean (%). RRMSE: original Equation 5. Conversion uses the same observations, so this is not independent validation.')
    else:
        st.info('Upload ground-station data to calculate performance metrics.')
with t5:
    if report.exists():
        st.download_button('📊  Excel report', workbook_bytes(report), report.name, width='stretch')
    st.download_button('All outputs (ZIP)', english_zip(directory), f'{directory.name}.zip', width='stretch')
    st.caption(f'Directory: {directory.resolve()}')
st.divider()
st.header('Comparison across wildfires')
from wildfire_gas.comparison import read, maximum_charts, measurement_charts, selected_year_cases
_all_cases, _errors = read(Path(output))
_year = int(year)
_cases = selected_year_cases(_all_cases, _year)
st.caption(f'Selected year: {_year}. All comparisons, statistics, maximum-day maps and batch downloads below use only fires starting in this year and observations within this calendar year.')
if _errors:
    with st.expander('Records excluded from comparison'):
        for _err in _errors:
            st.write(_err)
if _cases:
    st.caption(f'Completed analyses: {len(_cases)} fire records. Gas comparisons use converted μg/m3 data. Gases without conversion or observations are marked ‘No data’.')
    _scope = st.selectbox('Period for maximum concentrations', ['During fire', 'Full analysis period'], key='comparison_period')
    with st.spinner('Preparing wildfire maxima...'):
        _max_paths = maximum_charts(Path(output), _cases, _scope)
    for _path in _max_paths:
        _chart_show(_path)
    st.header('Maximum gas concentrations by year')
    from wildfire_gas.annual import annual_charts
    _year_count = sum((c['periods']['During fire'][0].year == _year for c in _cases.values()))
    st.caption(f'{_year}: {_year_count} completed analysis records · Period: {_scope}. Each gas peak date is matched to its own daily weather record.')
    with st.spinner('Preparing annual maxima and same-day weather...'):
        _year_paths = annual_charts(Path(output), _cases, _year, _scope)
    if not _year_paths:
        st.info('No completed analyses are available for this year. This section will populate automatically after analysis.')
    for _path in _year_paths:
        _chart_show(_path)
    st.header(f'Measurement statistics — Turkey, {_year}')
    from wildfire_gas.comparison import statistics_archive_path, all_statistics_archive
    st.caption(f'{len(_cases)} completed fire analyses available. Each fire has separate gas and meteorological charts for its own before, during and after periods. Newly completed analyses are included automatically. This is not a complete national wildfire inventory.')
    _bulk_dpi = st.selectbox('Resolution for all statistics charts', [300, 600], format_func=lambda x: f'{x} DPI', key='all_stats_dpi')
    _bulk_path = statistics_archive_path(Path(output), _cases, _bulk_dpi)
    if not _bulk_path.exists():
        if st.button('Generate statistics charts for all analysed fires', key='all_stats_generate'):
            with st.spinner('Preparing gas and meteorological charts for all analysed fires...'):
                _progress = st.progress(0.0)
                _bulk_path = all_statistics_archive(Path(output), _cases, _bulk_dpi, _progress.progress)
                _progress.empty()
    if _bulk_path.exists():
        with _bulk_path.open('rb') as _bulk_file:
            st.download_button(f'Download all statistics charts — {_bulk_dpi} DPI (ZIP)', _bulk_file, _bulk_path.name, mime='application/zip', key='all_stats_download')
    st.subheader('Preview statistics for an individual fire')
    _names = list(_cases)
    _case_name = st.selectbox('Wildfire for measurement statistics', _names, index=_names.index(directory.name) if directory.name in _names else 0, key=f'statistics_fire_{_year}')
    with st.spinner('Preparing measurement statistics...'):
        _stat_paths = measurement_charts(Path(output), _case_name, _cases[_case_name])
    for _path in _stat_paths:
        _chart_show(_path)
    st.subheader(f'Highest gas concentrations and their wildfire locations — {_year}')
    from wildfire_gas.synthesis import maxima_chart, map_inputs, maxima_maps
    _chart_show(maxima_chart(Path(output), _cases, _scope))
    _map_inputs, _map_errors = map_inputs(Path(output), Path(output), _cases, _scope)
    if _map_errors:
        if st.button('Download missing maximum-day maps', key='fetch_maximum_maps'):
            with st.spinner('Downloading exact-day Sentinel-5P maps...'):
                _map_inputs, _map_errors = map_inputs(Path(output), Path(output), _cases, _scope, fetch=True)
        for _message in _map_errors:
            st.warning(_message)
    _chart_show(maxima_maps(Path(output), _cases, _scope, _map_inputs))
    st.subheader(f'Min, max, mean and SD across analysed fires — {_year}')
    st.caption('One combined figure for completed fire analyses in the selected year only. Min, max, mean and sample SD summarize individual fire-period means with equal fire weights; n shows the number with valid data. Detected fires without completed measurements are excluded.')
    from wildfire_gas.comparison import mean_fire_chart
    with st.spinner('Preparing the combined average chart...'):
        _mean_path = mean_fire_chart(Path(output), _cases)
    _chart_show(_mean_path)
    st.subheader(f'Performance (converted satellite vs. ground station) — {_year}')
    from wildfire_gas.performance_summary import performance_data, performance_chart
    with st.spinner('Comparing performance across fires in the selected year...'):
        _performance, _performance_notes, _performance_key = performance_data(Path(output), _cases)
    if _performance.empty:
        st.info('No eligible paired satellite–station performance data are available for this year (at least 3 matched days per gas).')
    else:
        _chart_show(performance_chart(Path(output), _cases, _performance, _performance_key))
    if _performance_notes:
        with st.expander('Performance data coverage'):
            for _note in _performance_notes:
                st.write(_note)
else:
    st.info(f'No completed fire analyses are available for {_year}. Analyse fires from this year to generate these charts.')
