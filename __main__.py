"""Implementation of the wildfire gas analysis module."""
from .english import tr as _best
import argparse
from .english import install_cli_language
install_cli_language()
import sys
from pathlib import Path
from .config import AnalysisSettings, GASES

def _factor(s):
    d = {}
    for p in (s or '').split(','):
        if '=' in p:
            k, v = p.split('=')
            d[k.strip().upper()] = float(v)
    return d

def _streamlit_environment():
    """Implementation of _streamlit_environment."""
    import os
    d = Path.home() / '.streamlit'
    d.mkdir(exist_ok=True)
    k = d / 'credentials.toml'
    if not k.exists():
        k.write_text('[general]\nemail = ""\n', encoding='utf-8')
    return dict(os.environ)

def settings_create(a) -> AnalysisSettings:
    return AnalysisSettings(before_day=a.before_day, after_day=a.after_day, z_threshold=a.z_threshold, z_strong=a.z_strong, consecutive_day=a.consecutive_day, conversion=a.conversion, manual_factor=_factor(a.factor), station_daily=a.station_daily, gee_product=a.product, gee_reduction=a.reduction, gee_project=a.project, map_download=not a.map_none, point_buffer_km=a.buffer_km, station_automatic=not a.station_none, station_max_km=a.station_km, station_selection=a.station_selection, wind_sector=a.wind_sector)

def main(argv=None):
    p = argparse.ArgumentParser(prog='wildfire_gas', description=_best('Wildfire gas analysis (Sentinel-5P + ground stations)'))
    g = p.add_argument_group(_best('Fire'))
    g.add_argument('--inventory', help=_best('Fire inventory: SHP/GeoJSON/GPKG or CSV/XLSX (name, start, end, [latitude, longitude])'))
    g.add_argument('--shp', help=_best('Single fire polygon (SHP/GeoJSON/GPKG)'))
    g.add_argument('--name', help=_best('Fire name'))
    g.add_argument('--start', help=_best('Start date (DD.MM.YYYY); detected automatically if omitted'))
    g.add_argument('--end', help=_best('End date (DD.MM.YYYY); detected automatically if omitted'))
    g.add_argument('--name-column', help=_best('SHP fire-name column (if not detected automatically)'))
    g.add_argument('--epsg', type=int, help=_best('Coordinate reference system if SHP has no .prj (e.g. 5254)'))
    g.add_argument('--min-area-ha', type=float, default=0, help=_best('Skip smaller fires (Sentinel-5P ~5.5x3.5 km; suggested minimum ≥100 ha)'))
    g.add_argument('--only-date', action='store_true', help=_best('Detect dates using FIRMS/MCD64A1 only and export to Excel'))
    g.add_argument('--year', type=int, help=_best('Search year for fires with unknown dates (e.g. 2025 → 01.01–31.12)'))
    g.add_argument('--search-start', help=_best('FIRMS search start when fire start is unknown'))
    g.add_argument('--search-end', help=_best('FIRMS search end when fire start is unknown'))
    v = p.add_argument_group(_best('Data'))
    v.add_argument('--satellite', help=_best('Existing daily Sentinel-5P CSV/XLSX instead of GEE (mol/m2)'))
    v.add_argument('--station', help=_best('Ground station (havaizleme.gov.tr) CSV/XLSX, μg/m3'))
    v.add_argument('--station-none', action='store_true', help=_best('Disable automatic station-service download when no station file is supplied'))
    v.add_argument('--station-km', type=float, default=150, help=_best('Nearest-station search radius (km)'))
    v.add_argument('--station-selection', choices=['Wind-aligned', 'Nearest'], default='Wind-aligned', help=_best('Station matching method'))
    v.add_argument('--wind-sector', type=float, default=45, help=_best('Transport-direction tolerance, ±degrees (0–90)'))
    v.add_argument('--station-directory', help=_best('Batch station files: <fire name>.xlsx/.csv'))
    v.add_argument('--weather', help=_best('Weather CSV/XLSX (ERA5-Land via GEE if omitted)'))
    v.add_argument('--gases', default=','.join(GASES), help=_best('E.g. CO,NO2,SO2'))
    s = p.add_argument_group(_best('Settings'))
    s.add_argument('--before-day', type=int, default=10)
    s.add_argument('--after-day', type=int, default=8)
    s.add_argument('--z-threshold', type=float, default=2.5)
    s.add_argument('--z-strong', type=float, default=3.5)
    s.add_argument('--consecutive-day', type=int, default=2)
    s.add_argument('--conversion', choices=['station', 'Manual', 'none'], default='station')
    s.add_argument('--factor', help=_best('Manual factors: CO=24630,NO2=383000 (μg m-3 / mol m-2)'))
    s.add_argument('--station-daily', choices=['Mean', 'max'], default='Mean')
    s.add_argument('--product', choices=['OFFL', 'NRTI'], default='OFFL')
    s.add_argument('--reduction', choices=['mean', 'max'], default='mean')
    s.add_argument('--project', help=_best('Google Cloud / Earth Engine project ID'))
    s.add_argument('--map-none', action='store_true')
    s.add_argument('--buffer-km', type=float, default=5.0)
    p.add_argument('--output', default='output')
    p.add_argument('--desktop_ui', action='store_true', help=_best('Open desktop interface'))
    p.add_argument('--map', action='store_true', help=_best('Web map interface: select and analyze wildfires in Turkey'))
    p.add_argument('--demo', action='store_true', help=_best('Run with sample data'))
    p.add_argument('--station_service-test', action='store_true', help=_best('Test the national station service connection'))
    a = p.parse_args(argv)
    if a.station_service_test:
        from .station_service import connection_test
        return connection_test()
    if a.map:
        import subprocess
        path = Path(__file__).with_name('application.py')
        return subprocess.call([sys.executable, '-m', 'streamlit', 'run', str(path), '--server.maxUploadSize', '200', '--browser.gatherUsageStats', 'false', '--server.headless', 'false'], env=_streamlit_environment())
    if a.desktop_ui:
        from .desktop_ui import run
        return run()
    if a.demo:
        from .demo import demo
        return demo(a.output)
    from .analysis import dates_find_batch, batch_analysis, fire_analysis
    from .data_reading import read_station, weather_read, satellite_series_read, fire_inventory_read
    settings = settings_create(a)
    import pandas as pd
    if a.year and (not a.search_start):
        a.search_start, a.search_end = (f'01.01.{a.year}', f'31.12.{a.year}')
    search_start = pd.to_datetime(a.search_start, dayfirst=True) if a.search_start else None
    search_end = pd.to_datetime(a.search_end, dayfirst=True) if a.search_end else None
    gases = [x.strip().upper() for x in a.gases.split(',') if x.strip().upper() in GASES]
    list_stats = read_station(a.station, settings.station_daily) if a.station else None
    met = weather_read(a.weather) if a.weather else None
    gee_use = a.satellite is None
    if a.shp and (not a.inventory) and (not a.name):
        tmp = fire_inventory_read(a.shp, settings.point_buffer_km, a.epsg, a.name_column)
        if len(tmp) > 1:
            print(_best(f'In SHP: {len(tmp)} separate fires → running batch analysis.'))
            a.inventory = a.shp
    if a.inventory:
        env = fire_inventory_read(a.inventory, settings.point_buffer_km, a.epsg, a.name_column)
        if a.min_area_ha:
            n0 = len(env)
            env = env[env['area_ha_geometry'] >= a.min_area_ha].reset_index(drop=True)
            print(_best(f'{n0 - len(env)} fire {a.min_area_ha:g} skipped: below the area threshold (ha).'))
        if a.only_date:
            if search_start is None:
                p.error(_best('--only-date requires --year (or --search-start/--search-end)'))
            dates_find_batch(env, search_start, search_end, settings, a.output)
            return 0
        if env['start'].isna().any() and gee_use and (search_start is None):
            p.error(_best('Inventory dates missing: specify a search year, e.g. --year 2025'))
        list_stats_s = {}
        if a.station_directory:
            for f in Path(a.station_directory).glob('*'):
                if f.suffix.lower() in ('.csv', '.xlsx', '.xls'):
                    list_stats_s[f.stem] = read_station(f, settings.station_daily)
        if list_stats is not None:
            list_stats_s['*'] = list_stats
        print(_best(f'{len(env)} fires found.'))
        df = batch_analysis(env, a.output, settings, gee_use, a.satellite, list_stats_s, search_start, search_end, gases)
        print(_best(df.to_string()))
        return 0
    geo = None
    if a.shp:
        geo = fire_inventory_read(a.shp, settings.point_buffer_km, a.epsg, a.name_column)
        geo = geo.dissolve().reset_index(drop=True)
    if gee_use and geo is None:
        p.error(_best('GEE mode requires --shp or --inventory; alternatively supply an existing series using --satellite'))
    name = a.name or (Path(a.shp).stem if a.shp else 'fire')
    satellite = satellite_series_read(a.satellite) if a.satellite else None
    start = pd.to_datetime(a.start, dayfirst=True) if a.start else None
    end = pd.to_datetime(a.end, dayfirst=True) if a.end else None
    if gee_use and start is None and (search_start is None):
        p.error(_best('Start date missing: specify a search year, e.g. --year 2025'))
    summary = fire_analysis(name, start, end, geo, satellite, list_stats, met, settings, a.output, gee_use, search_start, search_end, gases)
    for k, v in summary.items():
        print(_best(f'  {k}: {v}'))
    return 0
if __name__ == '__main__':
    sys.exit(main())
