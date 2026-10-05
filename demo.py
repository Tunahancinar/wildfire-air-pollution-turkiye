"""Implementation of the wildfire gas analysis module."""
from pathlib import Path
import numpy as np
import pandas as pd
_PATTERN = {'CO': [1030, 990, 1030, 950, 5492, 2060, 2940, 2800, 1820, 1160, 900, 1300, 1040, 1130, 1260, 1050, 980], 'NO2': [112, 96, 128, 140, 383, 276, 276, 192, 350, 140, 200, 137, 95, 105, 136, 140, 130], 'SO2': [15, 53, 13, 12, 12, 38, 13, 39, 12, 13, 42, 18, 11, 71, 14, 12, 13], 'O3': [150, 160, 156, 130, 81, 26, 48, 52, 55, 23, 5.5, 82, 84, 72, 160, 156, 126], 'CH2O': [7, 3, 8, 6, 12.8, 11.2, 10.4, 10.8, 10.6, 8.1, 7.4, 6.2, 2.5, 8.1, 8.2, 7.8, 8.0]}
_FACTOR = {'CO': 5492.63 / 0.223, 'NO2': 383.52 / 0.001, 'SO2': 71.49 / 0.0011, 'O3': 160.92 / 0.139, 'CH2O': 12.83 / 0.0008}

def example_data_generate(directory: Path):
    rng = np.random.default_rng(42)
    directory.mkdir(parents=True, exist_ok=True)
    days = pd.date_range('2025-07-15', '2025-08-16', freq='D')
    pattern_start = pd.Timestamp('2025-07-25')
    satellite, real = ({}, {})
    for g, d in _PATTERN.items():
        d = np.array(d, float)
        baseline = d[:4].mean()
        series = []
        for t in days:
            i = (t - pattern_start).days
            v = d[i] if 0 <= i < len(d) else baseline * (1 + rng.normal(0, 0.12))
            series.append(v)
        series = np.array(series) * (1 + rng.normal(0, 0.03, len(series)))
        real[g] = series
        satellite[g] = series / _FACTOR[g]
    s5p = pd.DataFrame(satellite, index=days)
    s5p.iloc[[5, 22], :] = np.nan
    s5p.index.name = 'date'
    s5p.rename(columns={'CO': 'CO_column_number_density', 'NO2': 'NO2_column_number_density', 'SO2': 'SO2_column_number_density', 'O3': 'O3_column_number_density', 'CH2O': 'tropospheric_HCHO_column_number_density'}).reset_index().assign(date=lambda x: x.date.dt.strftime('%d.%m.%Y')).to_csv(directory / 'example_s5p_series.csv', index=False)
    hour = pd.date_range(days[0], days[-1] + pd.Timedelta(hours=23), freq='h')
    list_stats = {'Date': hour.strftime('%d.%m.%Y %H:%M')}
    for g, name in (('CO', 'CO ( μg/m3 )'), ('NO2', 'NO2 ( μg/m3 )'), ('SO2', 'SO2 ( μg/m3 )'), ('O3', 'O3 ( μg/m3 )')):
        day_change = pd.Series(real[g], index=days).reindex(hour.normalize()).values
        day_error = pd.Series(1 + rng.normal(0, 0.15, len(days)), index=days).reindex(hour.normalize()).values
        daytime = day_error + 0.15 * np.sin((hour.hour - 8) / 24 * 2 * np.pi)
        list_stats[name] = np.round(day_change * 0.85 * daytime * (1 + rng.normal(0, 0.12, len(hour))), 2)
    pd.DataFrame(list_stats).to_excel(directory / 'example_station_hourly.xlsx', index=False)
    fire = (hour >= '2025-07-28') & (hour < '2025-08-09')
    met = pd.DataFrame({'Date': hour.strftime('%d.%m.%Y %H:%M'), 'Temperature': np.round(28 + 7 * np.sin((hour.hour - 9) / 24 * 2 * np.pi) + rng.normal(0, 1, len(hour)), 1), 'Humidity': np.round(np.clip(50 - 25 * np.sin((hour.hour - 9) / 24 * 2 * np.pi) - 12 * fire + rng.normal(0, 4, len(hour)), 8, 100), 1), 'Wind_speed': np.round(np.abs(4 + 2.5 * fire + 2 * np.sin(hour.hour / 24 * 2 * np.pi) + rng.normal(0, 1, len(hour))), 2), 'Wind_direction': np.round((np.where(fire, 225, 300) + rng.normal(0, 25, len(hour))) % 360, 0)})
    met.to_csv(directory / 'example_weather_hourly.csv', index=False)
    try:
        import geopandas as gpd
        from shapely.geometry import Polygon
        polygon = Polygon([(31.35, 36.88), (31.62, 36.86), (31.7, 36.98), (31.55, 37.08), (31.38, 37.02)])
        gpd.GeoDataFrame({'name': ['Example_Fire'], 'start': ['28.07.2025'], 'end': ['08.08.2025'], 'area_ha': [58798]}, geometry=[polygon], crs='EPSG:4326').to_file(directory / 'example_fire_inventory.shp', encoding='utf-8')
    except Exception as e:
        print('Could not write the sample shapefile:', e)
    return directory

def demo(output='output_demo'):
    from .analysis import fire_analysis
    from .config import AnalysisSettings
    from .data_reading import read_station, weather_read, satellite_series_read
    k = example_data_generate(Path(output) / 'example_data')
    settings = AnalysisSettings(manual_factor={'CH2O': 16000.0})
    summary = fire_analysis('Example_Fire_2025', start='2025-07-28', end='2025-08-08', satellite=satellite_series_read(k / 'example_s5p_series.csv'), station=read_station(k / 'example_station_hourly.xlsx'), weather=weather_read(k / 'example_weather_hourly.csv'), settings=settings, output=output)
    for a, v in summary.items():
        print(f'  {a}: {v}')
    return 0
