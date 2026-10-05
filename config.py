"""Implementation of the wildfire gas analysis module."""
from dataclasses import dataclass, field
from typing import Dict, Optional

@dataclass
class GasInfo:
    code: str
    name: str
    gee_offl: str
    gee_nrti: str
    band: str
    molar_mass: float
    direction: str = 'Increase'
    who_threshold: Optional[float] = None
    station_names: tuple = ()
GASES: Dict[str, GasInfo] = {'CO': GasInfo('CO', 'CO', 'COPERNICUS/S5P/OFFL/L3_CO', 'COPERNICUS/S5P/NRTI/L3_CO', 'CO_column_number_density', 28.01, 'Increase', None, ('co', 'carbon_monoxide', 'carbon monoxide')), 'NO2': GasInfo('NO2', 'NO$_2$', 'COPERNICUS/S5P/OFFL/L3_NO2', 'COPERNICUS/S5P/NRTI/L3_NO2', 'NO2_column_number_density', 46.006, 'Increase', 40.0, ('no2', 'nitrogen_dioxide', 'nitrogen dioxide')), 'CH2O': GasInfo('CH2O', 'CH$_2$O', 'COPERNICUS/S5P/OFFL/L3_HCHO', 'COPERNICUS/S5P/NRTI/L3_HCHO', 'tropospheric_HCHO_column_number_density', 30.026, 'Increase', None, ('ch2o', 'hcho', 'formaldehyde')), 'SO2': GasInfo('SO2', 'SO$_2$', 'COPERNICUS/S5P/OFFL/L3_SO2', 'COPERNICUS/S5P/NRTI/L3_SO2', 'SO2_column_number_density', 64.066, 'Increase', 40.0, ('so2', 'sulfur_dioxide', 'sulfur dioxide', 'sulfur dioxide')), 'O3': GasInfo('O3', 'O$_3$', 'COPERNICUS/S5P/OFFL/L3_O3', 'COPERNICUS/S5P/NRTI/L3_O3', 'O3_column_number_density', 47.998, 'two_sided', 160.0, ('o3', 'ozone'))}
GAS_ALIASES = {'CO': ['co', 'co_column_number_density'], 'NO2': ['no2', 'no2_column_number_density', 'tropospheric_no2_column_number_density'], 'CH2O': ['ch2o', 'hcho', 'tropospheric_hcho_column_number_density'], 'SO2': ['so2', 'so2_column_number_density'], 'O3': ['o3', 'o3_column_number_density']}

@dataclass
class AnalysisSettings:
    before_day: int = 10
    after_day: int = 8
    z_threshold: float = 2.5
    z_strong: float = 3.5
    consecutive_day: int = 2
    cusum_drift: float = 0.5
    cusum_h: float = 4.0
    search_before_day: int = 1
    min_baseline_observation: int = 3
    pelt_penalty: Optional[float] = None
    conversion: str = 'station'
    manual_factor: Dict[str, float] = field(default_factory=dict)
    station_automatic: bool = True
    station_max_km: float = 150.0
    station_daily: str = 'Mean'
    station_selection: str = 'Wind-aligned'
    wind_sector: float = 45.0
    gee_product: str = 'OFFL'
    gee_reduction: str = 'mean'
    gee_scale: int = 1113
    gee_project: Optional[str] = None
    firms_use: bool = True
    map_download: bool = True
    point_buffer_km: float = 5.0
