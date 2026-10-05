# User Guide

1. Install the dependencies and open the application using the README commands.
2. Enable Demo mode for synthetic examples, or authenticate Earth Engine for live observations.
3. Select a year, minimum area, forest fraction and burned-area source. Load wildfires.
4. Select a fire and review its boundary, dates and station matching.
5. Configure the before/after periods, onset threshold, satellite product and station radius. Supply station observations if automatic retrieval is unavailable.
6. Run the analysis. Inspect gas series, meteorology, change onset and calibration charts.
7. Recalculate saved test results to update the year-filtered onset, confidence and Mann–Whitney figures.
8. Export figures or numerical tables.

## Interpretation

Threshold, CUSUM and PELT detection agreement determines the confidence class. Missing estimates and no detection remain visible. Satellite column values use mol/m²; surface conversions use µg/m³ and depend on calibration or supplied factors. Raw-column fallback is explicitly marked. Station performance using calibration observations is not independent validation.

Mann–Whitney compares valid observations before and during each fire. Samples below three observations are marked exploratory. Weather, seasonal variation and dependence between days affect interpretation. P-values are unadjusted.

## Input and output

Use English headers: date, name, start, end, latitude, longitude and area_ha. Gas codes are CO, NO2, CH2O, SO2 and O3. Keep dates unambiguous and units explicit. Regenerate or migrate historical records from earlier editions before importing.

Upload the source directory to GitHub. Exclude environments, output, caches, credentials and downloaded observations. Configure a personal Earth Engine project locally. Label demo observations as synthetic when sharing them.
