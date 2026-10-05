# Wildfire Gas Analysis

Wildfire pollutant analysis using Sentinel-5P, ground stations and ERA5-Land. Source code, configuration keys, the interface and exports use English.

## Install and run

Python 3.12 is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run wildfire_gas/application.py --theme.base light
```

Enable Demo mode for synthetic observations without Earth Engine. For live data, authenticate Earth Engine and enter your own enabled Cloud project ID.

```powershell
.\.venv\Scripts\earthengine.exe authenticate
.\.venv\Scripts\python.exe -m wildfire_gas --demo --output output
```

## Visual results

Select a year to inspect saved analyses for that year. Charts show CUSUM and PELT onset dates, consensus, confidence classes and Mann–Whitney statistics. Export PNG, PDF, Excel or CSV.

Confidence describes agreement within one day: High = three methods, Medium = two, Low = one. No detection and unavailable estimates remain separate. Confidence is not a probability or proof of wildfire causation.

Mann–Whitney comparisons are two-sided and exploratory. P-values have no multiplicity correction; daily independence has not been verified. Samples with fewer than three days are flagged. Unconverted satellite column results are marked separately and do not estimate surface concentrations.

## Compatibility and publishing

This edition uses the `wildfire_gas` package and English metadata keys. Earlier saved records require migration or regeneration. Output belongs in `output/`, which is excluded from Git. Environments, credentials and downloaded observations are not included. External service URLs retain their official hostnames.

See [USER_GUIDE.md](USER_GUIDE.md). No open-source license has been selected; choose one before granting reuse rights.
