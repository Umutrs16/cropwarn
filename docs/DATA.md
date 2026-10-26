# Data

## Demo path

`cropwarn run-demo` builds a mechanistic synthetic world that encodes H1–H4.
It is for method verification, figure generation and CI. It is **not**
observational evidence.

## Official observational path (headline)

1. FAOSTAT QCL via `python scripts/download_public_data.py` or the bulk ZIP.
   Headline yield targets: wheat and soybean only.
2. FAOSTAT Detailed Trade Matrix (year t−1 only).
3. National cropland-weighted monthly means: `python scripts/extract_gee_national.py`.
   Mask is year-specific MCD12Q1 cropland (12/14). SMAP L4 is one mid-month image.
4. GOSIF v2 monthly: `python scripts/extract_gosif.py` (true SIF, not MOD17 GPP).
5. VODCA crop-point extract in `data_interim/vodca_crop.csv` (ingested).

See `data_raw/EXTERNAL_DATA.md` after `cropwarn init`.

## Held-out extracts (not in official gates)

| Extract | Labels | Remote sensing | Why held out |
|---|---|---|---|
| India APY kharif/Aman rice | state / district | GEE rice points | Canopy vs APY r≈0 |
| China NBS early rice | provincial, calendar 90–195 | GEE + GOSIF | Couples (harvest R²≈0.61) but B4 adds no lead; P_full harvest R²≈0.13 |
| FAO national maize | country | national GEE | All leads R² negative |
| US NASS maize | county and state | GEE + GOSIF | In-sample July SIF r≈0.39; OOS harvest R² < 0 |
| Eurostat NUTS maize | NUTS, yield = production/area | GEE | Harvest R²≈0.07; no earliest_lead |

When `data_interim/usa_maize.csv` exists, FAO maize is dropped from the observational world. Headline filtering still removes maize before gates.

## Hard boundaries

- GlobalCropYield5min is not an independent label.
- GOSIF is reconstructed (Li & Xiao 2019); do not write it as TROPOMI/GOME-2.
- AlphaEarth annual embeddings leak post-harvest information.
- WorldCereal 2021 is not a 24-year crop-type time series.
- Do not put held-out rice or maize into `headline_crops` unless official gates actually GO.
