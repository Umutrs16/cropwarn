# CropWarn

Analysis code for pre-harvest wheat and soybean yield forecasts and lagged trade-exposure ranking.

The repository contains the `cropwarn` package, configuration files, tests, and the scripts that download public inputs, build features, fit rolling-origin models, and score importer exposure. It does not contain the Word manuscript, and it does not redistribute FAOSTAT or Google Earth Engine extracts.

## What the meteorological–thermal block contains

The block used in the source comparison is not reanalysis temperature alone. It is MODIS Terra MOD11A2 daytime land-surface temperature (`LST_Day_1km`) together with ERA5-Land total precipitation, potential evaporation, and vapour-pressure deficit calculated from ERA5-Land 2 m maximum temperature and dewpoint. The 2 m temperature is not itself a model feature.

## Install

```powershell
python -m pip install -e ".[dev]"
```

Earth Engine extraction also needs the optional `full` extra and an authenticated Earth Engine project.

## Reproduce the exposure scores

From the repository root, after the interim panel, trade, and season-series tables are in place:

```powershell
python scripts/run_obs_experiments.py
python scripts/run_as_revision.py
python scripts/run_as_p0b.py
python scripts/run_as_source_exposure.py
python scripts/plot_as_figures.py
```

`run_obs_experiments.py` writes the rolling-origin prediction file. The later scripts score that file. They do not replace it. Official prediction rows and the evaluation tables cited in the article are distributed with the manuscript, not in this repository.

## Inputs

Public production and trade tables come from FAOSTAT. Optical indices, land-surface temperature, and ERA5-Land variables are sampled in Google Earth Engine. Acquisition notes are in `docs/DATA.md`. Split rules are in `docs/LEAKAGE.md`.

## License

Code is released under the MIT license. FAOSTAT, MODIS, ERA5-Land, GOSIF, VODCA, and the other source products keep their own licenses.
