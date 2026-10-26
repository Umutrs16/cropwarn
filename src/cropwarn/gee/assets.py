"""GEE collection IDs verified against the paper catalogue (2026-09-01)."""

GEE_ASSETS = {
    "mod13q1": "MODIS/061/MOD13Q1",
    "mcd15a3h": "MODIS/061/MCD15A3H",
    "mod11a2": "MODIS/061/MOD11A2",
    "mcd12q2": "MODIS/061/MCD12Q2",
    "era5_land": "ECMWF/ERA5_LAND/DAILY_AGGR",
    "smap_l4": "NASA/SMAP/SPL4SMGP/008",
    "smap_l3": "NASA/SMAP/SPL3SMP_E/005",
    "chirps": "UCSB-CHG/CHIRPS/DAILY",
    "s1": "COPERNICUS/S1_GRD",
    "s2": "COPERNICUS/S2_SR_HARMONIZED",
    "dynamic_world": "GOOGLE/DYNAMICWORLD/V1",
    "worldcereal_models": "ESA/WorldCereal/2021/MODELS/v100",
    "worldcereal_aez": "ESA/WorldCereal/AEZ/v100",
    "worldcereal_markers": "ESA/WorldCereal/2021/MARKERS/v100",
    "alphaearth": "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL",
}

LEAKAGE_NOTES = {
    "alphaearth": "Annual embeddings contain post-harvest information. Use year t-1 only.",
    "worldcereal_models": "2021 seasonal maps are not a 2001-2024 crop-type time series.",
    "mcd12q2": "Land-surface phenology is not crop-specific; apply inside a crop mask.",
}


def leakage_notes() -> dict[str, str]:
    return dict(LEAKAGE_NOTES)
