"""Crop sample points: WorldCereal 2021 near breadbaskets, plus CROPGRIDS/SPAM when present.

WorldCereal is a 2021 check, not a yearly map. Points are used only as spatial
weights for 2001–2024 reducers.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from cropwarn.constants import BREADBASKET_POINTS, GEE_PROJECT, TYPICAL_AREA_MHA
from cropwarn.gee.export import init_ee
from cropwarn.gee.national import study_iso3s

WC_PRODUCT = {
    "wheat": ["wintercereals", "springcereals"],
    "maize": ["maize"],
}


def crop_points_worldcereal(root: Path, n_per: int = 14, buffer_m: int = 250_000) -> pd.DataFrame:
    cache = root / "data_interim" / "crop_points.csv"
    ee = init_ee(GEE_PROJECT)
    wc = ee.ImageCollection("ESA/WorldCereal/2021/MODELS/v100")
    rows = []
    iso3s = study_iso3s()
    for crop, products in WC_PRODUCT.items():
        mask = ee.Image(0)
        for prod in products:
            img = wc.filter(ee.Filter.eq("product", prod)).select("classification").mosaic()
            mask = mask.Or(img.gt(0))
        mask = mask.selfMask()
        for iso3 in iso3s:
            if iso3 not in TYPICAL_AREA_MHA.get(crop, {}) or iso3 not in BREADBASKET_POINTS:
                continue
            lat, lon = BREADBASKET_POINTS[iso3]
            region = ee.Geometry.Point([lon, lat]).buffer(buffer_m)
            try:
                pts = mask.sample(
                    region=region,
                    scale=1000,
                    numPixels=n_per,
                    seed=11,
                    geometries=True,
                    dropNulls=True,
                ).getInfo()
            except Exception as exc:
                print(f"SKIP wc {iso3} {crop}: {exc}", flush=True)
                continue
            n = 0
            for p in pts.get("features", []):
                coords = p.get("geometry", {}).get("coordinates")
                if not coords:
                    continue
                rows.append({"iso3": iso3, "crop": crop, "lon": coords[0], "lat": coords[1], "weight": 1.0, "source": "worldcereal2021"})
                n += 1
            print(f"wc {iso3} {crop} n={n}", flush=True)
    out = pd.DataFrame(rows)
    if cache.exists() and cache.stat().st_size > 50:
        old = pd.read_csv(cache)
        extra = old[~old["crop"].isin(WC_PRODUCT)]
        out = pd.concat([out, extra], ignore_index=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    if len(out):
        out.to_csv(cache, index=False)
    return out
