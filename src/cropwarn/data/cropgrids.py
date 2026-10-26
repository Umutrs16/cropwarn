"""CROPGRIDS 2020 harvested-area maps (figshare). Long-run weights only."""

from __future__ import annotations

from pathlib import Path
import zipfile
import urllib.request

import pandas as pd

CROPGRIDS_ZIP = "https://ndownloader.figshare.com/files/44950942"
CROP_ALIASES = {
    "wheat": ("wheat", "whea"),
    "rice": ("rice",),
    "maize": ("maize", "maiz", "corn"),
    "soybean": ("soybean", "soyabean", "soyb", "soy"),
}


def download_cropgrids(root: Path) -> Path:
    dest = root / "data_raw" / "cropgrids"
    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / "CROPGRIDSv1.08_NC_maps.zip"
    if zip_path.exists() and zip_path.stat().st_size > 100_000_000:
        return dest
    tmp = zip_path.with_suffix(".zip.part")
    req = urllib.request.Request(CROPGRIDS_ZIP, headers={"User-Agent": "cropwarn/nature-food"})
    with urllib.request.urlopen(req, timeout=600) as resp, open(tmp, "wb") as fh:
        n = 0
        while True:
            chunk = resp.read(1024 * 512)
            if not chunk:
                break
            fh.write(chunk)
            n += len(chunk)
            if n % (50 * 1024 * 1024) < 512 * 1024:
                print(f"CROPGRIDS {n/1e6:.0f} MB", flush=True)
    tmp.replace(zip_path)
    print(f"CROPGRIDS zip {zip_path.stat().st_size}", flush=True)
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        keep = []
        for name in names:
            low = name.lower()
            if any(alias in low for aliases in CROP_ALIASES.values() for alias in aliases):
                keep.append(name)
        if not keep:
            zf.extractall(dest)
        else:
            for name in keep:
                zf.extract(name, dest)
                print("extracted", name, flush=True)
    return dest


def nc_path(root: Path, crop: str) -> Path:
    return root / "data_raw" / "cropgrids" / "CROPGRIDSv1.08_NC_maps" / f"CROPGRIDSv1.08_{crop}.nc"


def crop_points_from_cropgrids(
    root: Path,
    bounds: pd.DataFrame,
    n_per_country: int = 16,
    min_ha: float = 20.0,
    polygons: dict | None = None,
) -> pd.DataFrame:
    import numpy as np
    import xarray as xr

    from cropwarn.constants import TYPICAL_AREA_MHA

    rows = []
    for crop in ("wheat", "rice", "maize", "soybean"):
        path = nc_path(root, crop)
        if not path.exists():
            print(f"SKIP missing {path}", flush=True)
            continue
        ds = xr.open_dataset(path)
        ha = ds["harvarea"]
        for rec in bounds.itertuples(index=False):
            iso3 = rec.iso3
            if iso3 not in TYPICAL_AREA_MHA.get(crop, {}):
                continue
            sub = ha.sel(lon=slice(float(rec.west), float(rec.east)), lat=slice(float(rec.south), float(rec.north)))
            vals = np.asarray(sub.values, dtype=float)
            lats = sub["lat"].values
            lons = sub["lon"].values
            if vals.size == 0:
                continue
            vals[~np.isfinite(vals)] = 0
            geom = polygons.get(iso3) if polygons else None
            if geom is not None and vals.size:
                from affine import Affine
                from rasterio.features import geometry_mask

                res_x = float(lons[1] - lons[0]) if len(lons) > 1 else 0.05
                res_y = float(lats[1] - lats[0]) if len(lats) > 1 else 0.05
                transform = Affine.translation(float(lons[0]) - res_x / 2, float(lats[0]) - res_y / 2) * Affine.scale(res_x, res_y)
                inside = geometry_mask([geom.__geo_interface__], out_shape=vals.shape, transform=transform, invert=True)
                vals = np.where(inside, vals, 0)
            flat = vals.ravel()
            if flat.max() <= 0:
                continue
            k = min(n_per_country, int((flat >= min_ha).sum()) or int((flat > 0).sum()) or 1)
            idx = np.argpartition(flat, -k)[-k:]
            idx = idx[np.argsort(flat[idx])[::-1]]
            width = vals.shape[1]
            for i in idx:
                r, c = divmod(int(i), width)
                w = float(flat[i])
                if w <= 0:
                    continue
                rows.append(
                    {
                        "iso3": iso3,
                        "crop": crop,
                        "lon": float(lons[c]),
                        "lat": float(lats[r]),
                        "weight": w,
                        "source": "cropgrids2020",
                    }
                )
        ds.close()
        print(f"CROPGRIDS points {crop} {sum(1 for r in rows if r['crop']==crop)}", flush=True)
    out = pd.DataFrame(rows)
    cache = root / "data_interim" / "crop_points.csv"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if len(out):
        out.to_csv(cache, index=False)
    return out

