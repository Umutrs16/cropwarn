"""SPAM 2020 harvested-area weights and crop-specific sample points.

CROPGRIDS 2020 is the paper's preferred long-run weight; SPAM 2020 v2r2 is
the accessible GeoTIFF stand-in. It is a single-year snapshot, not a 2001–2024
crop-type time series.
"""

from __future__ import annotations

from pathlib import Path
import zipfile
import urllib.request

import numpy as np
import pandas as pd

from cropwarn.constants import TYPICAL_AREA_MHA

SPAM_TIF_URLS = (
    "https://dataverse.harvard.edu/api/access/datafile/13827040?gbrecs=true",
    "https://dataverse.harvard.edu/api/access/datafile/13827040",
    "https://s3.amazonaws.com/mapspam/2020/v2.2/geotiff/spam2020V2r2_global_harvested_area.geotiff.zip",
    "https://s3.amazonaws.com/mapspam/2010/v2.0/geotiff/spam2010v2r0_global_harv_area.geotiff.zip",
)
CROPGRIDS_ZIP = "https://ndownloader.figshare.com/files/44950942"
SPAM_CROP_CODE = {
    "wheat": "WHEA",
    "rice": "RICE",
    "maize": "MAIZ",
    "soybean": "SOYB",
}


def spam_dir(root: Path) -> Path:
    d = root / "data_raw" / "spam2020"
    d.mkdir(parents=True, exist_ok=True)
    return d


def download_spam_harvested(root: Path) -> Path:
    dest = spam_dir(root)
    zip_path = dest / "spam2020V2r2_global_harvested_area.geotiff.zip"
    if not zip_path.exists() or zip_path.stat().st_size < 1_000_000:
        tmp = zip_path.with_suffix(".zip.part")
        last_err = None
        for url in SPAM_TIF_URLS:
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "cropwarn/nature-food"})
                with urllib.request.urlopen(req, timeout=300) as resp, open(tmp, "wb") as fh:
                    while True:
                        chunk = resp.read(1024 * 256)
                        if not chunk:
                            break
                        fh.write(chunk)
                if tmp.stat().st_size > 1_000_000:
                    tmp.replace(zip_path)
                    print(f"SPAM zip {zip_path.stat().st_size} from {url}", flush=True)
                    last_err = None
                    break
            except Exception as exc:
                last_err = exc
                print(f"SKIP {url}: {exc}", flush=True)
        if last_err and (not zip_path.exists() or zip_path.stat().st_size < 1_000_000):
            raise last_err
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    return dest


def _find_tif(root: Path, crop: str) -> Path:
    code = SPAM_CROP_CODE[crop]
    dest = spam_dir(root)
    hits = list(dest.rglob(f"*{code}*A.tif")) + list(dest.rglob(f"*{code}*A.tiff"))
    if not hits:
        hits = [p for p in dest.rglob("*.tif") if code.lower() in p.name.lower() and "r" not in p.stem.lower()[-2:]]
    if not hits:
        raise FileNotFoundError(f"No SPAM tif for {crop} ({code}) under {dest}")
    # Prefer total (A) over rainfed/irrigated splits.
    for p in hits:
        if p.stem.upper().endswith(f"{code}_A") or f"_{code}_A" in p.stem.upper():
            return p
    return hits[0]


def crop_points_from_spam(
    root: Path,
    bounds: pd.DataFrame,
    n_per_country: int = 16,
    min_ha: float = 50.0,
) -> pd.DataFrame:
    """Top harvested-area pixels inside each country bbox."""

    import rasterio
    from rasterio.windows import from_bounds

    rows = []
    for crop in SPAM_CROP_CODE:
        path = _find_tif(root, crop)
        with rasterio.open(path) as src:
            for rec in bounds.itertuples(index=False):
                iso3 = rec.iso3
                if iso3 not in TYPICAL_AREA_MHA.get(crop, {}):
                    continue
                west, south, east, north = rec.west, rec.south, rec.east, rec.north
                try:
                    window = from_bounds(west, south, east, north, src.transform)
                    data = src.read(1, window=window, boundless=True, fill_value=0)
                    transform = src.window_transform(window)
                except Exception as exc:
                    print(f"SKIP spam window {iso3} {crop}: {exc}", flush=True)
                    continue
                arr = np.array(data, dtype=float)
                arr[~np.isfinite(arr)] = 0
                arr[arr < min_ha] = 0
                if not arr.any():
                    arr = np.array(data, dtype=float)
                    arr[~np.isfinite(arr)] = 0
                flat = arr.ravel()
                if not np.isfinite(flat).any() or flat.max() <= 0:
                    continue
                k = min(n_per_country, int((flat > 0).sum()) or 1)
                idx = np.argpartition(flat, -k)[-k:]
                idx = idx[np.argsort(flat[idx])[::-1]]
                height, width = arr.shape
                for i in idx:
                    r, c = divmod(int(i), width)
                    lon, lat = rasterio.transform.xy(transform, r, c)
                    w = float(arr[r, c])
                    if w <= 0:
                        continue
                    rows.append({"iso3": iso3, "crop": crop, "lon": float(lon), "lat": float(lat), "weight": w})
        print(f"SPAM points {crop} {sum(1 for r in rows if r['crop']==crop)}", flush=True)
    out = pd.DataFrame(rows)
    cache = root / "data_interim" / "crop_points.csv"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if len(out):
        out.to_csv(cache, index=False)
    return out
