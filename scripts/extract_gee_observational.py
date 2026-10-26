from pathlib import Path

from cropwarn.constants import TYPICAL_AREA_MHA
from cropwarn.gee.export import init_ee
from cropwarn.gee.extract import EXTRACTORS, extract_country

ROOT = Path(__file__).resolve().parents[1]
iso3s = sorted({iso for crop in TYPICAL_AREA_MHA.values() for iso in crop})
priority = ["optical", "lst", "era5", "lsp", "lai", "gpp"]
print("countries", iso3s)
ee = init_ee()
for iso3 in iso3s:
    extract_country(iso3, ROOT, ee=ee, sensors=priority)
    print("done", iso3, flush=True)
print("GEE extract complete")
