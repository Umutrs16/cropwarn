"""Download notes and light adapters for non-GEE physiology / area products."""

from __future__ import annotations

from pathlib import Path

from cropwarn.config import Config

README = """# Non-GEE data that must be fetched manually or via institutional access

These products are required for the full observational paper run. The demo
pipeline synthesizes equivalent country-crop-year features so the method can
be tested without multi-terabyte downloads.

| Product | Why | Constraint | Entry |
|---|---|---|---|
| GOSIF v2 | Photosynthesis / SIFyield | Model-reconstructed; add TROPOMI/GOME-2 sensitivity | https://globalecology.unh.edu/data/GOSIF.html |
| TROPOMI / ANNSIF | Independent SIF check | Short record | product-specific |
| VODCA v2 | Canopy water / biomass | Coarse; do not mix CXKu and L-band eras | https://researchdata.tuwien.ac.at/records/t74ty-tcx62 |
| SMAP MT-DCA VOD | L-band VOD check | 2015+ | NSIDC / authors |
| CROPGRIDS | Crop spatial weights | Single reference year | https://doi.org/10.1038/s41597-024-03247-7 |
| MIRCA-OS | Irrigation / multi-cropping | Snapshot years | https://doi.org/10.1038/s41597-024-04313-w |
| Global cropping intensity | Single/double/triple | Not crop-specific after 2019 | Figshare |
| African Crop Calendar | Africa phenology validation | ~6 countries | https://doi.org/10.1038/s41597-026-07906-9 |
| GROW-Africa | Subnational Africa yields | Heterogeneous sources | https://doi.org/10.1038/s41597-025-05257-5 |
| CEPII BACI | Fine HS6 trade | Map HS to crops carefully | https://www.cepii.fr |
| SPAM v2020 | Alternate crop weights | Not a dynamic yearly map | https://doi.org/10.7910/DVN/SWPENT |

Hard boundaries
- GlobalCropYield5min is a modelled product. Never use it as the main label.
- AlphaEarth annual embeddings contain post-harvest information. Use year t-1 only.
- WorldCereal 2021 is not a 2001-2024 crop-type time series.
"""


def write_external_readme(cfg: Config) -> Path:
    path = cfg.path("data_raw") / "EXTERNAL_DATA.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(README, encoding="utf-8")
    return path
