"""Optional Earth Engine export helpers. Require a local `earthengine authenticate`."""

from __future__ import annotations

from cropwarn.gee.assets import GEE_ASSETS


def init_ee(project: str | None = None):
    import ee

    from cropwarn.constants import GEE_PROJECT

    ee.Initialize(project=project or GEE_PROJECT)
    return ee


def crop_weighted_country_means(ee, year: int, start: str, end: str, countries):
    """Template reducer: MODIS EVI mean inside a country, 2000–cutoff only."""

    col = (
        ee.ImageCollection(GEE_ASSETS["mod13q1"])
        .filterDate(start, end)
        .select(["EVI", "SummaryQA"])
        .map(lambda img: img.updateMask(img.select("SummaryQA").lte(1)).select("EVI").multiply(0.0001))
    )
    img = col.mean()
    return img.reduceRegions(collection=countries, reducer=ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True), scale=250)
