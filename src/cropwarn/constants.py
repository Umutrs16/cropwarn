"""Shared scientific constants and country metadata."""

from __future__ import annotations

from dataclasses import dataclass

CROPS = ("wheat", "rice", "maize", "soybean")
LEAD_DAYS = (160, 128, 96, 64, 48, 32, 16, 0)
PHENOLOGY_WINDOWS = ((0, 20), (20, 40), (40, 60), (60, 80), (80, 100))
MODALITIES = ("optical", "thermal", "climate", "sm", "sif", "vod")

FAOSTAT_ITEMS = {
    "wheat": 15,
    "rice": 27,
    "maize": 56,
    "soybean": 236,
}

FAOSTAT_ELEMENTS = {
    "area_harvested": 5312,
    "yield": 5412,  # kg/ha in current FAOSTAT QCL normalized files
    "production": 5510,
}

# HS-6 headings used to map BACI / FAOSTAT trade to the four staples.
CROP_HS6 = {
    "wheat": ["100111", "100119", "100191", "100199"],
    "rice": ["100610", "100620", "100630", "100640"],
    "maize": ["100510", "100590"],
    "soybean": ["120110", "120190"],
}


@dataclass(frozen=True)
class Country:
    iso3: str
    name: str
    fao_code: int
    region: str
    latitude_band: str  # temperate | subtropical | tropical
    multi_crop: bool
    exporter: bool
    irrigation: float  # 0-1
    tech_level: float  # 0-1, drives long-run yield level


# Representative producing / importing countries spanning Chen-style clusters.
COUNTRIES: tuple[Country, ...] = (
    Country("USA", "United States", 231, "NorthAmerica", "temperate", False, True, 0.15, 0.95),
    Country("CAN", "Canada", 33, "NorthAmerica", "temperate", False, True, 0.05, 0.90),
    Country("MEX", "Mexico", 138, "NorthAmerica", "subtropical", True, False, 0.35, 0.55),
    Country("BRA", "Brazil", 21, "SouthAmerica", "tropical", True, True, 0.10, 0.70),
    Country("ARG", "Argentina", 9, "SouthAmerica", "subtropical", False, True, 0.08, 0.75),
    Country("CHL", "Chile", 40, "SouthAmerica", "temperate", False, False, 0.45, 0.70),
    Country("FRA", "France", 68, "Europe", "temperate", False, True, 0.12, 0.92),
    Country("DEU", "Germany", 79, "Europe", "temperate", False, True, 0.08, 0.93),
    Country("UKR", "Ukraine", 230, "Europe", "temperate", False, True, 0.05, 0.55),
    Country("RUS", "Russian Federation", 185, "Europe", "temperate", False, True, 0.04, 0.50),
    Country("ESP", "Spain", 203, "Europe", "subtropical", False, False, 0.40, 0.80),
    Country("ITA", "Italy", 106, "Europe", "subtropical", False, False, 0.35, 0.82),
    Country("POL", "Poland", 173, "Europe", "temperate", False, False, 0.05, 0.75),
    Country("ROU", "Romania", 183, "Europe", "temperate", False, True, 0.10, 0.60),
    Country("GBR", "United Kingdom", 229, "Europe", "temperate", False, False, 0.04, 0.88),
    Country("TUR", "Turkiye", 223, "CentralAsia", "subtropical", False, False, 0.25, 0.60),
    Country("KAZ", "Kazakhstan", 108, "CentralAsia", "temperate", False, True, 0.06, 0.45),
    Country("CHN", "China", 351, "EastAsia", "subtropical", True, True, 0.55, 0.80),
    Country("IND", "India", 100, "SouthAsia", "tropical", True, False, 0.40, 0.45),
    Country("PAK", "Pakistan", 165, "SouthAsia", "subtropical", True, False, 0.70, 0.40),
    Country("BGD", "Bangladesh", 16, "SouthAsia", "tropical", True, False, 0.55, 0.35),
    Country("IDN", "Indonesia", 101, "SoutheastAsia", "tropical", True, False, 0.20, 0.40),
    Country("VNM", "Viet Nam", 237, "SoutheastAsia", "tropical", True, True, 0.45, 0.50),
    Country("THA", "Thailand", 216, "SoutheastAsia", "tropical", True, True, 0.30, 0.50),
    Country("PHL", "Philippines", 171, "SoutheastAsia", "tropical", True, False, 0.25, 0.40),
    Country("JPN", "Japan", 110, "EastAsia", "temperate", False, False, 0.55, 0.90),
    Country("KOR", "Republic of Korea", 117, "EastAsia", "temperate", False, False, 0.50, 0.88),
    Country("AUS", "Australia", 10, "Oceania", "subtropical", False, True, 0.06, 0.85),
    Country("NZL", "New Zealand", 156, "Oceania", "temperate", False, False, 0.10, 0.85),
    Country("EGY", "Egypt", 59, "Africa", "subtropical", True, False, 0.95, 0.55),
    Country("NGA", "Nigeria", 159, "Africa", "tropical", True, False, 0.05, 0.25),
    Country("ETH", "Ethiopia", 238, "Africa", "tropical", True, False, 0.04, 0.20),
    Country("KEN", "Kenya", 114, "Africa", "tropical", True, False, 0.08, 0.25),
    Country("ZAF", "South Africa", 202, "Africa", "subtropical", False, True, 0.15, 0.55),
    Country("TZA", "Tanzania", 215, "Africa", "tropical", True, False, 0.05, 0.20),
    Country("GHA", "Ghana", 81, "Africa", "tropical", True, False, 0.04, 0.22),
    Country("MAR", "Morocco", 143, "Africa", "subtropical", False, False, 0.20, 0.40),
    Country("SAU", "Saudi Arabia", 194, "WestAsia", "subtropical", False, False, 0.80, 0.55),
    Country("IRN", "Iran", 102, "WestAsia", "subtropical", False, False, 0.40, 0.45),
    Country("IRQ", "Iraq", 103, "WestAsia", "subtropical", False, False, 0.50, 0.30),
)

COUNTRY_BY_ISO3 = {c.iso3: c for c in COUNTRIES}

# Breadbasket sample points (lat, lon) for GEE getRegion extraction.
BREADBASKET_POINTS: dict[str, tuple[float, float]] = {
    "USA": (41.20, -93.50),
    "CAN": (51.00, -106.00),
    "MEX": (20.50, -101.00),
    "BRA": (-13.00, -55.50),
    "ARG": (-33.70, -60.50),
    "CHL": (-34.50, -71.00),
    "FRA": (48.50, 4.00),
    "DEU": (51.50, 11.00),
    "UKR": (49.00, 32.00),
    "RUS": (51.50, 40.50),
    "ESP": (41.60, -4.70),
    "ITA": (45.10, 9.50),
    "POL": (52.20, 19.00),
    "ROU": (44.50, 26.00),
    "GBR": (52.60, -1.20),
    "TUR": (39.00, 33.00),
    "KAZ": (51.20, 71.40),
    "CHN": (34.80, 113.70),
    "IND": (26.50, 80.30),
    "PAK": (31.50, 73.00),
    "BGD": (24.00, 90.00),
    "IDN": (-7.00, 110.40),
    "VNM": (10.80, 106.00),
    "THA": (16.00, 100.50),
    "PHL": (15.30, 120.90),
    "JPN": (36.00, 140.00),
    "KOR": (36.50, 127.00),
    "AUS": (-31.00, 117.00),
    "NZL": (-43.50, 172.00),
    "EGY": (30.60, 31.00),
    "NGA": (11.00, 8.00),
    "ETH": (9.00, 38.70),
    "KEN": (0.50, 36.00),
    "ZAF": (-27.00, 27.00),
    "TZA": (-6.20, 35.80),
    "GHA": (7.50, -1.00),
    "MAR": (32.30, -7.60),
    "SAU": (24.50, 46.70),
    "IRN": (35.00, 51.00),
    "IRQ": (32.50, 44.40),
}

GEE_PROJECT = "hallowed-chain-458005-g9"

# FAOSTAT sometimes publishes China as 41 and China mainland as 351.
FAO_CODE_ALIASES = {41: "CHN", 351: "CHN"}


# Static crop calendars (day-of-year SOS / EOS). Southern-hemisphere harvests
# may wrap into the next calendar year; EOS > 365 encodes that wrap.
STATIC_CALENDAR: dict[str, dict[str, tuple[int, int]]] = {
    "wheat": {
        "USA": (270, 200),
        "CAN": (120, 250),
        "FRA": (290, 200),
        "DEU": (280, 210),
        "UKR": (260, 200),
        "RUS": (250, 220),
        "AUS": (130, 330),
        "ARG": (150, 350),
        "CHN": (280, 170),
        "IND": (320, 110),
        "PAK": (310, 120),
        "TUR": (300, 190),
        "KAZ": (120, 250),
        "EGY": (320, 130),
        "MAR": (320, 150),
        "IRN": (300, 170),
        "IRQ": (310, 150),
        "GBR": (270, 210),
        "POL": (270, 220),
        "ROU": (280, 200),
        "ESP": (320, 170),
        "ITA": (310, 180),
        "ZAF": (140, 330),
        "ETH": (170, 320),
        "default": (280, 200),
    },
    "rice": {
        # CHN Jiangxi / middle-late rice: climatological EVI peaks in August.
        # The old 100–280 window mixed early and late crops.
        "CHN": (145, 275),
        "IND": (170, 320),
        "BGD": (160, 330),
        "IDN": (300, 150),
        # Mekong winter–spring crop (peak February), not the summer calendar.
        "VNM": (330, 130),
        "THA": (180, 320),
        "PHL": (160, 300),
        "JPN": (130, 270),
        "KOR": (140, 270),
        "EGY": (130, 270),
        "NGA": (150, 300),
        "USA": (120, 260),
        "BRA": (310, 120),
        "PAK": (170, 310),
        "default": (150, 300),
    },
    "maize": {
        "USA": (120, 280),
        "BRA": (280, 150),
        "ARG": (280, 140),
        "CHN": (130, 270),
        "IND": (170, 300),
        "MEX": (150, 300),
        "FRA": (120, 270),
        "UKR": (120, 270),
        "ROU": (120, 270),
        "NGA": (140, 300),
        "ETH": (150, 310),
        "ZAF": (300, 150),
        "IDN": (280, 120),
        "default": (130, 280),
    },
    "soybean": {
        "USA": (130, 280),
        "BRA": (280, 140),
        "ARG": (300, 130),
        "CHN": (140, 270),
        "IND": (170, 300),
        "CAN": (140, 270),
        "default": (140, 280),
    },
}

# Second rice season for countries where FAO annual yield sums two crops.
# Primary calendar stays in STATIC_CALENDAR; these windows are extra features only.
RICE_DUAL_SEASONS: dict[str, tuple[int, int]] = {
    "CHN": (90, 195),
    "VNM": (150, 280),
    "BGD": (340, 145),
    "PHL": (350, 130),
    "IND": (350, 130),
    "IDN": (150, 270),
}

# Southern-hemisphere maize (austral summer). Trained separately from NH maize.
MAIZE_SOUTH_ISO3 = frozenset({"BRA", "ARG", "ZAF", "IDN"})

# Typical harvested-area weights (million ha scale, relative).
TYPICAL_AREA_MHA = {
    "wheat": {
        "CHN": 23.0, "IND": 30.0, "RUS": 27.0, "USA": 15.0, "FRA": 5.0,
        "UKR": 6.5, "AUS": 12.0, "CAN": 10.0, "PAK": 9.0, "TUR": 7.0,
        "DEU": 3.0, "KAZ": 12.0, "IRN": 6.0, "ARG": 6.0, "GBR": 1.8,
        "POL": 2.4, "ROU": 2.1, "ESP": 1.9, "ITA": 1.2, "EGY": 1.4,
        "MAR": 2.8, "ETH": 1.8, "IRQ": 1.2,
    },
    "rice": {
        "CHN": 30.0, "IND": 44.0, "BGD": 11.5, "IDN": 10.5, "VNM": 7.2,
        "THA": 10.0, "PHL": 4.8, "JPN": 1.5, "KOR": 0.7, "USA": 1.1,
        "BRA": 1.6, "EGY": 0.5, "NGA": 3.6, "PAK": 3.3,
    },
    "maize": {
        "USA": 33.0, "CHN": 42.0, "BRA": 20.0, "ARG": 7.5, "IND": 10.0,
        "MEX": 7.0, "UKR": 5.0, "FRA": 1.6, "ROU": 2.5, "NGA": 6.0,
        "ETH": 2.3, "ZAF": 2.6, "IDN": 3.5, "CAN": 1.4,
    },
    "soybean": {
        "USA": 34.0, "BRA": 42.0, "ARG": 16.0, "CHN": 9.0, "IND": 12.0,
        "CAN": 2.2,
    },
}

BASE_YIELD_THA = {
    "wheat": 3.4,
    "rice": 4.6,
    "maize": 5.8,
    "soybean": 2.6,
}

REGIONS = (
    "NorthAmerica",
    "SouthAmerica",
    "Europe",
    "CentralAsia",
    "EastAsia",
    "SouthAsia",
    "SoutheastAsia",
    "Oceania",
    "Africa",
    "WestAsia",
)
