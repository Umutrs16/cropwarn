"""China NBS provincial early-rice yields (早稻).

Early rice is harvested around mid-July (calendar 90–195). That is the
season remote sensing can actually see: January 192-day leads have no
canopy, so optical harvest skill should collapse at long leads while
May–June SIF/VOD can still move. FAO annual rice stacks early + middle
+ late and is not a valid target.
"""

from __future__ import annotations

from pathlib import Path
import re
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from cropwarn.data.cropgrids import nc_path

EARLY_RICE_CALENDAR = (90, 195)

PROVINCES = {
    "zhejiang": "ZJ",
    "anhui": "AH",
    "fujian": "FJ",
    "jiangxi": "JX",
    "hubei": "HB",
    "hunan": "HN",
    "guangdong": "GD",
    "guangxi": "GX",
    "guangxi zhuang": "GX",
    "hainan": "HI",
    "yunnan": "YN",
    "浙江": "ZJ",
    "安徽": "AH",
    "福建": "FJ",
    "江西": "JX",
    "湖北": "HB",
    "湖南": "HN",
    "广东": "GD",
    "广西": "GX",
    "海南": "HI",
    "云南": "YN",
}

PROVINCE_NAMES = {
    "ZJ": "Zhejiang",
    "AH": "Anhui",
    "FJ": "Fujian",
    "JX": "Jiangxi",
    "HB": "Hubei",
    "HN": "Hunan",
    "GD": "Guangdong",
    "GX": "Guangxi",
    "HI": "Hainan",
    "YN": "Yunnan",
}

# Published NBS early-rice bulletins (English / Chinese). Extra years come
# from Wayback + data.stats.gov.cn when the live site moves a permalink.
BULLETIN_URLS = {
    2015: [
        "https://www.stats.gov.cn/sj/zxfb/202302/t20230203_1900176.html",
        "https://www.stats.gov.cn/tjsj/zxfb/201508/t20150824_1234197.html",
    ],
    2016: [
        "https://www.stats.gov.cn/sj/zxfb/202302/t20230203_1900120.html",
        "https://www.stats.gov.cn/tjsj/zxfb/201608/t20160824_1391883.html",
    ],
    2017: [
        "https://www.stats.gov.cn/sj/zxfb/202302/t20230203_1900078.html",
        "https://www.stats.gov.cn/tjsj/zxfb/201708/t20170825_1526123.html",
        "https://read01.com/2Gd7kRj.html",
    ],
    2018: [
        "https://www.stats.gov.cn/sj/zxfb/202302/t20230203_1900038.html",
        "https://www.stats.gov.cn/tjsj/zxfb/201808/t20180824_1619470.html",
    ],
    2019: [
        "https://www.stats.gov.cn/sj/zxfb/202302/t20230203_1899988.html",
        "https://www.stats.gov.cn/tjsj/zxfb/201908/t20190826_1692494.html",
    ],
    2020: [
        "https://www.stats.gov.cn/english/PressRelease/202008/t20200820_1784289.html",
        "https://www.stats.gov.cn/sj/zxfb/202302/t20230203_1902155.html",
    ],
    2021: [
        "https://www.stats.gov.cn/english/PressRelease/202108/t20210825_1821028.html",
        "https://www.stats.gov.cn/sj/zxfb/202302/t20230203_1901743.html",
    ],
    2022: [
        "https://www.stats.gov.cn/english/PressRelease/202208/t20220826_1887653.html",
    ],
    2023: [
        "https://www.stats.gov.cn/english/PressRelease/202308/t20230824_1942290.html",
        "https://www.stats.gov.cn/sj/zxfb/202308/t20230823_1942209.html",
    ],
    2024: [
        "https://www.stats.gov.cn/english/PressRelease/202409/t20240906_1956283.html",
        "https://www.stats.gov.cn/sj/zxfb/202408/t20240823_1956083.html",
    ],
    2025: [
        "https://www.stats.gov.cn/english/PressRelease/202508/t20250826_1960878.html",
        "https://www.stats.gov.cn/sj/zxfb/202508/t20250822_1960862.html",
    ],
}

# Official NBS bulletin cells transcribed when the live site TLS fails.
# Values are (area_1000ha, yield_kg_ha). Production is not stored.
# Gotohui republishes NBS 农村统计 provincial yield (kg/ha). Used to
# backfill 2010–2015 / 2018 so the 2020 flood year can be an anomaly.
GOTOHUI_BACKFILL = {
    "ZJ": {2010: 5393.64, 2011: 6111.31, 2012: 6037.22, 2013: 6229.69, 2014: 6143.67, 2015: 5806.21},
    "AH": {2010: 5326.50, 2011: 5355.19, 2012: 5556.96, 2013: 5554.14, 2014: 5694.63, 2015: 5748.88},
    "FJ": {2010: 5782.86, 2011: 6014.55, 2012: 6020.39, 2013: 5996.30, 2014: 6020.45, 2015: 6059.62, 2018: 6357.11},
    "JX": {2010: 5078.88, 2011: 5571.35, 2012: 5601.73, 2013: 5719.64, 2014: 5736.87, 2015: 5749.29, 2018: 5746.11},
    "HB": {2010: 5566.93, 2011: 5689.29, 2012: 5937.69, 2013: 5776.56, 2014: 5787.34, 2015: 5964.30, 2018: 5929.01},
    "HN": {2010: 5625.01, 2011: 5774.96, 2012: 5746.50, 2013: 5947.51, 2014: 5881.53, 2015: 5944.50, 2018: 6101.60},
    "GD": {2010: 5429.53, 2011: 5683.53, 2012: 5710.47, 2013: 5755.28, 2014: 5857.48, 2015: 5902.52, 2018: 5958.01},
    "YN": {2010: 6479.85, 2011: 6280.48, 2012: 6313.94, 2013: 5259.93, 2014: 5293.91, 2015: 5313.74, 2018: 5593.56},
}


def _gotohui_backfill_frame() -> pd.DataFrame:
    rows = []
    for code, years in GOTOHUI_BACKFILL.items():
        for year, yld in years.items():
            if yld is None:
                continue
            rows.append(
                {
                    "iso3": "CHN",
                    "unit_id": f"CHN-{code}",
                    "state_code": code,
                    "crop": "rice",
                    "year": year,
                    "season": "early",
                    "yield_tha": yld / 1000.0,
                    "harvested_ha": np.nan,
                    "source": "nbs_rural_gotohui",
                }
            )
    return pd.DataFrame(rows)


NBS_SEED = {
    2016: {  # https://read01.com/EAGNNB.html reprint of NBS 2016 bulletin
        "ZJ": (115.5, 6391.3),
        "AH": (175.6, 5559.9),
        "FJ": (169.6, 6026.5),
        "JX": (1365.1, 5757.2),
        "HB": (412.1, 5244.9),
        "HN": (1420.5, 5872.0),
        "GD": (891.8, 5980.5),
        "GX": (883.7, 5994.0),
        "HI": (136.3, 5701.6),
        "YN": (49.8, 5461.8),
    },
    2017: {  # https://read01.com/2Gd7kRj.html reprint of NBS 2017-08-25 bulletin
        "ZJ": (115.4, 6104.7),
        "AH": (158.4, 6094.1),
        "FJ": (144.6, 6136.7),
        "JX": (1348.7, 5605.5),
        "HB": (375.4, 5795.7),
        "HN": (1383.1, 5845.6),
        "GD": (892.5, 5964.0),
        "GX": (865.1, 5797.5),
        "HI": (130.1, 5730.3),
        "YN": (50.1, 5548.9),
    },
    2019: {  # https://www.stats.gov.cn/sj/zxfb/202302/t20230203_1900433.html
        "ZJ": (98.8, 6104.3),
        "AH": (164.6, 6136.8),
        "FJ": (97.4, 6326.4),
        "JX": (1095.9, 5714.1),
        "HB": (142.5, 5911.0),
        "HN": (1094.6, 6042.4),
        "GD": (834.7, 5850.0),
        "GX": (767.9, 5893.9),
        "HI": (114.4, 6049.0),
        "YN": (39.3, 5521.6),
    },
    2020: {  # https://www.stats.gov.cn/english/PressRelease/202008/t20200820_1784289.html
        "ZJ": (101.2, 6202.0),
        "AH": (170.3, 5355.3),
        "FJ": (97.7, 6362.4),
        "JX": (1217.5, 5311.6),
        "HB": (122.4, 5578.6),
        "HN": (1225.7, 5863.7),
        "GD": (869.1, 5966.0),
        "GX": (805.2, 5921.0),
        "HI": (110.0, 5989.5),
        "YN": (31.5, 5806.9),
    },
    2021: {  # NBS 10-province survey as reported by chyxx.com/industry/202108/971862.html
        "ZJ": (102.0, 6160.9),
        "AH": (170.9, 5804.5),
        "FJ": (97.0, 6378.0),
        "JX": (1218.8, 5521.8),
        "HB": (120.2, 5937.8),
        "HN": (1219.6, 6098.7),
        "GD": (858.6, 6105.0),
        "GX": (807.5, 5944.0),
        "HI": (111.3, 6123.3),
        "YN": (28.2, 6119.8),
    },
    2022: {  # https://www.stats.gov.cn/english/PressRelease/202208/t20220826_1887653.html
        "ZJ": (113.1, 6206.5),
        "AH": (171.4, 5898.5),
        "FJ": (97.4, 6326.9),
        "JX": (1220.1, 5550.2),
        "HB": (126.3, 5991.8),
        "HN": (1212.8, 6112.5),
        "GD": (864.2, 6018.0),
        "GX": (810.7, 5926.7),
        "HI": (112.3, 6063.4),
        "YN": (26.8, 6178.5),
    },
    2024: {  # https://www.stats.gov.cn/sj/zxfb/202408/t20240823_1956083.html
        "ZJ": (125.8, 6230.2),
        "AH": (169.9, 5882.4),
        "FJ": (95.3, 6334.1),
        "JX": (1206.8, 5607.0),
        "HB": (128.1, 6087.5),
        "HN": (1210.9, 6052.5),
        "GD": (869.7, 6067.5),
        "GX": (814.9, 5908.7),
        "HI": (112.5, 6124.1),
        "YN": (20.9, 6279.9),
    },
}


def _seed_frame() -> pd.DataFrame:
    rows = []
    for year, recs in NBS_SEED.items():
        for code, (area, yld) in recs.items():
            rows.append(
                {
                    "iso3": "CHN",
                    "unit_id": f"CHN-{code}",
                    "state_code": code,
                    "crop": "rice",
                    "year": year,
                    "season": "early",
                    "yield_tha": yld / 1000.0,
                    "harvested_ha": area * 1000.0,
                    "source": "nbs_bulletin_transcribed",
                }
            )
    return pd.DataFrame(rows)


SKIP_NAMES = {
    "national",
    "national total",
    "beijing",
    "tianjin",
    "hebei",
    "shanxi",
    "inner mongolia",
    "liaoning",
    "jilin",
    "heilongjiang",
    "shanghai",
    "jiangsu",
    "shandong",
    "henan",
    "chongqing",
    "sichuan",
    "guizhou",
    "tibet",
    "xizang",
    "shaanxi",
    "gansu",
    "qinghai",
    "ningxia",
    "xinjiang",
    "全国总计",
    "全国",
}

UA = "Mozilla/5.0 (compatible; NatureFood/1.0; +https://github.com/)"


def _norm_name(name: str) -> str:
    text = str(name).replace("\u3000", " ")
    text = re.sub(r"\s+", " ", text).strip().lower()
    for junk in (
        "special administrative region",
        "autonomous region",
        "municipality",
        "autonomous",
        "province",
        "省",
        "自治区",
        "壮族",
    ):
        text = text.replace(junk, " ")
    text = re.sub(r"[^a-z\u4e00-\u9fff]", "", text)
    aliases = {
        "zhejiang": "zhejiang",
        "anhui": "anhui",
        "fujian": "fujian",
        "jiangxi": "jiangxi",
        "hubei": "hubei",
        "hunan": "hunan",
        "guangdong": "guangdong",
        "guangzhou": "guangdong",  # geoBoundaries ADM1 mislabel
        "guangxi": "guangxi",
        "guangxizhuang": "guangxi",
        "hainan": "hainan",
        "yunnan": "yunnan",
        "浙江": "浙江",
        "安徽": "安徽",
        "福建": "福建",
        "江西": "江西",
        "湖北": "湖北",
        "湖南": "湖南",
        "广东": "广东",
        "广西": "广西",
        "海南": "海南",
        "云南": "云南",
    }
    return aliases.get(text, text)


def _code_for(name: str) -> str | None:
    key = _norm_name(name)
    if key in SKIP_NAMES or key in { _norm_name(s) for s in SKIP_NAMES }:
        return None
    return PROVINCES.get(key)


def _fetch(url: str, timeout: int = 40) -> str:
    import ssl

    ctx = ssl._create_unverified_context()
    req = Request(url, headers={"User-Agent": UA})
    with urlopen(req, timeout=timeout, context=ctx) as resp:
        raw = resp.read()
    for enc in ("utf-8", "gb18030", "gbk"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _parse_bulletin_html(html: str, year: int) -> pd.DataFrame:
    frames = []
    try:
        tables = pd.read_html(html)
    except ValueError:
        tables = []
    for tbl in tables:
        tbl = tbl.copy()
        tbl.columns = [_flatten_col(c) for c in tbl.columns]
        name_col = _find_col(tbl, ("region", "regions", "地区", "province"))
        if name_col is None:
            # first column is usually the region name
            name_col = tbl.columns[0]
        yld_col = _find_col(tbl, ("yield per unit", "单位面积产量", "kg/ha", "公斤/公顷", "yield"))
        area_col = _find_col(tbl, ("sown area", "播种面积", "thousand hectares", "千公顷"))
        prod_col = _find_col(tbl, ("total output", "总产量", "10,000 tons", "万吨"))
        if yld_col is None and (area_col is None or prod_col is None):
            continue
        rows = []
        for _, rec in tbl.iterrows():
            code = _code_for(rec.get(name_col, ""))
            if not code:
                continue
            yld = _to_float(rec.get(yld_col)) if yld_col else np.nan
            area = _to_float(rec.get(area_col)) if area_col else np.nan
            prod = _to_float(rec.get(prod_col)) if prod_col else np.nan
            if not np.isfinite(yld) and np.isfinite(area) and np.isfinite(prod) and area > 0:
                yld = (prod * 10_000.0) / (area * 1_000.0) * 1000.0  # kg/ha
            if not np.isfinite(yld) or yld < 2000 or yld > 12000:
                continue
            rows.append(
                {
                    "iso3": "CHN",
                    "unit_id": f"CHN-{code}",
                    "state_code": code,
                    "crop": "rice",
                    "year": year,
                    "season": "early",
                    "yield_tha": yld / 1000.0,
                    "harvested_ha": (area * 1000.0) if np.isfinite(area) else np.nan,
                    "source": "nbs_bulletin",
                }
            )
        if rows:
            frames.append(pd.DataFrame(rows))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).drop_duplicates(["unit_id", "year"])


def _flatten_col(col) -> str:
    if isinstance(col, tuple):
        return " ".join(str(x) for x in col if str(x) != "nan").strip().lower()
    return str(col).strip().lower()


def _find_col(tbl: pd.DataFrame, needles: tuple[str, ...]) -> str | None:
    for col in tbl.columns:
        text = str(col).lower()
        if any(n.lower() in text for n in needles):
            return col
    return None


def _to_float(val) -> float:
    if val is None:
        return np.nan
    text = str(val).replace(",", "").replace(" ", "").strip()
    if not text or text in {"nan", "-", "—", "–"}:
        return np.nan
    try:
        return float(text)
    except ValueError:
        return np.nan


def _wayback_candidates(year: int) -> list[str]:
    query = (
        "https://web.archive.org/cdx/search/cdx?url=stats.gov.cn/*"
        f"&matchType=prefix&filter=original:.*{year}08.*"
        "&filter=original:.*(Rice|rice|zaodao|早稻).*"
        "&output=json&fl=original,timestamp&collapse=urlkey&limit=30"
    )
    try:
        raw = _fetch(query, timeout=30)
        import json

        rows = json.loads(raw)
    except Exception:
        return []
    urls = []
    for row in rows[1:]:
        original = row[0]
        ts = row[1]
        if "rice" not in original.lower():
            continue
        urls.append(f"https://web.archive.org/web/{ts}/{original}")
        urls.append(original)
    return urls


GOTOHUI_YIELD = {
    "ZJ": "https://www.gotohui.com/ndata/show-1322684",
    "AH": "https://www.gotohui.com/ndata/show-1343383",
    "JX": "https://www.gotohui.com/ndata/show-1693233",
    "HN": "https://www.gotohui.com/ndata/show-1654235",
    "GD": "https://www.gotohui.com/ndata/show-1327496",
}


def _gotohui_yields(year_start: int, year_end: int) -> pd.DataFrame:
    """NBS rural-statistics series republished by Gotohui (kg/ha)."""

    rows = []
    for code, url in GOTOHUI_YIELD.items():
        try:
            html = _fetch(url, timeout=30)
        except Exception as exc:
            print(f"SKIP gotohui {code}: {exc}", flush=True)
            continue
        try:
            tables = pd.read_html(html)
        except ValueError:
            continue
        for tbl in tables:
            if len(tbl.columns) < 2:
                continue
            year_col, val_col = tbl.columns[0], tbl.columns[-1]
            for _, rec in tbl.iterrows():
                year = _to_float(rec[year_col])
                yld = _to_float(rec[val_col])
                if not (year_start <= year <= year_end) or not (2000 <= yld <= 12000):
                    continue
                rows.append(
                    {
                        "iso3": "CHN",
                        "unit_id": f"CHN-{code}",
                        "state_code": code,
                        "crop": "rice",
                        "year": int(year),
                        "season": "early",
                        "yield_tha": yld / 1000.0,
                        "harvested_ha": np.nan,
                        "source": "nbs_rural_gotohui",
                    }
                )
        print(f"gotohui {code} rows so far {sum(1 for r in rows if r['state_code']==code)}", flush=True)
    return pd.DataFrame(rows)


def _stats_gov_api(year_start: int, year_end: int) -> pd.DataFrame:
    """Try the public NBS easyquery endpoint for 早稻单位面积产量."""

    # A0D0D03 / A0D0C03 variants used across yearbook vintages.
    codes = ("A0D0D03", "A0D0D04", "A0D0D05", "A0D0103", "A0D0O03")
    rows = []
    for zb in codes:
        url = (
            "https://data.stats.gov.cn/easyquery.htm?m=QueryData&dbcode=fsnd"
            "&rowcode=reg&colcode=sj"
            f'&wds=[{{"wdcode":"zb","valuecode":"{zb}"}}]'
            f'&dfwds=[{{"wdcode":"sj","valuecode":"LAST20"}}]&k1=1'
        )
        try:
            raw = _fetch(url, timeout=25)
        except Exception:
            continue
        if "datanodes" not in raw and "data" not in raw:
            continue
        import json

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        nodes = payload.get("returndata", {}).get("datanodes") or payload.get("datanodes") or []
        wdnodes = payload.get("returndata", {}).get("wdnodes") or []
        reg_map = {}
        year_map = {}
        for wd in wdnodes:
            if wd.get("wdcode") == "reg":
                for n in wd.get("nodes", []):
                    reg_map[n.get("code")] = n.get("cname") or n.get("name")
            if wd.get("wdcode") == "sj":
                for n in wd.get("nodes", []):
                    year_map[n.get("code")] = n.get("cname") or n.get("name")
        for node in nodes:
            wds = {w.get("wdcode"): w.get("valuecode") for w in node.get("wds", [])}
            val = node.get("data", {}).get("data")
            if val is None:
                continue
            code = _code_for(reg_map.get(wds.get("reg"), ""))
            year_txt = str(year_map.get(wds.get("sj"), wds.get("sj", "")))[:4]
            if not code or not year_txt.isdigit():
                continue
            year = int(year_txt)
            if year < year_start or year > year_end:
                continue
            yld = float(val)
            if yld < 20:  # already t/ha in some vintages
                yld_tha = yld
            elif yld < 200:  # kg/mu
                yld_tha = yld * 15.0 / 1000.0
            else:
                yld_tha = yld / 1000.0
            if not (2.0 <= yld_tha <= 12.0):
                continue
            rows.append(
                {
                    "iso3": "CHN",
                    "unit_id": f"CHN-{code}",
                    "state_code": code,
                    "crop": "rice",
                    "year": year,
                    "season": "early",
                    "yield_tha": yld_tha,
                    "harvested_ha": np.nan,
                    "source": f"nbs_api_{zb}",
                }
            )
        if rows:
            break
    return pd.DataFrame(rows)


def fetch_nbs_early_rice(root: Path, year_start: int = 2010, year_end: int = 2024) -> pd.DataFrame:
    dest = root / "data_raw" / "china_nbs"
    dest.mkdir(parents=True, exist_ok=True)
    frames = []
    api = _stats_gov_api(year_start, year_end)
    if len(api):
        frames.append(api)
        print(f"NBS API rows={len(api)} years={sorted(api['year'].unique())}", flush=True)
    gh = _gotohui_yields(year_start, year_end)
    if len(gh):
        frames.append(gh)
        print(f"gotohui rows={len(gh)} years={sorted(gh['year'].unique())}", flush=True)

    for year in range(year_start, year_end + 1):
        cache = dest / f"early_rice_{year}.html"
        html = cache.read_text(encoding="utf-8") if cache.exists() and cache.stat().st_size > 500 else ""
        urls = list(BULLETIN_URLS.get(year, []))
        urls.extend(_wayback_candidates(year))
        if not html:
            for url in urls:
                try:
                    html = _fetch(url)
                except Exception as exc:
                    print(f"SKIP bulletin {year} {url}: {exc}", flush=True)
                    continue
                if "early rice" in html.lower() or "早稻" in html:
                    cache.write_text(html, encoding="utf-8")
                    print(f"saved bulletin {year} {url}", flush=True)
                    break
                html = ""
        if not html:
            continue
        parsed = _parse_bulletin_html(html, year)
        if len(parsed):
            frames.append(parsed)
            print(f"parsed bulletin {year} units={parsed['unit_id'].nunique()}", flush=True)
        else:
            print(f"empty parse {year}", flush=True)

    frames.append(_seed_frame())
    frames.append(_gotohui_backfill_frame())
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    rank = {
        "nbs_bulletin": 3,
        "nbs_bulletin_transcribed": 2,
        "nbs_rural_gotohui": 1,
    }
    out["_rank"] = out["source"].map(lambda s: rank.get(str(s), 0 if not str(s).startswith("nbs_api") else 1))
    out = out.sort_values(["_rank", "unit_id", "year"])
    out = out.drop_duplicates(["unit_id", "year"], keep="last").drop(columns=["_rank"])
    return out.reset_index(drop=True)


def parse_china_early_rice(root: Path, year_start: int = 2010, year_end: int = 2024) -> pd.DataFrame:
    cache = root / "data_interim" / "china_early_rice.csv"
    if cache.exists() and cache.stat().st_size > 200:
        out = pd.read_csv(cache)
        out = out[out["year"].between(year_start, year_end)].copy()
        if len(out) and out.groupby("unit_id")["year"].nunique().min() >= 7:
            return out
    out = fetch_nbs_early_rice(root, year_start, year_end)
    if out.empty:
        return out
    counts = out.groupby("unit_id")["year"].nunique()
    keep = set(counts[counts >= 6].index)
    out = out[out["unit_id"].isin(keep)].copy()
    cache.parent.mkdir(parents=True, exist_ok=True)
    if len(out):
        out.to_csv(cache, index=False)
    return out


def load_china_early_rice(root: Path) -> pd.DataFrame:
    path = root / "data_interim" / "china_early_rice.csv"
    if not path.exists() or path.stat().st_size < 50:
        return pd.DataFrame()
    return pd.read_csv(path)


def _adm1_path(root: Path) -> Path:
    return root / "data_raw" / "china_nbs" / "geoBoundaries-CHN-ADM1.geojson"


def download_china_adm1(root: Path) -> Path | None:
    dest = _adm1_path(root)
    if dest.exists() and dest.stat().st_size > 50_000:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    urls = [
        "https://github.com/wmgeolab/geoBoundaries/raw/main/releaseData/gbOpen/CHN/ADM1/geoBoundaries-CHN-ADM1.geojson",
        "https://raw.githubusercontent.com/wmgeolab/geoBoundaries/main/releaseData/gbOpen/CHN/ADM1/geoBoundaries-CHN-ADM1.geojson",
    ]
    for url in urls:
        try:
            raw = _fetch(url, timeout=90)
        except Exception as exc:
            print(f"SKIP ADM1 {url}: {exc}", flush=True)
            continue
        if len(raw) < 50_000 or "Feature" not in raw[:2000]:
            continue
        dest.write_text(raw, encoding="utf-8")
        print(f"saved China ADM1 {dest} bytes={dest.stat().st_size}", flush=True)
        return dest
    return dest if dest.exists() else None


def _province_polygons(root: Path) -> dict[str, object]:
    import json
    from shapely.geometry import shape

    path = download_china_adm1(root)
    if path is None or not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for feat in raw.get("features", []):
        props = feat.get("properties") or {}
        name = props.get("shapeName") or props.get("shapeNameEN") or props.get("name") or ""
        code = _code_for(name)
        if not code:
            continue
        out[code] = shape(feat["geometry"])
    return out


def china_early_rice_points(root: Path, n_per: int = 8, min_ha: float = 20.0) -> pd.DataFrame:
    import xarray as xr

    path = nc_path(root, "rice")
    if not path.exists():
        return pd.DataFrame()
    polys = _province_polygons(root)
    if not polys:
        return pd.DataFrame()
    ds = xr.open_dataset(path)
    ha = ds["harvarea"]
    rows = []
    for code, geom in polys.items():
        minx, miny, maxx, maxy = geom.bounds
        if float(ha.lat[0]) > float(ha.lat[-1]):
            sub = ha.sel(lon=slice(minx, maxx), lat=slice(maxy, miny))
        else:
            sub = ha.sel(lon=slice(minx, maxx), lat=slice(miny, maxy))
        vals = np.asarray(sub.values, dtype=float)
        lats = sub["lat"].values
        lons = sub["lon"].values
        if vals.size == 0:
            continue
        vals[~np.isfinite(vals)] = 0
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
        k = min(n_per, int((flat >= min_ha).sum()) or int((flat > 0).sum()) or 1)
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
                    "iso3": "CHN",
                    "unit_id": f"CHN-{code}",
                    "state_code": code,
                    "crop": "rice",
                    "lon": float(lons[c]),
                    "lat": float(lats[r]),
                    "weight": w,
                    "source": "cropgrids2020",
                }
            )
    ds.close()
    out = pd.DataFrame(rows)
    cache = root / "data_interim" / "china_early_rice_points.csv"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if len(out):
        out.to_csv(cache, index=False)
    return out
