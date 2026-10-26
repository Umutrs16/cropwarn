from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
url = "https://bulks-faostat.fao.org/production/Trade_DetailedTradeMatrix_E_All_Data_(Normalized).zip"
dest = ROOT / "data_raw" / "Trade_DetailedTradeMatrix_E_All_Data_(Normalized).zip"
dest.parent.mkdir(parents=True, exist_ok=True)
if dest.exists() and dest.stat().st_size > 1_000_000:
    print("already", dest.stat().st_size)
else:
    print("downloading TM")
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        with dest.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    print("wrote", dest.stat().st_size)
