import zipfile
from pathlib import Path

import pandas as pd

p = Path(r"C:\Users\3SOBS\Desktop\NatureFood\data_raw\Trade_DetailedTradeMatrix_E_All_Data_(Normalized).zip")
with zipfile.ZipFile(p) as zf:
    name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
    print("csv", name, "files", zf.namelist()[:5])
    with zf.open(name) as fh:
        df = pd.read_csv(fh, encoding="latin-1", nrows=3)
print(df.columns.tolist())
print(df.head(2).to_string())
