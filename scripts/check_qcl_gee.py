import zipfile
from pathlib import Path

import pandas as pd

p = Path(r"C:\Users\3SOBS\Desktop\NatureFood\data_raw\Production_Crops_Livestock_E_All_Data_(Normalized).zip")
with zipfile.ZipFile(p) as zf:
    name = next(n for n in zf.namelist() if n.lower().endswith(".csv") and "flag" not in n.lower())
    print("csv", name)
    with zf.open(name) as fh:
        df = pd.read_csv(fh, encoding="latin-1", usecols=["Item Code", "Element Code", "Element", "Unit"], low_memory=False)
print(df[df["Item Code"].isin([15, 27, 56, 236])][["Element Code", "Element", "Unit"]].drop_duplicates())

opt = pd.read_csv(Path(r"C:\Users\3SOBS\Desktop\NatureFood\data_raw\gee_ts\USA_optical.csv"))
print("optical finite", int(opt.evi.notna().sum()), "max", opt.evi.max())
print(opt[opt.date.str.contains("-07-")].head(3))
