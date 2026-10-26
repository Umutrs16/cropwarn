from pathlib import Path

from cropwarn.data.observational import parse_trade_zip

root = Path(__file__).resolve().parents[1]
src = root / "data_raw" / "Trade_DetailedTradeMatrix_E_All_Data_(Normalized).zip"
out = root / "data_interim" / "obs_trade_network.csv"
out.parent.mkdir(parents=True, exist_ok=True)
df = parse_trade_zip(src, year_start=2000)
df.to_csv(out, index=False)
print("trade rows", len(df), "years", df.year.min() if len(df) else None, df.year.max() if len(df) else None)
print(df.head(3).to_string())
