"""Ani fiyat hareketine yol açan olayların ölçümü: bilanço günleri, FOMC günleri, büyük gap'ler.
Soru: bu günlerde fiyat normal günlere göre ne kadar daha fazla oynuyor?"""
import pickle
import sys

sys.path.insert(0, "src")
import numpy as np
import pandas as pd

import backtest as B
import data as D

df = pd.read_parquet("data/daily.parquet")
p = B.load_panel(df)
earn = pd.read_parquet("data/earnings.parquet")
fomc = D.fomc_dates()
end = "2023-12-31"   # kilitli dönem dışı

ret = p.c.pct_change().loc[:end]
absret = ret.abs()
stocks = [t for t in p.tickers if t not in D.ETFS]

# bilanço tepki günü: kapanış sonrası açıklama -> ertesi gün, açılış öncesi -> aynı gün
e = earn.copy()
hr = e["ts"].dt.hour + e["ts"].dt.minute / 60
e = e[(hr >= 16) | ((hr > 0) & (hr < 9.5))]
pos = p.dates.searchsorted(e["ts"].dt.normalize().values)
pos = np.minimum(pos, len(p.dates) - 1)
on_day = p.dates[pos] == e["ts"].dt.normalize().values
react = np.where((hr.loc[e.index].values >= 16) & on_day, pos + 1, pos)
e = e.assign(r=np.minimum(react, len(p.dates) - 1))
e["rdate"] = p.dates[e["r"].values]
e = e[e["rdate"] <= end]
mask = pd.DataFrame(False, index=absret.index, columns=absret.columns)
col = {t: j for j, t in enumerate(absret.columns)}
for row in e.itertuples():
    j = col.get(row.ticker)
    if j is not None and row.rdate in mask.index:
        mask.iat[mask.index.get_loc(row.rdate), j] = True
a = absret[stocks]
m = mask[stocks]
earn_days = a[m].stack().dropna()
norm_days = a[~m].stack().dropna()

# bilanço sürprizi ile tepki yönü ilişkisi
e2 = e.dropna(subset=["surprise"]).copy()
e2["react_ret"] = [ret.iat[ret.index.get_loc(r.rdate), col[r.ticker]] if r.ticker in col and r.rdate in ret.index else np.nan
                   for r in e2.itertuples()]
e2 = e2.dropna(subset=["react_ret"])
e2["sbin"] = pd.cut(e2["surprise"], [-1e9, -10, -2, 2, 10, 1e9],
                    labels=["< -%10", "-%10..-%2", "-%2..+%2", "+%2..+%10", "> +%10"])
surprise_tbl = e2.groupby("sbin", observed=True)["react_ret"].agg(["count", "mean", "median",
                                                                   lambda x: (x > 0).mean()])
surprise_tbl.columns = ["adet", "ort_tepki", "medyan_tepki", "yükselme_oranı"]

# FOMC günleri (SPY)
spy_r = ret["SPY"].dropna()
spy = spy_r.abs()
fset = set(pd.to_datetime(fomc))
is_f = spy.index.isin(list(fset))
out = {
    "earn_n": int(len(earn_days)), "earn_abs_mean": float(earn_days.mean()), "norm_abs_mean": float(norm_days.mean()),
    "earn_big_share": float((earn_days > 0.05).mean()), "norm_big_share": float((norm_days > 0.05).mean()),
    "fomc_n": int(is_f.sum()), "fomc_abs_mean": float(spy[is_f].mean()), "spy_abs_mean": float(spy[~is_f].mean()),
    "fomc_mean_ret": float(spy_r[is_f].mean()), "spy_mean_ret": float(spy_r[~is_f].mean()),
    "surprise_tbl": surprise_tbl,
    "earn_tickers": int(earn["ticker"].nunique()), "earn_rows": int(len(earn)),
    "earn_first": str(earn["ts"].min().date()),
}
# en çok hareket eden günlerin ne kadarı bilanço günü?
top = a.stack().dropna()
thr = top.quantile(0.99)
big = a[a > thr]
out["top1pct_share_earn"] = float(m[a > thr].stack().mean())
for k, v in out.items():
    if not isinstance(v, pd.DataFrame):
        print(k, v)
print(surprise_tbl)
with open("results/events.pkl", "wb") as f:
    pickle.dump(out, f)
