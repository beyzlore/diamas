"""Gün içi ORB testleri (5 dakikalık ve 1 saatlik barlar).

Ücretsiz veri sınırı: 5 dk barlar yalnızca son ~60 işlem günü, 1 saatlik barlar son ~2 yıl.
Dönem ayrımı (gün sırasına göre): ilk %50 eğitim, sonraki %25 doğrulama, son %25 kilitli.
"""
import itertools
import pickle
import sys
from pathlib import Path

sys.path.insert(0, "src")
import numpy as np
import pandas as pd

import backtest as B
import data as D
import strategies_intraday as I

daily = pd.read_parquet("data/daily.parquet")
out = {}
summary = {}
TITLES = {"hisse": "Stocks in Play (en likit 60 hisse, günde en fazla 10)", "QQQ": "Yalnızca QQQ", "SPY": "Yalnızca SPY"}


def split_days(days):
    days = pd.DatetimeIndex(sorted(days))
    a, b = int(len(days) * 0.5), int(len(days) * 0.75)
    return days, (days[0], days[a - 1]), (days[a], days[b - 1]), (days[b], days[-1])


for interval, bar_min, or_list, stop_list in [
    ("5m", 5, [5, 15, 30], [0.05, 0.10, 0.25, None]),
    ("1h", 60, [60], [None, 0.5, 1.0]),
]:
    path = Path(f"data/intraday_{interval}.parquet")
    if not path.exists():
        print("veri yok:", interval)
        continue
    bars = pd.read_parquet(path)
    stocks = sorted(set(bars["ticker"]) - set(D.ETFS))
    for univ_name, tick, relvols, top_n in [("hisse", stocks, [0.0, 1.0, 2.0], 10),
                                            ("QQQ", ["QQQ"], [0.0], 1), ("SPY", ["SPY"], [0.0], 1)]:
        res = []
        for or_min in or_list:
            o, b = I.prepare(bars[bars["ticker"].isin(tick)], daily[daily["ticker"].isin(tick)], bar_min, or_min)
            days, TR, VA, LO = split_days(o["date"].unique())
            for stop, rv in itertools.product(stop_list, relvols):
                t = I.simulate_orb(o, b, or_min, stop, rv, top_n)
                d = I.daily_returns(t, days, top_n)
                res.append(dict(params=dict(or_min=or_min, stop_atr=stop, min_relvol=rv), trades=t, daily=d,
                                train=B.stats(d, t, *TR), valid=B.stats(d, t, *VA),
                                periods=(TR, VA, LO), k=top_n))
        ok = [r for r in res if r["train"]["işlem_sayısı"] >= 20]
        best = max(ok, key=lambda r: np.nan_to_num(r["train"]["sharpe"], nan=-9)) if ok else None
        key = f"orb_{interval}_{univ_name}"
        out[key] = res
        TR, VA, LO = res[0]["periods"]
        print(f"\n{key}: gün={len(days)} eğitim {TR[0].date()}..{TR[1].date()} doğrulama {VA[0].date()}..{VA[1].date()}")
        for r in res:
            tr_, va_ = r["train"], r["valid"]
            print(f"  {str(r['params']):55s} eğitim: n={tr_['işlem_sayısı']:4d} SR={tr_['sharpe']:6.2f} "
                  f"beklenen={tr_['beklenen_getiri']:.4%} | doğrulama: n={va_['işlem_sayısı']:4d} "
                  f"SR={va_['sharpe']:6.2f} beklenen={va_['beklenen_getiri']:.4%}")
        if best:
            print(f"  >> seçilen: {best['params']}  doğrulama SR={best['valid']['sharpe']:.2f}")
            v = best["valid"]
            ci = B.bootstrap_ci(best["trades"][(best["trades"]["date"] >= VA[0]) & (best["trades"]["date"] <= VA[1])]["net"].values) \
                if not best["trades"].empty else (np.nan, np.nan)
            enough = len(days) >= 250
            verdict = ("GEÇTİ" if enough and v["sharpe"] > 0.5 and np.isfinite(ci[0]) and ci[0] > 0
                       else ("ŞÜPHELİ" if v["sharpe"] > 0.5 else "BAŞARISIZ"))
            pr = best["params"]
            summary[key] = dict(title=f"ORB {interval} · {TITLES[univ_name]}", verdict=verdict, days=len(days),
                                params_txt=f"ilk {pr['or_min']} dk, stop {'aralığın karşı ucu' if pr['stop_atr'] is None else str(pr['stop_atr']) + '×ATR'}, "
                                           f"göreli hacim ≥ {pr['min_relvol']}",
                                train=best["train"], valid=v, locked=B.stats(best["daily"], best["trades"], *LO),
                                ci=ci, frac_pos=float(np.mean([r["valid"]["sharpe"] > 0 for r in res])))

with open("results/intraday_summary.pkl", "wb") as f:
    pickle.dump(summary, f)
