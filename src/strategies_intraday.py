"""Gün içi stratejiler: Açılış Aralığı Kırılımı (Opening Range Breakout, ORB).

Kaynak kural seti: Zarattini, Barbon & Aziz (2024) "A Profitable Day Trading Strategy For The
U.S. Equity Market" -> 'Stocks in Play' + 5 dakikalık ORB.

Kural:
  1) Açılış aralığı (OR): seansın ilk `or_min` dakikası. Bu süredeki en yüksek/en düşük fiyat.
  2) Yön: OR mumu yükselişle kapandıysa yalnızca AL, düşüşle kapandıysa yalnızca AÇIĞA SAT.
  3) Giriş: Fiyat OR zirvesini (alışta) / dibini (satışta) kırınca stop emriyle gir.
  4) Zarar durdur: giriş fiyatından 14 günlük ATR'nin `stop_atr` katı kadar uzakta
     (stop_atr=None ise OR'nin karşı ucu).
  5) Çıkış: stop olmazsa seans kapanışında.
  6) 'Stocks in Play' filtresi: ilk OR süresindeki hacim, son 14 günün aynı süredeki
     ortalamasının en az `min_relvol` katı olmalı; en yüksek göreli hacimli `top_n` hisse seçilir.

Geleceğe bakma önlemleri: ATR ve göreli hacim ortalaması bir önceki güne kadar olan veriyle
hesaplanır; aynı bar içinde hem giriş hem stop seviyesi görülürse en kötü durum (stop) varsayılır.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from metrics import model_spread_bps
from backtest import SLIPPAGE_BPS, STOP_SLIPPAGE_BPS, COMMISSION_PER_SHARE


def prepare(bars: pd.DataFrame, daily: pd.DataFrame, bar_min: int, or_min: int) -> pd.DataFrame:
    """Her (hisse, gün) için açılış aralığı özetini ve gün içi barları hazırlar."""
    bars = bars.sort_values(["ticker", "ts"]).copy()
    bars["tod"] = bars["ts"].dt.hour * 60 + bars["ts"].dt.minute - 570   # 09:30'dan itibaren dakika
    bars = bars[(bars["tod"] >= 0) & (bars["tod"] < 390)]
    n_or = or_min // bar_min
    g = bars.groupby(["ticker", "date"], sort=False)
    first = bars[bars["tod"] < or_min]
    o = first.groupby(["ticker", "date"]).agg(or_open=("open", "first"), or_close=("close", "last"),
                                              or_high=("high", "max"), or_low=("low", "min"),
                                              or_vol=("volume", "sum"), or_bars=("open", "size"))
    o = o[o["or_bars"] == n_or]
    o["day_close"] = g["close"].last().reindex(o.index)
    # göreli hacim: son 14 günün OR hacmi ortalamasına göre (bugün hariç)
    o = o.reset_index().sort_values(["ticker", "date"])
    o["or_vol_avg14"] = o.groupby("ticker")["or_vol"].transform(lambda s: s.shift(1).rolling(14, min_periods=10).mean())
    o["relvol"] = o["or_vol"] / o["or_vol_avg14"]
    # günlük ATR ve likidite (dünün verisi)
    d = daily.sort_values(["ticker", "date"]).copy()
    pc = d.groupby("ticker")["close"].shift(1)
    tr = np.maximum(d["high"] - d["low"], np.maximum((d["high"] - pc).abs(), (d["low"] - pc).abs()))
    d["atr14"] = tr.groupby(d["ticker"]).transform(lambda s: s.rolling(14).mean().shift(1))
    d["adv20"] = (d["close"] * d["volume"]).groupby(d["ticker"]).transform(lambda s: s.rolling(20).mean().shift(1))
    o = o.merge(d[["ticker", "date", "atr14", "adv20"]], on=["ticker", "date"], how="left")
    o["spread_bps"] = model_spread_bps(o["adv20"].fillna(1e8).values / 1e6)
    o["dir"] = np.sign(o["or_close"] - o["or_open"]).astype(int)
    return o.dropna(subset=["atr14", "relvol"]), bars


def simulate_orb(o: pd.DataFrame, bars: pd.DataFrame, or_min: int, stop_atr: float | None,
                 min_relvol: float, top_n: int, tickers: list[str] | None = None) -> pd.DataFrame:
    """Her işlem günü için seçilen hisselerde ORB işlemlerini simüle eder. İşlem başına net getiri döner."""
    cand = o[(o["dir"] != 0) & (o["relvol"] >= min_relvol)]
    if tickers is not None:
        cand = cand[cand["ticker"].isin(tickers)]
    cand = cand.sort_values(["date", "relvol"], ascending=[True, False]).groupby("date").head(top_n)
    key = bars.set_index(["ticker", "date"]).sort_index()
    rows = []
    for r in cand.itertuples():
        b = key.loc[(r.ticker, r.date)]
        b = b[b["tod"] >= or_min]
        if b.empty:
            continue
        H, L, O, C = b["high"].values, b["low"].values, b["open"].values, b["close"].values
        side = r.dir
        trig = r.or_high if side > 0 else r.or_low
        hit = np.where(H >= trig)[0] if side > 0 else np.where(L <= trig)[0]
        if len(hit) == 0:
            continue
        i = hit[0]
        entry = max(trig, O[i]) if side > 0 else min(trig, O[i])
        if stop_atr is None:
            stop = r.or_low if side > 0 else r.or_high
        else:
            stop = entry - side * stop_atr * r.atr14
        risk = abs(entry - stop)
        if risk <= 0:
            continue
        exit_px, stopped = C[-1], False
        for k in range(i, len(H)):
            # giriş barında stop da görüldüyse en kötü durumu varsay
            if (side > 0 and L[k] <= stop) or (side < 0 and H[k] >= stop):
                exit_px = min(stop, O[k]) if (side > 0 and k > i) else (max(stop, O[k]) if (side < 0 and k > i) else stop)
                stopped = True
                break
        gross = side * (exit_px / entry - 1)
        half = r.spread_bps / 2
        cost = ((half + STOP_SLIPPAGE_BPS) + (half + (STOP_SLIPPAGE_BPS if stopped else SLIPPAGE_BPS))) / 1e4 \
            + 2 * COMMISSION_PER_SHARE / entry
        rows.append(dict(ticker=r.ticker, date=r.date, side=side, entry=entry, exit=exit_px, stopped=stopped,
                         relvol=r.relvol, gross=gross, net=gross - cost, r_mult=(side * (exit_px - entry)) / risk))
    return pd.DataFrame(rows)


def daily_returns(trades: pd.DataFrame, days: pd.DatetimeIndex, k: int) -> pd.Series:
    """Her işlem sermayenin 1/k'sı ile; günlük portföy getirisi."""
    if trades.empty:
        return pd.Series(0.0, index=days)
    s = trades.groupby("date")["net"].sum() / k
    return s.reindex(days, fill_value=0.0)
