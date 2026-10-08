"""Araştırma düzeneği: parametre taraması, eğitim/doğrulama/kilitli test ayrımı, ileriye
yürüyen (walk-forward) seçim, piyasa rejimleri, maliyet duyarlılığı, aşırı uyum kontrolleri.

Dönemler
  EĞİTİM     2005-01-01 .. 2016-12-31  parametreler yalnızca burada seçilir
  DOĞRULAMA  2017-01-01 .. 2023-12-31  seçilen parametrenin görmediği veri
  KİLİTLİ    2024-01-01 .. son         araştırma bitene kadar BAKILMAZ; en sonda bir kez açılır
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd

import backtest as B

TRAIN = ("2005-01-01", "2016-12-31")
VALID = ("2017-01-01", "2023-12-31")
LOCKED = ("2024-01-01", "2099-12-31")
RESEARCH_END = VALID[1]


@dataclass
class Spec:
    name: str            # kısa kod
    title: str           # Türkçe ad
    rule: str            # kuralın sade anlatımı
    fn: callable         # fn(**params) -> trades
    grid: dict           # parametre ızgarası
    k: object            # eşzamanlı pozisyon sayısı (int veya params->int)
    family: str = ""


def run_spec(p: B.Panel, spec: Spec, cost_mult: float = 1.0) -> list[dict]:
    keys = list(spec.grid)
    out = []
    for vals in itertools.product(*[spec.grid[k] for k in keys]):
        prm = dict(zip(keys, vals))
        tr = spec.fn(**prm)
        k = spec.k(prm) if callable(spec.k) else spec.k
        daily, t = B.simulate(p, tr, k, cost_mult)
        # kilitli dönem araştırma sırasında görünmesin
        out.append(dict(params=prm, k=k, daily=daily, trades=t,
                        train=B.stats(daily, t, *TRAIN), valid=B.stats(daily, t, *VALID)))
    return out


def select(results: list[dict], period=TRAIN, min_trades: int = 30) -> dict | None:
    ok = [r for r in results if (r["train"] if period == TRAIN else B.stats(r["daily"], r["trades"], *period))
          ["işlem_sayısı"] >= min_trades]
    if not ok:
        return None
    return max(ok, key=lambda r: np.nan_to_num(r["train"]["sharpe"], nan=-9))


def walk_forward(results: list[dict], first_year: int = 2011, last_year: int = 2023,
                 lookback: int = 6) -> tuple[pd.Series, list]:
    """Her yıl için: önceki `lookback` yılda en iyi Sharpe'lı parametreyi seç, o yılı onunla oyna."""
    parts, choices = [], []
    for y in range(first_year, last_year + 1):
        a, b = f"{y - lookback}-01-01", f"{y - 1}-12-31"
        best, best_sr = None, -np.inf
        for r in results:
            d = r["daily"].loc[a:b]
            n = ((r["trades"]["date"] >= a) & (r["trades"]["date"] <= b)).sum()
            if n < 10 or d.std() == 0:
                continue
            sr = d.mean() / d.std()
            if sr > best_sr:
                best, best_sr = r, sr
        if best is None:
            parts.append(pd.Series(0.0, index=results[0]["daily"].loc[f"{y}-01-01":f"{y}-12-31"].index))
            choices.append((y, None))
            continue
        parts.append(best["daily"].loc[f"{y}-01-01":f"{y}-12-31"])
        choices.append((y, best["params"]))
    return pd.concat(parts), choices


def regimes(p: B.Panel, vix: pd.Series, daily: pd.Series, trades: pd.DataFrame,
            start=TRAIN[0], end=RESEARCH_END) -> pd.DataFrame:
    """Farklı piyasa koşullarında sonuçlar. Rejim, bir önceki günün kapanışıyla belirlenir."""
    spy = p.c["SPY"]
    bull = (spy > p.ind["sma200"]["SPY"]).shift(1)
    v = vix.reindex(p.dates).ffill().shift(1)
    vq = pd.cut(v, [0, 15, 25, 1000], labels=["VIX<15 (sakin)", "VIX 15-25 (normal)", "VIX>25 (korku)"])
    d = daily.loc[start:end]
    t = trades[(trades["date"] >= start) & (trades["date"] <= end)].copy()
    t["bull"] = bull.reindex(t["date"]).values
    t["vq"] = vq.reindex(t["date"]).values
    rows = []

    def add(label, mask_days, mask_tr):
        dd = d[mask_days.reindex(d.index).fillna(False).astype(bool)]
        tt = t[mask_tr]
        rows.append({"koşul": label, "gün": len(dd),
                     "yıllık_getiri_katkısı": dd.mean() * 252 if len(dd) else np.nan,
                     "sharpe": dd.mean() / dd.std() * np.sqrt(252) if len(dd) > 2 and dd.std() > 0 else np.nan,
                     "işlem": len(tt),
                     "kazanma_oranı": (tt["net"] > 0).mean() if len(tt) else np.nan,
                     "beklenen_getiri": tt["net"].mean() if len(tt) else np.nan})

    add("Boğa (SPY > 200g ort.)", bull == True, t["bull"] == True)       # noqa: E712
    add("Ayı (SPY < 200g ort.)", bull == False, t["bull"] == False)      # noqa: E712
    for lab in vq.cat.categories:
        add(lab, vq == lab, t["vq"] == lab)
    for lab, a, b in [("2008 krizi", "2007-10-01", "2009-03-31"), ("2020 Covid çöküşü", "2020-02-15", "2020-04-30"),
                      ("2022 ayı piyasası", "2022-01-01", "2022-10-31")]:
        m = pd.Series(False, index=p.dates)
        m.loc[a:b] = True
        add(lab, m, (t["date"] >= a) & (t["date"] <= b))
    return pd.DataFrame(rows)


def yearly(daily: pd.Series, start=TRAIN[0], end=RESEARCH_END) -> pd.Series:
    d = daily.loc[start:end]
    return d.groupby(d.index.year).apply(lambda x: (1 + x).prod() - 1)


def worst_window(daily: pd.Series, days: int = 252, start=TRAIN[0], end=RESEARCH_END):
    d = daily.loc[start:end]
    roll = (1 + d).rolling(days).apply(np.prod, raw=True) - 1
    if roll.dropna().empty:
        return np.nan, None
    e = roll.idxmin()
    return float(roll.min()), (d.index[max(d.index.get_loc(e) - days + 1, 0)], e)
