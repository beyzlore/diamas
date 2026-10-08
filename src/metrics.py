"""Hisse kalitesi ölçümleri: likidite, işlem hacmi, oynaklık, spread tahmini."""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_indicators(d: pd.DataFrame) -> pd.DataFrame:
    """Tek bir hissenin günlük verisine (tarihe göre sıralı) göstergeler ekler.

    Tüm göstergeler yalnızca o güne KADAR olan veriyi kullanır (geleceğe bakmaz).
    """
    d = d.sort_values("date").copy()
    c, h, l, o = d["close"], d["high"], d["low"], d["open"]
    prev_c = c.shift(1)
    d["ret"] = c.pct_change()
    d["dollar_vol"] = c * d["volume"]
    d["adv20"] = d["dollar_vol"].rolling(20).mean()          # 20 günlük ort. dolar hacmi
    d["vol20"] = d["volume"].rolling(20).mean()
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    d["atr14"] = tr.rolling(14).mean()
    d["atr_pct"] = d["atr14"] / c
    d["rv20"] = d["ret"].rolling(20).std() * np.sqrt(252)    # yıllık gerçekleşen oynaklık
    d["sma5"] = c.rolling(5).mean()
    d["sma50"] = c.rolling(50).mean()
    d["sma200"] = c.rolling(200).mean()
    d["gap"] = o / prev_c - 1                                 # açılış boşluğu
    d["rsi2"] = _rsi(c, 2)
    d["spread_est"] = _abdi_ranaldo(c, h, l)
    return d


def _rsi(c: pd.Series, n: int) -> pd.Series:
    delta = c.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _abdi_ranaldo(c: pd.Series, h: pd.Series, l: pd.Series, window: int = 60) -> pd.Series:
    """Abdi & Ranaldo (2017) kapanış-yüksek-düşük spread tahmincisi (oransal, ör. 0.0005 = 5 bps).

    Gerçek alış-satış kotasyonu verisi ücretsiz olmadığı için günlük fiyatlardan tahmin edilir.
    """
    lc, eta = np.log(c), (np.log(h) + np.log(l)) / 2
    prod = (lc - eta) * (lc - eta.shift(-1))
    # shift(-1) bir sonraki günü kullandığı için sonucu bir gün geri kaydırıyoruz -> geleceğe bakmaz
    s2 = 4 * prod.shift(1).rolling(window, min_periods=window // 2).mean()
    return np.sqrt(s2.clip(lower=0))


def universe_table(daily: pd.DataFrame, asof: pd.Timestamp | None = None) -> pd.DataFrame:
    """Her hisse için son 1 yılın ortalama likidite/oynaklık/spread ölçümleri."""
    asof = asof or daily["date"].max()
    rows = []
    for t, d in daily.groupby("ticker"):
        d = d[d["date"] <= asof]
        if len(d) < 260:
            continue
        d = add_indicators(d).tail(252)
        rows.append({
            "ticker": t,
            "price": d["close"].iloc[-1],
            "adv_musd": d["dollar_vol"].mean() / 1e6,
            "avg_volume_m": d["volume"].mean() / 1e6,
            "rv_annual": d["ret"].std() * np.sqrt(252),
            "atr_pct": d["atr_pct"].mean(),
            "spread_ar_bps": d["spread_est"].mean() * 1e4,
            "gap_abs_bps": d["gap"].abs().mean() * 1e4,
        })
    u = pd.DataFrame(rows).set_index("ticker")
    u["liq_rank"] = u["adv_musd"].rank(ascending=False)
    u["spread_bps"] = model_spread_bps(u["adv_musd"])
    return u.sort_values("adv_musd", ascending=False)


def model_spread_bps(adv_musd):
    """Likiditeye dayalı tipik alış-satış spread'i (bps).

    Abdi-Ranaldo tahmincisi çok likit hisselerde oynaklık yüzünden şişiyor (SPY için ~20 bps
    veriyor; gerçekte ~0.3 bps). Bu yüzden maliyet için ABD büyük hisselerinin bilinen tipik
    spread düzeylerine kalibre edilmiş, temkinli bir formül kullanıyoruz:
    ADV 46 milyar $ -> 0.3 bps, 10 milyar $ -> 0.6, 400 milyon $ -> 3, 40 milyon $ -> 9.5.
    """
    return np.clip(60 / np.sqrt(np.maximum(adv_musd, 1e-6)), 0.3, 30)
