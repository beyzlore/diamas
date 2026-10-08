"""Geriye dönük test motoru: maliyetler, portföy simülasyonu ve performans ölçümleri.

Temel kurallar (geleceğe bakmayı önlemek için):
- Sinyal t gününün KAPANIŞINDA bilinen veriyle üretilir; işlem en erken t+1 açılışında yapılır.
  (İstisna: önceden bilinen takvim olayları -FOMC gibi- veya o anda bilinen açılış fiyatına
  dayanan kararlar; bunlar strateji kodunda açıkça belirtilir.)
- Her alış ve satışta spread + kayma + komisyon düşülür.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from metrics import model_spread_bps

SLIPPAGE_BPS = 1.0          # açılış/kapanış emirlerinde ek kayma (spread yarısına ek olarak)
STOP_SLIPPAGE_BPS = 3.0     # stop emirleri piyasa emrine dönüşür -> daha kötü dolum
COMMISSION_PER_SHARE = 0.005  # IBKR benzeri; Alpaca'da 0 ama temkinli olalım


# ---------------------------------------------------------------- panel (geniş tablo) veri
@dataclass
class Panel:
    dates: pd.DatetimeIndex
    tickers: list[str]
    o: pd.DataFrame
    h: pd.DataFrame
    l: pd.DataFrame
    c: pd.DataFrame
    v: pd.DataFrame
    ind: dict = field(default_factory=dict)

    def idx(self, d) -> int:
        return self.dates.get_loc(d)


def load_panel(daily: pd.DataFrame, start: str = "2004-01-01") -> Panel:
    daily = daily[daily["date"] >= start]
    wide = {k: daily.pivot(index="date", columns="ticker", values=k).sort_index()
            for k in ["open", "high", "low", "close", "volume"]}
    p = Panel(wide["close"].index, list(wide["close"].columns), wide["open"], wide["high"],
              wide["low"], wide["close"], wide["volume"])
    c, h, l = p.c, p.h, p.l
    prev_c = c.shift(1)
    tr = np.maximum(h - l, np.maximum((h - prev_c).abs(), (l - prev_c).abs()))
    p.ind["adv20"] = (c * p.v).rolling(20).mean()
    p.ind["atr14"] = tr.rolling(14).mean()
    p.ind["atr_pct"] = p.ind["atr14"] / c
    p.ind["sma5"] = c.rolling(5).mean()
    p.ind["sma200"] = c.rolling(200).mean()
    delta = c.diff()
    up = delta.clip(lower=0).ewm(alpha=0.5, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=0.5, adjust=False).mean()
    p.ind["rsi2"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    p.ind["gap"] = p.o / prev_c - 1
    p.ind["spread_bps"] = pd.DataFrame(model_spread_bps(p.ind["adv20"].values / 1e6),
                                       index=c.index, columns=c.columns)
    return p


def pit_universe(p: Panel, n: int, exclude: set[str]) -> pd.DataFrame:
    """Zamana göre (point-in-time) evren: her ay başında, o ana kadarki 20 günlük dolar hacmine
    göre en likit n hisse. Bugünün büyük hisselerini geçmişe taşımaz (seçim yanlılığını azaltır)."""
    adv = p.ind["adv20"].drop(columns=[t for t in exclude if t in p.c.columns])
    month_end = adv.groupby(adv.index.to_period("M")).tail(1)
    mask = month_end.rank(axis=1, ascending=False) <= n
    # ay sonu sıralaması, takip eden ay boyunca geçerli olur (shift ile bir sonraki güne kaydır)
    mask = mask.reindex(adv.index).ffill().shift(1).fillna(False).astype(bool)
    return mask


# ---------------------------------------------------------------- maliyet
def side_cost(p: Panel, ticker: str, i: int, price: float, stop: bool = False) -> float:
    spread = p.ind["spread_bps"].iat[i, p.tickers.index(ticker)]
    if not np.isfinite(spread):
        spread = 10.0
    slip = STOP_SLIPPAGE_BPS if stop else SLIPPAGE_BPS
    return (spread / 2 + slip) / 1e4 + COMMISSION_PER_SHARE / price


# ---------------------------------------------------------------- portföy simülasyonu
def simulate(p: Panel, trades: pd.DataFrame, k: int, cost_mult: float = 1.0) -> tuple[pd.Series, pd.DataFrame]:
    """trades sütunları: ticker, ei (giriş gün indeksi), et ('open'/'close'), ep (giriş fiyatı),
    xi, xt, xp, side (+1/-1), rank (küçük = öncelikli), stop_exit (bool, ops.).

    En fazla k pozisyon aynı anda; her pozisyon sermayenin 1/k'sı. Günlük piyasa değerine göre
    getiri serisi ve kabul edilen işlemlerin net getirileri döner.
    """
    if trades.empty:
        return pd.Series(0.0, index=p.dates), trades.assign(net=[])
    trades = trades.sort_values(["ei", "rank"]).reset_index(drop=True)
    accepted, active = [], []  # active: (son dolu gün indeksi)
    for row in trades.itertuples():
        start = row.ei if row.et == "open" else row.ei + 1
        active = [e for e in active if e >= start]
        if len(active) >= k:
            continue
        accepted.append(row.Index)
        last = row.xi - 1 if row.xt == "open" else row.xi
        active.append(last)
    t = trades.loc[accepted].copy()

    C = p.c.values
    O = p.o.values
    col = {tk: j for j, tk in enumerate(p.tickers)}
    daily = np.zeros(len(p.dates))
    nets = []
    base_w = 1.0 / k
    has_w = "weight" in t.columns
    for r in t.itertuples():
        w = base_w * (r.weight if has_w and np.isfinite(r.weight) else 1.0)
        j = col[r.ticker]
        ce = side_cost(p, r.ticker, r.ei, r.ep) * cost_mult
        cx = side_cost(p, r.ticker, r.xi, r.xp, stop=getattr(r, "stop_exit", False)) * cost_mult
        s = r.side
        gross = s * (r.xp / r.ep - 1)
        nets.append(gross - ce - cx)
        # günlük piyasa değeri
        if r.ei == r.xi:
            daily[r.ei] += w * (gross - ce - cx)
            continue
        first = r.ei if r.et == "open" else r.ei + 1
        prev = r.ep
        for i in range(first, r.xi + 1):
            if i == r.xi:
                px = r.xp
            else:
                px = C[i, j]
            if not np.isfinite(px):
                continue
            daily[i] += w * s * (px / prev - 1)
            prev = px
        daily[r.ei] -= w * ce
        daily[r.xi] -= w * cx
    t["net"] = nets
    return pd.Series(daily, index=p.dates), t


# ---------------------------------------------------------------- ölçümler
def stats(daily: pd.Series, trades: pd.DataFrame, start=None, end=None) -> dict:
    d = daily.loc[start:end]
    tr = trades
    if "date" in tr.columns:
        tr = tr[(tr["date"] >= (start or tr["date"].min())) & (tr["date"] <= (end or tr["date"].max()))]
    n = len(tr)
    wins = tr["net"][tr["net"] > 0]
    losses = tr["net"][tr["net"] <= 0]
    eq = (1 + d).cumprod()
    dd = eq / eq.cummax() - 1
    years = max(len(d) / 252, 1e-9)
    ann_ret = eq.iloc[-1] ** (1 / years) - 1 if len(d) else np.nan
    vol = d.std() * np.sqrt(252)
    sharpe = d.mean() / d.std() * np.sqrt(252) if d.std() > 0 else np.nan
    exp = tr["net"].mean() if n else np.nan
    tstat = exp / (tr["net"].std() / np.sqrt(n)) if n > 2 and tr["net"].std() > 0 else np.nan
    return {
        "işlem_sayısı": n,
        "kazanma_oranı": len(wins) / n if n else np.nan,
        "ort_kâr": wins.mean() if len(wins) else np.nan,
        "ort_zarar": losses.mean() if len(losses) else np.nan,
        "beklenen_getiri": exp,                 # işlem başına net ortalama getiri
        "kâr_faktörü": wins.sum() / -losses.sum() if len(losses) and losses.sum() < 0 else np.nan,
        "t_istatistiği": tstat,
        "yıllık_getiri": ann_ret,
        "yıllık_oynaklık": vol,
        "sharpe": sharpe,
        "maks_düşüş": dd.min() if len(d) else np.nan,
        "piyasada_kalma": (d != 0).mean() if len(d) else np.nan,
    }


def deflated_sharpe(sr_daily: float, n_obs: int, n_trials: int, sr_trials_std: float,
                    skew: float = 0.0, kurt: float = 3.0) -> float:
    """Bailey & López de Prado (2014) Deflated Sharpe Ratio: çok sayıda deneme yapıldığında
    en iyi sonucun şansla çıkma olasılığını hesaba katar. Döner: gerçek Sharpe>0 olasılığı."""
    from scipy.stats import norm
    if n_trials < 2 or not np.isfinite(sr_daily):
        return np.nan
    emc = 0.5772156649
    sr0 = sr_trials_std * ((1 - emc) * norm.ppf(1 - 1 / n_trials) + emc * norm.ppf(1 - 1 / (n_trials * np.e)))
    num = (sr_daily - sr0) * np.sqrt(n_obs - 1)
    den = np.sqrt(1 - skew * sr_daily + (kurt - 1) / 4 * sr_daily ** 2)
    return float(norm.cdf(num / den))


def bootstrap_ci(x: np.ndarray, n: int = 5000, seed: int = 0) -> tuple[float, float]:
    """İşlem başına ortalama getirinin %95 güven aralığı (yeniden örnekleme ile)."""
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if len(x) < 5:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
