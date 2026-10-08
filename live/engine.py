"""Canlı takip motoru: sanal (paper) portföy.

Testlerden geçen / en umut verici 3 strateji, 9 Ekim 2026'dan itibaren 100.000 $ sanal parayla
gerçek piyasa fiyatlarıyla "oynanır". Hiçbir gerçek emir gönderilmez.

Her çalıştırmada başlangıçtan bugüne tüm günler yeniden hesaplanır (kaçırılan gün olsa bile
sonuç aynı çıkar). Kurallar backtest'tekilerle birebir aynıdır.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import backtest as B  # noqa: E402
import data as D  # noqa: E402

START = pd.Timestamp("2026-10-09")   # ilk sanal işlem günü (8 Ekim kapanışındaki sinyallerle)
CAPITAL = 100_000.0

SLEEVES = {
    "trend": dict(title="Trend takibi (SPY)", share=1 / 3, k=1,
                  rule="SPY kapanışı 100 günlük ortalamanın üstündeyse SPY tut, altına inerse nakite geç."),
    "pead": dict(title="Bilanço sonrası yükseliş", share=1 / 3, k=10,
                 rule="Kâr beklentisini aşan ve ilk gün yükselen hisseyi ertesi açılışta al, 10 işlem günü sonra sat."),
    "rsi2": dict(title="RSI(2) dipten alım + filtreler", share=1 / 3, k=10,
                 rule="Yükseliş trendindeki likit hisse RSI(2) < 5'e düşerse al; kapanış 5 günlük ortalamayı geçince "
                      "veya 10 gün dolunca sat. Piyasa düşüşteyken, VIX ≥ 30 iken ya da yakında bilanço varsa girme."),
}


@dataclass
class Pos:
    sleeve: str
    ticker: str
    entry_i: int
    entry_px: float
    shares: float
    cost_basis: float
    exit_due: int | None = None     # pead: planlı çıkış günü; rsi2: en geç çıkış günü


@dataclass
class Order:
    sleeve: str
    ticker: str
    side: str          # 'AL' / 'SAT'
    reason: str
    pos: Pos | None = None


@dataclass
class State:
    cash: dict = field(default_factory=lambda: {s: CAPITAL * v["share"] for s, v in SLEEVES.items()})
    positions: list = field(default_factory=list)
    closed: list = field(default_factory=list)
    equity: list = field(default_factory=list)


def earnings_reaction(p: B.Panel, earn: pd.DataFrame) -> pd.DataFrame:
    """Her bilanço için tepki günü indeksini hesaplar (kapanış sonrası -> ertesi gün)."""
    e = earn.copy()
    hr = e["ts"].dt.hour + e["ts"].dt.minute / 60
    e = e[(hr >= 16) | ((hr > 0) & (hr < 9.5))].copy()
    d = e["ts"].dt.normalize()
    after = (e["ts"].dt.hour + e["ts"].dt.minute / 60) >= 16
    # gelecekteki tarihler için de çalışsın: iş günü takvimiyle
    bdays = pd.bdate_range(p.dates[0], p.dates[-1] + pd.Timedelta(days=120))
    react_date = []
    for dd, af in zip(d, after):
        k = bdays.searchsorted(dd)
        if k < len(bdays) and bdays[k] == dd and af:
            k += 1
        react_date.append(bdays[min(k, len(bdays) - 1)])
    e["react_date"] = react_date
    return e


def run(p: B.Panel, earn: pd.DataFrame, vix: pd.Series):
    n = len(p.dates)
    start_i = int(p.dates.searchsorted(START))
    U100 = B.pit_universe(p, 100, set(D.ETFS)).reindex(columns=p.c.columns, fill_value=False).values
    U200 = B.pit_universe(p, 200, set(D.ETFS)).reindex(columns=p.c.columns, fill_value=False).values
    col = {t: j for j, t in enumerate(p.tickers)}
    O, C = p.o.values, p.c.values
    RSI, S5, S200 = p.ind["rsi2"].values, p.ind["sma5"].values, p.ind["sma200"].values
    sma100_spy = p.c["SPY"].rolling(100).mean().values
    spy = col["SPY"]
    v = vix.reindex(p.dates).ffill().values
    er = earnings_reaction(p, earn)
    # bilanço tepki günü -> (ticker, surprise) listesi
    react_by_day: dict[pd.Timestamp, list] = {}
    for r in er.itertuples():
        react_by_day.setdefault(r.react_date, []).append((r.ticker, r.surprise))
    future_earn = er.groupby("ticker")["react_date"].apply(lambda s: np.sort(s.values))

    st = State()
    pending: list[Order] = []
    bench_shares = None

    def sleeve_value(s, i):
        val = st.cash[s]
        for ps in st.positions:
            if ps.sleeve == s:
                px = C[i, col[ps.ticker]]
                val += ps.shares * (px if np.isfinite(px) else ps.entry_px)
        return val

    def held(s, t=None):
        return [ps for ps in st.positions if ps.sleeve == s and (t is None or ps.ticker == t)]

    def earnings_soon(t, i, days):
        arr = future_earn.get(t)
        if arr is None or len(arr) == 0:
            return False
        a, b = p.dates[min(i + 1, n - 1)], p.dates[i] + pd.offsets.BDay(days)
        if i + 1 >= n:
            a = p.dates[i] + pd.offsets.BDay(1)
        return bool(((arr >= np.datetime64(a)) & (arr <= np.datetime64(b))).any())

    if start_i == 0:
        raise RuntimeError("Başlangıç tarihi verinin içinde değil")
    for i in range(start_i - 1, n):
        # 1) bekleyen emirleri bu günün açılışında doldur
        if i >= start_i:
            if bench_shares is None:
                bench_shares = CAPITAL / O[i, spy]
            sv_prev = {s: sleeve_value(s, i - 1) for s in SLEEVES}
            for od in pending:
                j = col[od.ticker]
                px = O[i, j]
                if not np.isfinite(px):
                    continue
                c = B.side_cost(p, od.ticker, i, px)
                if od.side == "SAT" and od.pos in st.positions:
                    ps = od.pos
                    proceeds = ps.shares * px * (1 - c)
                    st.cash[od.sleeve] += proceeds
                    st.positions.remove(ps)
                    st.closed.append(dict(sleeve=od.sleeve, ticker=ps.ticker, entry=p.dates[ps.entry_i],
                                          exit=p.dates[i], entry_px=ps.entry_px, exit_px=px,
                                          pnl=proceeds - ps.cost_basis, ret=proceeds / ps.cost_basis - 1, reason=od.reason))
                elif od.side == "AL":
                    k = SLEEVES[od.sleeve]["k"]
                    alloc = min(sv_prev[od.sleeve] / k, st.cash[od.sleeve])
                    if alloc <= 1:
                        continue
                    sh = alloc / (px * (1 + c))
                    st.cash[od.sleeve] -= alloc
                    due = None
                    if od.sleeve == "pead":
                        due = i + 10
                    elif od.sleeve == "rsi2":
                        due = i + 10
                    st.positions.append(Pos(od.sleeve, od.ticker, i, px, sh, alloc, due))
            pending = []
            # 2) gün sonu değerleme
            row = {"date": p.dates[i], **{s: sleeve_value(s, i) for s in SLEEVES},
                   "SPY": bench_shares * C[i, spy]}
            row["toplam"] = sum(row[s] for s in SLEEVES)
            st.equity.append(row)

        # 3) bu günün kapanışıyla yarının emirlerini üret
        orders: list[Order] = []
        # --- trend
        up = C[i, spy] > sma100_spy[i]
        h = held("trend")
        if up and not h:
            orders.append(Order("trend", "SPY", "AL", "SPY 100 günlük ortalamanın üstünde"))
        elif not up and h:
            orders.append(Order("trend", "SPY", "SAT", "SPY 100 günlük ortalamanın altına indi", h[0]))
        # --- pead: çıkışlar
        for ps in held("pead"):
            if ps.exit_due is not None and i + 1 >= ps.exit_due:
                orders.append(Order("pead", ps.ticker, "SAT", "10 günlük tutma süresi doldu", ps))
        # --- pead: girişler
        cands = []
        for t, surpr in react_by_day.get(p.dates[i], []):
            j = col.get(t)
            if j is None or not U200[i, j] or not np.isfinite(surpr) or surpr < 0 or held("pead", t):
                continue
            if i < 1 or not (C[i, j] / C[i - 1, j] - 1 > 0):
                continue
            cands.append((abs(surpr), t, surpr, C[i, j] / C[i - 1, j] - 1))
        free = SLEEVES["pead"]["k"] - (len(held("pead")) - sum(o.side == "SAT" for o in orders if o.sleeve == "pead"))
        for _, t, surpr, rr in sorted(cands, reverse=True)[:max(free, 0)]:
            orders.append(Order("pead", t, "AL", f"Kâr sürprizi %{surpr:.1f}, ilk gün tepkisi %{rr * 100:.1f}"))
        # --- rsi2: çıkışlar
        for ps in held("rsi2"):
            j = col[ps.ticker]
            if C[i, j] > S5[i, j]:
                orders.append(Order("rsi2", ps.ticker, "SAT", "Kapanış 5 günlük ortalamanın üstüne çıktı", ps))
            elif ps.exit_due is not None and i + 1 >= ps.exit_due:
                orders.append(Order("rsi2", ps.ticker, "SAT", "10 günlük azami süre doldu", ps))
        # --- rsi2: girişler
        market_ok = C[i, spy] > S200[i, spy] and v[i] < 30
        cands = []
        if market_ok:
            for t, j in col.items():
                if not U100[i, j] or held("rsi2", t):
                    continue
                if RSI[i, j] < 5 and C[i, j] > S200[i, j] and not earnings_soon(t, i, 12):
                    cands.append((RSI[i, j], t))
        free = SLEEVES["rsi2"]["k"] - (len(held("rsi2")) - sum(o.side == "SAT" for o in orders if o.sleeve == "rsi2"))
        for r, t in sorted(cands)[:max(free, 0)]:
            orders.append(Order("rsi2", t, "AL", f"RSI(2) = {r:.1f}, uzun vadeli trend yukarı"))
        pending = orders

    ctx = dict(spy_close=C[-1, spy], spy_sma200=S200[-1, spy], spy_sma100=sma100_spy[-1], vix=v[-1],
               market_ok=bool(C[-1, spy] > S200[-1, spy] and v[-1] < 30), trend_up=bool(C[-1, spy] > sma100_spy[-1]))
    # izleme listesi: RSI(2) sinyaline yakın hisseler
    watch = []
    j_idx = [col[t] for t in p.tickers if U100[-1, col[t]]]
    for j in j_idx:
        if C[-1, j] > S200[-1, j] and RSI[-1, j] < 15:
            watch.append(dict(ticker=p.tickers[j], rsi2=RSI[-1, j], close=C[-1, j],
                              earn_soon=earnings_soon(p.tickers[j], n - 1, 12)))
    # yaklaşan bilançolar (en likit 100 + elde tutulanlar)
    last = p.dates[-1]
    watch_set = {p.tickers[j] for j in j_idx} | {ps.ticker for ps in st.positions}
    up_e = er[(er["react_date"] > last) & (er["react_date"] <= last + pd.Timedelta(days=10)) & er["ticker"].isin(watch_set)]
    up_e = up_e.sort_values("ts")[["ticker", "ts", "eps_est"]]
    return st, pending, ctx, sorted(watch, key=lambda w: w["rsi2"]), up_e, p.dates[-1]
