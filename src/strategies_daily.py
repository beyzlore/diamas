"""Günlük veriyle çalışan, kuralları açık stratejiler.

Her fonksiyon işlem listesi (trades DataFrame) üretir. Sütunlar backtest.simulate ile uyumlu:
ticker, ei, et, ep, xi, xt, xp, side, rank, (stop_exit), date
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtest import Panel


def _mk(rows: list[dict], p: Panel) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["ticker", "ei", "et", "ep", "xi", "xt", "xp", "side", "rank", "date"])
    t = pd.DataFrame(rows)
    t = t[np.isfinite(t["ep"]) & np.isfinite(t["xp"]) & (t["ep"] > 0) & (t["xp"] > 0)]
    t["date"] = p.dates[t["ei"].values]
    return t


def _runs_to_trades(p: Panel, ticker: str, sig: np.ndarray, side: int = 1, rank: float = 0) -> list[dict]:
    """sig[t] True ise t kapanışında pozisyon istenir -> t+1 açılışında gir, sinyal kapanınca
    ertesi açılışta çık."""
    O = p.o[ticker].values
    rows, n = [], len(sig)
    i = 0
    while i < n - 1:
        if sig[i] and (i == 0 or not sig[i - 1]):
            j = i
            while j < n - 1 and sig[j]:
                j += 1
            ei, xi = i + 1, min(j + 1, n - 1)
            rows.append(dict(ticker=ticker, ei=ei, et="open", ep=O[ei], xi=xi, xt="open", xp=O[xi],
                             side=side, rank=rank))
            i = j
        i += 1
    return rows


# ------------------------------------------------------------------ 1) Trend takibi (SMA)
def trend_sma(p: Panel, tickers: list[str], n: int) -> pd.DataFrame:
    """Kural: Kapanış > n günlük ortalama ise ertesi açılışta al; altına inince ertesi açılışta sat."""
    rows = []
    for t in tickers:
        c = p.c[t]
        sig = (c > c.rolling(n).mean()).values & c.notna().values
        rows += _runs_to_trades(p, t, sig)
    return _mk(rows, p)


def buy_hold(p: Panel, ticker: str) -> pd.DataFrame:
    c = p.c[ticker]
    first = int(np.argmax(c.notna().values))
    rows = [dict(ticker=ticker, ei=first, et="open", ep=p.o[ticker].iat[first], xi=len(c) - 1,
                 xt="close", xp=c.iat[-1], side=1, rank=0)]
    return _mk(rows, p)


# ------------------------------------------------------------------ 2) RSI(2) ortalamaya dönüş
def rsi2_reversion(p: Panel, univ: pd.DataFrame, th: float, max_hold: int = 10) -> pd.DataFrame:
    """Kural (Connors): Kapanış > 200 günlük ort. VE RSI(2) < th ise ertesi açılışta al.
    Kapanış 5 günlük ortalamanın üstüne çıkınca (veya max_hold gün dolunca) ertesi açılışta sat.
    Aynı gün çok sinyal varsa en düşük RSI'lı olanlar öncelikli."""
    rsi, sma200, sma5 = p.ind["rsi2"], p.ind["sma200"], p.ind["sma5"]
    sig = (rsi < th) & (p.c > sma200) & univ.reindex(columns=p.c.columns, fill_value=False)
    C, O = p.c.values, p.o.values
    S5 = sma5.values
    rows = []
    n = len(p.dates)
    ii, jj = np.where(sig.values[:-1])
    for i, j in zip(ii, jj):
        ei = i + 1
        x = None
        for s in range(ei, min(ei + max_hold, n - 1)):
            if C[s, j] > S5[s, j]:
                x = s + 1
                break
        if x is None:
            x = min(ei + max_hold, n - 1)
        rows.append(dict(ticker=p.tickers[j], ei=ei, et="open", ep=O[ei, j], xi=x, xt="open", xp=O[x, j],
                         side=1, rank=rsi.values[i, j]))
    return _dedupe(_mk(rows, p))


def _dedupe(t: pd.DataFrame) -> pd.DataFrame:
    """Aynı hissede açık pozisyon varken yeni giriş yapılmasın."""
    if t.empty:
        return t
    t = t.sort_values(["ticker", "ei"])
    keep, last_x = [], {}
    for r in t.itertuples():
        if r.ei >= last_x.get(r.ticker, -1):
            keep.append(r.Index)
            last_x[r.ticker] = r.xi
    return t.loc[keep]


# ------------------------------------------------------------------ 3) Açılış boşluğu (gap)
def gap_intraday(p: Panel, univ: pd.DataFrame, g: float, mode: str) -> pd.DataFrame:
    """Açılış fiyatı belli olduğu anda karar verilir, gün sonunda (kapanışta) çıkılır.
    gap, bir önceki günün ATR%'sine göre ölçülür (dünün verisi -> geleceğe bakmaz).
      mode='fade_down' : gap < -g*ATR%  -> açılışta AL (düşüşün geri dönmesine oyna)
      mode='go_up'     : gap > +g*ATR%  -> açılışta AL (yükselişin devamına oyna)
      mode='fade_up'   : gap > +g*ATR%  -> açılışta AÇIĞA SAT
    """
    atr_prev = p.ind["atr_pct"].shift(1)
    gap = p.ind["gap"]
    u = univ.reindex(columns=p.c.columns, fill_value=False)
    z = gap / atr_prev
    if mode == "fade_down":
        sig, side = (z < -g) & u, 1
    elif mode == "go_up":
        sig, side = (z > g) & u, 1
    else:
        sig, side = (z > g) & u, -1
    O, C = p.o.values, p.c.values
    ii, jj = np.where(sig.fillna(False).values)
    rows = [dict(ticker=p.tickers[j], ei=i, et="open", ep=O[i, j], xi=i, xt="close", xp=C[i, j],
                 side=side, rank=-abs(z.values[i, j])) for i, j in zip(ii, jj)]
    return _mk(rows, p)


# ------------------------------------------------------------------ 4) Bilanço sonrası sürüklenme
def pead(p: Panel, earn: pd.DataFrame, univ: pd.DataFrame, min_surprise: float, hold: int,
         need_up_reaction: bool = True, short_side: bool = False) -> pd.DataFrame:
    """Post-Earnings Announcement Drift: Beklentiyi aşan (sürpriz > min_surprise %) ve piyasanın
    ilk gün olumlu karşıladığı bilançolardan sonra hisse günlerce yükselmeye devam eder mi?
    Kural: tepki gününün ertesi açılışında al, `hold` işlem günü sonra açılışta sat.
    short_side=True: negatif sürpriz + düşüş tepkisi -> açığa sat."""
    e = earn.dropna(subset=["surprise"]).copy()
    hr = e["ts"].dt.hour + e["ts"].dt.minute / 60
    e = e[(hr >= 16) | ((hr > 0) & (hr < 9.5))]           # saati belirsiz (00:00) olanları at
    e["d"] = e["ts"].dt.normalize()
    after = hr.loc[e.index] >= 16
    pos = p.dates.searchsorted(e["d"].values)                # o gün veya sonraki ilk işlem günü
    on_day = (pos < len(p.dates)) & (p.dates[np.minimum(pos, len(p.dates) - 1)] == e["d"].values)
    react = np.where(after & on_day, pos + 1, pos)
    e["r"] = react
    col = {t: j for j, t in enumerate(p.tickers)}
    C, O = p.c.values, p.o.values
    U = univ.reindex(columns=p.c.columns, fill_value=False).values
    rows, n = [], len(p.dates)
    for row in e.itertuples():
        j = col.get(row.ticker)
        r = row.r
        if j is None or r < 1 or r + 1 + hold >= n or not U[r, j]:
            continue
        reaction = C[r, j] / C[r - 1, j] - 1
        if not short_side:
            if row.surprise < min_surprise or (need_up_reaction and not reaction > 0):
                continue
            side = 1
        else:
            if row.surprise > -min_surprise or (need_up_reaction and not reaction < 0):
                continue
            side = -1
        ei, xi = r + 1, r + 1 + hold
        rows.append(dict(ticker=row.ticker, ei=ei, et="open", ep=O[ei, j], xi=xi, xt="open", xp=O[xi, j],
                         side=side, rank=-abs(row.surprise)))
    return _dedupe(_mk(rows, p))


# ------------------------------------------------------------------ 5) Donchian kırılımı
def donchian(p: Panel, tickers: list[str], n: int) -> pd.DataFrame:
    """Kapanış son n günün en yükseğini aşınca ertesi açılışta al; son n/2 günün en düşüğünün
    altına inince ertesi açılışta sat (Kaplumbağa tüccarları kuralı)."""
    m = max(n // 2, 5)
    rows = []
    for t in tickers:
        c, h, l = p.c[t], p.h[t], p.l[t]
        hi, lo = h.rolling(n).max().shift(1), l.rolling(m).min().shift(1)
        C, HI, LO = c.values, hi.values, lo.values
        sig = np.zeros(len(c), bool)
        on = False
        for i in range(len(c)):
            if not np.isfinite(C[i]):
                on = False
            elif not on and C[i] > HI[i]:
                on = True
            elif on and C[i] < LO[i]:
                on = False
            sig[i] = on
        rows += _runs_to_trades(p, t, sig)
    return _mk(rows, p)


# ------------------------------------------------------------------ 6) Kesitsel momentum
def xs_momentum(p: Panel, univ: pd.DataFrame, lookback_m: int, top: int, trend_filter: bool,
                spy: str = "SPY") -> pd.DataFrame:
    """Her ay sonu: son `lookback_m` ayın (son ay hariç) getirisine göre en güçlü `top` hisseyi
    seç, ertesi ayın ilk açılışında al, bir sonraki ay başı açılışta sat.
    trend_filter: SPY 200 günlük ortalamanın altındaysa o ay nakitte kal."""
    c = p.c
    me = c.groupby(c.index.to_period("M")).tail(1).index      # ay sonu günleri
    pos = p.dates.get_indexer(me)
    U = univ.reindex(columns=c.columns, fill_value=False)
    rows = []
    for a in range(lookback_m + 1, len(pos) - 1):
        i = pos[a]
        if trend_filter and not (c[spy].iat[i] > p.ind["sma200"][spy].iat[i]):
            continue
        past, skip = pos[a - lookback_m], pos[a - 1]
        mom = c.iloc[skip] / c.iloc[past] - 1
        mom = mom[U.iloc[i] & mom.notna()]
        pick = mom.nlargest(top)
        ei, xi = i + 1, pos[a + 1] + 1
        if xi >= len(p.dates):
            break
        for t, m in pick.items():
            rows.append(dict(ticker=t, ei=ei, et="open", ep=p.o[t].iat[ei], xi=xi, xt="open",
                             xp=p.o[t].iat[xi], side=1, rank=-m))
    return _mk(rows, p)


# ------------------------------------------------------------------ 7) FOMC öncesi sürüklenme
def fomc_drift(p: Panel, fomc: pd.Series, ticker: str = "SPY", entry: str = "prev_close") -> pd.DataFrame:
    """FOMC tarihleri haftalar önceden ilan edilir (geleceğe bakma yok).
    entry='prev_close': karardan önceki gün kapanışta al, karar günü kapanışta sat.
    entry='open'      : karar günü açılışta al, kapanışta sat."""
    rows = []
    for d in fomc:
        if d not in p.dates:
            continue
        i = p.idx(d)
        if entry == "prev_close":
            rows.append(dict(ticker=ticker, ei=i - 1, et="close", ep=p.c[ticker].iat[i - 1], xi=i, xt="close",
                             xp=p.c[ticker].iat[i], side=1, rank=0))
        else:
            rows.append(dict(ticker=ticker, ei=i, et="open", ep=p.o[ticker].iat[i], xi=i, xt="close",
                             xp=p.c[ticker].iat[i], side=1, rank=0))
    return _mk(rows, p)


# ------------------------------------------------------------------ 8) Ay dönümü etkisi
def turn_of_month(p: Panel, ticker: str, before: int, after: int) -> pd.DataFrame:
    """Ayın son `before` işlem gününden önceki kapanışta al, yeni ayın `after`. işlem günü
    kapanışında sat. (Takvime dayalı; geleceğe bakma yok.)"""
    per = p.dates.to_period("M")
    first_idx = np.where(per[1:] != per[:-1])[0] + 1          # her ayın ilk işlem günü
    rows = []
    c = p.c[ticker].values
    for f in first_idx:
        ei, xi = f - before - 1, f + after - 1
        if ei < 0 or xi >= len(c) or not np.isfinite(c[ei]):
            continue
        rows.append(dict(ticker=ticker, ei=ei, et="close", ep=c[ei], xi=xi, xt="close", xp=c[xi],
                         side=1, rank=0))
    return _mk(rows, p)


# ================================================================== 2. TUR stratejileri
# 1. turun eğitim+doğrulama sonuçlarından çıkan derslerle tasarlandı (kilitli döneme bakılmadı).

def rsi2_regime(p: Panel, univ: pd.DataFrame, vix: pd.Series, th: float, vix_max: float,
                max_hold: int = 10) -> pd.DataFrame:
    """RSI(2) dipten alım + piyasa rejim filtresi. 1. turda strateji krizlerde (2008, 2020, 2022)
    zarar ettiği için: yalnızca SPY 200 günlük ortalamanın üstündeyken VE VIX < vix_max iken
    (bir önceki kapanış verisiyle) yeni pozisyon aç."""
    t = rsi2_reversion(p, univ, th, max_hold)
    if t.empty:
        return t
    ok_spy = (p.c["SPY"] > p.ind["sma200"]["SPY"])
    v = vix.reindex(p.dates).ffill()
    sig_day = t["ei"].values - 1                               # sinyal günü (dünün kapanışı)
    keep = ok_spy.values[sig_day] & (v.values[sig_day] < vix_max)
    return t[keep]


def dual_momentum(p: Panel, risky: list[str], safe: str, lookback_m: int, top: int = 1) -> pd.DataFrame:
    """Antonacci tipi çift momentum (ETF): her ay sonu riskli ETF'ler arasından son L ayda en çok
    kazanan `top` tanesini seç; seçilenin getirisi güvenli varlığınkinden (TLT) düşükse onun yerine
    güvenli varlığı tut. Ertesi ay başı açılışta al, bir sonraki ay başı açılışta sat."""
    c = p.c
    me = c.groupby(c.index.to_period("M")).tail(1).index
    pos = p.dates.get_indexer(me)
    rows = []
    for a in range(lookback_m, len(pos) - 1):
        i, past = pos[a], pos[a - lookback_m]
        mom = (c.iloc[i] / c.iloc[past] - 1)
        rk = mom[risky].dropna().nlargest(top)
        safe_m = mom.get(safe, np.nan)
        picks = [t if (np.isfinite(safe_m) and m > safe_m) or not np.isfinite(safe_m) else safe for t, m in rk.items()]
        ei, xi = i + 1, pos[a + 1] + 1
        if xi >= len(p.dates):
            break
        for t in picks:
            if not np.isfinite(p.o[t].iat[ei]):
                continue
            rows.append(dict(ticker=t, ei=ei, et="open", ep=p.o[t].iat[ei], xi=xi, xt="open",
                             xp=p.o[t].iat[xi], side=1, rank=0))
    return _mk(rows, p)


def vix_spike(p: Panel, vix: pd.Series, ratio: float, hold: int, ticker: str = "SPY") -> pd.DataFrame:
    """Panik alımı: VIX kendi 20 günlük ortalamasının `ratio` katına çıkarsa (korku zirvesi),
    ertesi açılışta SPY al, `hold` gün sonra açılışta sat."""
    v = vix.reindex(p.dates).ffill()
    sig = (v / v.rolling(20).mean() > ratio).values
    O = p.o[ticker].values
    rows, i, n = [], 20, len(p.dates)
    while i < n - hold - 1:
        if sig[i]:
            ei, xi = i + 1, i + 1 + hold
            rows.append(dict(ticker=ticker, ei=ei, et="open", ep=O[ei], xi=xi, xt="open", xp=O[xi],
                             side=1, rank=0))
            i = xi
        else:
            i += 1
    return _mk(rows, p)


def trend_voltarget(p: Panel, n: int, target_vol: float, ticker: str = "SPY") -> pd.DataFrame:
    """Trend takibi + oynaklık hedefleme: trend yukarıyken (kapanış > n günlük ort.) SPY tut;
    pozisyon büyüklüğü = hedef oynaklık / son 20 günün oynaklığı (en fazla 1 = kaldıraçsız).
    Ağırlık her ay başı güncellenir; işlem listesi aylık parçalara bölünür ve 'weight' sütunu taşır."""
    c = p.c[ticker]
    sig = (c > c.rolling(n).mean()).values
    rv = c.pct_change().rolling(20).std().values * np.sqrt(252)
    O = p.o[ticker].values
    me = c.groupby(c.index.to_period("M")).tail(1).index
    pos = p.dates.get_indexer(me)
    rows = []
    # Aylık yeniden dengeleme + sinyal günlük: her gün için pozisyon ağırlığını belirle, sonra
    # ağırlığın sabit kaldığı blokları işlem olarak yaz.
    w = np.zeros(len(c))
    cur_cap = 1.0
    month_set = set(pos.tolist())
    for i in range(len(c)):
        if i in month_set or i == 0:
            cur_cap = min(1.0, target_vol / rv[i]) if np.isfinite(rv[i]) and rv[i] > 0 else 1.0
        w[i] = cur_cap if sig[i] else 0.0
    i = 0
    while i < len(c) - 1:
        if w[i] > 0:
            j = i
            while j < len(c) - 1 and w[j] == w[i]:
                j += 1
            ei, xi = i + 1, min(j + 1, len(c) - 1)
            rows.append(dict(ticker=ticker, ei=ei, et="open", ep=O[ei], xi=xi, xt="open", xp=O[xi],
                             side=1, rank=0, weight=w[i]))
            i = j
        else:
            i += 1
    return _mk(rows, p)


# ================================================================== 3. TUR stratejileri
# 2. turun dersleri: PEAD ve RSI(2) ayı piyasalarında zarar ediyor; RSI(2) işlemleri bilanço
# günlerine denk gelince ani sıçramalara maruz kalıyor.

def pead_regime(p: Panel, earn: pd.DataFrame, univ: pd.DataFrame, min_surprise: float, hold: int) -> pd.DataFrame:
    """PEAD + piyasa filtresi: yalnızca SPY 200 günlük ortalamanın üstündeyken (giriş öncesi
    kapanış) yeni pozisyon aç."""
    t = pead(p, earn, univ, min_surprise, hold, True, False)
    if t.empty:
        return t
    ok = (p.c["SPY"] > p.ind["sma200"]["SPY"]).values
    return t[ok[t["ei"].values - 1]]


def earnings_days_mask(p: Panel, earn: pd.DataFrame) -> pd.DataFrame:
    """Her hisse için bilanço tepki günleri (True). Bilanço tarihleri şirketlerce haftalar önce
    duyurulur; yine de bu bir yaklaşıktır (veri gerçekleşen tarihi verir)."""
    e = earn.copy()
    hr = e["ts"].dt.hour + e["ts"].dt.minute / 60
    d = e["ts"].dt.normalize().values
    pos = np.minimum(p.dates.searchsorted(d), len(p.dates) - 1)
    on_day = p.dates[pos] == d
    r = np.minimum(np.where((hr.values >= 16) & on_day, pos + 1, pos), len(p.dates) - 1)
    m = pd.DataFrame(False, index=p.dates, columns=p.c.columns)
    col = {t: j for j, t in enumerate(p.tickers)}
    for tk, ri in zip(e["ticker"].values, r):
        j = col.get(tk)
        if j is not None:
            m.iat[ri, j] = True
    return m


def rsi2_no_earnings(p: Panel, univ: pd.DataFrame, vix: pd.Series, earn_mask: pd.DataFrame, th: float,
                     vix_max: float, buffer: int) -> pd.DataFrame:
    """RSI(2) + kriz filtresi + bilanço kaçınma: giriş ile çıkış (+buffer gün) arasında bilanço
    tepki günü olan işlemleri hiç açma."""
    t = rsi2_regime(p, univ, vix, th, vix_max)
    if t.empty:
        return t
    M = earn_mask.values
    col = {tk: j for j, tk in enumerate(p.tickers)}
    keep = []
    for r in t.itertuples():
        j = col[r.ticker]
        keep.append(not M[r.ei:min(r.xi + 1 + buffer, len(p.dates)), j].any())
    return t[np.array(keep)]
