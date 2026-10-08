"""Sinyal kartları: her yeni sinyal için giriş / çıkış koşulu / hedef / olumsuz senaryo / risk-ödül,
geçmiş benzer işlemlerden hesaplanan olasılık senaryoları ve güven seviyesi.

Tüm sayılar geriye dönük testteki aynı kuralın gerçekleşmiş işlemlerinden gelir (maliyetler düşülmüş).
Bunlar tahmin değil, geçmişin dağılımıdır; gelecekte farklı olabilir.
"""
from __future__ import annotations

import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
KEY = {"trend": "trend_spy", "pead": "pead_long", "rsi2": "rsi2_noearn"}
BAND = {"trend": 0.02, "pead": 0.01, "rsi2": 0.01}          # "yatay" sayılacak getiri aralığı (±)
LOG = ROOT / "results" / "signal_log.csv"


def _p(x, d=1):
    return ("−" if x < 0 else "+") + f"%{abs(x) * 100:.{d}f}"


def load_hist(path=ROOT / "results" / "daily_final.pkl") -> dict:
    S = pickle.load(open(path, "rb"))["summary"]
    out = {}
    for sl, k in KEY.items():
        tr = S[k]["trades"]
        out[sl] = tr[tr["date"] <= "2026-10-08"].copy()      # canlı dönem öncesi
    return out


def rule_stats(tr: pd.DataFrame, band: float) -> dict:
    x = tr["net"].dropna().values
    if len(x) == 0:
        return {}
    w, l = x[x > 0], x[x <= 0]
    se = x.std() / np.sqrt(len(x)) if len(x) > 1 else np.nan
    t = x.mean() / se if se and se > 0 else np.nan
    return dict(n=len(x), win=float((x > 0).mean()), avg_win=float(w.mean()) if len(w) else np.nan,
                avg_loss=float(l.mean()) if len(l) else np.nan, mean=float(x.mean()),
                p10=float(np.percentile(x, 10)), p90=float(np.percentile(x, 90)), worst=float(x.min()),
                up=float((x > band).mean()), flat=float(((x >= -band) & (x <= band)).mean()),
                down=float((x < -band).mean()), t=float(t),
                rr=float(w.mean() / -l.mean()) if len(w) and len(l) and l.mean() < 0 else np.nan)


def confidence(st: dict) -> str:
    """Güven seviyesi: geçmiş örnek sayısı ve istatistiksel anlamlılığa göre (kâr olasılığı değildir)."""
    if not st:
        return "Bilinmiyor"
    if st["n"] >= 300 and st["t"] > 3:
        return "Orta-yüksek"
    if st["n"] >= 100 and st["t"] > 2:
        return "Orta"
    return "Düşük"


def cards(pending, p, hist: dict, ctx: dict) -> list[dict]:
    out = []
    for o in pending:
        if o.side != "AL":
            continue
        tr = hist.get(o.sleeve, pd.DataFrame())
        st = rule_stats(tr, BAND[o.sleeve])
        same = tr[tr["ticker"] == o.ticker] if len(tr) else tr
        st_same = rule_stats(same, BAND[o.sleeve]) if len(same) >= 3 else {}
        px = float(p.c[o.ticker].dropna().iloc[-1])
        atr = float(p.ind["atr14"][o.ticker].dropna().iloc[-1])
        if o.sleeve == "trend":
            stop_lvl = ctx["spy_sma100"]
            exit_rule = f"SPY kapanışı 100 günlük ortalamanın ({stop_lvl:,.2f}) altına inerse ertesi açılışta sat"
            stop_txt = f"{stop_lvl:,.2f} (100g ort., şu an {_p(stop_lvl / px - 1)} uzakta; ortalama yükseldikçe yükselir)"
        elif o.sleeve == "pead":
            exit_rule = "10 işlem günü sonra açılışta sat (fiyattan bağımsız)"
            stop_txt = f"Kuralda sabit stop yok. Geçmişte en kötü %10'luk sonuç: {px * (1 + st['p10']):,.2f} ({_p(st['p10'])})"
        else:
            exit_rule = "Kapanış 5 günlük ortalamanın üstüne çıkınca ertesi açılışta sat; en geç 10 gün sonra sat"
            stop_txt = f"Kuralda sabit stop yok. Geçmişte en kötü %10'luk sonuç: {px * (1 + st['p10']):,.2f} ({_p(st['p10'])})"
        out.append(dict(
            ticker=o.ticker, sleeve=o.sleeve, reason=o.reason, price=px, atr_pct=atr / px,
            entry=f"Yarın açılışta (referans son kapanış {px:,.2f})", exit_rule=exit_rule, stop=stop_txt,
            target=f"{px * (1 + st['avg_win']):,.2f} (kazanan işlemlerin ortalaması {_p(st['avg_win'])})",
            rr=st.get("rr"), stats=st, stats_same=st_same, conf=confidence(st),
            model_size=1 / 3 / {"trend": 1, "pead": 10, "rsi2": 10}[o.sleeve],
            breakers=_breakers(o.sleeve, ctx),
        ))
    return out


def _breakers(sleeve, ctx) -> str:
    """Senaryoyu bozabilecek gelişmeler."""
    common = "Genel piyasada sert satış (SPY 200 günlük ortalamanın altına inerse kural yeni alımları durdurur)"
    if sleeve == "pead":
        return ("Şirketle ilgili olumsuz haber (dava, rehberlik indirimi), sektör çapında satış, " + common.lower() +
                ". 2020 ve 2022 ayı piyasalarında bu kural zarar etti.")
    if sleeve == "rsi2":
        return ("Düşüşün şirkete özgü kötü bir habere bağlı olması (geçici değil kalıcı düşüş), "
                "VIX'in 30'u aşması, " + common.lower() + ".")
    return "Ani piyasa çöküşü: trend kuralı geç tepki verir, ilk düşüşün bir kısmı yaşanır. " + \
        "Yatay ve dalgalı piyasada sık al-sat yapıp küçük zararlar biriktirebilir."


def log_signals(cs: list[dict], asof: pd.Timestamp):
    """Her sinyali, o anki beklentisiyle birlikte kaydeder (isabet ölçümü için)."""
    rows = [dict(signal_date=asof.date(), logged=datetime.now().strftime("%Y-%m-%d %H:%M"), ticker=c["ticker"],
                 sleeve=c["sleeve"], ref_price=round(c["price"], 4), exp_mean=c["stats"]["mean"],
                 exp_win=c["stats"]["win"], exp_up=c["stats"]["up"], exp_down=c["stats"]["down"], conf=c["conf"])
            for c in cs]
    if not rows:
        return
    new = pd.DataFrame(rows)
    if LOG.exists():
        old = pd.read_csv(LOG)
        key = lambda d: d["signal_date"].astype(str) + d["ticker"] + d["sleeve"]  # noqa: E731
        new = new[~key(new).isin(set(key(old)))]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(LOG, index=False)


def accuracy(closed: list[dict], hist: dict) -> list[dict]:
    """Canlı (sanal) sonuçlar ile geçmiş testin beklentisini strateji bazında karşılaştırır."""
    out = []
    df = pd.DataFrame(closed)
    for sl in KEY:
        st = rule_stats(hist[sl], BAND[sl])
        live = df[df["sleeve"] == sl] if len(df) else df
        n = len(live)
        out.append(dict(sleeve=sl, n=n, live_win=float((live["ret"] > 0).mean()) if n else np.nan,
                        live_mean=float(live["ret"].mean()) if n else np.nan,
                        exp_win=st["win"], exp_mean=st["mean"],
                        dir_hit=float(((live["ret"] > 0)).mean()) if n else np.nan))
    return out


FORECAST = ROOT / "results" / "forecast_log.csv"


def forecast_scores(spy_close: pd.Series) -> dict | None:
    """Günlük analizdeki SPY 5 günlük senaryo olasılıklarını gerçekleşenle karşılaştırır.
    forecast_log.csv sütunları: date, p_up, p_flat, p_down (ufuk 5 işlem günü, eşik ±%1)."""
    if not FORECAST.exists():
        return None
    f = pd.read_csv(FORECAST, parse_dates=["date"])
    rows = []
    s = spy_close.dropna()
    for r in f.itertuples():
        i = s.index.searchsorted(r.date) - 1           # tahmin anında bilinen son kapanış
        if i < 0 or i + 5 >= len(s):
            continue
        ret = s.iloc[i + 5] / s.iloc[i] - 1
        real = "up" if ret > 0.01 else ("down" if ret < -0.01 else "flat")
        p = {"up": r.p_up, "flat": r.p_flat, "down": r.p_down}
        tot = sum(p.values()) or 1
        p = {k: v / tot for k, v in p.items()}
        brier = sum((p[k] - (1 if k == real else 0)) ** 2 for k in p)
        rows.append(dict(hit=max(p, key=p.get) == real, brier=brier))
    if not rows:
        return dict(n=0, pending=len(f))
    d = pd.DataFrame(rows)
    # karşılaştırma: hep 1/3-1/3-1/3 deyen "bilgisiz tahmin" Brier puanı = 0.667
    return dict(n=len(d), hit=float(d["hit"].mean()), brier=float(d["brier"].mean()), pending=len(f) - len(d))
