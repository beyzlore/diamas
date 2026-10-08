"""Veri toplama: günlük fiyatlar, gün içi fiyatlar, VIX, bilanço tarihleri, FOMC takvimi.

Her şey data/ klasörüne parquet olarak önbelleğe alınır; tekrar çalıştırınca internetten
yeniden indirilmez (force=True ile yenilenir).
"""
from __future__ import annotations

import io
import re
import time
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0 research script"}

# Hayatta kalma yanlılığı (survivorship bias) içermeyen, uzun geçmişli ETF'ler
ETFS = ["SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "XLV", "XLY", "XLP",
        "XLI", "XLU", "XLB", "TLT", "GLD", "EFA", "EEM"]


def _today_cutoff() -> pd.Timestamp:
    """Bugünün yarım kalmış verisini atmak için. NY saatiyle 16:30'dan sonra bugünün kapanışı kesinleşmiştir."""
    now = pd.Timestamp.now(tz="America/New_York")
    day = now.normalize().tz_localize(None)
    return day + pd.Timedelta(days=1) if now.hour * 60 + now.minute >= 16 * 60 + 30 else day


def sp500_list(force: bool = False) -> pd.DataFrame:
    p = DATA / "sp500.parquet"
    if p.exists() and not force:
        return pd.read_parquet(p)
    r = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", headers=UA, timeout=30)
    t = pd.read_html(io.StringIO(r.text))[0]
    t = t.rename(columns={"Symbol": "ticker", "Security": "name", "GICS Sector": "sector"})
    t["ticker"] = t["ticker"].str.replace(".", "-", regex=False)
    t = t[["ticker", "name", "sector"]]
    t.to_parquet(p)
    return t


def daily_prices(tickers: list[str], start: str = "2004-01-01", force: bool = False) -> pd.DataFrame:
    """Uzun formatta günlük OHLCV (bölünme/temettü düzeltilmiş). Sütunlar: date,ticker,open,high,low,close,volume."""
    p = DATA / "daily.parquet"
    if p.exists() and not force:
        df = pd.read_parquet(p)
        missing = sorted(set(tickers) - set(df["ticker"].unique()))
        if not missing:
            return df[df["ticker"].isin(tickers)]
        new = _download_daily(missing, start)
        df = pd.concat([df, new], ignore_index=True)
    else:
        df = _download_daily(tickers, start)
    df.to_parquet(p)
    return df[df["ticker"].isin(tickers)]


def _download_daily(tickers: list[str], start: str) -> pd.DataFrame:
    frames = []
    for i in range(0, len(tickers), 100):
        chunk = tickers[i:i + 100]
        raw = yf.download(chunk, start=start, interval="1d", auto_adjust=True, progress=False,
                          group_by="ticker", threads=True)
        for t in chunk:
            try:
                d = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
            except KeyError:
                continue
            d = d.dropna(how="all")
            if d.empty:
                continue
            d = d.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].copy()
            d["ticker"] = t
            d.index.name = "date"
            frames.append(d.reset_index())
        time.sleep(1)
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df[df["date"] < _today_cutoff()]
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df[(df["close"] > 0) & (df["high"] >= df["low"])]
    return df


def intraday(tickers: list[str], interval: str, force: bool = False) -> pd.DataFrame:
    """Gün içi barlar. 5m: son ~60 işlem günü; 1h: son ~730 gün (Yahoo'nun ücretsiz sınırı)."""
    p = DATA / f"intraday_{interval}.parquet"
    if p.exists() and not force:
        df = pd.read_parquet(p)
        if set(tickers) <= set(df["ticker"].unique()):
            return df[df["ticker"].isin(tickers)]
    period = {"5m": "60d", "15m": "60d", "1h": "730d"}[interval]
    frames = []
    for i in range(0, len(tickers), 50):
        chunk = tickers[i:i + 50]
        raw = yf.download(chunk, period=period, interval=interval, auto_adjust=True, progress=False,
                          group_by="ticker", prepost=False, threads=True)
        for t in chunk:
            try:
                d = raw[t]
            except KeyError:
                continue
            d = d.dropna(how="all")
            if d.empty:
                continue
            d = d.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].copy()
            d.index = d.index.tz_convert("America/New_York").tz_localize(None)
            d.index.name = "ts"
            d["ticker"] = t
            frames.append(d.reset_index())
        time.sleep(1)
    df = pd.concat(frames, ignore_index=True)
    df["date"] = df["ts"].dt.normalize()
    df = df[df["date"] < _today_cutoff()]
    df.to_parquet(p)
    return df


def vix(force: bool = False) -> pd.Series:
    p = DATA / "vix.parquet"
    if p.exists() and not force:
        return pd.read_parquet(p)["vix"]
    v = yf.download("^VIX", start="1990-01-01", progress=False, auto_adjust=True)
    s = v["Close"].squeeze().rename("vix")
    s.index = pd.to_datetime(s.index).tz_localize(None)
    s.to_frame().to_parquet(p)
    return s


def earnings_dates(tickers: list[str], force: bool = False) -> pd.DataFrame:
    """Bilanço açıklama tarihleri + beklenti/gerçekleşen EPS. Sütunlar: ticker, ts, eps_est, eps_act, surprise."""
    p = DATA / "earnings.parquet"
    have = pd.DataFrame()
    if p.exists() and not force:
        have = pd.read_parquet(p)
        tickers = sorted(set(tickers) - set(have["ticker"].unique()))
        if not tickers:
            return have
    rows = []
    for t in tickers:
        for attempt in range(3):
            try:
                e = yf.Ticker(t).get_earnings_dates(limit=100)
                break
            except Exception:
                time.sleep(3)
                e = None
        if e is None or e.empty:
            continue
        e = e.reset_index()
        e.columns = ["ts", "eps_est", "eps_act", "surprise"][: len(e.columns)] + list(e.columns[4:])
        e["ts"] = pd.to_datetime(e["ts"], utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
        e["ticker"] = t
        rows.append(e[["ticker", "ts", "eps_est", "eps_act", "surprise"]])
        time.sleep(0.4)
    new = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    df = pd.concat([have, new], ignore_index=True)
    df.to_parquet(p)
    return df


def _month(name: str, months: dict) -> int | None:
    """'Jan/Feb' gibi iki aylı başlıklarda karar günü ikinci aydadır."""
    last = name.split("/")[-1].strip()[:3].lower()
    for full, i in months.items():
        if full[:3].lower() == last:
            return i
    return None


def fomc_dates(force: bool = False) -> pd.Series:
    """FOMC faiz kararı günleri (toplantının son günü). Kaynak: federalreserve.gov."""
    p = DATA / "fomc.parquet"
    if p.exists() and not force:
        return pd.read_parquet(p)["date"]
    months = {m: i for i, m in enumerate(["January", "February", "March", "April", "May", "June", "July",
                                          "August", "September", "October", "November", "December"], 1)}
    out = []
    for y in range(2004, 2021):
        r = requests.get(f"https://www.federalreserve.gov/monetarypolicy/fomchistorical{y}.htm", headers=UA, timeout=30)
        for m, days in re.findall(r"<h5[^>]*>\s*([A-Za-z/]+)\s+([\d\-]+)\s+(?:\(unscheduled\)\s+)?Meeting", r.text):
            m = _month(m, months)
            if m:
                out.append(pd.Timestamp(y, m, int(days.split("-")[-1])))
        time.sleep(0.5)
    txt = requests.get("https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm", headers=UA, timeout=30).text
    for y in range(2021, 2028):
        i = txt.find(f"{y} FOMC Meetings")
        if i < 0:
            continue
        j = txt.find("FOMC Meetings", i + 20)
        seg = txt[i: j if j > 0 else i + 40000]
        for m, days in re.findall(r"fomc-meeting__month[^>]*>\s*<strong>([^<]+)</strong>.*?fomc-meeting__date[^>]*>([^<]+)<", seg, re.S):
            m = _month(m, months)
            d = re.sub(r"[^\d\-]", "", days).split("-")[-1]
            if m and d:
                out.append(pd.Timestamp(y, m, int(d)))
    s = pd.Series(sorted(set(out)), name="date")
    s.to_frame().to_parquet(p)
    return s


def update_earnings(tickers: list[str]) -> pd.DataFrame:
    """Verilen hisselerin bilanço kayıtlarını yeniler; diğerlerine dokunmaz."""
    p = DATA / "earnings.parquet"
    have = pd.read_parquet(p) if p.exists() else pd.DataFrame(columns=["ticker"])
    if not tickers:
        return have
    tmp = DATA / "_earn_tmp.parquet"
    rows = []
    for t in tickers:
        try:
            e = yf.Ticker(t).get_earnings_dates(limit=12)
        except Exception:
            continue
        if e is None or e.empty:
            continue
        e = e.reset_index()
        e.columns = ["ts", "eps_est", "eps_act", "surprise"][: len(e.columns)] + list(e.columns[4:])
        e["ts"] = pd.to_datetime(e["ts"], utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
        e["ticker"] = t
        rows.append(e[["ticker", "ts", "eps_est", "eps_act", "surprise"]])
        time.sleep(0.3)
    if not rows:
        return have
    new = pd.concat(rows, ignore_index=True)
    # yeni gelen kayıtlar aynı (hisse, tarih) için eskisinin yerine geçer
    key = lambda d: d["ticker"] + "|" + d["ts"].dt.strftime("%Y-%m-%d")  # noqa: E731
    have = have[~key(have).isin(set(key(new)))]
    df = pd.concat([have, new], ignore_index=True).sort_values(["ticker", "ts"])
    df.to_parquet(tmp)
    tmp.replace(p)
    return df
