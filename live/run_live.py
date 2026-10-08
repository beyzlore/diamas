"""Günlük takip: veriyi güncelle -> sanal portföyü yeniden hesapla -> panoyu (Diamas) üret -> bildirim gönder.

Kullanım:
  .venv/bin/python live/run_live.py            # tam güncelleme (internetten veri indirir)
  .venv/bin/python live/run_live.py --offline  # sadece mevcut veriyle panoyu yeniden üret
"""
from __future__ import annotations

import html
import json
import pickle
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "live"))
import backtest as B  # noqa: E402
import data as D  # noqa: E402
import engine as E  # noqa: E402
import signals as SG  # noqa: E402

OUT = ROOT / "rapor" / "takip.html"
LOG = ROOT / "logs" / "live.log"


def log(msg):
    LOG.parent.mkdir(exist_ok=True)
    line = f"[{datetime.now():%Y-%m-%d %H:%M}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def notify(title, text):
    if sys.platform != "darwin":
        return
    t = text.replace('"', "'")
    subprocess.run(["osascript", "-e", f'display notification "{t}" with title "{title}" sound name "Glass"'],
                   check=False)


def update_data():
    sp = D.sp500_list(force=datetime.now().weekday() == 5)          # cumartesi listeyi yenile
    tick = sorted(set(sp["ticker"]) | set(D.ETFS))
    log(f"günlük fiyatlar indiriliyor ({len(tick)} sembol)")
    D.daily_prices(tick, force=True)
    D.vix(force=True)
    if datetime.now().weekday() == 5:
        D.fomc_dates(force=True)
    # bilançosu yakın/yeni olan hisselerin kayıtlarını yenile (sürpriz verisi için)
    earn = pd.read_parquet(D.DATA / "earnings.parquet")
    now = pd.Timestamp.now()
    near = earn[(earn["ts"] >= now - pd.Timedelta(days=6)) & (earn["ts"] <= now + pd.Timedelta(days=21))]
    stale = earn.groupby("ticker")["ts"].max()
    no_future = stale[stale < now].index                              # gelecek tarihi bilinmeyenler
    todo = sorted(set(near["ticker"]) | set(no_future))
    log(f"bilanço kayıtları yenileniyor ({len(todo)} hisse)")
    D.update_earnings(todo)


# ---------------------------------------------------------------- yardımcılar
def esc(x):
    return html.escape(str(x))


def money(x):
    return f"${x:,.0f}".replace(",", ".")


def pct(x, d=1, sign=True):
    if x is None or not np.isfinite(x):
        return "–"
    prefix = "−" if x < 0 else ("+" if sign and x > 0 else "")
    return f"{prefix}%{abs(x) * 100:.{d}f}"


def _sanitize(h: str) -> str:
    """Bulutta yazılan analiz metnini herkese açık sayfaya koymadan önce temizler:
    yalnızca basit biçim etiketleri kalır, betik/olay/javascript bağlantıları atılır."""
    import re
    h = re.sub(r"(?is)<(script|style|iframe|object|embed|form)[^>]*>.*?</\1>", "", h)
    h = re.sub(r"(?is)<(script|style|iframe|object|embed|form|meta|link|img|svg)[^>]*/?>", "", h)
    h = re.sub(r"(?i)\son\w+\s*=\s*(\"[^\"]*\"|'[^']*'|[^\s>]+)", "", h)
    h = re.sub(r"(?i)href\s*=\s*([\"'])\s*javascript:[^\"']*\1", 'href="#"', h)
    return h


def strategy_health(summary_path: Path):
    """Backtest serisinin son dönem performansı ve geçmişteki en kötü düşüşe göre durum."""
    if not summary_path.exists():
        return {}
    S = pickle.load(open(summary_path, "rb"))["summary"]
    out = {}
    for sleeve, key in [("trend", "trend_spy"), ("pead", "pead_long"), ("rsi2", "rsi2_noearn")]:
        s = S.get(key)
        if not s:
            continue
        d = s["daily"]
        d = d.loc[: d.ne(0).iloc[::-1].idxmax()] if d.ne(0).any() else d
        eq = (1 + d).cumprod()
        dd_now = eq.iloc[-1] / eq.cummax().iloc[-1] - 1
        dd_hist = (eq / eq.cummax() - 1).loc[:"2023-12-31"].min()
        r3 = eq.iloc[-1] / eq.iloc[-63] - 1
        r12 = eq.iloc[-1] / eq.iloc[-252] - 1
        status = "normal"
        if dd_now < dd_hist:
            status = "alarm"
        elif dd_now < 0.6 * dd_hist:
            status = "dikkat"
        out[sleeve] = dict(r3=r3, r12=r12, dd_now=dd_now, dd_hist=dd_hist, status=status, verdict=s["verdict"],
                           valid_sr=s["valid"]["sharpe"], locked_sr=s["locked"]["sharpe"], asof=d.index[-1])
    return out


def build(st, pending, ctx, watch, up_e, asof, health, fomc, cs=(), acc=()):
    eq = pd.DataFrame(st.equity)
    total = eq["toplam"].iloc[-1] if len(eq) else E.CAPITAL
    spy_v = eq["SPY"].iloc[-1] if len(eq) else E.CAPITAL
    ret = total / E.CAPITAL - 1
    spy_ret = spy_v / E.CAPITAL - 1
    days = len(eq)
    next_fomc = next((d for d in sorted(fomc) if d > asof), None)

    # yarının emirleri
    def order_rows():
        if not pending:
            return "<div class='empty'>Yarın için yeni işlem yok. Sistem mevcut pozisyonları tutmaya devam ediyor.</div>"
        rows = []
        for o in sorted(pending, key=lambda o: (o.side != "SAT", o.sleeve)):
            cls = "buy" if o.side == "AL" else "sell"
            rows.append(f"<li class='ord'><span class='pill {cls}'>{o.side}</span><b>{esc(o.ticker)}</b>"
                        f"<span class='muted'>{esc(E.SLEEVES[o.sleeve]['title'])}</span><span class='why'>{esc(o.reason)}</span></li>")
        return "<ul class='orders'>" + "".join(rows) + "</ul>"

    pos_rows = []
    for ps in sorted(st.positions, key=lambda x: (x.sleeve, x.ticker)):
        last = st_last_price.get(ps.ticker, np.nan)
        val = ps.shares * last
        r = val / ps.cost_basis - 1
        due = {"trend": "trend bozulunca", "pead": f"{E_bday(ps.exit_due)} açılışı",
               "rsi2": f"5g ort. üstü kapanışta, en geç {E_bday(ps.exit_due)}"}[ps.sleeve]
        pos_rows.append(f"<tr><td class='name'>{esc(ps.ticker)}</td><td>{esc(E.SLEEVES[ps.sleeve]['title'])}</td>"
                        f"<td class='n'>{E_dates[ps.entry_i]:%d.%m}</td><td class='n'>{ps.entry_px:,.2f}</td>"
                        f"<td class='n'>{last:,.2f}</td><td class='n'>{money(val)}</td>"
                        f"<td class='n {'pos' if r > 0 else 'neg'}'>{pct(r, 1)}</td><td>{esc(due)}</td></tr>")
    pos_html = ("<div class='scroll'><table><thead><tr><th>Hisse</th><th>Strateji</th><th class='n'>Giriş</th>"
                "<th class='n'>Giriş fiyatı</th><th class='n'>Son fiyat</th><th class='n'>Değer</th><th class='n'>Kâr/Zarar</th>"
                "<th>Planlanan çıkış</th></tr></thead><tbody>" + "".join(pos_rows) + "</tbody></table></div>") if pos_rows \
        else "<div class='empty'>Şu an açık pozisyon yok.</div>"

    closed = st.closed[-25:][::-1]
    cl_html = ("<div class='scroll'><table><thead><tr><th>Hisse</th><th>Strateji</th><th class='n'>Giriş</th><th class='n'>Çıkış</th>"
               "<th class='n'>Getiri</th><th class='n'>Kâr/Zarar</th><th>Neden</th></tr></thead><tbody>" + "".join(
                   f"<tr><td class='name'>{esc(c['ticker'])}</td><td>{esc(E.SLEEVES[c['sleeve']]['title'])}</td>"
                   f"<td class='n'>{c['entry']:%d.%m}</td><td class='n'>{c['exit']:%d.%m}</td>"
                   f"<td class='n {'pos' if c['ret'] > 0 else 'neg'}'>{pct(c['ret'], 1)}</td><td class='n'>{money(c['pnl'])}</td>"
                   f"<td class='muted'>{esc(c['reason'])}</td></tr>" for c in closed) + "</tbody></table></div>") if closed \
        else "<div class='empty'>Henüz kapanan işlem yok.</div>"

    # strateji kartları
    cards = []
    for s, meta in E.SLEEVES.items():
        v = eq[s].iloc[-1] if len(eq) else E.CAPITAL * meta["share"]
        r = v / (E.CAPITAL * meta["share"]) - 1
        h = health.get(s, {})
        stt = h.get("status", "normal")
        badge = {"normal": ("good", "✓", "Normal"), "dikkat": ("warn", "!", "Dikkat"), "alarm": ("bad", "✕", "Alarm")}[stt]
        npos = sum(1 for ps in st.positions if ps.sleeve == s)
        cards.append(f"""<div class='scard'>
<div class='shead'><b>{esc(meta['title'])}</b><span class='badge {badge[0]}'><span aria-hidden='true'>{badge[1]}</span> {badge[2]}</span></div>
<p class='muted small'>{esc(meta['rule'])}</p>
<div class='srow'><div><span>Sanal bakiye</span><b>{money(v)}</b></div><div><span>Başlangıçtan beri</span><b>{pct(r, 2)}</b></div>
<div><span>Açık pozisyon</span><b>{npos} / {meta['k']}</b></div></div>
<div class='srow sub'><div><span>Geçmiş test · son 3 ay</span><b>{pct(h.get('r3', np.nan), 1)}</b></div>
<div><span>son 12 ay</span><b>{pct(h.get('r12', np.nan), 1)}</b></div>
<div><span>Şu anki düşüş / geçmişteki en kötü</span><b>{pct(h.get('dd_now', np.nan), 1)} / {pct(h.get('dd_hist', np.nan), 0)}</b></div></div>
</div>""")

    w_html = "".join(f"<tr><td class='name'>{esc(w['ticker'])}</td><td class='n'>{w['rsi2']:.1f}</td><td class='n'>{w['close']:,.2f}</td>"
                     f"<td>{'Bilanço yakın: girilmez' if w['earn_soon'] else ('Sinyal (RSI &lt; 5)' if w['rsi2'] < 5 else 'Yaklaşıyor')}</td></tr>"
                     for w in watch[:15])
    w_html = (f"<div class='scroll'><table><thead><tr><th>Hisse</th><th class='n'>RSI(2)</th><th class='n'>Kapanış</th><th>Durum</th></tr></thead>"
              f"<tbody>{w_html}</tbody></table></div>") if watch else "<div class='empty'>Sinyale yakın hisse yok.</div>"
    e_html = "".join(f"<tr><td class='name'>{esc(r.ticker)}</td><td class='n'>{r.ts:%d.%m %H:%M}</td>"
                     f"<td>{'kapanış sonrası' if r.ts.hour >= 16 else 'açılış öncesi'}</td>"
                     f"<td class='n'>{'' if not np.isfinite(r.eps_est) else f'{r.eps_est:.2f}'}</td></tr>" for r in up_e.itertuples())
    e_html = (f"<div class='scroll'><table><thead><tr><th>Hisse</th><th class='n'>Tarih (NY saati)</th><th>Zaman</th><th class='n'>Beklenen EPS</th></tr></thead>"
              f"<tbody>{e_html}</tbody></table></div>") if len(up_e) else "<div class='empty'>Önümüzdeki 10 günde takip listesinde bilanço yok.</div>"

    chart = {"dates": [d.strftime("%Y-%m-%d") for d in eq["date"]] if len(eq) else [],
             "port": [round(x, 2) for x in eq["toplam"]] if len(eq) else [],
             "spy": [round(x, 2) for x in eq["SPY"]] if len(eq) else []}

    SLT = {k: v["title"] for k, v in E.SLEEVES.items()}

    def sig_card(c):
        st, ss = c["stats"], c["stats_same"]
        same = (f"<p class='small muted'>Bu hissede aynı kural geçmişte {ss['n']} kez sinyal verdi: kazanma {ss['win'] * 100:.0f}%, "
                f"ortalama {pct(ss['mean'], 1)}.</p>") if ss else ""
        bar = (f"<div class='scen'><div class='up' style='flex:{st['up']:.3f}'>Yükseliş {st['up'] * 100:.0f}%</div>"
               f"<div class='flat' style='flex:{st['flat']:.3f}'>Yatay {st['flat'] * 100:.0f}%</div>"
               f"<div class='down' style='flex:{st['down']:.3f}'>Düşüş {st['down'] * 100:.0f}%</div></div>")
        return f"""<div class='sig'>
<div class='shead'><div><b class='tk'>{esc(c['ticker'])}</b> <span class='muted small'>{esc(SLT[c['sleeve']])}</span></div>
<span class='badge {"good" if c["conf"].startswith("Orta") else "warn"}'>Güven: {esc(c['conf'])}</span></div>
<p class='small'><b>Gerekçe:</b> {esc(c['reason'])}</p>
<dl class='kv'>
<dt>Giriş</dt><dd>{esc(c['entry'])}</dd>
<dt>Çıkış koşulu</dt><dd>{esc(c['exit_rule'])}</dd>
<dt>Hedef (geçmiş ortalama kazanç)</dt><dd>{esc(c['target'])}</dd>
<dt>Olumsuz senaryo / stop</dt><dd>{esc(c['stop'])}</dd>
<dt>Risk / ödül (ort. kazanç ÷ ort. kayıp)</dt><dd>{'–' if not np.isfinite(c['rr']) else f"{c['rr']:.2f}"}</dd>
<dt>Model portföyde pay</dt><dd>sermayenin %{c['model_size'] * 100:.1f}'i</dd>
</dl>
<p class='small muted' style='margin:8px 0 4px'>Geçmişteki {st['n']} benzer işlemin dağılımı (maliyet sonrası):</p>{bar}
{same}
<p class='small'><b>Senaryoyu bozabilecekler:</b> {esc(c['breakers'])}</p>
</div>"""

    sig_html = "".join(sig_card(c) for c in cs) if cs else \
        "<div class='panel empty'>Bugün yeni alım sinyali yok. Uygun koşul oluşmadığında sistem işlem önermez.</div>"
    acc_rows = "".join(
        f"<tr><td class='name'>{esc(SLT[a['sleeve']])}</td><td class='n'>{a['n']}</td>"
        f"<td class='n'>{pct(a['live_win'], 0, sign=False) if a['n'] else '–'}</td>"
        f"<td class='n'>{pct(a['exp_win'], 0, sign=False)}</td><td class='n'>{pct(a['live_mean'], 2)}</td><td class='n'>{pct(a['exp_mean'], 2)}</td></tr>"
        for a in acc)
    acc_html = ("<div class='scroll'><table><thead><tr><th>Strateji</th><th class='n'>Kapanan işlem</th><th class='n'>Gerçekleşen kazanma</th>"
                "<th class='n'>Beklenen kazanma</th><th class='n'>Gerçekleşen ort.</th><th class='n'>Beklenen ort.</th></tr></thead>"
                f"<tbody>{acc_rows}</tbody></table></div>")
    fs = SG.forecast_scores(SPY_CLOSE)
    if fs is None or fs["n"] == 0:
        pend = 0 if fs is None else fs["pending"]
        acc_html += (f"<p class='small muted'>Günlük analizdeki SPY senaryo tahminleri: {pend} tahmin kaydedildi, "
                     "sonuçları 5 işlem günü sonra ölçülmeye başlanacak.</p>")
    else:
        acc_html += (f"<p class='small'><b>Günlük analiz tahmin isabeti (SPY, 5 gün):</b> {fs['n']} tahmin, en olası senaryo "
                     f"%{fs['hit'] * 100:.0f} doğru çıktı. Brier puanı {fs['brier']:.3f}; düşük olan daha iyidir, "
                     f"hep eşit olasılık veren tahmin 0,667 alır.</p>")
    brief_path = ROOT / "results" / "briefing.html"
    brief_html = _sanitize(brief_path.read_text()) if brief_path.exists() else \
        "<div class='empty'>Günlük haber ve senaryo analizi henüz etkinleştirilmedi.</div>"

    regime = "Yükseliş (boğa)" if ctx["spy_close"] > ctx["spy_sma200"] else "Düşüş (ayı)"
    vix_lbl = "sakin" if ctx["vix"] < 15 else ("normal" if ctx["vix"] < 25 else "korku")
    tpl = (ROOT / "live" / "dashboard_template.html").read_text()
    repl = {
        "{{ASOF}}": asof.strftime("%d.%m.%Y"), "{{UPDATED}}": datetime.now().strftime("%d.%m.%Y %H:%M"),
        "{{TOTAL}}": money(total), "{{RET}}": pct(ret, 2), "{{SPY_RET}}": pct(spy_ret, 2),
        "{{DIFF}}": pct(ret - spy_ret, 2), "{{NPOS}}": str(len(st.positions)), "{{DAYS}}": str(days),
        "{{REGIME}}": regime, "{{REGIME_CLS}}": "good" if "boğa" in regime else "bad",
        "{{SPY_CLOSE}}": f"{ctx['spy_close']:,.2f}", "{{SPY_VS200}}": pct(ctx["spy_close"] / ctx["spy_sma200"] - 1, 1),
        "{{VIX}}": f"{ctx['vix']:.1f}", "{{VIX_LBL}}": vix_lbl, "{{VIX_CLS}}": "good" if ctx["vix"] < 25 else ("warn" if ctx["vix"] < 30 else "bad"),
        "{{TREND}}": "SPY tut" if ctx["trend_up"] else "Nakit", "{{RSI_OK}}": "Açık" if ctx["market_ok"] else "Kapalı (yeni alım yok)",
        "{{ORDERS}}": order_rows(), "{{N_ORDERS}}": str(len(pending)), "{{POSITIONS}}": pos_html, "{{CLOSED}}": cl_html,
        "{{CARDS}}": "".join(cards), "{{WATCH}}": w_html, "{{EARN}}": e_html,
        "{{FOMC}}": next_fomc.strftime("%d.%m.%Y") if next_fomc is not None else "–",
        "{{CHART}}": json.dumps(chart), "{{SIGNALS}}": sig_html, "{{N_SIG}}": str(len(cs)),
        "{{ACCURACY}}": acc_html, "{{BRIEFING}}": brief_html,
    }
    for k, v in repl.items():
        tpl = tpl.replace(k, v)
    OUT.write_text(tpl)
    return total, ret, spy_ret


def main():
    global st_last_price, E_dates
    offline = "--offline" in sys.argv
    try:
        if not offline:
            update_data()
        daily = pd.read_parquet(D.DATA / "daily.parquet")
        p = B.load_panel(daily)
        earn = pd.read_parquet(D.DATA / "earnings.parquet")
        vix = D.vix()
        st, pending, ctx, watch, up_e, asof = E.run(p, earn, vix)
        st_last_price = {t: p.c[t].dropna().iloc[-1] for t in {ps.ticker for ps in st.positions}}
        E_dates = p.dates
        global SPY_CLOSE
        SPY_CLOSE = p.c["SPY"]
        health = strategy_health(ROOT / "results" / "daily_final.pkl")
        hist = SG.load_hist()
        cs = SG.cards(pending, p, hist, ctx)
        SG.log_signals(cs, asof)
        acc = SG.accuracy(st.closed, hist)
        total, ret, spy_ret = build(st, pending, ctx, watch, up_e, asof, health, D.fomc_dates(), cs, acc)
        buys = [o.ticker for o in pending if o.side == "AL"]
        sells = [o.ticker for o in pending if o.side == "SAT"]
        msg = f"Portföy {money(total)} ({pct(ret, 2)}). Yarın: {len(buys)} alım, {len(sells)} satış."
        if buys:
            msg += " AL: " + ", ".join(buys[:5])
        log(f"tamam ({asof:%Y-%m-%d}) {msg}")
        if not offline:
            notify("Diamas güncellendi", msg)
    except Exception as ex:  # noqa: BLE001
        log("HATA: " + "".join(traceback.format_exception(ex))[-1500:])
        notify("Diamas HATASI", str(ex)[:120])
        raise


def E_bday(i):
    if i is None:
        return "–"
    if i < len(E_dates):
        return E_dates[i].strftime("%d.%m")
    return (E_dates[-1] + pd.offsets.BDay(i - len(E_dates) + 1)).strftime("%d.%m")


st_last_price, E_dates, SPY_CLOSE = {}, None, None

if __name__ == "__main__":
    main()
