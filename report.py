"""Tüm sonuçlardan tek sayfalık HTML rapor üretir: rapor/rapor.html"""
import html
import json
import math
import pickle
import sys
from pathlib import Path

sys.path.insert(0, "src")
import numpy as np
import pandas as pd

import research as R

ROUND = sys.argv[1] if len(sys.argv) > 1 else "final"
S = pickle.load(open(f"results/daily_{ROUND}.pkl", "rb"))
EV = pickle.load(open("results/events.pkl", "rb"))
INTRA = pickle.load(open("results/intraday_summary.pkl", "rb")) if Path("results/intraday_summary.pkl").exists() else None
U = pd.read_parquet("data/universe.parquet")
summ = {k: v for k, v in S["summary"].items() if v}
bench = S["bench"]
Path("rapor").mkdir(exist_ok=True)


def esc(x):
    return html.escape(str(x))


def pct(x, d=1):
    return "–" if x is None or not np.isfinite(x) else f"%{x * 100:.{d}f}".replace("%-", "−%")


def num(x, d=2):
    return "–" if x is None or not np.isfinite(x) else f"{x:.{d}f}".replace("-", "−")


def badge(v):
    m = {"GEÇTİ": ("good", "✓"), "ŞÜPHELİ": ("warn", "!"), "BAŞARISIZ": ("bad", "✕")}
    c, i = m.get(v, ("warn", "?"))
    return f'<span class="badge {c}"><span aria-hidden="true">{i}</span> {esc(v)}</span>'


def prm_txt(p):
    names = {"n": "ortalama günü", "th": "RSI eşiği", "mh": "en uzun tutma", "g": "boşluk (ATR katı)",
             "lb": "geriye bakış (ay)", "top": "seçilen sayı", "f": "trend filtresi", "entry": "giriş",
             "b": "ay sonundan önce gün", "a": "yeni ayda gün", "s": "min. sürpriz %", "h": "tutma günü",
             "vmax": "en yüksek VIX", "r": "VIX oranı", "tv": "hedef oynaklık"}
    vals = {True: "açık", False: "kapalı", "prev_close": "önceki kapanış", "open": "karar günü açılış"}
    return ", ".join(f"{names.get(k, k)} = {vals.get(v, v) if not isinstance(v, (int, float)) or isinstance(v, bool) else v}"
                     for k, v in p.items())


STOCK_STRATS = {"rsi2_stk", "rsi2_regime", "rsi2_noearn", "xsmom", "pead_long", "pead_short", "pead_regime",
                "gap_fade_down", "gap_go_up", "gap_fade_up"}
order = {"GEÇTİ": 0, "ŞÜPHELİ": 1, "BAŞARISIZ": 2}
items = sorted(summ.values(), key=lambda s: (order[s["verdict"]], -np.nan_to_num(s["valid"]["sharpe"], nan=-9)))
passed = [s for s in items if s["verdict"] == "GEÇTİ"]
susp = [s for s in items if s["verdict"] == "ŞÜPHELİ"]
failed = [s for s in items if s["verdict"] == "BAŞARISIZ"]
n_trials = S["n_trials"]


def bstats(d, a, b):
    d = d.loc[a:b]
    eq = (1 + d).cumprod()
    yrs = len(d) / 252
    return dict(ret=eq.iloc[-1] ** (1 / yrs) - 1, sr=d.mean() / d.std() * math.sqrt(252),
                dd=(eq / eq.cummax() - 1).min())


spy = bench["SPY al-tut"]
spy_v, spy_l = bstats(spy, *R.VALID), bstats(spy, *R.LOCKED)
ew = bench["En likit 100 hisse eşit ağırlık"]
ew_v, ew_l = bstats(ew, *R.VALID), bstats(ew, *R.LOCKED)

# ---------------------------------------------------------------- birleşik portföy (yalnızca GEÇTİ olanlar)
combo = None
if passed:
    dfc = pd.concat([s["daily"] for s in passed], axis=1).fillna(0)
    combo_d = dfc.mean(axis=1)
    combo = dict(daily=combo_d, valid=bstats(combo_d, *R.VALID), locked=bstats(combo_d, *R.LOCKED),
                 train=bstats(combo_d, *R.TRAIN))

# ---------------------------------------------------------------- grafik verisi
chart_series = [("SPY al-tut (karşılaştırma)", spy, "bench")]
for i, s in enumerate(passed[:3]):
    chart_series.append((s["title"], s["daily"], f"s{i + 1}"))
if combo is not None and len(passed) > 1:
    chart_series.append(("Geçen stratejilerin eşit karışımı", combo["daily"], "s4"))
idx = spy.loc["2005-01-01":].index
weekly = idx[::5]
chart = {"dates": [d.strftime("%Y-%m-%d") for d in weekly], "series": []}
for name, d, cls in chart_series:
    eq = (1 + d.reindex(idx).fillna(0)).cumprod()
    chart["series"].append({"name": name, "cls": cls, "values": [round(float(x), 4) for x in eq.reindex(weekly)]})
chart["bands"] = [{"from": "2005-01-01", "to": R.TRAIN[1], "label": "Eğitim"},
                  {"from": R.VALID[0], "to": R.VALID[1], "label": "Doğrulama"},
                  {"from": R.LOCKED[0], "to": chart["dates"][-1], "label": "Kilitli test"}]

# ---------------------------------------------------------------- tablolar
def score_row(s):
    v, l = s["valid"], s["locked"]
    return f"""<tr>
<td class="name"><a href="#{s['name']}">{esc(s['title'])}</a></td><td>{badge(s['verdict'])}</td>
<td class="n">{v['işlem_sayısı']}</td><td class="n">{pct(v['kazanma_oranı'], 0)}</td>
<td class="n">{pct(v['ort_kâr'], 2)}</td><td class="n">{pct(v['ort_zarar'], 2)}</td>
<td class="n">{pct(v['beklenen_getiri'], 2)}</td><td class="n">{pct(v['maks_düşüş'], 0)}</td>
<td class="n">{num(v['sharpe'])}</td><td class="n">{pct(v['yıllık_getiri'], 1)}</td>
<td class="n lk">{num(l['sharpe'])}</td><td class="n lk">{pct(l['yıllık_getiri'], 1)}</td><td class="n lk">{pct(l['maks_düşüş'], 0)}</td></tr>"""


def regime_table(reg):
    rows = []
    for r in reg.itertuples():
        e = r.beklenen_getiri
        cls = "pos" if np.isfinite(e) and e > 0 else ("neg" if np.isfinite(e) else "")
        rows.append(f"<tr><td>{esc(r.koşul)}</td><td class='n'>{r.işlem}</td><td class='n'>{pct(r.kazanma_oranı, 0)}</td>"
                    f"<td class='n {cls}'>{pct(e, 2)}</td><td class='n'>{num(r.sharpe)}</td></tr>")
    return ("<table class='mini'><thead><tr><th>Piyasa koşulu</th><th>İşlem</th><th>Kazanma</th>"
            "<th>İşlem başı ort.</th><th>Sharpe</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>")


def year_strip(yr):
    cells = []
    for y, v in yr.items():
        cls = "pos" if v > 0 else "neg"
        cells.append(f"<div class='yr {cls}' title='{y}: {pct(v, 1)}'><span>{str(y)[2:]}</span><b>{pct(v, 0)}</b></div>")
    return "<div class='years'>" + "".join(cells) + "</div>"


def card(s):
    checks = "".join(f"<li class='{'ok' if v else 'no'}'><span aria-hidden='true'>{'✓' if v else '✕'}</span> {esc(k)}</li>"
                     for k, v in s["checks"].items())
    ci = s["ci"]
    cost = s["cost"]
    ww = s["worst_12m_win"]
    weak = ", ".join(s["weak_regimes"]) if s["weak_regimes"] else "Belirgin bir zayıf koşul bulunmadı"
    badyrs = ", ".join(map(str, s["bad_years"])) if s["bad_years"] else "yok"
    grid_rows = "".join(f"<tr><td>{esc(prm_txt(p))}</td><td class='n'>{num(a)}</td><td class='n'>{num(b)}</td></tr>"
                        for p, a, b in s["grid"])
    v, t = s["valid"], s["train"]
    surv = ""
    if s["name"] in STOCK_STRATS:
        surv = (f"<p class='callout warn' style='margin:10px 0'><b>Yanlılık uyarısı:</b> Bu strateji hisse senetlerinde "
                f"çalışıyor ve evren bugünkü S&amp;P 500 üyelerinden kuruldu. İflas eden veya endeksten düşen şirketler veride yok. "
                f"Bu yüzden sonuç olduğundan iyi görünebilir. Ölçüt olarak aynı evrenin eşit ağırlıklı al-tut'una bakın: "
                f"doğrulamada Sharpe {num(ew_v['sr'])} / yıllık {pct(ew_v['ret'], 1)}, kilitli dönemde Sharpe {num(ew_l['sr'])} / yıllık {pct(ew_l['ret'], 1)}.</p>")
    return f"""
<details class="card" id="{s['name']}">
<summary><span class="ttl">{esc(s['title'])}</span> {badge(s['verdict'])}
<span class="sub">Doğrulama Sharpe {num(v['sharpe'])} · {v['işlem_sayısı']} işlem</span></summary>
<div class="body">
{surv}<p class="rule"><b>Kural:</b> {esc(s['rule'])}</p>
<p><b>Eğitim döneminde seçilen ayar:</b> {esc(prm_txt(s['params']))} (aynı anda en fazla {s['k']} pozisyon; {s['n_grid']} ayar denendi)</p>
<div class="two">
<div><h4>Kontroller</h4><ul class="checks">{checks}</ul></div>
<div><h4>Sayılar</h4><dl class="kv">
<dt>Eğitim Sharpe → Doğrulama Sharpe</dt><dd>{num(t['sharpe'])} → {num(v['sharpe'])}</dd>
<dt>İleriye yürüyen test Sharpe</dt><dd>{num(s['wf_sharpe'])}</dd>
<dt>İşlem başı getiri %95 aralığı</dt><dd>{pct(ci[0], 2)} … {pct(ci[1], 2)}</dd>
<dt>Maliyet 0× / 2× / 3× Sharpe</dt><dd>{num(cost.get(0.0))} / {num(cost.get(2.0))} / {num(cost.get(3.0))}</dd>
<dt>Şans düzeltmeli Sharpe (DSR)</dt><dd>{pct(s['dsr'], 0)}</dd>
<dt>Ayarların doğrulamada kârlı olan payı</dt><dd>{pct(s['frac_valid_pos'], 0)}</dd>
<dt>En kötü 12 ay</dt><dd>{pct(s['worst_12m'], 1)}{(' (' + ww[0].strftime('%m/%Y') + '–' + ww[1].strftime('%m/%Y') + ')') if ww else ''}</dd>
<dt>Zirveyi geri almak için en uzun bekleyiş</dt><dd>{s['longest_dd_days']} işlem günü</dd>
</dl></div></div>
<h4>Ne zaman çalışmıyor?</h4>
<p>Zararlı çıktığı koşullar: <b>{esc(weak)}</b>. Zararlı yıllar (2005–2023): <b>{esc(badyrs)}</b>.</p>
{regime_table(s['regimes'])}
<h4>Yıllara göre getiri (2005–2023)</h4>{year_strip(s['yearly'])}
<details class="inner"><summary>Denenen tüm ayarlar (eğitim → doğrulama Sharpe)</summary>
<table class="mini"><thead><tr><th>Ayar</th><th>Eğitim</th><th>Doğrulama</th></tr></thead><tbody>{grid_rows}</tbody></table></details>
<p class="lockline"><b>Kilitli test (2024 – bugün, en sonda bir kez açıldı):</b> {s['locked']['işlem_sayısı']} işlem, Sharpe {num(s['locked']['sharpe'])}, yıllık {pct(s['locked']['yıllık_getiri'], 1)}, en büyük düşüş {pct(s['locked']['maks_düşüş'], 0)}, işlem başı ort. {pct(s['locked']['beklenen_getiri'], 2)}</p>
</div></details>"""


uni_rows = "".join(
    f"<tr><td class='name'>{esc(t)}</td><td class='n'>${r.price:,.0f}</td><td class='n'>{r.adv_musd:,.0f}</td>"
    f"<td class='n'>{pct(r.rv_annual, 0)}</td><td class='n'>{pct(r.atr_pct, 1)}</td><td class='n'>{num(r.spread_bps, 1)}</td>"
    f"<td class='n'>{num(r.spread_ar_bps, 0)}</td><td class='n'>{num(r.gap_abs_bps / 100, 2)}</td></tr>"
    for t, r in U.head(25).iterrows())

st = EV["surprise_tbl"]
surp_rows = "".join(f"<tr><td>{esc(i)}</td><td class='n'>{int(r.adet)}</td><td class='n'>{pct(r.ort_tepki, 2)}</td>"
                    f"<td class='n'>{pct(r.yükselme_oranı, 0)}</td></tr>" for i, r in st.iterrows())

# ---------------------------------------------------------------- gün içi
intra_html = ""
if INTRA:
    rows = []
    for key, s in INTRA.items():
        rows.append(f"<tr><td class='name'>{esc(s['title'])}</td><td>{badge(s['verdict'])}</td><td class='n'>{s['days']}</td>"
                    f"<td>{esc(s['params_txt'])}</td>"
                    f"<td class='n'>{s['train']['işlem_sayısı']}</td><td class='n'>{num(s['train']['sharpe'])}</td>"
                    f"<td class='n'>{s['valid']['işlem_sayısı']}</td><td class='n'>{pct(s['valid']['kazanma_oranı'], 0)}</td>"
                    f"<td class='n'>{pct(s['valid']['beklenen_getiri'], 3)}</td><td class='n'>{num(s['valid']['sharpe'])}</td>"
                    f"<td class='n lk'>{num(s['locked']['sharpe'])}</td><td class='n lk'>{pct(s['locked']['beklenen_getiri'], 3)}</td></tr>")
    intra_html = "".join(rows)

# ---------------------------------------------------------------- pozisyon büyüklüğü örneği
def kelly(s):
    tr = s["trades"]
    v = tr[(tr["date"] >= R.VALID[0]) & (tr["date"] <= R.VALID[1])]["net"]
    if len(v) < 20:
        return np.nan
    w = (v > 0).mean()
    aw, al = v[v > 0].mean(), -v[v <= 0].mean()
    if not al or not np.isfinite(al):
        return np.nan
    b = aw / al
    return w - (1 - w) / b


def g(k):
    return summ[k]


def L(k, f="sharpe"):
    return g(k)["locked"][f]


best_locked = max(summ.values(), key=lambda s: np.nan_to_num(s["locked"]["sharpe"], nan=-9))
BOTTOM = f"""
<p><b>Kısa sonuç:</b> {len(summ)} günlük strateji ve {n_trials} ayar denendi. Gün içi ORB ayrıca test edildi. <b>"Kesin para makinesi" çıkmadı.</b> Testleri geçen stratejilerin avantajı, piyasadan çok daha fazla kazandırmaları değil, <b>daha az kaybettirmeleri</b>.</p>
<p><b>1. En sağlam bulgu, trend takibi:</b> SPY 100 günlük ortalamasının üstündeyken tutulur, altına inince nakite geçilir. Tüm kontrollerden geçti; denenen her ayar görmediği dönemde kârlıydı. Kilitli testte yıllık {pct(L('trend_spy', 'yıllık_getiri'), 1)} getirdi, SPY al-tut {pct(spy_l['ret'], 1)} getirdi. En büyük düşüş ise {pct(L('trend_spy', 'maks_düşüş'), 0)} oldu, SPY'da {pct(spy_l['dd'], 0)}. Yani biraz daha az kazanç, çok daha az stres.</p>
<p><b>2. Bilanço sonrası sürüklenme gerçek bir etki:</b> Beklentiyi aşan ve ilk gün yükselen hisse sonraki 10 günde ortalama yükselmeye devam ediyor. Doğrulamada geçti, kilitli testte de kârlı kaldı (Sharpe {num(L('pead_long'))}). Ama o dönemde SPY'ın gerisinde kaldı ve ayı piyasalarında zarar ediyor.</p>
<p><b>3. En umut verici ama henüz kanıtlanmamış aday:</b> "RSI(2) dipten alım + kriz filtresi + bilanço kaçınma". Kilitli testte en iyi risk-getiri sonucunu verdi: Sharpe {num(L('rsi2_noearn'))}, yıllık {pct(L('rsi2_noearn', 'yıllık_getiri'), 0)}, en büyük düşüş {pct(L('rsi2_noearn', 'maks_düşüş'), 0)}. Yine de iki nedenle "şüpheli" sayılıyor. Hisse verisindeki hayatta kalma yanlılığı sonucu şişiriyor olabilir. Ayrıca bu kadar çok deneme yapınca şans payı yüksek. Sanal parayla canlı izlenmeye en çok değer aday bu.</p>
<p><b>4. Başarısız olanlar:</b> Açılış boşluğu (gap) stratejileri, Fed günü alımı, panik alımı, bilanço sonrası açığa satış ve saatlik ORB görmediği dönemde para kaybettirdi.</p>
<p><b>Dürüst gerçek:</b> Kilitli dönem (2024–2026) güçlü bir yükseliş piyasasıydı. Sadece SPY tutmak yıllık {pct(spy_l['ret'], 1)} getirdi (Sharpe {num(spy_l['sr'])}). Yeni başlayan biri için en zor rakip, hiçbir şey yapmadan endeks fonu tutmaktır.</p>
"""
NEXT = """
<li><b>Sanal parayla canlı test (en az 3 ay):</b> Testleri geçen ve en umut verici stratejiler Alpaca'nın ücretsiz sanal hesabında otomatik çalıştırılmalı. Asıl soru şu: Gerçek zamanlı sonuçlar geçmiş testlerle uyuşuyor mu?</li>
<li><b>Daha iyi veri:</b> Alpaca hesabı açılınca yıllarca geriye giden 1 dakikalık veri ücretsiz gelir. ORB ancak bu veriyle düzgün test edilebilir. Ücretsiz 5 dakikalık veride sıkı stop seviyeleri doğru ölçülemiyor; burada her belirsiz durumda en kötü senaryo varsayıldı.</li>
<li><b>Yanlılıksız hisse verisi:</b> İflas eden şirketleri de içeren veriyle (ör. Norgate, ücretli) hisse stratejileri tekrar sınanmalı.</li>
<li><b>Gerçek para kararı sende:</b> Sanal sonuçlar tutarlıysa küçük başla. Tek işlemde sermayenin %1'inden fazlasını riske atma, stratejileri karıştırarak riski dağıt.</li>
<li><b>Otomatik yenileme:</b> Bu araştırma her ay yeni verilerle kendiliğinden tekrar çalıştırılabilir. Bir strateji bozulmaya başlarsa erken fark edilir.</li>
"""
repl_extra = {"{{BOTTOM_LINE}}": BOTTOM, "{{NEXT_STEPS}}": NEXT}

TEMPLATE = Path("src/report_template.html").read_text()
out = TEMPLATE
repl = {
    "{{N_STRAT}}": str(len(summ)), "{{N_TRIALS}}": str(n_trials),
    "{{N_PASS}}": str(len(passed)), "{{N_SUSP}}": str(len(susp)), "{{N_FAIL}}": str(len(failed)),
    "{{PASS_NAMES}}": ", ".join(esc(s["title"]) for s in passed) or "hiçbiri",
    "{{SCORE_ROWS}}": "".join(score_row(s) for s in items),
    "{{CARDS_PASS}}": "".join(card(s) for s in passed + susp),
    "{{CARDS_FAIL}}": "".join(card(s) for s in failed),
    "{{CHART}}": json.dumps(chart),
    "{{UNI_ROWS}}": uni_rows, "{{N_UNI}}": str(len(U)),
    "{{SPY_V_SR}}": num(spy_v["sr"]), "{{SPY_V_RET}}": pct(spy_v["ret"], 1), "{{SPY_V_DD}}": pct(spy_v["dd"], 0),
    "{{SPY_L_SR}}": num(spy_l["sr"]), "{{SPY_L_RET}}": pct(spy_l["ret"], 1), "{{SPY_L_DD}}": pct(spy_l["dd"], 0),
    "{{EW_V_SR}}": num(ew_v["sr"]), "{{EW_V_RET}}": pct(ew_v["ret"], 1), "{{EW_V_DD}}": pct(ew_v["dd"], 0),
    "{{EW_L_SR}}": num(ew_l["sr"]), "{{EW_L_RET}}": pct(ew_l["ret"], 1), "{{EW_L_DD}}": pct(ew_l["dd"], 0),
    "{{COMBO_V}}": (f"Sharpe {num(combo['valid']['sr'])}, yıllık {pct(combo['valid']['ret'], 1)}, en büyük düşüş {pct(combo['valid']['dd'], 0)}" if combo else "–"),
    "{{COMBO_L}}": (f"Sharpe {num(combo['locked']['sr'])}, yıllık {pct(combo['locked']['ret'], 1)}, en büyük düşüş {pct(combo['locked']['dd'], 0)}" if combo else "–"),
    "{{EARN_N}}": f"{EV['earn_n']:,}".replace(",", "."), "{{EARN_ABS}}": pct(EV["earn_abs_mean"], 2),
    "{{NORM_ABS}}": pct(EV["norm_abs_mean"], 2), "{{EARN_BIG}}": pct(EV["earn_big_share"], 0),
    "{{NORM_BIG}}": pct(EV["norm_big_share"], 1), "{{TOP1_EARN}}": pct(EV["top1pct_share_earn"], 0),
    "{{FOMC_N}}": str(EV["fomc_n"]), "{{FOMC_ABS}}": pct(EV["fomc_abs_mean"], 2), "{{SPY_ABS}}": pct(EV["spy_abs_mean"], 2),
    "{{FOMC_RET}}": pct(EV["fomc_mean_ret"], 3), "{{SPY_RET}}": pct(EV["spy_mean_ret"], 3),
    "{{SURP_ROWS}}": surp_rows, "{{EARN_TICKERS}}": str(EV["earn_tickers"]), "{{EARN_FIRST}}": EV["earn_first"],
    "{{INTRA_ROWS}}": intra_html or "<tr><td colspan='12'>Gün içi veri yok</td></tr>",
    "{{KELLY_ROWS}}": "".join(
        f"<tr><td class='name'>{esc(s['title'])}</td><td class='n'>{pct(kelly(s), 0)}</td><td class='n'>{pct(max(kelly(s), 0) / 2, 0) if np.isfinite(kelly(s)) else '–'}</td></tr>"
        for s in passed + susp),
}
repl.update(repl_extra)
for k, v in repl.items():
    out = out.replace(k, v)
Path("rapor/rapor.html").write_text(out)
# PDF için: tüm kartlar açık
Path("rapor/rapor_print.html").write_text(out.replace('<details class="card"', '<details open class="card"'))
print("yazıldı: rapor/rapor.html", len(out) // 1024, "KB")
