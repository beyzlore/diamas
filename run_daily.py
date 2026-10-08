"""1. tur: günlük stratejilerin tamamını test eder, sonuçları results/daily_round.pkl'e yazar.

Kullanım:  .venv/bin/python run_daily.py [tur_adı]
"""
import pickle
import sys
from pathlib import Path

sys.path.insert(0, "src")
import numpy as np
import pandas as pd

import backtest as B
import data as D
import research as R
import strategies_daily as S
import analyze as A

ROUND = sys.argv[1] if len(sys.argv) > 1 else "r1"
OUT = Path("results")
OUT.mkdir(exist_ok=True)

daily_df = pd.read_parquet("data/daily.parquet")
p = B.load_panel(daily_df)
vix = D.vix()
fomc = D.fomc_dates()
earn = pd.read_parquet("data/earnings.parquet") if Path("data/earnings.parquet").exists() else None

U100 = B.pit_universe(p, 100, set(D.ETFS))
EMASK = S.earnings_days_mask(p, earn) if earn is not None else None
U200 = B.pit_universe(p, 200, set(D.ETFS))
UETF = pd.DataFrame(False, index=p.dates, columns=p.c.columns)
UETF[D.ETFS] = True

specs = [
    R.Spec("trend_spy", "Trend takibi (SPY)",
           "SPY kapanışı n günlük ortalamanın üstündeyse ertesi açılışta al, altına inince sat.",
           lambda n: S.trend_sma(p, ["SPY"], n), {"n": [50, 100, 150, 200, 250]}, 1, "trend"),
    R.Spec("trend_etf", "Trend takibi (17 ETF)",
           "Her ETF için: kapanış n günlük ortalamanın üstündeyse tut, altındaysa nakit. Sermaye 17'ye bölünür.",
           lambda n: S.trend_sma(p, D.ETFS, n), {"n": [50, 100, 200]}, len(D.ETFS), "trend"),
    R.Spec("donchian_etf", "Donchian kırılımı (17 ETF)",
           "Kapanış son n günün zirvesini aşınca al; son n/2 günün dibini kırınca sat.",
           lambda n: S.donchian(p, D.ETFS, n), {"n": [20, 55, 100, 200]}, len(D.ETFS), "trend"),
    R.Spec("rsi2_stk", "RSI(2) dipten alım (en likit 100 hisse)",
           "Hisse 200 günlük ortalamanın üstündeyken RSI(2) eşiğin altına düşerse ertesi açılışta al; "
           "kapanış 5 günlük ortalamayı geçince sat. En fazla 10 pozisyon.",
           lambda th, mh: S.rsi2_reversion(p, U100, th, mh), {"th": [5, 10, 15, 20], "mh": [5, 10]}, 10, "ortalamaya dönüş"),
    R.Spec("rsi2_etf", "RSI(2) dipten alım (17 ETF)",
           "Aynı kural, ama hisse yerine ETF'lerde (iflas/listeden çıkma yanlılığı yok). En fazla 5 pozisyon.",
           lambda th, mh: S.rsi2_reversion(p, UETF, th, mh), {"th": [5, 10, 15, 20], "mh": [5, 10]}, 5, "ortalamaya dönüş"),
    R.Spec("gap_fade_down", "Aşağı boşluğu alma (gün içi)",
           "Hisse normal oynaklığının g katı aşağı açılırsa açılışta al, kapanışta sat.",
           lambda g: S.gap_intraday(p, U100, g, "fade_down"), {"g": [1.0, 1.5, 2.0, 3.0]}, 10, "olay/gap"),
    R.Spec("gap_go_up", "Yukarı boşluğu takip (gün içi)",
           "Hisse normal oynaklığının g katı yukarı açılırsa açılışta al, kapanışta sat.",
           lambda g: S.gap_intraday(p, U100, g, "go_up"), {"g": [1.0, 1.5, 2.0, 3.0]}, 10, "olay/gap"),
    R.Spec("gap_fade_up", "Yukarı boşluğa karşı açığa satış (gün içi)",
           "Hisse normal oynaklığının g katı yukarı açılırsa açılışta açığa sat, kapanışta kapat.",
           lambda g: S.gap_intraday(p, U100, g, "fade_up"), {"g": [1.0, 1.5, 2.0, 3.0]}, 10, "olay/gap"),
    R.Spec("xsmom", "Kesitsel momentum (en likit 100 hisse)",
           "Her ay sonu son L ayın (son ay hariç) en çok yükselen N hissesini al, bir ay tut. "
           "Filtreli sürümde SPY 200 günlük ortalamanın altındaysa nakitte kal.",
           lambda lb, top, f: S.xs_momentum(p, U100, lb, top, f),
           {"lb": [3, 6, 12], "top": [10, 20], "f": [False, True]}, lambda prm: prm["top"], "momentum"),
    R.Spec("fomc", "FOMC öncesi sürüklenme (SPY)",
           "Fed faiz kararı gününden önceki kapanışta (veya karar günü açılışta) SPY al, karar günü kapanışta sat.",
           lambda entry: S.fomc_drift(p, fomc, "SPY", entry), {"entry": ["prev_close", "open"]}, 1, "takvim/olay"),
    R.Spec("tom", "Ay dönümü etkisi (SPY)",
           "Ay bitmeden b gün önce kapanışta SPY al, yeni ayın a. günü kapanışta sat.",
           lambda b, a: S.turn_of_month(p, "SPY", b, a), {"b": [1, 2, 3], "a": [2, 3, 4]}, 1, "takvim/olay"),
]
# ---- 2. tur (1. turun eğitim+doğrulama derslerinden; kilitli döneme bakılmadı)
specs += [
    R.Spec("rsi2_regime", "RSI(2) dipten alım + kriz filtresi (2. tur)",
           "RSI(2) kuralı aynen; ama yalnızca SPY 200 günlük ortalamanın üstünde ve VIX belirli bir "
           "seviyenin altındayken yeni alım yapılır (1. turda krizlerde zarar ettiği için).",
           lambda th, vmax: S.rsi2_regime(p, U100, vix, th, vmax), {"th": [5, 10, 15], "vmax": [20, 25, 30]}, 10,
           "ortalamaya dönüş"),
    R.Spec("dual_mom", "Çift momentum (ETF rotasyonu, 2. tur)",
           "Her ay sonu SPY, QQQ, IWM, EFA, EEM, GLD arasından son L ayda en çok kazananı al; "
           "o bile tahvil ETF'i TLT'den kötüyse TLT tut. Ayda bir işlem.",
           lambda lb, top: S.dual_momentum(p, ["SPY", "QQQ", "IWM", "EFA", "EEM", "GLD"], "TLT", lb, top),
           {"lb": [3, 6, 9, 12], "top": [1, 2]}, lambda prm: prm["top"], "momentum"),
    R.Spec("vix_spike", "Panik alımı (VIX sıçraması, 2. tur)",
           "VIX kendi 20 günlük ortalamasının r katına fırlarsa ertesi açılışta SPY al, h gün tut.",
           lambda r, h: S.vix_spike(p, vix, r, h), {"r": [1.2, 1.3, 1.5], "h": [5, 10, 20]}, 1, "takvim/olay"),
    R.Spec("trend_vt", "Trend + oynaklık hedefleme (SPY, 2. tur)",
           "SPY trend yukarıyken tut, ama piyasa çok dalgalıysa pozisyonu küçült "
           "(büyüklük = hedef oynaklık / son 20 gün oynaklığı, en fazla %100).",
           lambda n, tv: S.trend_voltarget(p, n, tv), {"n": [100, 200], "tv": [0.10, 0.15, 0.20]}, 1, "trend"),
]
if earn is not None:
    specs += [
        R.Spec("pead_long", "Bilanço sonrası yükseliş sürüklenmesi",
               "Şirket kâr beklentisini en az s% aşar ve hisse ilk gün yükselirse, ertesi açılışta al, h gün tut.",
               lambda s, h: S.pead(p, earn, U200, s, h, True, False), {"s": [0, 5, 10, 20], "h": [5, 10, 20, 40]}, 10,
               "olay/bilanço"),
        R.Spec("pead_regime", "Bilanço sonrası sürüklenme + piyasa filtresi (3. tur)",
               "PEAD kuralı aynen; ama yalnızca SPY 200 günlük ortalamanın üstündeyken yeni alım yapılır "
               "(2. turda ayı piyasalarında zarar ettiği için).",
               lambda s, h: S.pead_regime(p, earn, U200, s, h), {"s": [0, 5, 10], "h": [10, 20, 40]}, 10, "olay/bilanço"),
        R.Spec("rsi2_noearn", "RSI(2) + kriz filtresi + bilanço kaçınma (3. tur)",
               "RSI(2) dipten alım ve kriz filtresi; ek olarak elde tutma süresine bilanço açıklaması "
               "denk gelecekse işleme hiç girilmez (ani sıçrama riskini önlemek için).",
               lambda th, vmax, buf: S.rsi2_no_earnings(p, U100, vix, EMASK, th, vmax, buf),
               {"th": [5, 10], "vmax": [25, 30], "buf": [0, 2]}, 10, "ortalamaya dönüş"),
        R.Spec("pead_short", "Bilanço sonrası düşüş sürüklenmesi (açığa satış)",
               "Şirket beklentinin en az s% altında kalır ve hisse ilk gün düşerse, ertesi açılışta açığa sat, h gün tut.",
               lambda s, h: S.pead(p, earn, U200, s, h, True, True), {"s": [0, 5, 10, 20], "h": [5, 10, 20, 40]}, 10,
               "olay/bilanço"),
    ]

only = set(sys.argv[2].split(",")) if len(sys.argv) > 2 else None
results = {}
spec_by = {}
for sp in specs:
    if only and sp.name not in only:
        continue
    res = R.run_spec(p, sp)
    results[sp.name] = (sp.title, sp.rule, sp.family, res)
    spec_by[sp.name] = sp
    best = R.select(res)
    if best:
        print(f"{sp.name:14s} deneme={len(res):2d} seçilen={best['params']}  "
              f"eğitim SR={best['train']['sharpe']:.2f}  doğrulama SR={best['valid']['sharpe']:.2f}  "
              f"doğ. beklenen={best['valid']['beklenen_getiri']:.4%}  doğ. işlem={best['valid']['işlem_sayısı']}",
              flush=True)

# benchmarklar
bench = {}
for name, tr, k in [("SPY al-tut", S.buy_hold(p, "SPY"), 1)]:
    bench[name] = B.simulate(p, tr, k)[0]
# evren eşit ağırlık: en likit 100 hissenin hepsini eşit ağırlıkla tut (aylık yenile)
ret = p.c.pct_change().where(U100.shift(1, fill_value=False))
bench["En likit 100 hisse eşit ağırlık"] = ret.mean(axis=1).fillna(0)

# --- analiz: tüm denemeler üzerinden şans düzeltmesi
all_tr_sr = [A.daily_sr(r["daily"].loc[R.TRAIN[0]:R.TRAIN[1]]) for (_, _, _, res) in results.values() for r in res]
n_trials = len(all_tr_sr)
sr_std = float(np.nanstd(all_tr_sr))
spy_valid_sr = A.daily_sr(bench["SPY al-tut"].loc[R.VALID[0]:R.VALID[1]]) * np.sqrt(252)
summary = {}
for name, (title, rule, fam, res) in results.items():
    sp = spec_by[name]

    def resim(prm, m, sp=sp):
        k = sp.k(prm) if callable(sp.k) else sp.k
        return B.simulate(p, sp.fn(**prm), k, m)
    summary[name] = A.analyze_spec(p, vix, name, title, rule, fam, res, n_trials, sr_std, resim, spy_valid_sr)
    s_ = summary[name]
    if s_:
        print(f"{name:14s} {s_['verdict']:9s} doğ.SR={s_['valid']['sharpe']:.2f} WF={s_['wf_sharpe']:.2f} "
              f"2xmaliyet={s_['cost'].get(2.0, np.nan):.2f} DSR={s_['dsr']:.2f} ayarların %{100*s_['frac_valid_pos']:.0f}'i kârlı "
              f"zayıf={s_['weak_regimes']}", flush=True)

with open(OUT / f"daily_{ROUND}.pkl", "wb") as f:
    pickle.dump({"summary": summary, "bench": bench, "n_trials": n_trials, "sr_std": sr_std,
                 "spy_valid_sr": spy_valid_sr}, f)
print("kaydedildi", OUT / f"daily_{ROUND}.pkl")
