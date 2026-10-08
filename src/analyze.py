"""Her strateji için karar raporu: seçilen ayar, eğitim/doğrulama sonuçları, ileriye yürüyen test,
maliyet duyarlılığı, aşırı uyum kontrolleri, rejim analizi, en kötü dönem ve karar."""
from __future__ import annotations

import numpy as np
import pandas as pd

import backtest as B
import research as R


def daily_sr(d: pd.Series) -> float:
    return d.mean() / d.std() if d.std() > 0 else np.nan


def analyze_spec(p, vix, name, title, rule, family, res, n_trials_total, sr_std_total,
                 resim=None, bench_valid_sr=None) -> dict | None:
    best = R.select(res)
    if best is None:
        return None
    tr, dl = best["trades"], best["daily"]
    v_tr = tr[(tr["date"] >= R.VALID[0]) & (tr["date"] <= R.VALID[1])]
    # ileriye yürüyen
    wf, choices = R.walk_forward(res)
    wf_v = wf.loc[R.VALID[0]:R.VALID[1]]
    wf_sr = daily_sr(wf_v) * np.sqrt(252)
    # aşırı uyum: ızgaradaki tüm ayarların doğrulama sonuçları
    v_srs = np.array([r["valid"]["sharpe"] for r in res], dtype=float)
    t_srs = np.array([r["train"]["sharpe"] for r in res], dtype=float)
    frac_pos = np.nanmean(v_srs > 0)
    rank_corr = pd.Series(t_srs).corr(pd.Series(v_srs), method="spearman") if len(res) > 2 else np.nan
    d_tr = dl.loc[R.TRAIN[0]:R.TRAIN[1]]
    dsr = B.deflated_sharpe(daily_sr(d_tr), len(d_tr), n_trials_total, sr_std_total,
                            float(d_tr.skew()), float(d_tr.kurt() + 3))
    ci = B.bootstrap_ci(v_tr["net"].values)
    # maliyet duyarlılığı (doğrulama dönemi)
    cost = {}
    if resim is not None:
        for m in (0.0, 2.0, 3.0):
            d2, t2 = resim(best["params"], m)
            cost[m] = B.stats(d2, t2, *R.VALID)["sharpe"]
    reg = R.regimes(p, vix, dl, tr)
    yr = R.yearly(dl)
    ww, wwin = R.worst_window(dl)
    # en uzun su altı (zirveden uzak kalma) süresi
    eq = (1 + dl.loc[R.TRAIN[0]:R.RESEARCH_END]).cumprod()
    under = eq < eq.cummax()
    runs = under.groupby((~under).cumsum()).sum()
    longest_dd_days = int(runs.max()) if len(runs) else 0

    vs = best["valid"]
    checks = {
        "Doğrulamada Sharpe > 0.5": vs["sharpe"] > 0.5,
        "Doğrulamada işlem başı getiri > 0 (%95 güvenle)": np.isfinite(ci[0]) and ci[0] > 0,
        "İleriye yürüyen testte Sharpe > 0.3": wf_sr > 0.3,
        "Maliyetler 2 katına çıksa da Sharpe > 0": cost.get(2.0, np.nan) > 0,
        "Izgaradaki ayarların çoğu doğrulamada kârlı (>%60)": frac_pos > 0.6,
        "Şans düzeltmeli Sharpe (DSR) > %90": dsr > 0.9,
        "Doğrulamada en az 30 işlem": vs["işlem_sayısı"] >= 30,
    }
    if bench_valid_sr is not None:
        checks["Doğrulamada SPY al-tut'tan iyi Sharpe"] = vs["sharpe"] > bench_valid_sr
    passed = sum(bool(v) for v in checks.values())
    hard = checks["Doğrulamada Sharpe > 0.5"] and checks["Maliyetler 2 katına çıksa da Sharpe > 0"] \
        and checks["Doğrulamada işlem başı getiri > 0 (%95 güvenle)"]
    verdict = "GEÇTİ" if hard and passed >= len(checks) - 1 else ("ŞÜPHELİ" if vs["sharpe"] > 0.3 and passed >= 4 else "BAŞARISIZ")

    weak = reg[(reg["işlem"] >= 10) & (reg["beklenen_getiri"] < 0)]["koşul"].tolist()
    bad_years = [int(y) for y, v in yr.items() if v < 0]
    # kilitli dönem: hesaplanır ama araştırma bitene kadar raporlanmaz
    locked = B.stats(dl, tr, *R.LOCKED)
    return dict(locked=locked, trades=tr[["ticker", "date", "side", "net"]].copy(), name=name, title=title, rule=rule, family=family, params=best["params"], k=best["k"],
                n_grid=len(res), train=best["train"], valid=vs, wf_sharpe=wf_sr, wf_choices=choices,
                frac_valid_pos=frac_pos, rank_corr=rank_corr, dsr=dsr, ci=ci, cost=cost, regimes=reg,
                yearly=yr, worst_12m=ww, worst_12m_win=wwin, longest_dd_days=longest_dd_days,
                checks=checks, verdict=verdict, weak_regimes=weak, bad_years=bad_years,
                daily=dl, grid=[(r["params"], r["train"]["sharpe"], r["valid"]["sharpe"]) for r in res])
