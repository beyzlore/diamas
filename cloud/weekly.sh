#!/usr/bin/env bash
# Haftalık: tüm araştırmayı yeni verilerle tekrar çalıştır, raporu ve PDF'i yenile (bulut sürümü).
set -e
PY=${PY:-python}
CHROME=${CHROME:-google-chrome}
$PY - <<'PYEOF'
import sys; sys.path.insert(0, "src")
import pandas as pd, data as D, metrics as M
sp = D.sp500_list(force=True)
D.daily_prices(sorted(set(sp.ticker) | set(D.ETFS)), force=True)
D.earnings_dates(sorted(sp.ticker), force=True)
D.fomc_dates(force=True); D.vix(force=True)
u = M.universe_table(pd.read_parquet("data/daily.parquet")); u.to_parquet("data/universe.parquet")
top = [t for t in u.index if t not in D.ETFS][:60]
tk = sorted(set(top) | {"SPY", "QQQ", "IWM", "DIA"})
D.intraday(tk, "5m", force=True); D.intraday(tk, "1h", force=True)
PYEOF
$PY run_events.py
$PY run_daily.py final
$PY run_intraday.py
$PY report.py final
"$CHROME" --headless=new --no-sandbox --disable-gpu --no-pdf-header-footer --virtual-time-budget=5000 \
  --window-size=760,1200 --print-to-pdf="$PWD/rapor/Strateji_Arastirma_Raporu.pdf" "file://$PWD/rapor/rapor_print.html"
$PY live/run_live.py --offline
