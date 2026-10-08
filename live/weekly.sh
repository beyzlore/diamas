#!/bin/zsh
# Haftalık: tüm araştırmayı yeni verilerle tekrar çalıştır, raporu ve PDF'i yenile.
cd /Users/m2/Desktop/trade-arastirma
PY=.venv/bin/python
{
  echo "[$(date '+%Y-%m-%d %H:%M')] haftalık başladı"
  $PY -c "import sys; sys.path.insert(0,'src'); import data as D; sp=D.sp500_list(force=True); D.daily_prices(sorted(set(sp.ticker)|set(D.ETFS)), force=True); D.earnings_dates(sorted(sp.ticker), force=True); D.fomc_dates(force=True); D.vix(force=True)
import pandas as pd, metrics as M; df=pd.read_parquet('data/daily.parquet'); u=M.universe_table(df); u.to_parquet('data/universe.parquet')
top=[t for t in u.index if t not in D.ETFS][:60]; tk=sorted(set(top)|{'SPY','QQQ','IWM','DIA'}); D.intraday(tk,'5m',force=True); D.intraday(tk,'1h',force=True)" &&
  $PY run_events.py && $PY run_daily.py final && $PY run_intraday.py && $PY report.py final &&
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --disable-gpu --no-pdf-header-footer \
     --virtual-time-budget=5000 --window-size=760,1200 --print-to-pdf="$PWD/rapor/Strateji_Arastirma_Raporu.pdf" "file://$PWD/rapor/rapor_print.html" &&
  $PY live/run_live.py --offline &&
  osascript -e 'display notification "Haftalık araştırma ve PDF rapor yenilendi." with title "Diamas" sound name "Glass"' &&
  echo "[$(date '+%Y-%m-%d %H:%M')] haftalık bitti" || {
  osascript -e 'display notification "Haftalık yenilemede hata oluştu, logs/weekly.log dosyasına bakın." with title "Diamas HATASI"'
  echo "[$(date '+%Y-%m-%d %H:%M')] HATA"; }
} >> logs/weekly.log 2>&1
