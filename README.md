# Strateji Araştırma Sistemi

ABD hisse ve ETF'leri için kuralları açık stratejileri geçmiş veride test eden araştırma düzeneği.
Yatırım tavsiyesi değildir.

## Sırayla çalıştırma

```bash
cd ~/Desktop/trade-arastirma
.venv/bin/python scripts_fetch_events.py      # bilanço tarihleri + gün içi veri (~10 dk)
.venv/bin/python run_events.py               # olay analizi (bilanço / FOMC günleri)
.venv/bin/python run_daily.py final          # tüm günlük stratejiler + kontroller
.venv/bin/python run_intraday.py             # gün içi ORB testleri
.venv/bin/python report.py final             # rapor/rapor.html
```

Günlük fiyatları yenilemek için `data/daily.parquet` dosyasını silip `src/data.py` içindeki
`daily_prices(..., force=True)` ile tekrar indirin.

## Klasörler

- `src/data.py`: veri toplama (Yahoo Finance, federalreserve.gov, Wikipedia)
- `src/metrics.py`: likidite, hacim, oynaklık, spread ölçümleri
- `src/backtest.py`: test motoru, maliyetler, performans ölçümleri, şans düzeltmesi
- `src/strategies_daily.py` / `src/strategies_intraday.py`: strateji kuralları
- `src/research.py`: eğitim/doğrulama/kilitli dönem ayrımı, ileriye yürüyen test, rejimler
- `src/analyze.py`: her strateji için kontrol listesi ve karar
- `rapor/rapor.html`: sonuç raporu

## Dönemler

| Dönem | Tarih | Kullanım |
|---|---|---|
| Eğitim | 2005–2016 | ayar seçimi |
| Doğrulama | 2017–2023 | karar |
| Kilitli test | 2024–bugün | araştırma bitince bir kez açıldı |
