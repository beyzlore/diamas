import sys; sys.path.insert(0,"src")
import pandas as pd, data as D
sp=D.sp500_list()
u=pd.read_parquet("data/universe.parquet")
e=D.earnings_dates(sorted(sp.ticker)); print("earnings rows",len(e), e.ticker.nunique(), e.ts.min(), flush=True)
top=[t for t in u.index if t not in D.ETFS][:60]
intr=sorted(set(top)|{"SPY","QQQ","IWM","DIA"})
m5=D.intraday(intr,"5m"); print("5m",m5.shape, m5.ticker.nunique(), m5.date.nunique(), flush=True)
h1=D.intraday(intr,"1h"); print("1h",h1.shape, h1.ticker.nunique(), h1.date.nunique(), flush=True)
