"""
ACCURATE BACKTEST v9.0 — M1 DATA + RESAMPLE
=============================================
Downloads real M1 gold data (7 days) and resamples to M5/M15/H1/H4
for genuine multi-timeframe bars. Tests all v9.0 features.
"""
import os, sys, warnings, json
from datetime import datetime, timedelta
import pandas as pd, numpy as np
warnings.filterwarnings("ignore")
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT); sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'trading_bot_mt5'))
import logging; logging.disable(logging.CRITICAL)
from trading_bot.indicators.technical_indicators import compute_all_indicators, compute_sr_levels
from trading_bot.strategy.gold_scalping_strategy import GoldScalpingStrategy
import candle_patterns as cp, sr_levels_mtf as sr_mtf

# Config
BAL = 500.0; FIXED_RISK = 0.05; SL_M = 2.5; TP_M = 5.0
BE_M = 2.0; BE_BUF = 50; BE_USD = 40; SPREAD = 0.30
MIN_SCORE = 40; MIN_S_BUY = 40; MIN_S_SELL = 40; HS_THRESH = 70; HS_RISK = 0.10
SR_ONLY_AT_LEVELS = True; SR_BOUNCE_BUFFER = 6.0; SR_NO_TRADE_BUFFER = 3.0

def download_resample():
    import yfinance as yf
    print("Downloading M1 gold data (7d)...")
    df = yf.download("GC=F", period="7d", interval="1m", progress=False)
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex): df.columns = [c[0].lower() for c in df.columns]
    else: df.columns = [c.lower() for c in df.columns]
    df.index = pd.to_datetime(df.index, utc=True)
    df = df[~df.index.duplicated(keep='last')].sort_index()
    df['tick_volume'] = df.get('volume', 1)
    print(f"  M1: {len(df)} candles ({df.index[0]} -> {df.index[-1]})")

    def r(rule): return df.resample(rule).agg({'open':'first','high':'max','low':'min','close':'last','tick_volume':'sum'}).dropna()
    m5 = r('5min'); m15 = r('15min'); h1 = r('1h'); h4 = r('4h')

    # Inject micro-noise to differentiate bars (Yahoo bars have flat periods)
    np.random.seed(42)
    for d in [m5, m15, h1, h4]:
        if d is not None and len(d) > 0:
            n = np.random.uniform(0.00002, 0.00008, len(d))
            d['high'] = d['high'] * (1 + n); d['low'] = d['low'] * (1 - n)
    print(f"  M5={len(m5)} M15={len(m15)} H1={len(h1)} H4={len(h4)}")
    return df, m5, m15, h1, h4

def run():
    print("="*55); print("  ACCURATE BACKTEST v9.0 — M1 RESAMPLE"); print("="*55)
    m1, m5, m15, h1, h4 = download_resample()
    if m15 is None or len(m15) < 30: print("  Not enough data"); return

    sdate = m15.index[30]; edate = m15.index[-1]
    print(f"\n  Period: {sdate} -> {edate}")

    st = GoldScalpingStrategy(); st._max_trades_per_day = 50; st._max_open_positions = 3
    sr_eng = sr_mtf.MultiTFSupportResistance()
    bal = BAL; pos = []; closed = []; partials = []; dpnl = 0.0; lastd = None; peak = BAL

    for i in range(30, len(m15)):
        ct = m15.index[i]
        pc = float(m15["close"].iloc[i]); ph = float(m15["high"].iloc[i]); pl = float(m15["low"].iloc[i])
        cdate = ct.date()
        if lastd != cdate: dpnl = 0.0; lastd = cdate
        if dpnl <= -bal * 0.05: continue

        m5w = m5[m5.index <= ct].tail(100).copy()
        m15w = m15.iloc[max(0,i-100):i+1].copy()
        h1w = h1[h1.index <= ct].tail(50).copy()
        h4w = h4[h4.index <= ct].tail(50).copy()
        if len(m5w) < 50 or len(m15w) < 30: continue

        i5 = compute_all_indicators(m5w); i15 = compute_all_indicators(m15w)
        if i5 is None or i15 is None: continue
        atr = float(i15["atr"].iloc[-1]) if i15.get("atr") is not None and len(i15["atr"]) > 0 else 1.0
        if atr < 0.2: continue

        # Position mgmt with intra-bar high/low
        surv = []
        for p in pos:
            e, d, sl, tp, lot = p['entry'], p['dir'], p['sl'], p['tp'], p['lot']
            sd = abs(e-sl) if sl > 0 else atr*SL_M
            td = abs(tp-e) if tp > 0 else sd*(TP_M/SL_M)

            # Track MFE
            mfe = (ph-e)*lot*100 if d=="BUY" else (e-pl)*lot*100
            if mfe > 0: p['_mfe'] = max(p.get('_mfe',0), mfe)

            # BE triggers
            if not p.get('be') and p.get('be_target'):
                if (d=="BUY" and ph >= p['be_target']) or (d=="SELL" and pl <= p['be_target']): p['be']=True; p['sl']=e
            if not p.get('be') and BE_USD > 0:
                pp = BE_USD/(lot*100) if lot>0 else 999
                bp = e+pp if d=="BUY" else e-pp
                if (d=="BUY" and ph >= bp) or (d=="SELL" and pl <= bp):
                    p['be']=True; p['sl']=e+(BE_BUF*0.01) if d=="BUY" else e-(BE_BUF*0.01)

            # Progressive trailing (use close for trail)
            profit = pc-e if d=="BUY" else e-pc
            ppct = profit/td if td>0 else 0
            tm = 999
            if ppct >= 0.60: tm = 0.10
            elif ppct >= 0.40: tm = 0.20
            elif ppct >= 0.20: tm = 0.40
            if tm < 999:
                ns = pc-(sd*tm) if d=="BUY" else pc+(sd*tm)
                if (d=="BUY" and ns>p['sl']+0.1) or (d=="SELL" and ns<p['sl']-0.1):
                    p['sl']=round(ns,2); p['trail_stage']=f"s{int(ppct*100)}"

            # Partial close at 80%
            if ppct >= 0.80 and not p.get('partial'):
                hl = round(lot/2,2)
                if hl >= 0.01:
                    ppnl = profit*hl*100-SPREAD*hl*100
                    bal+=ppnl; dpnl+=ppnl; p['lot']=round(lot-hl,2); p['partial']=True
                    partials.append({'pnl':round(ppnl,2)})

            # Exit check (intra-bar)
            hit=False; pnl=0.0; reason=""; xp=pc
            if d=="BUY":
                if tp and ph >= tp: pnl=(tp-e)*lot*100; xp=tp; reason="TP"; hit=True
                elif sl and pl <= sl: pnl=(sl-e)*lot*100; xp=sl; reason="TRAIL" if p.get('trail_stage') else ("BE_SL" if p.get('be') else "SL"); hit=True
            else:
                if tp and pl <= tp: pnl=(e-tp)*lot*100; xp=tp; reason="TP"; hit=True
                elif sl and ph >= sl: pnl=(e-sl)*lot*100; xp=sl; reason="TRAIL" if p.get('trail_stage') else ("BE_SL" if p.get('be') else "SL"); hit=True
            if hit:
                pnl-=SPREAD*lot*100; bal+=pnl; dpnl+=pnl
                p['pnl']=round(pnl,2); p['reason']=reason; p['close_price']=xp; p['close_time']=ct
                closed.append(p)
                if p.get('_mfe'): p['_mfe']=round(p['_mfe'],2)
            else: surv.append(p)
        pos = surv
        if len(pos) >= 3: continue

        # New entry
        try:
            em = {"rsi":pd.Series([50]),"emas":pd.DataFrame(),"macd":pd.Series([0])}
            res = st.analyze(m1_indicators=em,m5_indicators=i5,m15_indicators=i15,
                             m1_ohlcv=m5w.tail(20),m5_ohlcv=m5w,m15_ohlcv=m15w)
        except: continue
        d = res.get("direction","NONE"); sc = res.get("setup_score",0)
        if d=="NONE" or sc<MIN_SCORE: continue

        # Candle boost
        try:
            sw = cp.detect_swing_levels(m15w); ca = cp.analyze_full(m15w,sw)
            if ca.get("signal")==d: sc=min(100,sc+max(1,ca.get("confidence",0)//3))
        except: pass

        # Multi-TF S/R filter
        try:
            mtf = sr_eng.compute_all(h4w,h1w,m15w,m5w,pc)
            # 1. Block trades against major S/R
            blk, rsn = sr_eng.is_in_no_trade_zone(pc,d,mtf.get("no_buy_zones",[]),mtf.get("no_sell_zones",[]),SR_NO_TRADE_BUFFER)
            if blk: continue
            # 2. SR_ONLY_AT_LEVELS: buy near support, sell near resistance
            if SR_ONLY_AT_LEVELS:
                nearest_r = mtf.get("nearest_resistance", {}).get("level", pc + 50)
                nearest_s = mtf.get("nearest_support", {}).get("level", pc - 50)
                dist_to_r = nearest_r - pc
                dist_to_s = pc - nearest_s
                if d == "BUY":
                    if dist_to_s > SR_BOUNCE_BUFFER: continue
                elif d == "SELL":
                    if dist_to_r > SR_BOUNCE_BUFFER: continue
        except: pass

        ms = MIN_S_BUY if d=="BUY" else MIN_S_SELL
        if sc < ms: continue

        # RSI check
        try:
            r5=i5['rsi'].iloc[-1]; r15=i15['rsi'].iloc[-1]
            if d=="BUY" and (r5<25 or r15<25): continue
            if d=="SELL" and (r5>75 or r15>75): continue
        except: pass

        sd = atr*SL_M; td = atr*TP_M
        sl = round(pc-sd,2) if d=="BUY" else round(pc+sd,2)
        tp = round(pc+td,2) if d=="BUY" else round(pc-td,2)
        rp = HS_RISK if sc >= HS_THRESH else FIXED_RISK
        lot = max(0.01,round((bal*rp)/(sd*100),2))
        if len(pos)==1: lot*=0.5
        lot = max(0.01,round(lot,2))
        bt = pc+(atr*BE_M if d=="BUY" else -atr*BE_M)
        pos.append({"entry":pc,"sl":sl,"tp":tp,"lot":lot,"dir":d,"open_time":ct,"score":sc,"be_target":bt,"be":False,"partial":False,"_mfe":0})

    # Results
    nt = len(closed); partial_pnl = sum(x['pnl'] for x in partials)
    if nt == 0: print("\n  No trades."); return
    df = pd.DataFrame(closed)
    net = df['pnl'].sum() + partial_pnl
    wins = len(df[df['pnl']>0]); losses = len(df[df['pnl']<=0])
    wr = wins/nt*100 if nt>0 else 0
    pf = abs(sum(t['pnl'] for t in closed if t['pnl']>0)/sum(t['pnl'] for t in closed if t['pnl']<=0)) if any(t['pnl']<=0 for t in closed) else 999
    # Drawdown
    eq = np.cumsum([BAL]+[t['pnl'] for t in closed])
    peak_e = np.maximum.accumulate(eq); dd = (peak_e-eq)/peak_e*100; mdd = dd.max()

    reasons = {}; dirs = {"BUY":0,"SELL":0}; total_mfe = 0; mfe_lost = 0
    for t in closed:
        reasons[t.get('reason','?')]=reasons.get(t.get('reason','?'),0)+1
        dirs[t.get('dir','?')]=dirs.get(t.get('dir','?'),0)+1
        if t.get('_mfe',0)>0: total_mfe+=t['_mfe']; mfe_lost+=t['_mfe']-max(t.get('pnl',0),0)

    print("\n"+"="*55)
    print("  ACCURATE BACKTEST RESULTS — v9.0")
    print("="*55)
    print(f"  Balance:   ${BAL:.0f} -> ${BAL+net:.2f}")
    print(f"  Net PnL:   ${net:+.2f} ({(net/BAL)*100:+.1f}%)")
    print(f"  Partials:  {len(partials)} (${partial_pnl:+.2f})")
    print(f"  Max DD:    {mdd:.1f}%")
    print(f"  PF:        {pf:.2f}")
    print(f"  Win Rate:  {wr:.1f}% ({wins}W/{losses}L)")
    print(f"  Trades:    {nt} | BUY={dirs.get('BUY',0)} SELL={dirs.get('SELL',0)}")
    print(f"  Reasons:   {json.dumps(reasons)}")
    print(f"  Avg Win:   ${df[df['pnl']>0]['pnl'].mean():+.2f}" if wins>0 else "  Avg Win:   N/A")
    print(f"  Avg Loss:  ${df[df['pnl']<=0]['pnl'].mean():+.2f}" if losses>0 else "  Avg Loss:  N/A")
    if nt > 0:
        mfe_pct = (mfe_lost/total_mfe*100) if total_mfe>0 else 0
        print(f"  MFE Lost:  ${mfe_lost:.0f} ({mfe_pct:.0f}% of peak profit given back)")

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"backtest_v9_accurate_results.txt")
    with open(path,"w") as f:
        f.write(f"ACCURATE BACKTEST v9.0\n{sdate} -> {edate}\n")
        f.write(f"Net: ${net:+.2f} | DD: {mdd:.1f}% | WR: {wr:.1f}% | Trades: {nt}\n")
    print(f"\n  Saved: {path}")

if __name__=="__main__": run()