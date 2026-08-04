"""
PURE S/R BACKTEST — No Indicators, Only Support & Resistance
==============================================================
D1/H4/H1/M15/M5 levels. BUY at support + bullish candle. SELL at resistance + bearish candle.
Tight SL (1.0x ATR), tight TP (2.0x ATR). Aggressive trail. High hit-rate scalping.

Goal: Prove S/R trading is profitable, then apply to live bot.
"""
import os, sys, warnings, json
warnings.filterwarnings("ignore")
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT); sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'trading_bot_mt5'))
import pandas as pd, numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import logging; logging.disable(logging.CRITICAL)
import sr_levels_mtf as sr_mtf

# ── CONFIG ────────────────────────────────────────────────────
BAL = 500.0; RISK_PER_TRADE = 0.02  # 2% per trade
SL_MULT = 1.0   # 1x ATR = tight stop
TP_MULT = 2.0   # 2x ATR = quick target
TRAIL_START = 0.15  # Start trail at 15% of TP
TRAIL_TIGHT = 0.08  # Trail distance at stage 2
TRAIL_AGGR = 0.04   # Trail distance at stage 3 (near TP)
SR_BUFFER = 2.5     # Must be within this many pts of a level to enter
SR_BOUNCE_BUY = 6.0  # Max distance below support to bounce-buy
SR_BOUNCE_SELL = 6.0 # Max distance below resistance to reject-sell
SPREAD = 0.30

def download_m1():
    print("Downloading M1 gold data (7d)...")
    df = yf.download("GC=F", period="7d", interval="1m", progress=False)
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex): df.columns = [c[0].lower() for c in df.columns]
    else: df.columns = [c.lower() for c in df.columns]
    df.index = pd.to_datetime(df.index, utc=True); df = df[~df.index.duplicated(keep='last')].sort_index()
    df['tick_volume'] = df.get('volume', 1)
    return df

def resample(m1):
    def r(rule): return m1.resample(rule).agg({'open':'first','high':'max','low':'min','close':'last','tick_volume':'sum'}).dropna()
    return r('5min'), r('15min'), r('1h'), r('4h'), r('1D')

def compute_atr(ohlc, period=14):
    h, l, c = ohlc['high'], ohlc['low'], ohlc['close']
    tr = pd.concat([(h-l).abs(), (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

def is_bullish_candle(o, h, l, c):
    body = abs(c-o); lower_wick = min(o,c)-l; upper_wick = h-max(o,c); total = h-l
    if total <= 0: return False
    # Hammer or engulfing or strong bullish body with lower wick
    if c > o and lower_wick > body*1.5 and upper_wick < body*0.5: return True  # Hammer
    if c > o and lower_wick > body and body/total > 0.3: return True  # Reversal
    if c > o and body/total > 0.7: return True  # Strong bullish
    return False

def is_bearish_candle(o, h, l, c):
    body = abs(c-o); lower_wick = min(o,c)-l; upper_wick = h-max(o,c); total = h-l
    if total <= 0: return False
    if c < o and upper_wick > body*1.5 and lower_wick < body*0.5: return True  # Shooting star
    if c < o and upper_wick > body and body/total > 0.3: return True  # Reversal
    if c < o and body/total > 0.7: return True  # Strong bearish
    return False

def get_level_strength(price, levels, is_buy):
    """Find nearest support (for buy) or resistance (for sell) and return distance."""
    if is_buy:
        supports = [l for l in levels if l < price]
        if not supports: return 999
        nearest = max(supports)
        return price - nearest
    else:
        resistances = [l for l in levels if l > price]
        if not resistances: return 999
        nearest = min(resistances)
        return nearest - price

def collect_all_levels(sr_engine, m5, m15, h1, h4, d1, price):
    """Collect ALL S/R levels from all timeframes by computing swings on each TF separately."""
    all_r = []; all_s = []
    from sr_levels_mtf import detect_swings, cluster_levels
    
    for tf_name, data in [("D1", d1), ("H4", h4), ("H1", h1), ("M15", m15), ("M5", m5)]:
        if data is None or len(data) < 10: continue
        try:
            swing_highs, swing_lows = detect_swings(data, window=3)
            res = cluster_levels(swing_highs, threshold_pct=0.002)
            sup = cluster_levels(swing_lows, threshold_pct=0.002)
            # Keep only levels within 100 pts of current price
            all_r.extend([r for r in res if abs(r - price) <= 100])
            all_s.extend([s for s in sup if abs(s - price) <= 100])
        except: pass
    
    # Merge nearby levels (cluster within 1 point)
    def cluster(lst):
        if not lst: return []
        s = sorted(set(round(x,1) for x in lst))
        clusters = [[s[0]]]
        for v in s[1:]:
            if v - clusters[-1][-1] <= 1.0: clusters[-1].append(v)
            else: clusters.append([v])
        return [sum(c)/len(c) for c in clusters]
    
    return cluster(all_r), cluster(all_s)

def run():
    print("="*60); print("  PURE S/R BACKTEST — D1/H4/H1/M15/M5"); print("="*60)
    m1 = download_m1()
    if m1 is None: return
    m5, m15, h1, h4, d1 = resample(m1)
    
    # Need enough D1 candles
    while d1 is not None and len(d1) < 10:
        # Extend download to 30 days
        m1 = yf.download("GC=F", period="30d", interval="1m", progress=False)
        if isinstance(m1.columns, pd.MultiIndex): m1.columns = [c[0].lower() for c in m1.columns]
        else: m1.columns = [c.lower() for c in m1.columns]
        m1.index = pd.to_datetime(m1.index, utc=True); m1 = m1[~m1.index.duplicated(keep='last')].sort_index()
        m1['tick_volume'] = m1.get('volume', 1)
        m5, m15, h1, h4, d1 = resample(m1)
        break
    
    sr_engine = sr_mtf.MultiTFSupportResistance()
    
    sdate = m15.index[50]; edate = m15.index[-1]
    print(f"  Period: {sdate} -> {edate}")
    print(f"  Data: M1={len(m1)} M5={len(m5)} M15={len(m15)} H1={len(h1)} H4={len(h4)} D1={len(d1)}")
    
    bal = BAL; pos = []; closed = []; dpnl = 0.0; lastd = None; peak = BAL
    
    for i in range(50, len(m15)):
        ct = m15.index[i]
        pc = float(m15["close"].iloc[i]); ph = float(m15["high"].iloc[i]); pl = float(m15["low"].iloc[i])
        cdate = ct.date()
        if lastd != cdate: dpnl = 0.0; lastd = cdate
        if dpnl <= -bal * 0.03: continue  # 3% daily loss halt
        
        # Slice windows
        m5w = m5[m5.index <= ct].tail(150).copy()
        m15w = m15.iloc[max(0,i-150):i+1].copy()
        h1w = h1[h1.index <= ct].tail(100).copy()
        h4w = h4[h4.index <= ct].tail(100).copy()
        d1w = d1[d1.index <= ct].tail(50).copy()
        
        if len(m5w) < 50: continue
        
        atr_s = compute_atr(m15w, 14)
        if len(atr_s) == 0: continue
        atr = float(atr_s.iloc[-1])
        if atr < 0.3: continue
        
        # Collect ALL S/R levels
        all_res, all_sup = collect_all_levels(sr_engine, m5w, m15w, h1w, h4w, d1w, pc)
        if not all_res or not all_sup: continue
        
        dist_to_r = get_level_strength(pc, all_res, is_buy=False)
        dist_to_s = get_level_strength(pc, all_sup, is_buy=True)
        
        # Position management
        surv = []
        for p in pos:
            e, d, sl, tp, lot = p['entry'], p['dir'], p['sl'], p['tp'], p['lot']
            td = abs(tp-e) if tp > 0 else atr*TP_MULT
            sd = abs(e-sl) if sl > 0 else atr*SL_MULT
            mfe = (ph-e)*lot*100 if d=="BUY" else (e-pl)*lot*100
            if mfe>0: p['_mfe'] = max(p.get('_mfe',0), mfe)
            
            # Aggressive trail (much tighter than before)
            profit = pc-e if d=="BUY" else e-pc
            ppct = profit/td if td>0 else 0
            
            if ppct >= 0.33:  # Stage 3: super tight
                trail_m = TRAIL_AGGR
            elif ppct >= 0.20:  # Stage 2: tight
                trail_m = TRAIL_TIGHT
            elif ppct >= 0.10:  # Stage 1: start trailing
                trail_m = TRAIL_START
            else:
                trail_m = 999
            
            if trail_m < 999:
                ns = pc-(sd*trail_m) if d=="BUY" else pc+(sd*trail_m)
                if (d=="BUY" and ns>p['sl']+0.05) or (d=="SELL" and ns<p['sl']-0.05):
                    p['sl']=round(ns,2); p['trail']=True
            
            # Exit
            hit=False; pnl=0.0; reason=""
            if d=="BUY":
                if tp and ph >= tp: pnl=(tp-e)*lot*100; reason="TP"; hit=True
                elif sl and pl <= sl: pnl=(sl-e)*lot*100; reason="TRAIL" if p.get('trail') else "SL"; hit=True
            else:
                if tp and pl <= tp: pnl=(e-tp)*lot*100; reason="TP"; hit=True
                elif sl and ph >= sl: pnl=(e-sl)*lot*100; reason="TRAIL" if p.get('trail') else "SL"; hit=True
            if hit:
                pnl-=SPREAD*lot*100; bal+=pnl; dpnl+=pnl
                p['pnl']=round(pnl,2); p['reason']=reason; p['close_time']=ct
                closed.append(p)
            else: surv.append(p)
        pos = surv
        
        if len(pos) >= 2: continue
        
        # ── S/R ENTRY LOGIC ──────────────────────────────
        # Check M15 candle pattern
        try:
            o15 = float(m15w["open"].iloc[-2]); h15 = float(m15w["high"].iloc[-2])
            l15 = float(m15w["low"].iloc[-2]); c15 = float(m15w["close"].iloc[-2])
        except: continue
        
        bull_candle = is_bullish_candle(o15, h15, l15, c15)
        bear_candle = is_bearish_candle(o15, h15, l15, c15)
        
        direction = "NONE"
        entry_reason = ""
        
        # BUY: Near support + bullish reversal candle
        if bull_candle and dist_to_s <= SR_BUFFER:
            direction = "BUY"
            entry_reason = f"supp_bounce_${dist_to_s:.1f}"
        # SELL: Near resistance + bearish reversal candle
        elif bear_candle and dist_to_r <= SR_BUFFER:
            direction = "SELL"
            entry_reason = f"res_reject_${dist_to_r:.1f}"
        
        if direction == "NONE": continue
        
        # Sizing
        sd = atr * SL_MULT
        sl_price = pc - sd if direction == "BUY" else pc + sd
        if direction == "BUY" and sl_price >= pc: continue
        if direction == "SELL" and sl_price <= pc: continue
        
        tp_price = pc + (sd * TP_MULT) if direction == "BUY" else pc - (sd * TP_MULT)
        lot = max(0.01, round((bal * RISK_PER_TRADE) / (sd * 100), 2))
        lot = max(0.01, min(lot, 5.0))
        
        pos.append({
            "entry": pc, "sl": round(sl_price,2), "tp": round(tp_price,2),
            "lot": lot, "dir": direction, "open_time": ct, "reason_entry": entry_reason,
            "_mfe": 0, "trail": False
        })
    
    # Results
    nt = len(closed)
    if nt == 0: print("\n  No trades."); return
    df = pd.DataFrame(closed)
    net = df['pnl'].sum()
    wins = len(df[df['pnl']>0]); losses = len(df[df['pnl']<=0])
    wr = wins/nt*100
    
    win_pnls = [t['pnl'] for t in closed if t['pnl']>0]
    loss_pnls = [t['pnl'] for t in closed if t['pnl']<=0]
    pf = abs(sum(win_pnls)/sum(loss_pnls)) if loss_pnls and sum(loss_pnls)!=0 else 999
    
    eq = np.cumsum([BAL]+[t['pnl'] for t in closed])
    peak_e = np.maximum.accumulate(eq); mdd = ((peak_e-eq)/peak_e*100).max()
    
    reasons = {}; dirs = {"BUY":0,"SELL":0}
    for t in closed:
        reasons[t.get('reason','?')] = reasons.get(t.get('reason','?'),0)+1
        dirs[t.get('dir','?')] = dirs.get(t.get('dir','?'),0)+1
    
    print("\n"+"="*60)
    print("  PURE S/R BACKTEST RESULTS")
    print("="*60)
    print(f"  Balance:  ${BAL:.0f} -> ${BAL+net:.2f} ({(net/BAL)*100:+.1f}%)")
    print(f"  Trades:   {nt} | Win: {wr:.1f}% ({wins}W/{losses}L)")
    print(f"  PF:       {pf:.2f} | MaxDD: {mdd:.1f}%")
    print(f"  BUY={dirs.get('BUY',0)} SELL={dirs.get('SELL',0)}")
    print(f"  Reasons:  {json.dumps(reasons)}")
    if wins>0: print(f"  Avg Win:  ${np.mean(win_pnls):+.2f}")
    if losses>0: print(f"  Avg Loss: ${np.mean(loss_pnls):+.2f}")
    
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sr_only_results.txt")
    with open(path,"w") as f:
        f.write(f"PURE S/R BACKTEST\n{net:+.2f} | {wr:.1f}% WR | {nt} trades\n")
    print(f"\n  Saved: {path}")
    return net, wr, nt  # Return for caller

if __name__ == "__main__":
    net, wr, nt = run()