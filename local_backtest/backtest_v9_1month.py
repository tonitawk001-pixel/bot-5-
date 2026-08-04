"""
BACKTEST v9.0 — ONE MONTH with ALL NEW FEATURES
=================================================
Features tested:
  - Pattern scoring boost (+30)
  - Multi-TF S/R filter (H4/H1/M15/M5)
  - Progressive multi-tier trailing stop
  - Partial close at 80% TP
  - Breakeven at $40 (untouched)

Period: Last 30 calendar days from today
Data: Yahoo Finance (resampled to H4/H1/M15/M5)
"""

import os, sys, warnings, json
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Suppress all logging
import logging
logging.disable(logging.CRITICAL)

# ── CONFIG (matches main_super.py v9) ──────────────────────────
SYMBOL = "XAUUSD"
STARTING_BALANCE = 500.0
MIN_SCORE = 30
SL_ATR_MULT = 2.5
TP_ATR_MULT = 5.0
BE_ATR_MULT = 2.0
BE_BUFFER_POINTS = 50
BE_PROFIT_USD = 40
FIXED_RISK = 0.05
MIN_SCORE_BUY = 35
MIN_SCORE_SELL = 35

# Progressive trailing
TRAIL_STAGE1_PCT = 0.30
TRAIL_STAGE2_PCT = 0.50
TRAIL_STAGE3_PCT = 0.70
TRAIL_STAGE1_MULT = 0.60
TRAIL_STAGE2_MULT = 0.35
TRAIL_STAGE3_MULT = 0.15

# Partial close
PARTIAL_CLOSE_ENABLED = True
PARTIAL_CLOSE_PCT = 0.80

# Spread cost (realistic)
SPREAD_COST_PIP = 0.50

# ── DATA DOWNLOAD ──────────────────────────────────────────────
def download_gold_data(days=35):
    """Download gold futures data from Yahoo Finance (1-hour, resample to all TFs)."""
    import yfinance as yf

    print(f"\n  [Download] Fetching XAUUSD data from Yahoo Finance (last {days} days, 1h)...")
    # Try XAUUSD first, then GC=F
    try:
        df = yf.download("GC=F", period=f"{days}d", interval="1h", progress=False)
    except:
        df = yf.download("XAUUSD=X", period=f"{days}d", interval="1h", progress=False)

    if df.empty:
        print("  [ERROR] No data downloaded!")
        return None, None, None, None, None

    # Flatten multi-index columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]

    # Ensure we have OHLC
    required = {'open', 'high', 'low', 'close'}
    if not required.issubset(set(df.columns)):
        print(f"  [ERROR] Missing columns. Got: {list(df.columns)}")
        return None, None, None, None, None

    # Add tick_volume col if missing
    if 'volume' in df.columns:
        df['tick_volume'] = df['volume']
    else:
        df['tick_volume'] = 1

    # Resample to M5, M15, H1, H4
    print(f"  [Resample] 1h → M5, M15, H1, H4...")

    # Need to create pseudo-M1 for accurate resample
    def resample_ohlcv(df_1h, rule):
        return df_1h.resample(rule).agg({
            'open': 'first', 'high': 'max', 'low': 'min',
            'close': 'last', 'tick_volume': 'sum',
        }).dropna()

    m5 = resample_ohlcv(df, '5min')
    m15 = resample_ohlcv(df, '15min')
    h1 = resample_ohlcv(df, '1h')  # same as original
    h4 = resample_ohlcv(df, '4h')

    print(f"  [Download] M5:{len(m5) if m5 is not None else 0} M15:{len(m15)} H1:{len(h1)} H4:{len(h4)} candles")
    return m5, m15, h1, h4, df


# ── STRATEGY IMPORT ────────────────────────────────────────────
from trading_bot.indicators.technical_indicators import compute_all_indicators, compute_sr_levels
from trading_bot.strategy.gold_scalping_strategy import GoldScalpingStrategy

# Import new modules
sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'trading_bot_mt5'))
import candle_patterns as cp
import sr_levels_mtf as sr_mtf


# ── RUN BACKTEST ───────────────────────────────────────────────
def run_backtest_v9():
    print("=" * 60)
    print("  BACKTEST v9.0 — 1 MONTH GOLD")
    print("  Pattern + S/R + Progressive Trail + Partial Close")
    print("=" * 60)

    # Download data
    m5_raw, m15_raw, h1_raw, h4_raw, raw_1h = download_gold_data(days=35)

    if m15_raw is None or len(m15_raw) < 100:
        print("  [ERROR] Not enough M15 data")
        return

    # Set up date range
    end_date = m15_raw.index[-1]
    start_date = end_date - timedelta(days=30)
    warmup_start = start_date - timedelta(days=5)

    # Slice data
    def slice_data(df, warmup_start, end_date):
        if df is None:
            return None
        return df[(df.index >= warmup_start) & (df.index <= end_date)].sort_index()

    m15_df = slice_data(m15_raw, warmup_start, end_date)
    m5_df = slice_data(m5_raw, warmup_start, end_date) if m5_raw is not None else None
    h1_df = slice_data(h1_raw, warmup_start, end_date) if h1_raw is not None else None
    h4_df = slice_data(h4_raw, warmup_start, end_date) if h4_raw is not None else None

    if m5_df is None:
        # Resample M15 to pseudo M5
        print("  [Warning] No M5 data — using M15 as fallback")
        m5_df = m15_df.copy()

    print(f"\n  Backtest Period: {start_date.date()} to {end_date.date()}")
    print(f"  M15 candles: {len(m15_df)} | M5: {len(m5_df)} | H1: {len(h1_df) if h1_df is not None else 0} | H4: {len(h4_df) if h4_df is not None else 0}")

    # Create strategy and S/R engine
    strategy = GoldScalpingStrategy()
    strategy._max_trades_per_day = 50
    strategy._max_open_positions = 3
    sr_engine = sr_mtf.MultiTFSupportResistance()

    balance = STARTING_BALANCE
    positions = []
    closed_trades = []
    daily_pnl = 0.0
    consecutive_losses = 0
    halt_until = None
    last_date = None
    last_processed = None
    daily_trades = 0

    # Track partial closes for reporting
    partial_closes = []

    # Main loop — iterate M15 bars
    for idx in range(200, len(m15_df)):
        ct = m15_df.index[idx]
        price = float(m15_df["close"].iloc[idx])

        if ct < start_date:
            continue

        # Friday close
        if ct.weekday() == 4 and ct.hour >= 21:
            for p in positions:
                pnl = (price - p['entry']) * p['lot'] * 100 if p['dir'] == "BUY" else (p['entry'] - price) * p['lot'] * 100
                pnl -= SPREAD_COST_PIP * p['lot'] * 100
                balance += pnl
                p['pnl'] = pnl; p['reason'] = "FRIDAY_CLOSE"; p['close_price'] = price; p['close_time'] = ct
                closed_trades.append(p)
            positions = []
            continue

        # Daily reset
        ct_date = ct.date() if hasattr(ct, 'date') else ct
        if hasattr(ct_date, '__call__'):
            ct_date = pd.Timestamp(ct).date()
        if last_date != ct_date:
            daily_pnl = 0.0
            last_date = ct_date
            daily_trades = 0

        if halt_until and ct < halt_until:
            continue
        if daily_pnl <= -balance * 0.05:
            continue

        # Get window data
        m5_window = m5_df[m5_df.index <= ct].tail(100).copy()
        m15_window = m15_df.iloc[max(0, idx-100):idx+1].copy()
        h1_window = h1_df[h1_df.index <= ct].tail(50).copy() if h1_df is not None else None
        h4_window = h4_df[h4_df.index <= ct].tail(50).copy() if h4_df is not None else None

        if len(m5_window) < 50 or len(m15_window) < 50:
            continue

        # Compute indicators
        ind5 = compute_all_indicators(m5_window)
        ind15 = compute_all_indicators(m15_window)

        if ind5 is None or ind15 is None:
            continue

        atr_val = float(ind15["atr"].iloc[-1]) if ind15.get("atr") is not None and len(ind15["atr"]) > 0 else 1.0
        if atr_val < 0.3:
            continue

        # ── POSITION MANAGEMENT ──────────────────────────
        surviving_positions = []
        for p in positions:
            entry, direction, sL, tp, lot = p['entry'], p['dir'], p['sl'], p['tp'], p['lot']
            pv = lot * 100
            sl_distance = abs(entry - sL) if sL and sL > 0 else atr_val * SL_ATR_MULT
            tp_dist = abs(tp - entry) if tp and tp > 0 else sl_distance * (TP_ATR_MULT / SL_ATR_MULT)
            progress_pct = 0

            # BE trigger
            if not p.get('be', False) and p.get('be_target'):
                if (direction == "BUY" and price >= p['be_target']) or (direction == "SELL" and price <= p['be_target']):
                    p['be'] = True
                    p['sl'] = entry  # Move SL to entry

            # BE at $40 profit (same as live)
            if not p.get('be', False) and BE_PROFIT_USD > 0:
                profit_points = BE_PROFIT_USD / (lot * 100) if lot > 0 else 999
                be_price = entry + profit_points if direction == "BUY" else entry - profit_points
                if (direction == "BUY" and price >= be_price) or (direction == "SELL" and price <= be_price):
                    p['be'] = True
                    p['sl'] = entry + (BE_BUFFER_POINTS * 0.01) if direction == "BUY" else entry - (BE_BUFFER_POINTS * 0.01)

            # Progressive trailing
            if direction == "BUY":
                profit = price - entry
                progress_pct = profit / tp_dist if tp_dist > 0 else 0
            else:
                profit = entry - price
                progress_pct = profit / tp_dist if tp_dist > 0 else 0

            trail_mult = 999
            if progress_pct >= TRAIL_STAGE3_PCT:
                trail_mult = TRAIL_STAGE3_MULT
            elif progress_pct >= TRAIL_STAGE2_PCT:
                trail_mult = TRAIL_STAGE2_MULT
            elif progress_pct >= TRAIL_STAGE1_PCT:
                trail_mult = TRAIL_STAGE1_MULT

            if trail_mult < 999:
                new_sl = price - (sl_distance * trail_mult) if direction == "BUY" else price + (sl_distance * trail_mult)
                if direction == "BUY" and new_sl > p['sl'] + 0.1:
                    p['sl'] = round(new_sl, 2)
                    p['trail_stage'] = f"s{int(progress_pct*100)}"
                elif direction == "SELL" and new_sl < p['sl'] - 0.1:
                    p['sl'] = round(new_sl, 2)
                    p['trail_stage'] = f"s{int(progress_pct*100)}"

            # Partial close at 80% TP
            if PARTIAL_CLOSE_ENABLED and progress_pct >= PARTIAL_CLOSE_PCT and not p.get('partial_done'):
                half_lot = round(lot / 2, 2)
                if half_lot >= 0.01:
                    pnl_partial = profit * half_lot * 100
                    pnl_partial -= SPREAD_COST_PIP * half_lot * 100
                    balance += pnl_partial
                    daily_pnl += pnl_partial
                    p['lot'] = round(lot - half_lot, 2)
                    p['partial_done'] = True
                    partial_closes.append({
                        'close_time': ct, 'dir': direction, 'entry': entry,
                        'close_price': price, 'pnl': round(pnl_partial, 2),
                        'lot': half_lot, 'progress': round(progress_pct * 100, 1),
                    })

            # Exit check
            sL_now, tp_now = p['sl'], p['tp']
            hit = False; pnl = 0.0; reason = ""

            if direction == "BUY":
                if tp_now and price >= tp_now:
                    pnl = (tp_now - entry) * p['lot'] * 100
                    reason = "TP"; hit = True
                elif sL_now and price <= sL_now:
                    pnl = (sL_now - entry) * p['lot'] * 100
                    reason = "TRAIL" if p.get('trail_stage') else ("BE_SL" if p.get('be') else "SL")
                    hit = True
            else:
                if tp_now and price <= tp_now:
                    pnl = (entry - tp_now) * p['lot'] * 100
                    reason = "TP"; hit = True
                elif sL_now and price >= sL_now:
                    pnl = (entry - sL_now) * p['lot'] * 100
                    reason = "TRAIL" if p.get('trail_stage') else ("BE_SL" if p.get('be') else "SL")
                    hit = True

            if hit:
                pnl -= SPREAD_COST_PIP * p['lot'] * 100
                balance += pnl
                daily_pnl += pnl
                p['pnl'] = round(pnl, 2)
                p['reason'] = reason
                p['close_price'] = price
                p['close_time'] = ct
                closed_trades.append(p)
                if pnl < 0:
                    consecutive_losses += 1
                else:
                    consecutive_losses = 0
            else:
                surviving_positions.append(p)

        positions = surviving_positions

        # ── NEW ENTRY ────────────────────────────────────
        if len(positions) >= 3:
            continue

        # Strategy analysis
        try:
            empty_m1 = {"rsi": pd.Series([50]), "emas": pd.DataFrame(), "macd": pd.Series([0])}
            result = strategy.analyze(
                m1_indicators=empty_m1, m5_indicators=ind5, m15_indicators=ind15,
                m1_ohlcv=m5_window.tail(20), m5_ohlcv=m5_window, m15_ohlcv=m15_window, news_context=None
            )
        except:
            continue

        direction = result.get("direction", "NONE")
        score = result.get("setup_score", 0)

        if direction == "NONE" or score < MIN_SCORE:
            continue

        # Boost score with candle confirmation
        try:
            swing_levels = cp.detect_swing_levels(m15_window)
            candle_analysis = cp.analyze_full(m15_window, swing_levels)
            candle_signal = candle_analysis.get("signal", "NONE")
            candle_conf = candle_analysis.get("confidence", 0)
            if direction != "NONE" and candle_signal == direction:
                boost = max(1, candle_conf // 3)
                score = min(100, score + boost)
        except:
            pass

        # Multi-TF S/R filter
        blocked_by_sr = False
        try:
            if h4_window is not None or h1_window is not None:
                mtf_sr = sr_engine.compute_all(h4_window, h1_window, m15_window, m5_window, price)
                blocked, sr_reason = sr_engine.is_in_no_trade_zone(
                    price, direction,
                    mtf_sr.get("no_buy_zones", []),
                    mtf_sr.get("no_sell_zones", []),
                    buffer_points=5.0,
                )
                if blocked:
                    blocked_by_sr = True
        except:
            pass

        if blocked_by_sr:
            continue

        # Directional min score
        min_sc = MIN_SCORE_BUY if direction == "BUY" else MIN_SCORE_SELL
        if score < min_sc:
            continue

        # RSI filter
        try:
            rsi5 = ind5['rsi'].iloc[-1]
            rsi15 = ind15['rsi'].iloc[-1]
            if direction == "BUY" and (rsi5 < 25 or rsi15 < 25):
                continue
            if direction == "SELL" and (rsi5 > 75 or rsi15 > 75):
                continue
        except:
            pass

        # TP/SL Calculation
        sl_dist = atr_val * SL_ATR_MULT
        tp_dist = atr_val * TP_ATR_MULT

        if direction == "BUY":
            sl = round(price - sl_dist, 2)
            tp = round(price + tp_dist, 2)
        else:
            sl = round(price + sl_dist, 2)
            tp = round(price - tp_dist, 2)

        risk_pct = FIXED_RISK
        if score >= 70:
            risk_pct = 0.10
        lot = max(0.01, round((balance * risk_pct) / (sl_dist * 100), 2))
        if len(positions) == 1:
            lot *= 0.5
        lot = max(0.01, round(lot, 2))

        be_target = price + (atr_val * BE_ATR_MULT if direction == "BUY" else -atr_val * BE_ATR_MULT)
        daily_trades += 1

        positions.append({
            "entry": price, "sl": sl, "tp": tp, "lot": lot, "dir": direction,
            "open_time": ct, "score": score, "be_target": be_target, "be": False,
            "partial_done": False, "_high": price, "_low": price,
        })

    # ── RESULTS ──────────────────────────────────────────
    total_trades = len(closed_trades)
    total_partials = len(partial_closes)

    if total_trades == 0:
        print("\n  No trades executed in this period.")
        return

    # Include partial closes in P&L
    partial_pnl = sum(pc['pnl'] for pc in partial_closes)
    df_res = pd.DataFrame(closed_trades)
    net_pnl = df_res['pnl'].sum() + partial_pnl
    wins = len(df_res[df_res['pnl'] > 0])
    losses = len(df_res[df_res['pnl'] <= 0])
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0

    # Equity curve
    all_trades = sorted(
        [{"time": t['close_time'], "pnl": t['pnl']} for t in closed_trades if 'close_time' in t] +
        [{"time": pc['close_time'], "pnl": pc['pnl']} for pc in partial_closes],
        key=lambda x: x['time']
    )
    eq = np.cumsum([STARTING_BALANCE] + [t['pnl'] for t in all_trades])
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / peak * 100
    maxdd = dd.max() if len(dd) > 0 else 0

    avg_win = df_res[df_res['pnl'] > 0]['pnl'].mean() if wins > 0 else 0
    avg_loss = df_res[df_res['pnl'] <= 0]['pnl'].mean() if losses > 0 else 0

    win_pnls = [t['pnl'] for t in closed_trades if t['pnl'] > 0]
    loss_pnls = [t['pnl'] for t in closed_trades if t['pnl'] <= 0]
    pf = abs(sum(win_pnls) / sum(loss_pnls)) if loss_pnls and sum(loss_pnls) != 0 else float('inf')

    reason_counts = {}
    for t in closed_trades:
        r = t.get('reason', '?')
        reason_counts[r] = reason_counts.get(r, 0) + 1

    dir_counts = {"BUY": 0, "SELL": 0}
    for t in closed_trades:
        dir_counts[t['dir']] = dir_counts.get(t['dir'], 0) + 1

    # Print results
    print("\n" + "=" * 60)
    print("  BACKTEST RESULTS — V9.0 (ALL FEATURES)")
    print("=" * 60)
    print(f"  Starting Balance:    ${STARTING_BALANCE:.2f}")
    print(f"  Final Balance:       ${STARTING_BALANCE + net_pnl:.2f}")
    print(f"  Net Profit:          ${net_pnl:+.2f} ({(net_pnl / STARTING_BALANCE) * 100:+.1f}%)")
    print(f"  Partial Close PnL:   ${partial_pnl:+.2f} ({total_partials} partials)")
    print(f"  Max Drawdown:        {maxdd:.1f}%")
    print(f"  Profit Factor:       {pf:.2f}")
    print(f"  Win Rate:            {win_rate:.1f}% ({wins}W / {losses}L)")
    print(f"  Total Trades:        {total_trades}")
    print(f"  Avg Win:             ${avg_win:+.2f}")
    print(f"  Avg Loss:            ${avg_loss:+.2f}")
    print(f"  Direction:           BUY={dir_counts.get('BUY',0)} SELL={dir_counts.get('SELL',0)}")
    print(f"  Exit Reasons:        {json.dumps(reason_counts)}")

    # Save report
    report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backtest_v9_results.txt")
    with open(report_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("BACKTEST v9.0 — 1 MONTH GOLD\n")
        f.write(f"Period: {start_date.date()} to {end_date.date()}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Net Profit:          ${net_pnl:+.2f} ({(net_pnl/500)*100:+.1f}%)\n")
        f.write(f"Partial Close PnL:   ${partial_pnl:+.2f}\n")
        f.write(f"Max Drawdown:        {maxdd:.1f}%\n")
        f.write(f"Profit Factor:       {pf:.2f}\n")
        f.write(f"Win Rate:            {win_rate:.1f}% ({wins}W/{losses}L)\n")
        f.write(f"Total Trades:        {total_trades}\n")
        f.write(f"Avg Win/Loss:        ${avg_win:+.2f} / ${avg_loss:+.2f}\n")
        f.write(f"Partial Closes:      {total_partials}\n")
        f.write(f"Exit Reasons:        {json.dumps(reason_counts)}\n")
        f.write(f"\nFeatures Active:\n")
        f.write(f"  - Pattern scoring: +30 (up from +15)\n")
        f.write(f"  - Multi-TF S/R filter: H4/H1/M15/M5\n")
        f.write(f"  - Progressive trailing: {TRAIL_STAGE1_PCT*100:.0f}%/{TRAIL_STAGE2_PCT*100:.0f}%/{TRAIL_STAGE3_PCT*100:.0f}%\n")
        f.write(f"  - Partial close: {PARTIAL_CLOSE_PCT*100:.0f}% TP\n")
        f.write(f"  - Breakeven: ${BE_PROFIT_USD}\n")

    print(f"\n  Report saved: {report_path}")

    # Trade log
    if total_trades > 0:
        trade_log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backtest_v9_trades.csv")
        export_df = df_res[['open_time', 'close_time', 'dir', 'entry', 'close_price', 'sl', 'tp', 'lot', 'pnl', 'score', 'reason']].copy()
        export_df.to_csv(trade_log_path, index=False)
        print(f"  Trades list: {trade_log_path}")


if __name__ == "__main__":
    run_backtest_v9()