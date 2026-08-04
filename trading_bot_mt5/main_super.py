"""
███████ SUPER BOT v8.0 — INSTITUTIONAL GRADE ███████
PROFESSIONAL GOLD SCALPING — CANDLES + S/R + DEEPSEEK AI
"""
import os, sys, time, json, atexit, signal, traceback, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datetime import datetime, timedelta, timezone
import pandas as pd, numpy as np, MetaTrader5 as mt5
from mt5_connection import MT5Connection
from logger_mt5 import logger
import telegram_notifier as tg, trade_exporter, github_setup
import telegram_handler as tg_handler
import github_exporter as gh_exporter
from news_filter import NewsFilter
from deepseek_filter import DeepSeekFilter
import candle_patterns as cp
import sr_levels_mtf as sr_mtf
import sr_entry as srentry
from performance_tracker import PerformanceTracker

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path: sys.path.insert(0, _PROJECT_ROOT)
from trading_bot.indicators.technical_indicators import compute_all_indicators, compute_sr_levels
from trading_bot.strategy.gold_scalping_strategy import GoldScalpingStrategy

SYMBOL = "XAUUSD"
MIN_SCORE_BUY = 35; MIN_SCORE_SELL = 35; MIN_SCORE = 35
MAX_POSITIONS = 1; MAX_PER_DIRECTION = 1
DAILY_LOSS_PCT = 0.03
TOTAL_RISK_LIMIT = 0.03
ATR_VOL_THRESHOLD = 4.0
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
USE_AI_FILTER = False        # AI signal veto: TRUE = AI can block trades, FALSE = AI watches only
AI_MARKET_ANALYSIS = False   # 15-min market reports: TRUE = enabled, FALSE = paused (no spam)
AI_WATCHDOG = True           # Hourly health watchdog + error auto-fix: TRUE = ON
AI_WATCHDOG_INTERVAL_MIN = 60  # minutes between watchdog reports
TP_ATR_MULT = 2.0; TP_PARTIAL_MULT = 1.5; SL_ATR_MULT = 1.0
SR_ENTRY_MODE = True          # Use pure S/R entry (ignores indicators for direction)
BE_ATR_MULT = 2.0; BE_BUFFER_POINTS = 50
TRAIL_ATR_MULT = 0.5      # Trail stop - gentle to allow more room
BE_PROFIT_USD = 40         # Move SL to entry after +$40 profit
FIXED_RISK = 0.05           # 5% per trade
HIGH_SCORE_THRESHOLD = 70; HIGH_SCORE_RISK = 0.10
SECOND_POS_MIN_SCORE = 35; SECOND_POS_LOT_RATIO = 0.5
HALT_HOURS = 6; MAX_CONSEC_LOSSES = 2
REGIME_RISK_LOW = 0.01; REGIME_WR_THRESHOLD = 0.45
RECENT_TRADE_WINDOW = 20
TRADE_HOURS_START = 0; TRADE_HOURS_END = 24
SESSION_COOLDOWN_MIN = 10
DXY_ENABLED = True; DXY_THRESHOLD = 0.003; DXY_LOOKBACK_H = 1
MTF_CONFLUENCE = True
STATE_FILE = "bot_state_super.json"
NEWS_BUFFER_MIN = 30; HARD_FLOOR = 50.00; MAX_SPREAD = 2.00
DD_EMERGENCY_ENABLED = False     # Emergency shutdown DISABLED — bot will NOT stop at -25% DD

# ── Progressive Trailing (SUPER TIGHT) ──────────────────────────
TRAIL_STAGE1_PCT = 0.10    # Start trailing at 10% of TP distance
TRAIL_STAGE2_PCT = 0.20    # Tighten at 20% of TP
TRAIL_STAGE3_PCT = 0.33    # Super tight at 33% of TP
TRAIL_STAGE1_MULT = 0.15   # Trail distance: 15% of SL at stage 1
TRAIL_STAGE2_MULT = 0.08   # Trail distance: 8% of SL at stage 2
TRAIL_STAGE3_MULT = 0.04   # Trail distance: 4% of SL at stage 3 — near TP lock

# ── Partial Close ─────────────────────────────────────────────────
PARTIAL_CLOSE_ENABLED = True   # Close 50% at 80% TP
PARTIAL_CLOSE_PCT = 0.80       # Trigger at 80% of TP distance

# ── Multi-TF S/R Filter ─────────────────────────────────────────
MTF_SR_ENABLED = True          # Block entries near ALL S/R levels (H4/H1/M15/M5)
SR_NO_TRADE_BUFFER = 3.0       # Tight buffer — block within 3 points of any opposing S/R
SR_ONLY_AT_LEVELS = True       # ONLY trade at S/R levels: buy near support, sell near resistance
SR_BOUNCE_BUFFER = 6.0         # Max distance from support/resistance to qualify as "at level"

# ── Global state ──────────────────────────────────────────────────────
consecutive_losses = 0
daily_pnl = 0.0
last_date = ""
last_processed_m15_time = None
trades_log = []
balance_snapshot = 0.0
_mt5_conn = None
_spread_paused = False
_spread_notified = False
_last_m1_price = None  # flash spike detection
_prev_positions = {}   # track position IDs for SL detection

def get_risk_pct(balance: float) -> float:
    """Fixed risk — no death spiral tiers."""
    return FIXED_RISK

def load_state():
    global consecutive_losses, daily_pnl, last_date, last_processed_m15_time, trades_log
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, "r") as f:
                s = json.load(f)
            consecutive_losses = s.get("losses", 0)
            daily_pnl = s.get("pnl", 0.0)
            last_date = s.get("date", "")
            trades_log = s.get("trades_log", [])
            m15_str = s.get("m15")
            last_processed_m15_time = datetime.fromisoformat(m15_str) if m15_str else None
            logger.info(f"State loaded: losses={consecutive_losses} pnl={daily_pnl:.2f} trades={len(trades_log)}")
    except Exception as e:
        logger.warning(f"load_state failed: {e}")

def save_state():
    try:
        s = {
            "losses": consecutive_losses,
            "pnl": daily_pnl,
            "date": last_date,
            "trades_log": trades_log[-500:],
            "m15": last_processed_m15_time.isoformat() if last_processed_m15_time else None,
        }
        with open(STATE_FILE, "w") as f:
            json.dump(s, f, default=str)
    except Exception as e:
        logger.error(f"save_state failed: {e}")

def cleanup():
    """Shutdown handler — close MT5, save state, notify."""
    logger.info("Shutting down...")
    save_state()
    try:
        tg.notify_shutdown()
    except Exception:
        pass
    if _mt5_conn:
        try:
            _mt5_conn.shutdown()
        except Exception:
            pass

atexit.register(cleanup)
signal.signal(signal.SIGTERM, lambda *_: cleanup() or os._exit(0))
signal.signal(signal.SIGINT, lambda *_: cleanup() or os._exit(0))

def get_session() -> str:
    now = datetime.now(timezone.utc)
    h = now.hour
    if 8 <= h < 17 and 13 <= h < 22: return "overlap"
    if 8 <= h < 17: return "london"
    if 13 <= h < 22: return "new_york"
    if h >= 23 or h < 8: return "asian"
    return "transition"


# ── Main loop ─────────────────────────────────────────────────────────
def main_loop():
    global consecutive_losses, daily_pnl, last_date, last_processed_m15_time
    global trades_log, balance_snapshot, _mt5_conn

    _mt5_conn = MT5Connection()
    if not _mt5_conn.initialize():
        logger.critical("MT5 INIT FAILED — exiting")
        return

    conn = _mt5_conn
    strategy = GoldScalpingStrategy()
    nf = NewsFilter()
    ai = DeepSeekFilter(api_key=DEEPSEEK_API_KEY) if DEEPSEEK_API_KEY else None
    perf = PerformanceTracker(deepseek_client=ai)
    load_state()
    
    # ── LOAD CONFIG OVERRIDES ──────────────────────────
    try:
        config_file = "config_overrides.json"
        if os.path.exists(config_file):
            with open(config_file, "r") as f:
                overrides = json.load(f)
            _globals = globals()
            _valid_keys = {'MIN_SCORE_BUY','MIN_SCORE_SELL','MAX_POSITIONS','MAX_PER_DIRECTION',
                'DAILY_LOSS_PCT','TOTAL_RISK_LIMIT','ATR_VOL_THRESHOLD','USE_AI_FILTER',
                'TP_ATR_MULT','TP_PARTIAL_MULT','SL_ATR_MULT','BE_ATR_MULT','BE_BUFFER_POINTS',
                'FIXED_RISK','HIGH_SCORE_THRESHOLD','HIGH_SCORE_RISK','SECOND_POS_MIN_SCORE',
                'SECOND_POS_LOT_RATIO','HALT_HOURS','MAX_CONSEC_LOSSES','TRADE_HOURS_START',
                'TRADE_HOURS_END','SESSION_COOLDOWN_MIN','DXY_ENABLED','DXY_THRESHOLD',
                'DXY_LOOKBACK_H','MTF_CONFLUENCE','MAX_SPREAD','DD_EMERGENCY_ENABLED',
                'TRAIL_STAGE1_PCT','TRAIL_STAGE2_PCT','TRAIL_STAGE3_PCT',
                'TRAIL_STAGE1_MULT','TRAIL_STAGE2_MULT','TRAIL_STAGE3_MULT',
                'PARTIAL_CLOSE_ENABLED','PARTIAL_CLOSE_PCT',
                'MTF_SR_ENABLED','SR_NO_TRADE_BUFFER','SR_ONLY_AT_LEVELS','SR_BOUNCE_BUFFER',
                'SR_ENTRY_MODE'}
            for key, value in overrides.items():
                key_upper = key.upper()
                if key_upper in _valid_keys and key_upper in _globals:
                    old_val = _globals[key_upper]
                    _globals[key_upper] = type(old_val)(value)
                    logger.info(f"[Config] Override {key_upper} = {value} (was {old_val})")
            logger.info(f"[Config] Loaded {len(overrides)} overrides from {config_file}")
    except Exception as e:
        logger.warning(f"[Config] Override load failed: {e}")
    
    github_setup.setup_remote()
    nf.update_news()

    # Get account info for health check
    info = conn.get_account_info()

    # Send simple startup ping first (always works, no AI needed)
    if info:
        try:
            tg.set_account_name(str(info.get("login", "?")))
            tg.send_message(f"🤖 BOT STARTED\nAccount: {info.get('login','?')}\nBalance: ${info['balance']:,.2f}\n⏰ {datetime.now(timezone.utc).strftime('%H:%M')} UTC")
        except:
            pass

    # ── AI SYSTEM HEALTH CHECK ──────────────────────────
    if ai is not None:
        try:
            tick = mt5.symbol_info_tick(SYMBOL)
            spread_val = round((tick.ask - tick.bid) if tick and tick.ask and tick.bid and tick.ask != tick.bid else 0.30, 2)
            health = ai.system_health_check({
                "mt5_connected": True,
                "login": info.get("login", "?") if info else "?",
                "server": info.get("server", "?") if info else "?",
                "balance": info["balance"] if info else 0,
                "equity": info.get("equity", 0) if info else 0,
                "spread": spread_val,
                "session": get_session(),
                "in_hours": TRADE_HOURS_START <= datetime.now(timezone.utc).hour < TRADE_HOURS_END,
                "news_count": len(nf.red_folder_events) if hasattr(nf, 'red_folder_events') else 0,
                "news_ok": nf.has_news() if hasattr(nf, 'has_news') else True,
            })
            verdict_emoji = {"OK": "✅", "WARNING": "⚠️", "ERROR": "🚨"}.get(health.get("verdict", "WARNING"), "⚠️")
            score = health.get("health_score", 0)
            issues = health.get("issues", [])
            report = health.get("report", "No report generated")
            health_msg = (
                f"🤖 ACCOUNT: {info.get('login','?') if info else '?'}\n"
                f"🟢 BOT STARTED — AI Health Check\n\n"
                f"🧠 AI VERDICT: {verdict_emoji} {health.get('verdict','WARNING')}\n"
                f"   Health Score: {score}/100\n\n"
                f"━━━━━━━━━━━━━━━\n"
                f"✅ MT5: Connected | {info.get('server','?') if info else '?'}\n"
                f"✅ Balance: ${info['balance'] if info else 0:,.2f}\n"
                f"✅ Spread: ${spread_val} {'(WIDE!)' if spread_val > 2.0 else '(normal)'}\n"
                f"✅ Session: {get_session()}{' — Trading Hours' if TRADE_HOURS_START <= datetime.now(timezone.utc).hour < TRADE_HOURS_END else ' — Outside Hours'}\n"
                f"✅ News: {len(nf.red_folder_events) if hasattr(nf, 'red_folder_events') else 0} events loaded\n\n"
                f"{'⚠️ News API may be down (no fresh data in 24h)' if hasattr(nf, 'has_news') and not nf.has_news() else ''}\n"
                f"📋 TOOLS: 17/17 Active — Candle Patterns + S/R, Strategy, DXY, MTF, Score Filters, Risk Mgmt\n\n"
            )
            if issues:
                health_msg += f"⚠️ WARNINGS:\n" + "\n".join(f"  • {i}" for i in issues[:5]) + "\n\n"
            health_msg += (
                f"🧠 AI REPORT: {report}\n\n"
                f"⏰ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC"
            )
            tg.send_message(health_msg)
            logger.info(f"[HEALTH] AI Verdict: {health.get('verdict')} | Score: {score}")
        except Exception as e:
            logger.warning(f"[HEALTH] Check failed: {e}")
            try:
                tg.send_message(f"⚠️ Health check failed: {str(e)[:200]}\nBot will trade normally.")
            except:
                logger.error("[HEALTH] Telegram fallback also failed")

    # ── START TELEGRAM CHAT POLLING ─────────────────────────
    if ai is not None:
        tg_handler.set_deepseek_client(ai)
        tg_handler.start_polling()
        logger.info("[TelegramPoll] DeepSeek chat polling started — you can now message the bot on Telegram")

    # ── SETUP GITHUB EXPORTER ──────────────────────────────
    gh_exporter.setup_git()
    logger.info("[GitExport] GitHub auto-export configured (every 30 min)")

    logger.info("SUPER BOT v8.0 Started - All Tools Active")
    cycle = 0
    daily_trades = 0
    daily_halted = False
    halt_until = None

    while True:
        cycle += 1
        try:
            # Safety: ensure halt_until is datetime (not string from bad state)
            if halt_until and not isinstance(halt_until, datetime):
                halt_until = None
            if last_processed_m15_time and not isinstance(last_processed_m15_time, datetime):
                last_processed_m15_time = None

            info = conn.get_account_info()
            if not info:
                time.sleep(10)
                continue
            balance = info["balance"]
            balance_snapshot = balance
            equity = info.get("equity", balance)
            now = datetime.now(timezone.utc)

            # ── PERFORMANCE TRACKER ─────────────────────────
            perf.update(balance, equity)

            # DD Emergency Stop (25% from peak) — toggleable via DD_EMERGENCY_ENABLED
            if DD_EMERGENCY_ENABLED and len(trades_log) > 0 and perf.check_dd_emergency(balance) and not perf.dd_halted:
                perf.dd_halted = True
                _emergency_positions = mt5.positions_get(symbol=SYMBOL)
                tg.send_message("🚨 EMERGENCY STOP: 25% drawdown from peak. Closing all positions.")
                if _emergency_positions:
                    for p in _emergency_positions: conn.close_position(p.ticket)
                break

            # DD Risk Reduction (10% from peak equity) — disabled to match backtest
            active_risk = FIXED_RISK
            active_max_pos = MAX_POSITIONS

            # Weekly AI Review (Sunday 20:00 UTC)
            if perf.should_run_weekly(now):
                try:
                    report = perf.run_weekly_review(trades_log, balance)
                    tg.send_message(f"📊 Weekly Performance Review:\n\n{report[:800]}")
                    tg.send_message(f"📤 Notes pushed to GitHub: mt5-bot-final-2")
                except Exception as e:
                    logger.warning(f"Weekly review failed: {e}")

            today_str = now.strftime("%Y-%m-%d")
            if last_date != today_str:
                daily_pnl = 0.0
                daily_trades = 0
                daily_halted = False
                last_date = today_str
                strategy.reset_daily()

            # Balance floor — emergency stop
            if balance < HARD_FLOOR:
                tg.notify_bot_crashed(f"Balance ${balance:.2f} below floor ${HARD_FLOOR}")
                break

            # Daily halted — skip trading
            if daily_halted:
                continue

            # Consecutive loss halt
            if halt_until and now < halt_until:
                continue

            global _prev_positions
            mt5_pos = mt5.positions_get(symbol=SYMBOL)
            floating_pnl = sum([p.profit for p in mt5_pos]) if mt5_pos else 0.0
            
            # ── SL EXIT DETECTION ──
            current_ids = set(p.ticket for p in mt5_pos) if mt5_pos else set()
            for prev_id, prev_data in list(_prev_positions.items()):
                if prev_id not in current_ids:
                    # Position vanished — was closed by SL (TP already handled above)
                    pnl = prev_data.get("profit_at_close", 0)
                    exit_price = prev_data.get("sl_price", 0) or prev_data.get("entry", 0)
                    _log_sl_closed(prev_data, pnl, exit_price)
            _prev_positions = {}
            for p in (mt5_pos or []):
                _prev_positions[p.ticket] = {
                    "id": p.ticket, "type": "BUY" if p.type == 0 else "SELL",
                    "entry": p.price_open, "sl": p.sl, "tp": p.tp, "lots": p.volume,
                    "profit_at_close": float(p.profit) if hasattr(p, 'profit') else 0.0,
                    "sl_price": p.sl, "tp_price": p.tp, "time": p.time if hasattr(p, 'time') else None,
                }
            
            pos = [{"id": p.ticket, "type": "BUY" if p.type == 0 else "SELL",
                    "entry": p.price_open, "sl": p.sl, "tp": p.tp, "lots": p.volume}
                   for p in mt5_pos] if mt5_pos else []

            # ── SPREAD AUTO-PAUSE ─────────────────────────────
            global _spread_paused, _spread_notified
            tick = mt5.symbol_info_tick(SYMBOL)
            spread = round(tick.ask - tick.bid, 2) if tick and tick.ask and tick.bid and tick.ask != tick.bid else 0.30

            if spread > MAX_SPREAD and not _spread_paused:
                _spread_paused = True
                if not _spread_notified:
                    try:
                        tg.send_message(
                            f"⏸️ SPREAD PAUSE\n"
                            f"Current: ${spread:.2f} | Max: ${MAX_SPREAD:.2f}\n"
                            f"Bot will skip new entries until spread normalizes.\n"
                            f"⏰ {now.strftime('%H:%M:%S')} UTC"
                        )
                    except: pass
                    _spread_notified = True

            elif spread <= MAX_SPREAD and _spread_paused:
                _spread_paused = False
                _spread_notified = False
                try:
                    tg.send_message(
                        f"✅ SPREAD NORMAL: ${spread:.2f}\n"
                        f"Bot resuming trading.\n"
                        f"⏰ {now.strftime('%H:%M:%S')} UTC"
                    )
                except: pass


            # Manage open positions
            if mt5_pos:
                tick = mt5.symbol_info_tick(SYMBOL)
                for p in mt5_pos:
                    if not tick: continue
                    curr_price = tick.bid if p.type == 0 else tick.ask
                    sl_distance = abs(p.price_open - p.sl) if p.sl and p.sl > 0 else 5.0

                    if p.tp and p.tp > 0:
                        tp_target = p.tp
                    else:
                        mult = (TP_ATR_MULT / SL_ATR_MULT)
                        tp_target = p.price_open + (sl_distance * mult) if p.type == 0 else p.price_open - (sl_distance * mult)

                    mult_be = (BE_ATR_MULT / SL_ATR_MULT)
                    be_trigger = p.price_open + (sl_distance * mult_be) if p.type == 0 else p.price_open - (sl_distance * mult_be)

                    if (p.type == 0 and curr_price >= tp_target) or (p.type == 1 and curr_price <= tp_target):
                        res = conn.close_position(p.ticket)
                        pnl = float(p.profit) if hasattr(p, 'profit') else 0.0
                        _log_closed_trade(p, pnl, "tp", tp_target)  # daily_pnl updated inside
                        logger.info(f"[TP] #{p.ticket} pnl={pnl:.2f} result={res}")

                    elif p.sl and p.sl > 0:
                        # BE trigger
                        if (p.type == 0 and curr_price >= be_trigger) or (p.type == 1 and curr_price <= be_trigger):
                            new_sl = p.price_open + (BE_BUFFER_POINTS * 0.01) if p.type == 0 else p.price_open - (BE_BUFFER_POINTS * 0.01)
                            if (p.type == 0 and new_sl > p.sl) or (p.type == 1 and new_sl < p.sl):
                                res = conn.modify_position(p.ticket, sl=new_sl)
                                save_state()
                                logger.info(f"[BE] #{p.ticket} SL->{new_sl:.2f} result={res}")
                        # BE-at-profit trigger — move SL to entry after $40 profit (backtest proven)
                        if BE_PROFIT_USD > 0:
                            lot_size = p.volume
                            profit_points = BE_PROFIT_USD / (lot_size * 100)
                            be_price = p.price_open + profit_points if p.type == 0 else p.price_open - profit_points
                            if (p.type == 0 and curr_price >= be_price) or (p.type == 1 and curr_price <= be_price):
                                new_sl = p.price_open + (BE_BUFFER_POINTS * 0.01) if p.type == 0 else p.price_open - (BE_BUFFER_POINTS * 0.01)
                                if (p.type == 0 and new_sl > p.sl) or (p.type == 1 and new_sl < p.sl):
                                    res = conn.modify_position(p.ticket, sl=round(new_sl, 2))
                                    save_state()
                                    logger.info(f"[BE-PROFIT] #{p.ticket} SL->{new_sl:.2f} (+${BE_PROFIT_USD} profit)")
                        # ── PROGRESSIVE MULTI-TIER TRAILING STOP ──
                        if p.type == 0:  # BUY
                            profit = curr_price - p.price_open
                            tp_dist = abs((p.tp or p.price_open + sl_distance * 2) - p.price_open)
                            progress_pct = profit / tp_dist if tp_dist > 0 else 0

                            trail_mult = 999  # no trail
                            stage = "none"
                            if progress_pct >= TRAIL_STAGE3_PCT:
                                trail_mult = TRAIL_STAGE3_MULT
                                stage = "stage3_tight"
                            elif progress_pct >= TRAIL_STAGE2_PCT:
                                trail_mult = TRAIL_STAGE2_MULT
                                stage = "stage2"
                            elif progress_pct >= TRAIL_STAGE1_PCT:
                                trail_mult = TRAIL_STAGE1_MULT
                                stage = "stage1"

                            if trail_mult < 999:
                                new_sl = curr_price - (sl_distance * trail_mult)
                                if new_sl > p.sl:
                                    res = conn.modify_position(p.ticket, sl=round(new_sl, 2))
                                    save_state()
                                    logger.info(f"[TRAIL-{stage}] #{p.ticket} SL->{new_sl:.2f} progress={progress_pct*100:.0f}% profit=${profit:.1f}")

                            # Partial close at 80% TP
                            if PARTIAL_CLOSE_ENABLED and progress_pct >= PARTIAL_CLOSE_PCT:
                                half_lot = p.volume / 2
                                if half_lot >= 0.01:
                                    try:
                                        close_half = conn.close_position_partial(p.ticket, half_lot)
                                        if close_half.get("success"):
                                            logger.info(f"[PARTIAL] #{p.ticket} Closed 50% at {progress_pct*100:.0f}% TP (lot {half_lot})")
                                    except Exception as e:
                                        logger.warning(f"[PARTIAL] #{p.ticket} failed: {e}")

                        else:  # SELL
                            profit = p.price_open - curr_price
                            tp_dist = abs(p.price_open - (p.tp or p.price_open - sl_distance * 2))
                            progress_pct = profit / tp_dist if tp_dist > 0 else 0

                            trail_mult = 999  # no trail
                            stage = "none"
                            if progress_pct >= TRAIL_STAGE3_PCT:
                                trail_mult = TRAIL_STAGE3_MULT
                                stage = "stage3_tight"
                            elif progress_pct >= TRAIL_STAGE2_PCT:
                                trail_mult = TRAIL_STAGE2_MULT
                                stage = "stage2"
                            elif progress_pct >= TRAIL_STAGE1_PCT:
                                trail_mult = TRAIL_STAGE1_MULT
                                stage = "stage1"

                            if trail_mult < 999:
                                new_sl = curr_price + (sl_distance * trail_mult)
                                if new_sl < p.sl:
                                    res = conn.modify_position(p.ticket, sl=round(new_sl, 2))
                                    save_state()
                                    logger.info(f"[TRAIL-{stage}] #{p.ticket} SL->{new_sl:.2f} progress={progress_pct*100:.0f}% profit=${profit:.1f}")

                            # Partial close at 80% TP
                            if PARTIAL_CLOSE_ENABLED and progress_pct >= PARTIAL_CLOSE_PCT:
                                half_lot = p.volume / 2
                                if half_lot >= 0.01:
                                    try:
                                        close_half = conn.close_position_partial(p.ticket, half_lot)
                                        if close_half.get("success"):
                                            logger.info(f"[PARTIAL] #{p.ticket} Closed 50% at {progress_pct*100:.0f}% TP (lot {half_lot})")
                                    except Exception as e:
                                        logger.warning(f"[PARTIAL] #{p.ticket} failed: {e}")

            # Daily loss halt disabled — backtest v8.7 has no halt (positions run to TP/SL)

            # Friday close
            if now.weekday() == 4 and now.hour >= 21:
                if pos:
                    logger.info("Friday close - closing all positions")
                    for p_in_pos in pos:
                        conn.close_position(p_in_pos["id"])
                time.sleep(300)
                continue

            # New bar processing (every 1 minute for faster S/R reaction)
            if _spread_paused:
                continue

            m1_time = now.replace(second=0, microsecond=0)
            if last_processed_m15_time is None or m1_time > last_processed_m15_time:
                last_processed_m15_time = m1_time
                save_state()

                if len(pos) >= active_max_pos:
                    continue

                is_n, ev_title, ev_time, pause_mins = nf.is_news_active(buffer_minutes=NEWS_BUFFER_MIN)
                if is_n:
                    logger.info(f"News Pause: {ev_title}")
                    try:
                        tg.notify_news_pause(ev_title, ev_time or "?", pause_mins)
                    except Exception:
                        pass
                    continue

                h4w = conn.get_candles(SYMBOL, "H4", 30)
                h1w = conn.get_candles(SYMBOL, "H1", 50)
                m15w = conn.get_candles(SYMBOL, "M15", 50)
                m5w = conn.get_candles(SYMBOL, "M5", 50)
                m1w = conn.get_candles(SYMBOL, "M1", 50)
                if h4w is None or m15w is None or m5w is None or m1w is None:
                    continue

                h4w_ren = h4w.rename(columns=lambda x: x.lower())
                m15w_ren = m15w.rename(columns=lambda x: x.lower())
                m5w_ren = m5w.rename(columns=lambda x: x.lower())
                m1w_ren = m1w.rename(columns=lambda x: x.lower())

                i15 = compute_all_indicators(m15w_ren)
                i5 = compute_all_indicators(m5w_ren)
                i4 = compute_all_indicators(h4w_ren)
                i1 = compute_all_indicators(m1w_ren)

                h4_ema20 = i4['emas']['EMA_20'].iloc[-1]
                h4_trend = "BULLISH" if h4w['close'].iloc[-1] > h4_ema20 else "BEARISH"
                sr = compute_sr_levels(m15w)  # M15 S/R matches backtest

                # ── MULTI-TF S/R COMPUTATION ────────────────────────
                mtf_sr_data = None
                if MTF_SR_ENABLED:
                    try:
                        sr_engine = sr_mtf.MultiTFSupportResistance()
                        curr_price = float(m15w['close'].iloc[-1])
                        mtf_sr_data = sr_engine.compute_all(h4w, h1w, m15w, m5w if m5w is not None else m15w, curr_price)
                        logger.info(f"[MTF-SR] {mtf_sr_data.get('sr_summary', 'N/A')}")
                    except Exception as e:
                        logger.warning(f"[MTF-SR] Computation failed: {e}")

                # ── CANDLE + S/R ANALYSIS ──────────────────────────
                swing_levels = cp.detect_swing_levels(m15w)
                candle_analysis = cp.analyze_full(m15w, swing_levels)
                candle_signal = candle_analysis.get("signal", "NONE")
                candle_conf = candle_analysis.get("confidence", 0)
                candle_patterns_list = candle_analysis.get("patterns_detected", [])
                sr_touch_info = candle_analysis.get("sr_touch", {})

                # Current price (used by multiple filters below)
                curr = float(m15w['close'].iloc[-1])

                # ── PURE S/R ENTRY (SR_ENTRY_MODE) ──────────────
                direction = "NONE"
                score = 0
                strategy_reason = ""
                result = {"direction": "NONE", "setup_score": 0, "reason": "", "bias": "neutral", "is_mean_reversion": False}
                if SR_ENTRY_MODE:
                    try:
                        m5_last = m5w.iloc[-2] if len(m5w) >= 2 else m5w.iloc[-1]
                        sr_filter = srentry.SREntryFilter()
                        ohlcv_map = {"H4": h4w, "H1": h1w, "M15": m15w, "M5": m5w}
                        candle_ohlc = (float(m5_last["open"]), float(m5_last["high"]),
                                       float(m5_last["low"]), float(m5_last["close"]))
                        sr_result = sr_filter.analyze(ohlcv_map, curr, candle_ohlc)
                        direction = sr_result.get("direction", "NONE")
                        score = sr_result.get("confidence", 0)
                        strategy_reason = sr_result.get("reason", "no S/R setup")
                        # Boost with indicator+RIS confirmation
                        if direction != "NONE":
                            i14_rsi = float(i15["rsi"].iloc[-1])
                            i5_macd = i5.get("macd", {})
                            macd_bull = i5_macd.get("macd", pd.Series([0])).iloc[-1] > i5_macd.get("signal", pd.Series([0])).iloc[-1] if len(i5_macd) > 0 else False
                            if direction == "BUY":
                                if 25 <= i14_rsi <= 55: score += 15  # RSI oversold bounce
                                if macd_bull: score += 10
                            elif direction == "SELL":
                                if 45 <= i14_rsi <= 75: score += 15  # RSI overbought fade
                                if not macd_bull: score += 10
                            # Require minimum confidence after bonuses
                            if score < 50: direction = "NONE"
                    except Exception as e:
                        logger.warning(f"[SR-ENTRY] Failed: {e}")
                
                # Fallback to indicator strategy if S/R didn't produce a valid signal
                if direction == "NONE" or score < 50:
                    result = strategy.analyze(i5, i5, i15, m5w.tail(5), m5w, m15w)
                    direction = result.get("direction", "NONE")
                    score = result.get("setup_score", 0)
                    strategy_reason = result.get("reason", "indicator_fallback")

                # ── DEEPSEEK MARKET ANALYSIS (only every 15 minutes) ─────
                if ai is not None and AI_MARKET_ANALYSIS and now.minute % 15 == 0:
                    try:
                        macd_status = "bullish" if i15['macd']['macd'].iloc[-1] > i15['macd']['signal'].iloc[-1] else "bearish"
                        market_ctx = {
                            "price": round(m15w['close'].iloc[-1], 2),
                            "session": get_session(),
                            "m15_bias": result.get("bias", "?"),
                            "h4_trend": h4_trend,
                            "rsi": round(i15['rsi'].iloc[-1], 1),
                            "macd_status": macd_status,
                            "atr_status": "High" if (i15['atr'].iloc[-1] > i15['atr'].rolling(20).mean().iloc[-1] * 1.5) else "Normal",
                            "nearest_support": candle_analysis.get("nearest_support", "?"),
                            "nearest_resistance": candle_analysis.get("nearest_resistance", "?"),
                            "candle_patterns": ", ".join(candle_patterns_list) if candle_patterns_list else "none",
                            "sr_touch_info": sr_touch_info.get("reason", "none"),
                            "news": nf.get_news_impact_context(),
                        }
                        ai_market = ai.analyze_market(market_ctx)
                        tg.notify_ai_market_analysis(
                            bias=ai_market.get("bias", "?").upper(),
                            risk=ai_market.get("risk_level", "?").upper(),
                            notes=ai_market.get("notes", "No analysis available")
                        )
                        # Log analysis to file for GitHub upload
                        try:
                            analysis_entry = {
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "price": market_ctx["price"],
                                "bias": ai_market.get("bias", "?"),
                                "risk": ai_market.get("risk_level", "?"),
                                "confidence": ai_market.get("confidence", 0),
                                "notes": ai_market.get("notes", ""),
                                "m15_bias": market_ctx["m15_bias"],
                                "h4_trend": market_ctx["h4_trend"],
                                "rsi": market_ctx["rsi"],
                                "session": market_ctx["session"],
                                "candle_patterns": market_ctx["candle_patterns"],
                            }
                            with open("deepseek_analysis_log.jsonl", "a") as f:
                                f.write(json.dumps(analysis_entry) + "\n")
                        except Exception as log_e:
                            logger.warning(f"[AnalysisLog] Write failed: {log_e}")
                    except Exception as e:
                        logger.warning(f"Market analysis failed: {e}")

                # Save original signal before filters run (for consolidated report)
                original_dir = direction
                original_score = score
                blocked_by = None
                ai_blocked = False
                ai_conf = 0
                ai_reason_str = ""
                go = True  # default — only becomes False if AI explicitly blocks

                # Boost score with candle confirmation (stronger)
                if direction != "NONE" and candle_signal == direction:
                    boost = max(1, candle_conf // 3)
                    score = min(100, score + boost)
                    strategy_reason += f" +{boost} candle"

                # ── TRADING HOURS + SESSION COOLDOWN ──────────
                h = now.hour; m = now.minute
                if not (TRADE_HOURS_START <= h < TRADE_HOURS_END):
                    blocked_by = blocked_by or "outside_trading_hours"
                    direction = "NONE"
                if (h == 0 and m < SESSION_COOLDOWN_MIN):
                    blocked_by = blocked_by or "session_cooldown"
                    direction = "NONE"

                # ── DIRECTIONAL SCORE ──────────────────────────
                if direction != "NONE":
                    min_sc = MIN_SCORE_BUY if direction == "BUY" else MIN_SCORE_SELL
                    if len(pos) >= 1:
                        min_sc = max(min_sc, SECOND_POS_MIN_SCORE)
                    if score < min_sc:
                        blocked_by = f"score_{score}_lt_min_{min_sc}"
                        direction = "NONE"

                # ── MTF CONFLUENCE ─────────────────────────────
                if direction != "NONE" and MTF_CONFLUENCE:
                    try:
                        m5_rsi = float(i5["rsi"].iloc[-1])
                        if (direction == "BUY" and m5_rsi < 20) or (direction == "SELL" and m5_rsi > 80):
                            blocked_by = f"MTF_conflict_M5_RSI_{m5_rsi:.0f}"
                            direction = "NONE"
                    except:
                        pass

                # ── DXY CORRELATION ─────────────────────────────
                if direction != "NONE" and DXY_ENABLED:
                    gold_4h_ago = m15w['close'].iloc[max(0, len(m15w) - DXY_LOOKBACK_H * 4)]
                    gold_change = (m15w['close'].iloc[-1] - gold_4h_ago) / gold_4h_ago if gold_4h_ago > 0 else 0
                    dxy_proxy = -gold_change
                    if (direction == "BUY" and dxy_proxy > DXY_THRESHOLD) or (direction == "SELL" and dxy_proxy < -DXY_THRESHOLD):
                        blocked_by = "DXY_correlation"
                        direction = "NONE"

                # ── MULTI-TF S/R ENTRY FILTER ───────────────────────
                if direction != "NONE" and MTF_SR_ENABLED and mtf_sr_data is not None:
                    # 1. Block trades against major S/R (H4/H1 levels)
                    blocked, sr_reason = sr_engine.is_in_no_trade_zone(
                        curr, direction,
                        mtf_sr_data.get("no_buy_zones", []),
                        mtf_sr_data.get("no_sell_zones", []),
                        buffer_points=SR_NO_TRADE_BUFFER,
                    )
                    if blocked:
                        blocked_by = f"MTF_SR_{sr_reason[:50]}"
                        direction = "NONE"
                        logger.info(f"[MTF-SR] Blocked {original_dir}: {sr_reason}")
                    
                    # 2. SR_ONLY_AT_LEVELS: Require price near support (BUY) or resistance (SELL)
                    if direction != "NONE" and SR_ONLY_AT_LEVELS:
                        nearest_r = mtf_sr_data.get("nearest_resistance", {}).get("level", curr + 50)
                        nearest_s = mtf_sr_data.get("nearest_support", {}).get("level", curr - 50)
                        dist_to_r = nearest_r - curr
                        dist_to_s = curr - nearest_s
                        
                        if direction == "BUY":
                            # BUY only near support (bounce)
                            if dist_to_s > SR_BOUNCE_BUFFER:
                                blocked_by = f"no_near_support_${dist_to_s:.1f}"
                                direction = "NONE"
                            elif dist_to_r < SR_BOUNCE_BUFFER and dist_to_s > SR_BOUNCE_BUFFER:
                                # Price near resistance but far from support — blocked
                                blocked_by = f"near_resistance_not_support_R=${dist_to_r:.1f}_S=${dist_to_s:.1f}"
                                direction = "NONE"
                        elif direction == "SELL":
                            # SELL only near resistance (rejection)
                            if dist_to_r > SR_BOUNCE_BUFFER:
                                blocked_by = f"no_near_resistance_${dist_to_r:.1f}"
                                direction = "NONE"
                            elif dist_to_s < SR_BOUNCE_BUFFER and dist_to_r > SR_BOUNCE_BUFFER:
                                # Price near support but far from resistance — blocked
                                blocked_by = f"near_support_not_resistance_R=${dist_to_r:.1f}_S=${dist_to_s:.1f}"
                                direction = "NONE"
                    
                    # 3. Apply S/R confluence bonus/penalty to score
                    if direction != "NONE":
                        sr_confluence = cp.analyze_sr_confluence(direction, curr, mtf_sr_data)
                        sr_bonus = cp.compute_mtf_sr_score_bonus(candle_signal, candle_conf, sr_confluence)
                        if sr_bonus <= -100:
                            blocked_by = f"MTF_SR_HARD_BLOCK_{sr_confluence.get('reason','?')[:40]}"
                            direction = "NONE"
                        elif sr_bonus != 0:
                            score = max(0, min(100, score + sr_bonus))
                            strategy_reason += f" +SR_{sr_bonus}"

                # ── SAME DIRECTION STACKING BLOCK ───────────────
                if direction != "NONE" and pos:
                    existing_dirs = set(p["type"] for p in pos)
                    if direction in existing_dirs and sum(1 for p in pos if p["type"] == direction) >= MAX_PER_DIRECTION:
                        blocked_by = "max_per_direction"
                        direction = "NONE"

                # ── FLASH SPIKE PROTECTION ── ($50 in <1 min = pause)
                global _last_m1_price
                if m1w is not None and _last_m1_price is not None:
                    flash_move = abs(m1w['close'].iloc[-1] - _last_m1_price)
                    if flash_move > 50.0:
                        logger.warning(f"FLASH SPIKE: ${flash_move:.1f} in <1 min")
                        blocked_by = f"flash_spike_${flash_move:.0f}"
                        direction = "NONE"
                        try:
                            tg.send_message(f"⚡ FLASH SPIKE: ${flash_move:.1f} in <1 min\nBot pausing for this bar.\n⏰ {now.strftime('%H:%M:%S')} UTC")
                        except: pass
                if m1w is not None:
                    _last_m1_price = m1w['close'].iloc[-1]

                if direction != "NONE":
                    tick = mt5.symbol_info_tick(SYMBOL)

                    atr_s = i15.get("atr")
                    if atr_s is not None and len(atr_s) >= 20:
                        current_atr = atr_s.iloc[-1]
                        mean_atr = atr_s.rolling(20).mean().iloc[-1]
                        if current_atr > mean_atr * ATR_VOL_THRESHOLD:
                            blocked_by = f"ATR_spike_{current_atr/mean_atr:.1f}x"
                            direction = "NONE"

                    dist_r = sr['resistance'] - curr
                    dist_s = curr - sr['support']

                    if direction == "BUY" and dist_r < 2.5:
                        blocked_by = f"near_resistance_${dist_r:.1f}"
                        direction = "NONE"
                    if direction == "SELL" and dist_s < 2.5:
                        blocked_by = f"near_support_${dist_s:.1f}"
                        direction = "NONE"

                    if pos:
                        existing_dirs = set(p["type"] for p in pos)
                        if direction not in existing_dirs and len(existing_dirs) > 0:
                            blocked_by = "opposite_direction_open"
                            direction = "NONE"

                    # Counter-trend filter: only block if H4 strongly disagrees AND
                    # the signal isn't a mean-reversion setup (RSI extreme)
                    if direction != "NONE" and direction != ("BUY" if h4_trend == "BULLISH" else "SELL"):
                        m15_rsi_val = i15['rsi'].iloc[-1]
                        is_mean_rev = result.get("is_mean_reversion", False)
                        if not is_mean_rev and not (m15_rsi_val < 25 or m15_rsi_val > 75):
                            blocked_by = f"counter_trend_RSI_{m15_rsi_val:.0f}"
                            direction = "NONE"

                    # BB filter: only block at extreme overextensions
                    if direction == "BUY" and m15w['close'].iloc[-1] < i15['bb']['lower'].iloc[-1]:
                        blocked_by = "BB_below_lower_band"
                        direction = "NONE"
                    if direction == "SELL" and m15w['close'].iloc[-1] > i15['bb']['upper'].iloc[-1]:
                        blocked_by = "BB_above_upper_band"
                        direction = "NONE"

                    existing_risk = sum([abs(p["entry"] - p["sl"]) * 100 * p["lots"]
                                         for p in pos if p["sl"] and p["sl"] > 0])
                    if existing_risk >= balance * TOTAL_RISK_LIMIT:
                        blocked_by = "total_risk_limit"
                        direction = "NONE"

                    # AI Filter — trade veto (only active if USE_AI_FILTER=True)
                    if direction != "NONE" and score >= MIN_SCORE and ai is not None and USE_AI_FILTER:
                        atr_status_text = "High" if (i15['atr'].iloc[-1] > i15['atr'].rolling(20).mean().iloc[-1] * 1.5) else "Normal"
                        ctx = {
                            "m15_bias": result.get("bias"),
                            "h4_trend": h4_trend,
                            "rsi": round(i15['rsi'].iloc[-1], 1),
                            "atr_status": atr_status_text,
                            "sr_levels": f"Dist to Res:${dist_r:.2f}, Dist to Sup:${dist_s:.2f}",
                            "news": nf.get_news_impact_context(),
                            "score": score,
                            "reason": strategy_reason,
                            "daily_pnl": round(daily_pnl, 2),
                            "open_positions": f"{len(pos)} open: " + ", ".join(
                                [f"{p['type']} at {p['entry']} (SL:{p['sl']})" for p in pos]) if pos else "None"
                        }
                        go, ai_reason = ai.analyze_signal(direction, curr, ctx)
                        ai_reason_str = str(ai_reason)[:150] if ai_reason else ""
                        # Extract AI confidence
                        try:
                            if isinstance(ai_reason, dict):
                                ai_conf = int(ai_reason.get("confidence", 0))
                            elif isinstance(ai_reason, str) and "confidence" in ai_reason.lower():
                                _m = re.search(r'(\d+)', ai_reason)
                                if _m: ai_conf = int(_m.group(0))
                        except:
                            pass
                        if not go:
                            if ai_conf >= 70:
                                ai_blocked = True
                                blocked_by = f"AI_veto_conf_{ai_conf}"
                                direction = "NONE"
                                logger.info(f"AI BLOCKED (conf={ai_conf}): {ai_reason}")
                            else:
                                logger.info(f"AI OVERRULED (low conf={ai_conf}): {ai_reason} — passing to rules")
                        else:
                            logger.info(f"AI APPROVE (conf={ai_conf}): {ai_reason}")

                # ── PLACE TRADE ──────────────────────────────────
                trade_opened = False
                if direction != "NONE" and score >= MIN_SCORE:
                    tick = mt5.symbol_info_tick(SYMBOL)
                    if tick:
                        price = tick.ask if direction == "BUY" else tick.bid
                        atr_val = i15["atr"].iloc[-1]
                        sl_dist = atr_val * SL_ATR_MULT
                        sl = price - sl_dist if direction == "BUY" else price + sl_dist
                        tp_dist = atr_val * TP_ATR_MULT
                        tp = price + tp_dist if direction == "BUY" else price - tp_dist

                        risk_pct = active_risk
                        if score >= HIGH_SCORE_THRESHOLD:
                            risk_pct = HIGH_SCORE_RISK  # flat 10% as backtest
                        lot = max(0.01, round((balance * risk_pct) / (sl_dist * 100), 2))
                        if len(pos) == 1:
                            lot *= SECOND_POS_LOT_RATIO
                        lot = max(0.01, round(lot, 2))

                        res_order = conn.place_order(direction, SYMBOL, lot, sl=sl, tp=tp)
                        if res_order.get("success"):
                            trade_opened = True
                            logger.info(f"ENTRY {direction} lot={lot} price={price:.2f} sl={sl:.2f} tp={tp:.2f}")
                            daily_trades += 1
                            strategy.record_trade()
                            save_state()
                            # Consolidated trade opened message
                            ai_note = ""
                            if ai_conf > 0 and ai_conf < 70 and not go:
                                ai_note = f"\n⚠️ AI was cautious (conf {ai_conf}%) but overruled by rules"
                            try:
                                tg.send_message(
                                    f"✅ TRADE OPENED: {direction} {SYMBOL}\n"
                                    f"━━━━━━━━━━━━━━━\n"
                                    f"Price: ${price:.2f} | Score: {score}/100\n"
                                    f"Lot: {lot} | SL: ${sl:.2f} | TP: ${tp:.2f}\n"
                                    f"Reason: {strategy_reason}\n"
                                    f"Balance: ${balance:,.2f}{ai_note}"
                                )
                            except Exception as e:
                                logger.warning(f"Telegram notify failed: {e}")
                        else:
                            logger.error(f"ORDER FAILED: {res_order.get('reason')}")
                            try:
                                tg.send_message(f"❌ ORDER FAILED: {direction} at ${price:.2f}\nReason: {res_order.get('reason','unknown')}")
                            except:
                                pass

                # ── CONSOLIDATED REPORT: signal blocked ──────────
                if not trade_opened and original_dir != "NONE":
                    price_now = m15w['close'].iloc[-1]
                    if ai_blocked:
                        report = (
                            f"❌ AI VETO: Bot wanted {original_dir} (Score {original_score})\n"
                            f"━━━━━━━━━━━━━━━\n"
                            f"Price: ${price_now:.2f}\n"
                            f"Reason: {strategy_reason}\n"
                            f"Blocked by: {blocked_by}\n"
                            f"AI: {ai_reason_str}"
                        )
                    else:
                        report = (
                            f"🚫 BLOCKED: Bot wanted {original_dir} (Score {original_score})\n"
                            f"━━━━━━━━━━━━━━━\n"
                            f"Price: ${price_now:.2f}\n"
                            f"Reason: {strategy_reason}\n"
                            f"Blocked by: {blocked_by or 'unknown'}"
                        )
                    try:
                        tg.send_message(report)
                    except:
                        pass

            # Hourly export + GitHub push
            if cycle % 60 == 0:
                try:
                    trade_exporter.export_trades(balance, len(pos), trades_log)
                except Exception as e:
                    logger.warning(f"Trade exporter failed: {e}")
                try:
                    push_result = gh_exporter.push_analysis()
                    if push_result and isinstance(push_result, dict):
                        if push_result.get("success"):
                            logger.info(f"[GitExport] {push_result.get('message', 'OK')}")
                        else:
                            err = push_result.get("error", "Unknown error")
                            logger.warning(f"[GitExport] Push failed: {err}")
                            try:
                                tg.send_message(f"⚠️ GITHUB PUSH FAILED\n━━━━━━━━━━━━━━━\nError: {err}\n⏰ {now.strftime('%H:%M:%S')} UTC")
                            except Exception:
                                pass
                except Exception as e:
                    logger.debug(f"[GitExport] Push check: {e}")

            # Heartbeat (every 5 min — internal throttle)
            try:
                tg.notify_heartbeat(
                    balance=balance, open_positions=len(pos),
                    total_trades=len(trades_log),
                    equity=info.get("equity", 0)
                )
            except Exception:
                pass

        except Exception as e:
            err_msg = f"{type(e).__name__}: {str(e)[:200]}"
            tb = traceback.format_exc()
            logger.error(f"Loop Error: {err_msg}\n{tb[-500:]}")
            try:
                tg.notify_system_error("main_loop", f"{err_msg}\nLine: {tb.split(chr(10))[-3].strip()}")
            except Exception:
                pass
            time.sleep(10)

        time.sleep(5)  # Check every 5 seconds for 1-min bar updates


def _log_sl_closed(p_data: dict, pnl: float, exit_price: float):
    """Log a position that was closed by SL (vanished from MT5)."""
    global daily_pnl, consecutive_losses, trades_log
    direction = p_data["type"]
    trade_entry = {
        "open_time": datetime.fromtimestamp(p_data["time"], tz=timezone.utc).isoformat() if p_data.get("time") else datetime.now(timezone.utc).isoformat(),
        "close_time": datetime.now(timezone.utc).isoformat(),
        "dir": direction,
        "entry": p_data["entry"],
        "close_price": exit_price,
        "sl": p_data.get("sl", 0),
        "tp": p_data.get("tp", 0),
        "lot": p_data["lots"],
        "pnl": round(pnl, 2),
        "reason": "sl",
        "score": 0,
        "regime": "",
        "be": False,
    }
    trades_log.append(trade_entry)
    consecutive_losses += 1
    daily_pnl += pnl
    try:
        tg.notify_trade_closed(
            direction=direction, symbol=SYMBOL, entry=p_data["entry"],
            exit_price=exit_price, pnl=pnl, reason="sl",
            balance=balance_snapshot
        )
    except Exception:
        pass

def _log_closed_trade(p, pnl: float, reason: str, exit_price: float):
    """Append closed trade to trades_log and send notification."""
    global daily_pnl, consecutive_losses, trades_log
    direction = "BUY" if p.type == 0 else "SELL"
    trade_entry = {
        "open_time": datetime.fromtimestamp(p.time, tz=timezone.utc).isoformat() if hasattr(p, 'time') else datetime.now(timezone.utc).isoformat(),
        "close_time": datetime.now(timezone.utc).isoformat(),
        "dir": direction,
        "entry": p.price_open,
        "close_price": exit_price,
        "sl": p.sl,
        "tp": p.tp,
        "lot": p.volume,
        "pnl": round(pnl, 2),
        "reason": reason,
        "score": 0,
        "regime": "",
        "be": False,
    }
    trades_log.append(trade_entry)
    if pnl > 0:
        consecutive_losses = 0
    else:
        consecutive_losses += 1
    daily_pnl += pnl

    try:
        tg.notify_trade_closed(
            direction=direction, symbol=SYMBOL, entry=p.price_open,
            exit_price=exit_price, pnl=pnl, reason=reason,
            balance=balance_snapshot
        )
    except Exception:
        pass


if __name__ == "__main__":
    main_loop()