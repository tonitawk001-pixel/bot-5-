"""
PURE S/R ENTRY MODULE — No indicators, only Support & Resistance
=================================================================
BUY: Price near support (across D1/H4/H1/M15/M5) + reversal candle = ENTRY
SELL: Price near resistance (across D1/H4/H1/M15/M5) + reversal candle = ENTRY
NO ENTRY in "no man's land" (between levels).

Uses sr_levels_mtf for multi-timeframe S/R computation.
"""
import pandas as pd, numpy as np
from typing import Optional, Dict, Tuple
from sr_levels_mtf import detect_swings, cluster_levels

# ── Candle detection ───────────────────────────────────────────
def _is_reversal_buy(o: float, h: float, l: float, c: float) -> bool:
    """Hammer, bullish engulfing, piercing line at support."""
    body = abs(c - o); lower = min(o, c) - l; upper = h - max(o, c); total = h - l
    if total <= 0: return False
    if c <= o: return False  # must close bullish
    # Hammer: long lower wick, small body, tiny upper wick
    if lower > body * 1.8 and upper < body * 0.4: return True
    # Bullish engulfing-like: strong body, significant lower wick
    if lower > body * 0.6 and body / total > 0.4: return True
    # Strong bullish marubozu with lower wick
    if lower > body * 0.3 and body / total > 0.65: return True
    return False

def _is_reversal_sell(o: float, h: float, l: float, c: float) -> bool:
    """Shooting star, bearish engulfing, dark cloud at resistance."""
    body = abs(c - o); lower = min(o, c) - l; upper = h - max(o, c); total = h - l
    if total <= 0: return False
    if c >= o: return False  # must close bearish
    # Shooting star: long upper wick, small body, tiny lower wick
    if upper > body * 1.8 and lower < body * 0.4: return True
    # Bearish engulfing-like: strong body, significant upper wick
    if upper > body * 0.6 and body / total > 0.4: return True
    # Strong bearish marubozu with upper wick
    if upper > body * 0.3 and body / total > 0.65: return True
    return False


class SREntryFilter:
    """
    S/R-only entry filter. Completely replaces indicator-based strategy
    for entry decisions. Only enters at S/R levels with reversal candles.
    """
    def __init__(self):
        self.sr_buffer = 3.0       # Must be within 3pts of a level to enter
        self.sr_proximity = 6.0    # Max distance to consider "near" a level (for rejection detection)
    
    def collect_levels(self, ohlcv_map: Dict[str, pd.DataFrame], price: float) -> Tuple[list, list]:
        """
        Collect clustered S/R levels from all timeframes.
        
        Args:
            ohlcv_map: {"D1": df, "H4": df, "H1": df, "M15": df, "M5": df}
            price: current price
        
        Returns:
            (all_resistances, all_supports) - sorted lists of merged levels
        """
        all_r = []; all_s = []
        
        for tf_name, data in ohlcv_map.items():
            if data is None or len(data) < 10:
                continue
            try:
                # Different window sizes per timeframe
                windows = {"D1": 2, "H4": 2, "H1": 3, "M15": 3, "M5": 4}
                w = windows.get(tf_name, 3)
                
                swing_h, swing_l = detect_swings(data, window=w)
                res = cluster_levels(swing_h, threshold_pct=0.002)
                sup = cluster_levels(swing_l, threshold_pct=0.002)
                
                # Only keep levels within 80pts (realistic XAUUSD range)
                all_r.extend([r for r in res if abs(r - price) <= 80])
                all_s.extend([s for s in sup if abs(s - price) <= 80])
            except:
                pass
        
        # Merge nearby levels (cluster within $1.5)
        def _merge(lst):
            if not lst: return []
            s = sorted(set(round(x, 1) for x in lst))
            clusters, current = [], [s[0]]
            for v in s[1:]:
                if v - current[-1] <= 1.5: current.append(v)
                else: clusters.append(sum(current)/len(current)); current = [v]
            clusters.append(sum(current)/len(current))
            return sorted(clusters)
        
        return _merge(all_r), _merge(all_s)
    
    def analyze(self, ohlcv_map: Dict[str, pd.DataFrame], current_price: float,
                candle_ohlc: Optional[tuple] = None) -> Dict:
        """
        Determine entry direction based purely on S/R levels and candle.
        
        Returns:
            {
                "direction": "BUY" | "SELL" | "NONE",
                "confidence": 0-100,
                "reason": str,
                "nearest_resistance": float,
                "nearest_support": float,
                "dist_to_resistance": float,
                "dist_to_support": float,
            }
        """
        resistances, supports = self.collect_levels(ohlcv_map, current_price)
        
        # Find nearest resistance and support
        r_above = [r for r in resistances if r > current_price]
        s_below = [s for s in supports if s < current_price]
        
        nearest_r = min(r_above) if r_above else current_price + 30
        nearest_s = max(s_below) if s_below else current_price - 30
        
        dist_r = nearest_r - current_price
        dist_s = current_price - nearest_s
        
        result = {
            "direction": "NONE",
            "confidence": 0,
            "reason": "No S/R setup",
            "nearest_resistance": nearest_r,
            "nearest_support": nearest_s,
            "dist_to_resistance": round(dist_r, 2),
            "dist_to_support": round(dist_s, 2),
            "all_resistances": resistances[:10],
            "all_supports": supports[:10],
        }
        
        if candle_ohlc is None:
            return result
        
        o, h, l, c = candle_ohlc
        bull_rev = _is_reversal_buy(o, h, l, c)
        bear_rev = _is_reversal_sell(o, h, l, c)
        
        # ── ENTRY RULES ──
        
        # BUY: within 3pts of support + bullish reversal candle
        if bull_rev and dist_s <= self.sr_buffer:
            confidence = 85
            # Boost confidence if support is from higher timeframe
            if dist_s <= 1.5: confidence = 95
            result["direction"] = "BUY"
            result["confidence"] = confidence
            result["reason"] = f"BUY: Support ${nearest_s:.1f} + reversal (dist=${dist_s:.1f})"
            return result
        
        # SELL: within 3pts of resistance + bearish reversal candle
        if bear_rev and dist_r <= self.sr_buffer:
            confidence = 85
            if dist_r <= 1.5: confidence = 95
            result["direction"] = "SELL"
            result["confidence"] = confidence
            result["reason"] = f"SELL: Resistance ${nearest_r:.1f} + reversal (dist=${dist_r:.1f})"
            return result
        
        # BUY: pattern at support (slightly wider) — still valid
        if bull_rev and dist_s <= self.sr_proximity:
            result["direction"] = "BUY"
            result["confidence"] = 65
            result["reason"] = f"BUY: Near support ${nearest_s:.1f} + reversal (dist=${dist_s:.1f})"
            return result
        
        # SELL: pattern at resistance (slightly wider) — still valid
        if bear_rev and dist_r <= self.sr_proximity:
            result["direction"] = "SELL"
            result["confidence"] = 65
            result["reason"] = f"SELL: Near resistance ${nearest_r:.1f} + reversal (dist=${dist_r:.1f})"
            return result
        
        # NO ENTRY: In no man's land or no pattern
        return result