"""
MULTI-TIMEFRAME SUPPORT & RESISTANCE ENGINE
============================================
Computes swing-based S/R levels on H4, H1, M15, M5 timeframes.
M5 uses pure price action (no indicators) — swing highs/lows only.
Levels are clustered, weighted, and merged across timeframes.

Usage:
    engine = MultiTFSupportResistance()
    sr_registry = engine.compute_all(h4_ohlcv, h1_ohlcv, m15_ohlcv, m5_ohlcv, current_price)
    # sr_registry has: all_levels, no_buy_zones, no_sell_zones, nearest_resistance, nearest_support
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────
#  SWING DETECTION (generic — works on any OHLCV)
# ─────────────────────────────────────────────────────────────

def detect_swings(ohlcv: pd.DataFrame, window: int = 3) -> Tuple[List[float], List[float]]:
    """
    Find swing highs and swing lows from OHLCV data.
    Uses a configurable window for pivot detection.
    """
    highs = ohlcv['high'].values
    lows = ohlcv['low'].values
    n = len(highs)
    
    if n < window * 2 + 1:
        return [], []
    
    swing_highs = []
    swing_lows = []
    
    for i in range(window, n - window):
        # Swing high: highest in window*2+1 bars
        if highs[i] == max(highs[i - window:i + window + 1]):
            swing_highs.append(highs[i])
        # Swing low: lowest in window*2+1 bars
        if lows[i] == min(lows[i - window:i + window + 1]):
            swing_lows.append(lows[i])
    
    return swing_highs, swing_lows


def cluster_levels(levels: List[float], threshold_pct: float = 0.002) -> List[float]:
    """
    Cluster nearby swing levels into a single representative level.
    Returns sorted list (high-to-low for resistance, low-to-high for support).
    """
    if not levels:
        return []
    
    sorted_levels = sorted(set(round(x, 1) for x in levels))
    clusters = []
    current = [sorted_levels[0]]
    
    for lvl in sorted_levels[1:]:
        if current[-1] > 0 and abs(lvl - current[-1]) / current[-1] < threshold_pct:
            current.append(lvl)
        else:
            clusters.append(sum(current) / len(current))
            current = [lvl]
    clusters.append(sum(current) / len(current))
    
    return sorted(clusters, reverse=True)


def filter_levels_by_proximity(levels: List[float], price: float, max_dist: float = 100.0) -> List[float]:
    """Keep only levels within max_dist dollars of current price."""
    return [lvl for lvl in levels if abs(lvl - price) <= max_dist]


# ─────────────────────────────────────────────────────────────
#  MULTI-TF S/R COMPUTATION
# ─────────────────────────────────────────────────────────────

class MultiTFSupportResistance:
    """
    Computes and merges S/R levels across H4, H1, M15, M5 timeframes.
    M5 uses pure price action (swing highs/lows only, no indicators).
    """
    
    def __init__(self):
        # Timeframe config: (window_size, threshold_pct, weight)
        self.tf_config = {
            "H4":  {"window": 3, "threshold": 0.003, "weight": 3.0, "label": "H4"},
            "H1":  {"window": 3, "threshold": 0.002, "weight": 2.0, "label": "H1"},
            "M15": {"window": 3, "threshold": 0.002, "weight": 1.5, "label": "M15"},
            "M5":  {"window": 4, "threshold": 0.001, "weight": 1.0, "label": "M5 (PA)"},
        }
        self._cached_levels: Dict = {}
    
    def compute_all(
        self,
        h4_ohlcv: Optional[pd.DataFrame],
        h1_ohlcv: Optional[pd.DataFrame],
        m15_ohlcv: pd.DataFrame,
        m5_ohlcv: pd.DataFrame,
        current_price: float,
    ) -> Dict:
        """
        Main entry point. Compute S/R on all available timeframes and merge.
        
        Returns:
            {
                "all_levels": [{"level": float, "type": "R"|"S", "timeframe": str, "weight": float}, ...],
                "no_buy_zones": [{"level": float, "strength": float, "timeframe": str}, ...],   # resistance levels
                "no_sell_zones": [{"level": float, "strength": float, "timeframe": str}, ...],   # support levels
                "nearest_resistance": {"level": float, "timeframe": str, "weight": float},
                "nearest_support": {"level": float, "timeframe": str, "weight": float},
                "h4_resistance": float, "h4_support": float,
                "h1_resistance": float, "h1_support": float,
                "m15_resistance": float, "m15_support": float,
                "m5_resistance": float, "m5_support": float,
                "sr_summary": str,
            }
        """
        all_entries = []  # (level, type, timeframe, weight, raw_weight)
        
        ohlcv_map = {"H4": h4_ohlcv, "H1": h1_ohlcv, "M15": m15_ohlcv, "M5": m5_ohlcv}
        
        for tf_name, ohlcv in ohlcv_map.items():
            if ohlcv is None or len(ohlcv) < 10:
                continue
            
            cfg = self.tf_config[tf_name]
            swing_highs, swing_lows = detect_swings(ohlcv, window=cfg["window"])
            
            resistances = cluster_levels(swing_highs, threshold_pct=cfg["threshold"])
            supports = cluster_levels(swing_lows, threshold_pct=cfg["threshold"])
            
            # Filter to near current price
            resistances = filter_levels_by_proximity(resistances, current_price, max_dist=150)
            supports = filter_levels_by_proximity(supports, current_price, max_dist=150)
            
            # Count touches for strength
            for res in resistances:
                touches = sum(1 for sh in swing_highs if abs(sh - res) / res < cfg["threshold"])
                strength = min(touches, 5) * cfg["weight"]
                all_entries.append({
                    "level": round(res, 2),
                    "type": "R",
                    "timeframe": tf_name,
                    "weight": cfg["weight"],
                    "strength": round(strength, 1),
                    "touches": touches,
                })
            
            for sup in supports:
                touches = sum(1 for sl in swing_lows if abs(sl - sup) / sup < cfg["threshold"])
                strength = min(touches, 5) * cfg["weight"]
                all_entries.append({
                    "level": round(sup, 2),
                    "type": "S",
                    "timeframe": tf_name,
                    "weight": cfg["weight"],
                    "strength": round(strength, 1),
                    "touches": touches,
                })
        
        # Sort by level descending
        all_entries.sort(key=lambda x: x["level"], reverse=True)
        
        # Find nearest resistance and support
        resistances = [e for e in all_entries if e["level"] > current_price]
        supports = [e for e in all_entries if e["level"] < current_price]
        
        resistances.sort(key=lambda x: x["level"])  # closest first
        supports.sort(key=lambda x: x["level"], reverse=True)  # closest first
        
        nearest_res = resistances[0] if resistances else {"level": current_price + 50, "timeframe": "N/A", "weight": 0, "strength": 0}
        nearest_sup = supports[0] if supports else {"level": current_price - 50, "timeframe": "N/A", "weight": 0, "strength": 0}
        
        # Build no-trade zones
        no_buy_zones = []
        for r in resistances:
            if r["weight"] >= 1.5:  # M15+ weight
                no_buy_zones.append({
                    "level": r["level"],
                    "strength": r["strength"],
                    "timeframe": r["timeframe"],
                })
        
        no_sell_zones = []
        for s in supports:
            if s["weight"] >= 1.5:
                no_sell_zones.append({
                    "level": s["level"],
                    "strength": s["strength"],
                    "timeframe": s["timeframe"],
                })
        
        # Extract per-timeframe nearest
        def _get_tf_nearest(tf_name: str, entries: List[Dict], above: bool):
            tf_entries = [e for e in entries if e["timeframe"] == tf_name]
            if above:
                tf_entries = [e for e in tf_entries if e["level"] > current_price]
                tf_entries.sort(key=lambda x: x["level"])
            else:
                tf_entries = [e for e in tf_entries if e["level"] < current_price]
                tf_entries.sort(key=lambda x: x["level"], reverse=True)
            return tf_entries[0]["level"] if tf_entries else (current_price + 30 if above else current_price - 30)
        
        h4_r = _get_tf_nearest("H4", all_entries, above=True)
        h4_s = _get_tf_nearest("H4", all_entries, above=False)
        h1_r = _get_tf_nearest("H1", all_entries, above=True)
        h1_s = _get_tf_nearest("H1", all_entries, above=False)
        m15_r = _get_tf_nearest("M15", all_entries, above=True)
        m15_s = _get_tf_nearest("M15", all_entries, above=False)
        m5_r = _get_tf_nearest("M5", all_entries, above=True)
        m5_s = _get_tf_nearest("M5", all_entries, above=False)
        
        # Build summary string
        summary_parts = []
        for tf in ["H4", "H1", "M15", "M5"]:
            tf_r = _get_tf_nearest(tf, all_entries, above=True)
            tf_s = _get_tf_nearest(tf, all_entries, above=False)
            summary_parts.append(f"{tf}: R={tf_r:.1f} S={tf_s:.1f}")
        
        return {
            "all_levels": all_entries,
            "no_buy_zones": no_buy_zones[:5],
            "no_sell_zones": no_sell_zones[:5],
            "nearest_resistance": nearest_res,
            "nearest_support": nearest_sup,
            "h4_resistance": h4_r,
            "h4_support": h4_s,
            "h1_resistance": h1_r,
            "h1_support": h1_s,
            "m15_resistance": m15_r,
            "m15_support": m15_s,
            "m5_resistance": m5_r,
            "m5_support": m5_s,
            "sr_summary": " | ".join(summary_parts),
        }
    
    def is_in_no_trade_zone(
        self,
        price: float,
        direction: str,
        no_buy_zones: List[Dict],
        no_sell_zones: List[Dict],
        buffer_points: float = 5.0,
    ) -> Tuple[bool, str]:
        """
        Check if current price is too close to a major S/R level.
        Returns (is_blocked, reason).
        """
        if direction == "BUY":
            for zone in no_buy_zones:
                dist = zone["level"] - price
                if 0 < dist <= buffer_points:
                    return True, f"{zone['timeframe']} resistance at ${zone['level']:.1f} (dist=${dist:.1f})"
                if price >= zone["level"]:
                    return True, f"Price ABOVE {zone['timeframe']} resistance ${zone['level']:.1f}"
        
        if direction == "SELL":
            for zone in no_sell_zones:
                dist = price - zone["level"]
                if 0 < dist <= buffer_points:
                    return True, f"{zone['timeframe']} support at ${zone['level']:.1f} (dist=${dist:.1f})"
                if price <= zone["level"]:
                    return True, f"Price BELOW {zone['timeframe']} support ${zone['level']:.1f}"
        
        return False, ""