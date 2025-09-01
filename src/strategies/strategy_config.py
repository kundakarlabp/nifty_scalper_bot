"""Utilities for loading and validating strategy configuration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from typing import Any, Dict, Optional
import os
import yaml  # type: ignore[import-untyped]


def _parse_time(s: str) -> time:
    """Parse ``HH:MM`` strings into :class:`datetime.time`."""
    hh, mm = s.split(":")
    return time(int(hh), int(mm))


@dataclass
class StrategyConfig:
    """Dataclass representing the declarative strategy configuration."""

    raw: Dict[str, Any]
    version: int
    name: str
    tz: str
    trading_windows: list
    no_trade: Dict[str, Any]
    atr_min: float
    atr_max: float
    score_trend_min: int
    score_range_min: int
    mtf_confirm: bool
    iv_percentile_limit: int
    adx_min_trend: int
    max_spread_pct_regular: float
    max_spread_pct_open: float
    max_spread_pct_last20m: float
    depth_multiplier: int
    depth_min_lots: int
    min_oi: int
    max_median_spread_pct: float
    delta_target: float
    delta_band: float
    delta_enable_score: int
    delta_min: float
    delta_max: float
    re_atm_drift_pct: float
    tp1_R_min: float
    tp1_R_max: float
    tp1_partial: float
    tp2_R_trend: float
    tp2_R_range: float
    trail_atr_mult: float
    time_stop_min: int
    gamma_enabled: bool
    gamma_after: time
    gamma_tp2_cap: float
    gamma_trail_mult: float
    gamma_time_stop_min: int
    min_atr_pct_nifty: float
    min_atr_pct_banknifty: float
    min_bars_required: int
    indicator_min_bars: int
    mtf_min_bars: int
    lower_score_temp: bool
    source_path: str
    mtime: float

    @classmethod
    def load(cls, path: str) -> "StrategyConfig":
        """Load configuration from ``path``."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        meta = data.get("meta", {})
        windows = data.get("windows", {})
        gates = data.get("gates", {})
        micro = data.get("micro", {})
        thresholds = data.get("thresholds", {})
        options = data.get("options", {})
        lc = data.get("lifecycle", {})
        warm = data.get("warmup", {})
        dbg = data.get("debug", {})

        tz = windows.get("timezone", "Asia/Kolkata")
        cfg = cls(
            raw=data,
            version=int(meta.get("version", 1)),
            name=str(meta.get("name", "default")),
            tz=tz,
            trading_windows=windows.get(
                "trading", [{"days": [1, 2, 3, 4, 5], "start": "09:25", "end": "15:05"}]
            ),
            no_trade=windows.get("no_trade", {}),
            atr_min=float(gates.get("atr_pct_min", 0.30)),
            atr_max=float(gates.get("atr_pct_max", 0.90)),
            score_trend_min=int(gates.get("score_trend_min", 9)),
            score_range_min=int(gates.get("score_range_min", 8)),
            mtf_confirm=bool(gates.get("mtf_confirm", True)),
            iv_percentile_limit=int(gates.get("iv_percentile_limit", 85)),
            adx_min_trend=int(gates.get("adx_min_trend", 18)),
            max_spread_pct_regular=float(micro.get("max_spread_pct_regular", 0.35)),
            max_spread_pct_open=float(micro.get("max_spread_pct_open", 0.30)),
            max_spread_pct_last20m=float(micro.get("max_spread_pct_last20m", 0.45)),
            depth_multiplier=int(micro.get("depth_multiplier", 5)),
            depth_min_lots=int(micro.get("depth_min_lots", 1)),
            min_oi=int(options.get("min_oi", 500000)),
            max_median_spread_pct=float(options.get("max_median_spread_pct", 0.35)),
            delta_target=float(options.get("delta_target", 0.40)),
            delta_band=float(options.get("delta_band", 0.05)),
            delta_enable_score=int(options.get("delta_enable_score", 999)),
            delta_min=float(options.get("delta_min", 0.35)),
            delta_max=float(options.get("delta_max", 0.60)),
            re_atm_drift_pct=float(options.get("re_atm_drift_pct", 0.35)),
            tp1_R_min=float(lc.get("tp1_R_min", 1.00)),
            tp1_R_max=float(lc.get("tp1_R_max", 1.10)),
            tp1_partial=float(lc.get("tp1_partial", 0.6)),
            tp2_R_trend=float(lc.get("tp2_R_trend", 1.80)),
            tp2_R_range=float(lc.get("tp2_R_range", 1.40)),
            trail_atr_mult=float(lc.get("trail_atr_mult", 0.80)),
            time_stop_min=int(lc.get("time_stop_min", 12)),
            gamma_enabled=bool(lc.get("gamma_mode", {}).get("enabled", True)),
            gamma_after=_parse_time(lc.get("gamma_mode", {}).get("thu_after", "14:45")),
            gamma_tp2_cap=float(lc.get("gamma_mode", {}).get("tp2_R_cap", 1.40)),
            gamma_trail_mult=float(lc.get("gamma_mode", {}).get("trail_atr_mult", 0.60)),
            gamma_time_stop_min=int(lc.get("gamma_mode", {}).get("time_stop_min", 8)),
            min_atr_pct_nifty=float(thresholds.get("min_atr_pct_nifty", 0.04)),
            min_atr_pct_banknifty=float(thresholds.get("min_atr_pct_banknifty", 0.06)),
            min_bars_required=int(warm.get("min_bars_required", 20)),
            indicator_min_bars=int(warm.get("indicator_min_bars", 14)),
            mtf_min_bars=int(warm.get("mtf_min_bars", 21)),
            lower_score_temp=bool(dbg.get("lower_score_temp", False)),
            source_path=path,
            mtime=os.path.getmtime(path),
        )
        return cfg


def resolve_config_path(env_key: str = "STRATEGY_CONFIG_FILE", default_path: str = "config/strategy.yaml") -> str:
    """Resolve the config file path from environment or fallback to default."""
    return os.environ.get(env_key, default_path)


def try_load(path: str, current: Optional[StrategyConfig]) -> StrategyConfig:
    """Attempt to load config; fall back to ``current`` on failure."""
    try:
        cfg = StrategyConfig.load(path)
        return cfg
    except Exception:
        if current:
            return current
        raise
