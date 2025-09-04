"""Utilities for loading and validating strategy configuration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from pathlib import Path
from typing import Any, Dict, Optional
from pathlib import Path
import logging
import os
import shutil
import yaml  # type: ignore[import-untyped]


log = logging.getLogger(__name__)


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
    mtf_confirm: bool
    iv_percentile_limit: int
    adx_min_trend: int
    depth_min_lots: int
    micro: Dict[str, Any]
    min_oi: int
    max_median_spread_pct: float
    delta_min: float
    delta_max: float
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
    max_tick_lag_s: int
    max_bar_lag_s: int
    min_bars_required: int
    enable_range_scoring: bool
    ema_fast: int
    ema_slow: int
    bb_period: int
    score_gate: float
    lower_score_temp: bool
    source_path: str
    mtime: float

    @classmethod
    def load(cls, path: str) -> "StrategyConfig":
        """Load configuration from ``path``.

        If ``path`` does not exist, gracefully fall back to the default
        configuration resolved by :func:`resolve_config_path`. This mirrors the
        behaviour used elsewhere in the application and prevents a missing file
        from crashing startup when an environment variable is misconfigured.
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except FileNotFoundError:
            fallback = resolve_config_path()
            if fallback != path:
                return cls.load(fallback)
            raise
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
            mtf_confirm=bool(gates.get("mtf_confirm", True)),
            iv_percentile_limit=int(gates.get("iv_percentile_limit", 85)),
            adx_min_trend=int(gates.get("adx_min_trend", 18)),
            depth_min_lots=int(micro.get("depth_min_lots", 1)),
            micro=micro,
            min_oi=int(options.get("min_oi", 500000)),
            max_median_spread_pct=float(options.get("max_median_spread_pct", 0.35)),
            delta_min=float(options.get("delta_min", 0.35)),
            delta_max=float(options.get("delta_max", 0.60)),
            tp1_R_min=float(lc.get("tp1_R_min", 1.00)),
            tp1_R_max=float(lc.get("tp1_R_max", 1.10)),
            tp1_partial=float(lc.get("tp1_partial", 0.6)),
            tp2_R_trend=float(lc.get("tp2_R_trend", 1.80)),
            tp2_R_range=float(lc.get("tp2_R_range", 1.40)),
            trail_atr_mult=float(lc.get("trail_atr_mult", 0.80)),
            time_stop_min=int(lc.get("time_stop_min", 12)),
            gamma_enabled=bool(lc.get("gamma_mode", {}).get("enabled", True)),
            gamma_after=_parse_time(lc.get("gamma_mode", {}).get("tue_after", "14:45")),
            gamma_tp2_cap=float(lc.get("gamma_mode", {}).get("tp2_R_cap", 1.40)),
            gamma_trail_mult=float(lc.get("gamma_mode", {}).get("trail_atr_mult", 0.60)),
            gamma_time_stop_min=int(lc.get("gamma_mode", {}).get("time_stop_min", 8)),
            min_atr_pct_nifty=float(thresholds.get("min_atr_pct_nifty", 0.04)),
            min_atr_pct_banknifty=float(thresholds.get("min_atr_pct_banknifty", 0.06)),
            max_tick_lag_s=int(thresholds.get("max_tick_lag_s", 8)),
            max_bar_lag_s=int(thresholds.get("max_bar_lag_s", 75)),
            min_bars_required=int(warm.get("min_bars", warm.get("min_bars_required", 20))),
            enable_range_scoring=bool(thresholds.get("enable_range_scoring", True)),
            ema_fast=int(thresholds.get("ema_fast", 9)),
            ema_slow=int(thresholds.get("ema_slow", 21)),
            bb_period=int(thresholds.get("bb_period", 20)),
            score_gate=float(thresholds.get("score_gate", 0.30)),
            lower_score_temp=bool(dbg.get("lower_score_temp", False)),
            source_path=path,
            mtime=os.path.getmtime(path),
        )
        return cfg

    @classmethod
    def try_load(cls, path: str | None) -> "StrategyConfig":
        """Locate and load a strategy config, seeding defaults if missing."""
        candidates: list[str | None] = [
            path,
            os.getenv("STRATEGY_CFG"),
            os.getenv("STRATEGY_CONFIG_FILE"),
            "config/strategy.yaml",
            "/app/config/strategy.yaml",
        ]
        for p in candidates:
            if p and Path(p).is_file():
                return cls.load(p)

        repo_default = Path("config/strategy.yaml")
        if repo_default.is_file():
            target = Path("config") / "strategy.yaml"
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.resolve() != repo_default.resolve():
                shutil.copy2(repo_default, target)
            log.warning("No external strategy config found. Seeded default at %s", target)
            return cls.load(str(target))

        raise FileNotFoundError(
            "strategy config not found. Create config/strategy.yaml or set STRATEGY_CFG env var."
        )


def resolve_config_path(
    env_key: str = "STRATEGY_CONFIG_FILE", default_path: str = "config/strategy.yaml"
) -> str:
    """Resolve the config file path from environment or fallback to default.

    If the environment variable is set but points to a non‑existent file,
    fall back to ``default_path``. This guards against typos like using the
    ``.yml`` extension when only ``.yaml`` exists or providing an absolute
    path in the wrong location.
    """

    env_path = os.environ.get(env_key)
    if env_path:
        # Use the path if it exists as‑is
        if os.path.exists(env_path):
            return env_path

        # Try swapping common YAML extensions (.yml <-> .yaml)
        base, ext = os.path.splitext(env_path)
        if ext in {".yml", ".yaml"}:
            alt_path = base + (".yaml" if ext == ".yml" else ".yml")
            if os.path.exists(alt_path):
                return alt_path

        # If an absolute path was provided, also check a relative variant
        if os.path.isabs(env_path):
            rel_path = env_path.lstrip("/")
            if os.path.exists(rel_path):
                return rel_path

    # If the default path exists relative to the current working directory,
    # return it verbatim to preserve existing behaviour and tests.
    if os.path.exists(default_path):
        return default_path

    # As a final fallback, resolve the path relative to the project root
    # (two levels up from this file). This helps when the working directory is
    # different, such as when the package is installed elsewhere.
    repo_default = Path(__file__).resolve().parents[2] / default_path
    if repo_default.exists():
        return str(repo_default)

    return default_path


def try_load(path: str | None, current: Optional[StrategyConfig]) -> StrategyConfig:
    """Attempt to load config; fall back to ``current`` on failure."""
    try:
        cfg = StrategyConfig.try_load(path)
        return cfg
    except Exception:
        if current:
            return current
        raise
