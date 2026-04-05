"""Event-driven strategy execution runner."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from ..regime import gate as regime_gate

if TYPE_CHECKING:
    from ..indicators.engine import IndicatorSnapshot
    from ..regime.detector import MarketRegime, RegimeType
    from ..risk.manager import RiskManager
    from .base import Strategy
    from .manager import StrategyManager
    from .signal import Signal


_DEDUP_WINDOW = 60  # seconds


class StrategyRunner:
    """Applies all active strategies to all valid indicator snapshots.

    Applies 8 signal filters per spec Section 6.3 before returning signals.
    """

    def __init__(
        self,
        strategy_manager: "StrategyManager",
        risk_manager: "RiskManager",
    ) -> None:
        self._mgr = strategy_manager
        self._risk = risk_manager
        self._recent_signals: dict[str, float] = {}  # key → timestamp

    def run(
        self,
        snapshots: dict[str, "IndicatorSnapshot"],
        regime: "MarketRegime",
        open_positions: list,
    ) -> list["Signal"]:
        """Run all strategies, apply filters, return approved signals."""
        raw: list["Signal"] = []

        for strategy in self._mgr.active_strategies():
            for symbol, indicators in snapshots.items():
                if not indicators.is_valid:
                    continue
                sig = strategy.generate_signal(symbol, indicators, regime)
                if sig and sig.confidence >= strategy.min_confidence:
                    raw.append(sig)

        return self._filter(raw, regime, open_positions)

    def _filter(
        self,
        signals: list["Signal"],
        regime: "MarketRegime",
        open_positions: list,
    ) -> list["Signal"]:
        approved = []
        open_symbols = {p.underlying for p in open_positions}
        open_dirs: dict[str, str] = {}
        for p in open_positions:
            open_dirs[p.underlying] = "BUY_CALL" if p.option_type == "CE" else "BUY_PUT"

        for sig in signals:
            # 1. Direction lock
            existing_dir = open_dirs.get(sig.symbol)
            if existing_dir and existing_dir == sig.direction.value:
                continue

            # 2. Concurrent positions
            if not self._risk.can_add_position():
                continue

            # 3. Risk gate
            ok, _ = self._risk.can_trade()
            if not ok:
                continue

            # 4. Regime gate
            if not regime_gate.allows(sig.strategy_name, regime.regime):
                continue

            # 5. Duplicate suppression (60s window)
            dedup_key = f"{sig.symbol}:{sig.direction.value}:{sig.strategy_name}"
            now = datetime.now(timezone.utc).timestamp()
            last = self._recent_signals.get(dedup_key, 0.0)
            if (now - last) < _DEDUP_WINDOW:
                continue

            self._recent_signals[dedup_key] = now
            approved.append(sig)

        return approved
