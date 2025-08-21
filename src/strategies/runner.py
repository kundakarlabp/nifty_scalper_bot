from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from src.config import settings
from src.utils.strike_selector import get_instrument_tokens
from src.risk.position_sizing import compute_lots
from src.strategies.scalping_strategy import make_signal
from src.execution.order_executor import OrderExecutor  # assumes your existing file
from src.data.source import LiveKiteSource  # assumes your existing file
from src.signals.regime_detector import detect_regime  # light helper

log = logging.getLogger(__name__)


class StrategyRunner:
    """
    Deterministic pipeline runner with per-stage diagnostics.
    """

    def __init__(self) -> None:
        self.kite_src = LiveKiteSource()
        self.executor = OrderExecutor(self.kite_src.kite)
        self._paused_until: Optional[datetime] = None
        self.quality_mode = "auto"     # auto|on|off
        self.regime_mode = "auto"      # auto|trend|range
        self.last_diag: Dict[str, Any] = {}

    # ----------- external controls -----------

    def set_quality_mode(self, mode: str) -> None:
        self.quality_mode = mode

    def set_regime_mode(self, mode: str) -> None:
        self.regime_mode = mode

    def pause(self, minutes: Optional[int]) -> None:
        self._paused_until = (datetime.now() + timedelta(minutes=minutes)) if minutes else datetime.max

    def resume(self) -> None:
        self._paused_until = None

    # ----------- diagnostics / status -----------

    def status(self) -> Dict[str, Any]:
        return {
            "time_ist": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "live_trading": bool(settings.enable_live_trading),
            "broker": "Kite",
            "active_orders": len(self.executor.list_open_orders()),
            "quality_mode": self.quality_mode,
            "regime_mode": self.regime_mode,
            "paused": bool(self._paused_until and self._paused_until > datetime.now()),
        }

    def diag(self) -> Dict[str, Any]:
        return self.last_diag or {"ok": False, "checks": []}

    # ----------- one cycle -----------

    def run_once(self) -> None:
        checks = []
        result: Dict[str, Any] = {"ok": False, "checks": checks}

        try:
            # 1) Market window
            market_ok = self.kite_src.is_market_open() or settings.allow_offhours_testing
            checks.append({"name": "market_open", "ok": bool(market_ok)})
            if not market_ok:
                self.last_diag = result
                return

            # Paused?
            if self._paused_until and self._paused_until > datetime.now():
                checks.append({"name": "paused", "ok": False})
                self.last_diag = result
                return

            # 2) Spot LTP
            spot_ltp = self.kite_src.get_spot_ltp()
            checks.append({"name": "spot_ltp", "ok": spot_ltp is not None, "value": spot_ltp})
            if spot_ltp is None:
                self.last_diag = result
                return

            # 3) Spot OHLC
            spot_df = self.kite_src.get_spot_ohlc()
            checks.append({"name": "spot_ohlc", "ok": len(spot_df) > 0, "rows": int(len(spot_df))})
            if spot_df is None or spot_df.empty:
                self.last_diag = result
                return

            # 4) Strike selection
            tk = get_instrument_tokens(
                self.kite_src.kite,
                symbol=settings.instruments.trade_symbol,
                exchange=settings.executor.exchange,
                expiry=None,
                spot_price=float(spot_ltp),
            )
            checks.append({"name": "strike_selection", "ok": True, "result": tk})

            # 5) Option OHLC (ATM)
            opt_df = self.kite_src.get_option_ohlc(token=tk["tokens"]["ce"], token_pe=tk["tokens"]["pe"])
            checks.append({"name": "option_ohlc", "ok": len(opt_df) > 0, "rows": int(len(opt_df))})
            if opt_df is None or opt_df.empty:
                self.last_diag = result
                return

            # 6) Indicators
            regime = detect_regime(spot_df) if self.regime_mode in ("auto",) else self.regime_mode
            checks.append({"name": "indicators", "ok": True, "regime": regime})

            # 7) Signal
            sig = make_signal(spot_df, opt_df, regime=regime)
            checks.append({"name": "signal", "ok": bool(sig), "side": getattr(sig, "side", None)})
            if not sig:
                self.last_diag = result
                return

            # 8) Sizing
            lots = compute_lots(
                equity=settings.risk.default_equity,
                risk_fraction=settings.risk.risk_per_trade,  # fraction, e.g., 0.01
                stop_points=max(sig.sl_points, 5.0),
                lot_size=settings.instruments.nifty_lot_size,
                min_lots=settings.instruments.min_lots,
                max_lots=settings.instruments.max_lots,
            )
            checks.append({"name": "sizing", "ok": lots > 0, "lots": lots})
            if lots <= 0:
                self.last_diag = result
                return

            # 9) Execution
            if settings.enable_live_trading:
                self.executor.place_entry(
                    side=sig.side,
                    lots=lots,
                    entry_price=sig.entry_price,
                    stop_loss_points=sig.sl_points,
                    target_points=sig.tp_points,
                    tokens=tk["tokens"],
                )
            checks.append({"name": "execution_ready", "ok": True, "live": bool(settings.enable_live_trading)})

            result["ok"] = True
            self.last_diag = result

        except Exception as e:
            log.exception("Runner error: %s", e)
            self.last_diag = {"ok": False, "error": str(e), "checks": checks}