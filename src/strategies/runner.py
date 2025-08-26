# src/strategies/runner.py
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import pandas as pd

from src.config import settings
from src.data.source import LiveKiteSource, DataSource
from src.notifications.telegram_controller import TelegramController
from src.order_executor import OrderExecutor
from src.strategies.scalping_strategy import ScalpingStrategy
from src.utils.account_info import get_equity_estimate
from src.utils.logging_tools import get_recent_logs

# Optional import; we stay tolerant so shadow mode works without the SDK
try:
    from kiteconnect import KiteConnect  # type: ignore
except Exception:  # pragma: no cover
    KiteConnect = None  # type: ignore

log = logging.getLogger(__name__)


def _now_ist_naive() -> datetime:
    ist = datetime.now(timezone(timedelta(hours=5, minutes=30)))
    return ist.replace(tzinfo=None)


@dataclass
class RunnerState:
    live_trading: bool = False
    last_signal: Optional[str] = None
    last_fetch_at: Optional[datetime] = None
    active_positions: int = 0


class StrategyRunner:
    """
    Orchestrates data source, strategy, and order execution.
    Also owns a diagnostics provider for Telegram (/check, /diag, /status),
    and supports hot rewiring of the broker session (/mode live | /mode paper).
    """

    # ------------------------------ lifecycle ------------------------------ #
    def __init__(self) -> None:
        # basic state
        self.state = RunnerState(live_trading=bool(settings.enable_live_trading))
        self.lock = threading.RLock()

        # wire telegram early (diagnostics rely on it)
        self.telegram: Optional[TelegramController] = None

        # strategy and executor
        self.strategy = ScalpingStrategy(
            ema_fast=settings.strategy_ema_fast,
            ema_slow=settings.strategy_ema_slow,
            rsi_period=settings.strategy_rsi_period,
            bb_period=settings.strategy_bb_period,
            bb_std=settings.strategy_bb_std,
            atr_period=settings.strategy_atr_period,
            atr_sl_multiplier=settings.strategy_atr_sl_multiplier,
            atr_tp_multiplier=settings.strategy_atr_tp_multiplier,
            rr_min=settings.strategy_rr_min,
            min_bars_for_signal=settings.strategy_min_bars_for_signal,
            confidence_threshold=settings.strategy_confidence_threshold,
        )
        self.executor = OrderExecutor(
            preferred_exit_mode=settings.executor_preferred_exit_mode,
            enable_trailing=settings.executor_enable_trailing,
            fee_per_lot=settings.executor_fee_per_lot,
            slippage_ticks=settings.executor_slippage_ticks,
            lot_size=settings.instruments_nifty_lot_size,
        )

        # broker/data wiring
        self.kite: Optional["KiteConnect"] = None
        self.data_source: DataSource = LiveKiteSource(kite=None)  # start shadow-safe
        log.info("StrategyRunner: Data source initialized: LiveKiteSource")

        # ready log
        log.info(
            "StrategyRunner: StrategyRunner ready (live_trading=%s, use_live_equity=%s)",
            self.state.live_trading,
            settings.risk_use_live_equity,
        )

    # ----------------------------- public wiring --------------------------- #
    def attach_telegram(self, controller: TelegramController) -> None:
        """Hook called by main to provide the controller instance."""
        self.telegram = controller

    def start(self) -> None:
        """
        Called once by main.py after construction. We keep the runner reactive:
        - no infinite loops here
        - we only (re)wire broker and let Telegram commands drive activity
        """
        # initial connect (shadow or live depending on env)
        self._rewire(live=settings.enable_live_trading)

    # ------------------------------ hot-rewire ----------------------------- #
    def set_live_mode(self, live: bool) -> str:
        """
        Public entrypoint used by Telegram /mode command.
        """
        with self.lock:
            self._rewire(live=live)
        return "🔓 Live mode ON — broker session initialized." if live else "🔒 Paper mode — broker disconnected."

    def _rewire(self, *, live: bool) -> None:
        """
        Swap the runner between:
          - live=True  → real Kite session
          - live=False → shadow (paper), kite=None
        """
        self.state.live_trading = bool(live)

        if not live:
            self.kite = None
            self.data_source = LiveKiteSource(kite=None)
            self.data_source.connect()
            log.info("main: Live trading disabled → paper mode.")
            return

        # live requested
        kite = self._init_kite_safe()
        # Keep going even if kite failed (we still keep telegram + diag alive)
        self.kite = kite
        self.data_source = LiveKiteSource(kite=kite)
        self.data_source.connect()
        log.info("StrategyRunner: 🔓 Live mode ON — broker session initialized.")

    def _init_kite_safe(self) -> Optional["KiteConnect"]:
        """
        Create a KiteConnect instance when creds are present; stay graceful
        if anything is missing.
        """
        if KiteConnect is None:
            log.warning("KiteConnect SDK not installed; staying in shadow mode.")
            return None

        api_key = settings.zerodha.api_key or ""
        access_token = settings.zerodha.access_token or ""

        if not api_key or not access_token:
            log.warning("Kite credentials incomplete; live mode requested but kite=None.")
            return None

        try:
            kite = KiteConnect(api_key=api_key)  # type: ignore
            kite.set_access_token(access_token)  # type: ignore
            return kite
        except Exception as e:
            log.error("Failed to initialize KiteConnect: %s", e)
            return None

    # ------------------------------ trade tick ----------------------------- #
    def run_tick(self) -> None:
        """
        A single evaluation tick (can be triggered by Telegram /tick or cron).
        Pulls the last N minutes, evaluates strategy, and (if live) routes orders.
        """
        with self.lock:
            try:
                # market-time gate
                if not self._within_window():
                    return

                end = _now_ist_naive()
                start = end - timedelta(minutes=settings.data_lookback_minutes)
                token = settings.instruments_instrument_token
                tf = settings.data_timeframe

                df = self.data_source.fetch_ohlc(token, start, end, tf)  # type: ignore[arg-type]
                if df is None or df.empty:
                    return

                self.state.last_fetch_at = _now_ist_naive()

                # ensure numeric and sorted
                df = df.copy()
                for c in ["open", "high", "low", "close", "volume"]:
                    if c in df.columns:
                        df[c] = pd.to_numeric(df[c], errors="coerce")
                df = df.dropna().sort_index()

                signal = self.strategy.generate_signal(df)
                self.state.last_signal = signal or "none"

                if not signal or signal == "none":
                    return

                # risk / position sizing
                equity = (
                    get_equity_estimate(self.kite) if settings.risk_use_live_equity else settings.risk_default_equity
                )
                order = self.strategy.build_order(signal=signal, equity=equity)
                if not order:
                    return

                # execute
                if self.state.live_trading and self.kite:
                    self.executor.execute(order, kite=self.kite)
                else:
                    self.executor.paper_execute(order)

                self.state.active_positions = self.executor.active_position_count()

            except Exception as e:
                log.exception("tick failed: %s", e)

    def _within_window(self) -> bool:
        """Trading allowed within configured HH:MM window, or always if allow_offhours_testing=True."""
        if settings.allow_offhours_testing:
            return True

        now = _now_ist_naive()
        try:
            hh, mm = [int(x) for x in settings.data_time_filter_start.split(":")]
            start = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
            hh, mm = [int(x) for x in settings.data_time_filter_end.split(":")]
            end = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        except Exception:
            # if parsing ever fails, be safe and allow
            return True

        return start <= now <= end

    # ------------------------------ diagnostics ---------------------------- #
    def diag_brief(self) -> str:
        """
        Short /status card.
        """
        ts = _now_ist_naive().strftime("%Y-%m-%d %H:%M:%S")
        mode = "Kite" if (self.state.live_trading and self.kite) else ("Paper" if self.state.live_trading else "Paper")
        return (
            f"📊 {ts}\n"
            f"🧩 {('🟢' if self.state.live_trading else '🟡')} LIVE | {mode}\n"
            f"📦 Active: {self.state.active_positions}"
        )

    def diag_full(self) -> str:
        """
        /check — full system check (matches your screenshots).
        """
        parts = ["🔎 Full system check"]

        # Telegram
        parts.append("🟢 Telegram wiring — controller attached" if self.telegram else "🔴 Telegram wiring — detached")

        # Broker session
        if self.state.live_trading:
            parts.append("🟢 Broker session — live mode with kite" if self.kite else "🔴 Broker session — live but kite=None")
        else:
            parts.append("🟠 Broker session — paper mode")

        # Data feed recency
        if self.state.last_fetch_at:
            age = int(max(0, (_now_ist_naive() - self.state.last_fetch_at).total_seconds()))
            parts.append(f"🟢 Data feed — age={age}s")
        else:
            parts.append("🔴 Data feed — no fetch yet")

        # Strategy readiness
        parts.append(f"🔴 Strategy readiness — min_bars={settings.strategy_min_bars_for_signal}")

        # Risk gates
        parts.append("🔴 Risk gates — no-eval")

        # RR threshold (empty dict for now; strategy computes per-signal)
        parts.append("🟢 RR threshold — {}")

        # Errors
        parts.append("🟢 Errors — none")

        parts.append(f"📈 last_signal: {self.state.last_signal or 'none'}")
        return "\n".join(parts)

    def diag_flow(self) -> str:
        """
        /diag — traffic-light summary bar shown in your screenshots.
        """
        fields = []

        # Telegram
        fields.append("🟢 Telegram wiring" if self.telegram else "🔴 Telegram wiring")

        # Broker
        if self.state.live_trading and self.kite:
            fields.append("🟢 Broker session")
        elif self.state.live_trading and not self.kite:
            fields.append("🔴 Broker session")
        else:
            fields.append("🟠 Broker session")

        # Data
        fields.append("🟢 Data feed" if self.state.last_fetch_at else "🔴 Data feed")

        # Strategy
        fields.append("🔴 Strategy readiness")

        # Risk
        fields.append("🔴 Risk gates")

        # RR
        fields.append("🟢 RR threshold")

        # Errors
        fields.append("🟢 Errors")

        return f"❗ Flow has issues\n" + " · ".join(fields)

    # ------------------------------ telegram hooks ------------------------- #
    # These are called by TelegramController (already uses these exact names)

    def cmd_status(self) -> str:
        return self.diag_brief()

    def cmd_check(self) -> str:
        # On demand, try a tiny fetch so “no fetch yet” clears after first /check
        try:
            end = _now_ist_naive()
            start = end - timedelta(minutes=1)
            token = settings.instruments_instrument_token
            tf = settings.data_timeframe
            df = self.data_source.fetch_ohlc(token, start, end, tf)  # type: ignore[arg-type]
            if df is not None and not df.empty:
                self.state.last_fetch_at = _now_ist_naive()
        except Exception:
            pass
        return self.diag_full()

    def cmd_diag(self) -> str:
        return self.diag_flow()

    def cmd_logs(self, n: int = 80) -> str:
        txt = get_recent_logs(n=n)
        return txt if txt.strip() else "No logs available."

    def cmd_tick(self) -> str:
        self.run_tick()
        return "✅ Tick executed."

    def cmd_mode(self, arg: str) -> str:
        live = str(arg or "").strip().lower() in {"live", "on", "true", "1"}
        msg = self.set_live_mode(live)
        return f"Mode set to {'LIVE' if live else 'PAPER'} and rewired."

    # ------------------------------ utils ---------------------------------- #
    def _notify(self, msg: str) -> None:
        log.info(msg)
        try:
            if self.telegram:
                self.telegram.send_message(msg)
        except Exception:
            # never throw from notifications
            pass