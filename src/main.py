# src/main.py
from __future__ import annotations

import logging
import os
import signal
import sys
from threading import Event, Thread
from typing import Dict, Any

from src.config import settings
from src.data.source import LiveKiteSource
from src.execution.order_executor import OrderExecutor
from src.notifications.telegram_controller import TelegramController
from src.risk.position_sizing import PositionSizer
from src.risk.session import TradingSession
from src.server.health import run as run_health_server
from src.strategies.runner import StrategyRunner
from src.utils import strike_selector as sel
from src.utils.kite_auth import build_kite_from_env
from src.strategies.scalping_strategy import EnhancedScalpingStrategy

# ---- logging ----
try:
    _LOG_LEVEL = getattr(logging, str(settings.log_level).upper())
except Exception:
    _LOG_LEVEL = logging.INFO
logging.basicConfig(level=_LOG_LEVEL, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("main")


class Application:
    def __init__(self) -> None:
        self._stop_event = Event()
        self.settings = settings

        # Ensure IST time base unless overridden by Railway/ENV
        os.environ.setdefault("TZ", os.getenv("TZ", "Asia/Kolkata"))

        # Build authenticated Kite client (validates token or auto-generates)
        try:
            self.kite = build_kite_from_env()
        except Exception as e:
            logger.critical("Kite init/auth failed: %s", e, exc_info=True)
            sys.exit(1)

        # Core components
        self.data_source = LiveKiteSource(self.kite)
        self.strategy = EnhancedScalpingStrategy(self.settings.strategy)
        self.executor = OrderExecutor(self.settings.executor, self.kite)
        self.sizer = PositionSizer(self.settings.risk)
        self.session = TradingSession(self.settings.risk, self.settings.executor, starting_equity=100_000.0)

        # Telegram (support both implementations: requests-based or PTB-based)
        self.telegram: TelegramController | None = None
        if getattr(self.settings, "enable_telegram", False) and getattr(self.settings, "telegram", None):
            bot_token = getattr(self.settings.telegram, "bot_token", "") or os.getenv("TELEGRAM_BOT_TOKEN", "")
            chat_id = getattr(self.settings.telegram, "chat_id", None) or os.getenv("TELEGRAM_CHAT_ID")
            if bot_token:
                try:
                    # Newer (requests-based) controller signature
                    self.telegram = TelegramController(
                        status_callback=self.get_status,
                        control_callback=self.control_bot,
                        summary_callback=lambda: f"Daily PnL: {self.session.daily_pnl:.2f}",
                        bot_token=bot_token,
                        chat_id=chat_id,
                    )
                except TypeError:
                    # Fallback to PTB-based controller
                    self.telegram = TelegramController(bot_token=bot_token, chat_id=chat_id)
                    if hasattr(self.telegram, "set_context"):
                        self.telegram.set_context(
                            session=self.session,
                            executor=self.executor,
                            health_getter=self.get_health,
                        )

        self.runner = StrategyRunner(
            data_source=self.data_source,
            strategy=self.strategy,
            order_executor=self.executor,
            trading_session=self.session,
            position_sizer=self.sizer,
            telegram_controller=self.telegram,
        )

        self._runner_thread: Thread | None = None
        self._health_thread: Thread | None = None

    # ---- lifecycle ----

    def _tg_start(self) -> None:
        if not self.telegram:
            return
        # Support both variants
        if hasattr(self.telegram, "start_polling"):
            self.telegram.start_polling()  # requests-based
        elif hasattr(self.telegram, "start"):
            self.telegram.start()          # PTB-based

    def _tg_stop(self) -> None:
        if not self.telegram:
            return
        if hasattr(self.telegram, "stop_polling"):
            self.telegram.stop_polling()
        elif hasattr(self.telegram, "stop"):
            self.telegram.stop()

    def start(self) -> None:
        signal.signal(signal.SIGINT, self._on_signal)
        signal.signal(signal.SIGTERM, self._on_signal)

        # Health server
        self._health_thread = Thread(target=run_health_server, args=(self.get_health,), daemon=True)
        self._health_thread.start()

        # Data source ready
        try:
            self.data_source.connect()
        except Exception as e:
            logger.critical("Data source connect failed: %s", e, exc_info=True)
            sys.exit(1)

        # Telegram UI
        self._tg_start()

        # Strategy runner
        self._runner_thread = Thread(target=self.runner.start, name="runner", daemon=True)
        self._runner_thread.start()

        logger.info("Application started. Live mode: %s, TZ=%s", self.settings.enable_live_trading, os.getenv("TZ"))
        self._stop_event.wait()
        logger.info("Application exiting.")

    def _on_signal(self, signum, _frame) -> None:
        logger.info("Signal received: %s – shutting down…", signum)
        self.stop()

    def stop(self) -> None:
        if not self._stop_event.is_set():
            try:
                self.runner.stop()
            except Exception:
                pass
            try:
                self._tg_stop()
            except Exception:
                pass
            self._stop_event.set()

    # ---- status & health ----

    def get_status(self) -> Dict[str, Any]:
        """
        Compact status blob for Telegram.
        """
        return {
            "is_trading": bool(self.runner and getattr(self.runner, "_running", False)),
            "live_mode": bool(self.settings.enable_live_trading),
            "open_positions": len(self.session.active_trades),
            "trades_today": self.session.trades_today,
            "daily_pnl": self.session.daily_pnl,
            "consecutive_losses": self.session.consecutive_losses,
            "session_date": str(self.session.start_day),
            "uptime_sec": self.session.uptime_sec,
            # optional quality/regime fields if your runner populates them:
            "quality_mode": getattr(self.strategy, "quality_mode", "auto"),
            "quality_auto": getattr(self.strategy, "quality_auto", True),
            "regime_mode": getattr(self.strategy, "regime_mode", "auto"),
        }

    def get_health(self) -> Dict[str, Any]:
        try:
            base = sel.health_check(self.kite)
        except Exception as e:
            base = {"overall_status": "ERROR", "message": f"health_check failed: {e}", "checks": {}}
        base["runner"] = {"running": bool(self.runner and getattr(self.runner, "_running", False)),
                          "live_mode": bool(self.settings.enable_live_trading)}
        base["session"] = {
            "open_positions": len(self.session.active_trades),
            "trades_today": self.session.trades_today,
            "daily_pnl": self.session.daily_pnl,
            "consecutive_losses": self.session.consecutive_losses,
            "paused": self.session.paused,
            "day_stopped": self.session.is_day_stopped(),
        }
        return base

    # ---- Telegram control bridge ----

    def control_bot(self, command: str, value: str) -> bool:
        """
        Handles Telegram commands. Return True on success.
        Only toggles in-memory flags; persist if you want via your config layer.
        """
        try:
            cmd = command.lower().strip()
            if cmd == "start":
                # if you implement a pause gate in runner, clear it here
                self.session.resume_entries()
                return True
            if cmd == "stop":
                self.stop()
                return True
            if cmd == "refresh":
                # plug any reload logic you want (configs/caches)
                return True
            if cmd == "health":
                return True
            if cmd == "emergency":
                try:
                    self.executor.cancel_all_orders()
                except Exception:
                    pass
                # Flatten session book quickly using executor LTP
                price_fn = lambda sym: self.executor.get_last_price(sym)
                self.session.flatten_all(price_fn)
                return True
            if cmd == "mode":
                v = (value or "live").lower()
                if v in {"live", "shadow"}:
                    self.settings.enable_live_trading = (v == "live")
                    return True
                return False
            if cmd == "quality":
                v = (value or "auto").lower()
                if hasattr(self.strategy, "set_quality_mode"):
                    self.strategy.set_quality_mode(v)  # type: ignore[attr-defined]
                    return True
                # else store on strategy for status visibility
                setattr(self.strategy, "quality_mode", v)
                return True
            if cmd == "regime":
                v = (value or "auto").lower()
                if hasattr(self.strategy, "set_regime_mode"):
                    self.strategy.set_regime_mode(v)  # type: ignore[attr-defined]
                    return True
                setattr(self.strategy, "regime_mode", v)
                return True
            if cmd == "risk":
                # value expected as percent (e.g., "0.5" = 0.5%)
                try:
                    pct = float(value)
                except Exception:
                    return False
                # store as fraction (0.005) if your PositionSizer expects that
                if hasattr(self.sizer, "set_risk_per_trade"):
                    self.sizer.set_risk_per_trade(pct / 100.0)  # type: ignore[attr-defined]
                else:
                    setattr(self.settings.risk, "risk_per_trade", pct / 100.0)
                return True
            if cmd == "pause":
                mins = 1
                try:
                    mins = max(1, int(float(value))) if value else 1
                except Exception:
                    mins = 1
                self.session.pause_entries(mins)
                return True
            if cmd == "resume":
                self.session.resume_entries()
                return True
        except Exception as e:
            logger.error("control_bot error for cmd=%s val=%s: %s", command, value, e, exc_info=True)
        return False


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "start":
        app = Application()
        app.start()
    else:
        print("Usage: python -m src.main start", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
