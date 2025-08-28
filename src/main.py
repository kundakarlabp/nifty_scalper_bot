# Path: src/main.py
from __future__ import annotations

import logging
import signal
import sys
import time
from typing import List, Optional

from src.config import settings, validate_critical_settings
from src.utils.logging_tools import RateLimitFilter
from src.strategies.runner import StrategyRunner

# Optional broker SDK
try:
    from kiteconnect import KiteConnect  # type: ignore
except Exception:
    KiteConnect = None  # type: ignore


# -----------------------------
# Logging
# -----------------------------
def _setup_logging() -> None:
    level = getattr(logging, (settings.log_level or "INFO").upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    root = logging.getLogger()
    root.addFilter(RateLimitFilter(interval=120.0))
    logging.getLogger("urllib3").setLevel(logging.WARNING)


# -----------------------------
# No-op Telegram fallback
# -----------------------------
class _NoopTelegram:
    """Minimal stand-in when Telegram is disabled."""

    def send_message(self, *_a, **_k) -> None:  # pragma: no cover - trivial
        return None

    def start_polling(self) -> None:  # pragma: no cover - trivial
        return None

    def stop_polling(self) -> None:  # pragma: no cover - trivial
        return None


# -----------------------------
# KiteConnect Builder
# -----------------------------
def _build_kite_session() -> Optional["KiteConnect"]:
    log = logging.getLogger("main")
    if not settings.enable_live_trading:
        log.info("Live trading disabled → paper mode.")
        return None

    if KiteConnect is None:
        raise RuntimeError("ENABLE_LIVE_TRADING=true but kiteconnect not installed.")

    api_key = settings.zerodha.api_key
    access_token = settings.zerodha.access_token
    if not api_key or not access_token:
        raise RuntimeError("Missing Zerodha credentials for live mode.")

    kite = KiteConnect(api_key=str(api_key))
    kite.set_access_token(str(access_token))
    log.info("✅ KiteConnect session initialized")
    return kite


# -----------------------------
# Tail logs for Telegram
# -----------------------------
def _tail_logs(n: int = 180, path: str = "trading_bot.log") -> List[str]:
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            block = 4096
            data = b""
            while size > 0 and data.count(b"\n") <= n:
                read_size = block if size >= block else size
                size -= read_size
                f.seek(size)
                data = f.read(read_size) + data
        text = data.decode(errors="ignore")
        return text.splitlines()[-n:]
    except Exception:
        return []


# -----------------------------
# Telegram Import Wrapper
# -----------------------------
def _import_telegram_class():
    try:
        from src.notifications.telegram_controller import (
            TelegramController,
        )  # type: ignore

        return TelegramController
    except Exception as e:
        logging.getLogger("main").error("TelegramController import failed: %s", e)
        return None


# -----------------------------
# Wire Telegram Controller
# -----------------------------
def _wire_real_telegram(runner: StrategyRunner) -> None:
    """Replace the temporary placeholder Telegram controller with the real one."""

    # Clear any placeholders so diagnostics don't see the no-op object
    runner.telegram_controller = None
    runner.telegram = None

    TelegramController = _import_telegram_class()
    if not TelegramController:
        return

    tg = TelegramController.create(
        # providers
        status_provider=getattr(runner, "get_status_snapshot", lambda: {"ok": False}),
        positions_provider=getattr(runner.executor, "get_positions_kite", None),
        actives_provider=getattr(runner.executor, "get_active_orders", None),
        diag_provider=getattr(runner, "build_diag", None),
        compact_diag_provider=getattr(runner, "get_compact_diag_summary", None),
        logs_provider=_tail_logs,
        last_signal_provider=getattr(runner, "get_last_signal_debug", None),
        # controls
        runner_pause=getattr(runner, "pause", None),
        runner_resume=getattr(runner, "resume", None),
        runner_tick=getattr(runner, "runner_tick", None),
        cancel_all=getattr(runner.executor, "cancel_all_orders", None),
        # strategy/bot mutators
        set_live_mode=runner.set_live_mode,
        set_min_score=getattr(runner, "set_min_score", None),
        set_conf_threshold=getattr(runner, "set_conf_threshold", None),
        set_atr_period=getattr(runner, "set_atr_period", None),
        set_sl_mult=getattr(runner, "set_sl_mult", None),
        set_tp_mult=getattr(runner, "set_tp_mult", None),
        set_trend_boosts=getattr(runner, "set_trend_boosts", None),
        set_range_tighten=getattr(runner, "set_range_tighten", None),
    )
    if tg is None:
        return

    runner.telegram_controller = tg  # back-compat
    runner.telegram = tg

    try:
        tg.start_polling()
        logging.getLogger("main").info("📡 Telegram polling started")
    except Exception:
        logging.getLogger("main").warning("Telegram polling failed to start")


# -----------------------------
# Lifecycle
# -----------------------------
_stop_flag = False


def _install_signal_handlers(_runner: StrategyRunner) -> None:
    def _handler(signum, _frame):
        global _stop_flag
        logging.getLogger("main").info("Signal %s received — shutting down…", signum)
        _stop_flag = True

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handler)
        except Exception:
            logging.getLogger("main").warning("Failed to set handler for %s", sig)


def main() -> int:
    _setup_logging()
    try:
        validate_critical_settings()
    except Exception as e:
        logging.getLogger("main").error("\u274c Config validation failed: %s", e)
        return 1
    log = logging.getLogger("main")

    kite = None
    try:
        kite = _build_kite_session()
    except Exception as e:
        log.error("❌ Kite session init failed: %s", e)

    runner = StrategyRunner(kite=kite or None, telegram_controller=_NoopTelegram())

    try:
        runner.set_live_mode(settings.enable_live_trading)
    except Exception as e:
        log.error("⚠️ Live mode setup failed: %s", e)

    _install_signal_handlers(runner)

    # Wire Telegram
    _wire_real_telegram(runner)

    # Announce
    try:
        mode = "LIVE" if settings.enable_live_trading else "DRY"
        runner.telegram_controller.send_message(f"🚀 Bot starting ({mode})")
    except Exception:
        log.warning("Telegram startup message failed")

    try:
        if hasattr(runner, "start"):
            runner.start()
    except Exception as e:
        log.exception("Runner start failed: %s", e)
        return 1

    try:
        while not _stop_flag:
            try:
                runner.process_tick(tick=None)
                runner.health_check()
                time.sleep(5)
            except Exception as e:
                log.exception("Main loop error: %s", e)
                time.sleep(1)
                continue
    finally:
        try:
            runner.shutdown()
        except Exception:
            log.exception("Runner shutdown failed")
        try:
            runner.telegram_controller.send_message("🛑 Bot stopped.")
        except Exception:
            log.warning("Failed to send shutdown message to Telegram")
        try:
            runner.telegram_controller.stop_polling()
        except Exception:
            log.warning("Failed to stop Telegram polling")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        logging.getLogger("main").exception("Fatal error in main: %s", e)
        sys.exit(1)
