# Path: src/main.py
from __future__ import annotations

import logging
import signal
import sys
import time
from typing import Optional, Callable, Any

from src.config import settings
from src.strategies.runner import StrategyRunner
from src.utils.logging_tools import log_buffer_handler  # ring buffer for /logs

# Optional broker SDK (live only)
try:
    from kiteconnect import KiteConnect  # type: ignore
except Exception:  # pragma: no cover
    KiteConnect = None  # type: ignore


# -----------------------------
# Logging
# -----------------------------
def _setup_logging() -> None:
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(level)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                                      datefmt="%Y-%m-%d %H:%M:%S"))
    root.addHandler(ch)

    # In-memory ring buffer (used by Telegram /logs)
    log_buffer_handler.setLevel(level)
    root.addHandler(log_buffer_handler)

    # Quiet noisy libs
    logging.getLogger("urllib3").setLevel(logging.WARNING)


# -----------------------------
# Tiny no-op Telegram (so runner ctor never fails)
# -----------------------------
class _NoopTelegram:
    def send_message(self, *_a, **_k) -> None:
        pass

    def start_polling(self) -> None:
        pass

    def stop_polling(self) -> None:
        pass


# -----------------------------
# Builders
# -----------------------------
def _build_kite() -> Optional["KiteConnect"]:
    log = logging.getLogger("main")
    if not settings.enable_live_trading:
        log.info("Live trading disabled → paper mode.")
        return None
    if KiteConnect is None:
        raise RuntimeError("ENABLE_LIVE_TRADING=true but kiteconnect is not installed.")
    api_key = settings.zerodha.api_key
    access_token = settings.zerodha.access_token
    if not api_key or not access_token:
        raise RuntimeError("Missing Zerodha credentials while live mode is enabled.")
    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)
    return kite


def _import_telegram_class():
    """
    Import inside a wrapper so any syntax/import issues in the controller don't crash boot.
    """
    try:
        from src.notifications.telegram_controller import TelegramController  # type: ignore
        return TelegramController
    except Exception as e:
        logging.getLogger("main").error(
            "TelegramController import failed — continuing with no-op Telegram. Cause: %s",
            e,
        )
        return None


def _wire_real_telegram(runner: StrategyRunner) -> None:
    TelegramController = _import_telegram_class()
    if not TelegramController:
        # Keep runner.telegram as the _NoopTelegram from construction
        return

    # Providers & controls — use getattr to avoid AttributeErrors if a hook is absent
    tg = TelegramController(
        # providers
        status_provider=runner.get_status_snapshot,
        positions_provider=getattr(runner.executor, "get_positions_kite", None),
        actives_provider=getattr(runner.executor, "get_active_orders", None),
        diag_provider=getattr(runner, "get_last_flow_debug", None),
        logs_provider=lambda n=120: log_buffer_handler.tail(n),  # ring buffer
        last_signal_provider=getattr(runner, "get_last_signal_debug", None),
        # controls
        runner_pause=getattr(runner, "pause", None),
        runner_resume=getattr(runner, "resume", None),
        runner_tick=getattr(runner, "runner_tick", None),  # accepts dry=bool
        cancel_all=getattr(runner.executor, "cancel_all_orders", None),
        # strategy/bot controls (only if runner implements them)
        set_live_mode=getattr(runner, "set_live_mode", None),
        set_min_score=getattr(runner, "set_min_score", None),
        set_conf_threshold=getattr(runner, "set_conf_threshold", None),
        set_atr_period=getattr(runner, "set_atr_period", None),
        set_sl_mult=getattr(runner, "set_sl_mult", None),
        set_tp_mult=getattr(runner, "set_tp_mult", None),
        set_trend_boosts=getattr(runner, "set_trend_boosts", None),
        set_range_tighten=getattr(runner, "set_range_tighten", None),
    )

    # Back-compat attributes the runner may expect
    runner.telegram_controller = tg
    runner.telegram = tg

    try:
        tg.start_polling()
        logging.getLogger("src.notifications.telegram_controller").info(
            "Telegram polling started (chat_id=%s).", getattr(tg, "_chat_id", "?")
        )
    except Exception:
        logging.getLogger("main").warning("Telegram polling failed to start; continuing.")


# -----------------------------
# App lifecycle
# -----------------------------
_stop_flag = False


def _install_signal_handlers(_runner: StrategyRunner) -> None:
    def _handler(signum, _frame):
        nonlocal _runner
        global _stop_flag
        logging.getLogger("main").info("Signal %s received — shutting down…", signum)
        _stop_flag = True

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handler)
        except Exception:
            pass


def main() -> int:
    _setup_logging()
    log = logging.getLogger("main")

    log.info(
        "Boot | live=%s window=%s-%s rr_min=%.2f",
        settings.enable_live_trading,
        settings.data.time_filter_start,
        settings.data.time_filter_end,
        getattr(settings.strategy, "rr_min", 0.0),
    )

    # Broker (None in paper mode)
    kite = _build_kite()

    # 1) Create runner with a temporary no-op telegram so ctor will never raise
    runner = StrategyRunner(kite=kite, telegram_controller=_NoopTelegram())
    _install_signal_handlers(runner)

    # 2) Build & wire the real Telegram controller (non-fatal if it fails)
    _wire_real_telegram(runner)

    # Announce (best-effort)
    try:
        if getattr(runner, "telegram_controller", None):
            runner.telegram_controller.send_message("🚀 Bot starting (shadow mode by default).")
    except Exception:
        log.warning("Telegram startup message failed (continuing).")

    # If your runner has start(), call it; else we just do the health loop
    try:
        if hasattr(runner, "start"):
            runner.start()
    except Exception as e:
        log.exception("Runner start failed: %s", e)
        return 1

    # Health loop
    try:
        while not _stop_flag:
            try:
                runner.health_check()
            except Exception as e:
                log.warning("Health check warn: %s", e)
            time.sleep(5)
    finally:
        try:
            runner.shutdown()
        except Exception:
            pass
        try:
            if getattr(runner, "telegram_controller", None):
                runner.telegram_controller.send_message("🛑 Bot stopped.")
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        logging.getLogger("main").exception("Fatal error in main: %s", e)
        sys.exit(1)