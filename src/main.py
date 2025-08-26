from __future__ import annotations

import logging
import signal
import sys
import time
from typing import Optional, List

from src.config import settings
from src.strategies.runner import StrategyRunner

# Optional broker SDK (used only at boot if live is already enabled)
try:
    from kiteconnect import KiteConnect  # type: ignore
except Exception:  # pragma: no cover
    KiteConnect = None  # type: ignore


# -----------------------------
# Logging
# -----------------------------
def _setup_logging() -> None:
    level = getattr(logging, (getattr(settings, "log_level", "INFO") or "INFO").upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)


# -----------------------------
# Tiny no-op Telegram (so runner ctor can never fail)
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
def _build_kite_if_live_enabled() -> Optional["KiteConnect"]:
    """Create a KiteConnect only if live is enabled at process boot."""
    log = logging.getLogger("main")
    if not bool(getattr(settings, "enable_live_trading", False)):
        log.info("Live trading disabled → paper mode.")
        return None

    if KiteConnect is None:
        raise RuntimeError("ENABLE_LIVE_TRADING=true but kiteconnect is not installed.")

    api_key = getattr(settings.zerodha, "api_key", None)
    access_token = getattr(settings.zerodha, "access_token", None)
    if not api_key or not access_token:
        raise RuntimeError("Missing Zerodha credentials while live mode is enabled.")

    kite = KiteConnect(api_key=str(api_key))
    kite.set_access_token(str(access_token))
    return kite


def _tail_logs(n: int = 180, path: str = "trading_bot.log") -> List[str]:
    """Return last n lines from log file; safe if file missing."""
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


def _import_telegram_class():
    """Import inside a wrapper so a TG file issue won't crash the bot."""
    try:
        from src.notifications.telegram_controller import TelegramController  # type: ignore
        return TelegramController
    except Exception as e:
        logging.getLogger("main").error(
            "TelegramController import failed — running with no-op Telegram.\nCause: %s",
            e,
        )
        return None


def _wire_real_telegram(runner: StrategyRunner) -> None:
    TelegramController = _import_telegram_class()
    if not TelegramController:
        return

    tg = TelegramController(
        # providers
        status_provider=runner.get_status_snapshot,
        positions_provider=getattr(runner.executor, "get_positions_kite", None),
        actives_provider=getattr(runner.executor, "get_active_orders", None),
        diag_provider=runner.get_last_flow_debug,  # ← correct provider
        logs_provider=_tail_logs,
        last_signal_provider=runner.get_last_signal_debug,
        # controls
        runner_pause=runner.pause,
        runner_resume=runner.resume,
        runner_tick=runner.runner_tick,  # accepts dry=bool
        cancel_all=getattr(runner.executor, "cancel_all_orders", None),
        # strategy/bot mutators (optional — use getattr to avoid AttributeError)
        set_live_mode=runner.set_live_mode,
        set_min_score=getattr(runner, "set_min_score", None),
        set_conf_threshold=getattr(runner, "set_conf_threshold", None),
        set_atr_period=getattr(runner, "set_atr_period", None),
        set_sl_mult=getattr(runner, "set_sl_mult", None),
        set_tp_mult=getattr(runner, "set_tp_mult", None),
        set_trend_boosts=getattr(runner, "set_trend_boosts", None),
        set_range_tighten=getattr(runner, "set_range_tighten", None),
    )

    # keep attribute for backwards compatibility
    runner.telegram_controller = tg  # type: ignore[attr-defined]
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
        getattr(settings, "enable_live_trading", False),
        getattr(settings.data, "time_filter_start", "09:20"),
        getattr(settings.data, "time_filter_end", "15:20"),
        float(getattr(settings.strategy, "rr_min", 0.0) or 0.0),
    )

    kite = _build_kite_if_live_enabled()

    # Create runner with a temporary no-op telegram so ctor never raises
    runner = StrategyRunner(kite=kite, telegram_controller=_NoopTelegram())
    _install_signal_handlers(runner)

    # Build the real TelegramController and wire providers from runner
    _wire_real_telegram(runner)

    # Announce
    try:
        runner.telegram_controller.send_message("🚀 Bot starting (shadow mode by default).")  # type: ignore[attr-defined]
    except Exception:
        log.warning("Telegram startup message failed (continuing).")

    # Optionally start a runner loop if provided by your code
    try:
        if hasattr(runner, "start"):
            runner.start()
    except Exception as e:
        log.exception("Runner start failed: %s", e)
        return 1

    # health loop
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
            runner.telegram_controller.send_message("🛑 Bot stopped.")  # type: ignore[attr-defined]
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        logging.getLogger("main").exception("Fatal error in main: %s", e)
        sys.exit(1)