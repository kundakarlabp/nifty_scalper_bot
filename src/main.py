# Path: src/main.py
from __future__ import annotations

import logging
import signal
import sys
import time
from logging.handlers import RotatingFileHandler
from typing import Optional

from src.config import settings
from src.strategies.runner import StrategyRunner

# Optional broker SDK (live only)
try:
    from kiteconnect import KiteConnect  # type: ignore
except Exception:
    KiteConnect = None  # type: ignore

# Your Telegram controller (the one you shared)
try:
    from src.notifications.telegram_controller import TelegramController  # type: ignore
except Exception:
    TelegramController = None  # type: ignore


# -----------------------------
# Logging
# -----------------------------
def _setup_logging() -> None:
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    # File logs → support /logs from Telegram
    try:
        fh = RotatingFileHandler("trading_bot.log", maxBytes=2_000_000, backupCount=3)
        fh.setLevel(level)
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        logging.getLogger().addHandler(fh)
    except Exception:
        # Non-fatal; console-only logging still works
        pass


# -----------------------------
# Tiny no-op Telegram used only to satisfy runner ctor
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
    if not settings.enable_live_trading:
        logging.getLogger("main").info("Live trading disabled → paper mode.")
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


def _tail_logs(n: int = 100, path: str = "trading_bot.log") -> list[str]:
    """Return last n lines from log file; used by /logs. Safe if file missing."""
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


def _wire_real_telegram(runner: StrategyRunner):
    """
    Build your real TelegramController and wire providers from the runner,
    keeping original names/flow. Minimal wiring; avoids attribute errors.
    """
    if TelegramController is None:
        raise RuntimeError("src.notifications.telegram_controller not found.")

    tg = TelegramController(
        status_provider=runner.get_status_snapshot,
        positions_provider=getattr(runner.executor, "get_positions_kite", None),
        actives_provider=getattr(runner.executor, "get_active_orders", None),
        diag_provider=runner.get_system_diag,          # << detailed /diag and /check
        logs_provider=_tail_logs,                      # enables /logs [n]
        last_signal_provider=runner.get_last_signal_debug,
        runner_pause=runner.pause,
        runner_resume=runner.resume,
        runner_tick=runner.runner_tick,                # accepts dry=bool in our runner
        cancel_all=getattr(runner.executor, "cancel_all_orders", None),
        set_risk_pct=None,
        toggle_trailing=None,
        set_trailing_mult=None,
        toggle_partial=None,
        set_tp1_ratio=None,
        set_breakeven_ticks=None,
        set_live_mode=runner.set_live_mode,
        set_min_score=None,
        set_conf_threshold=None,
        set_atr_period=None,
        set_sl_mult=None,
        set_tp_mult=None,
        set_trend_boosts=None,
        set_range_tighten=None,
    )

    # Keep both attributes to match old/new code paths
    runner.telegram_controller = tg
    runner.telegram = tg

    try:
        tg.start_polling()
    except Exception:
        logging.getLogger("main").warning("Telegram polling failed to start; continuing.")


# -----------------------------
# App lifecycle
# -----------------------------
_stop_flag = False


def _install_signal_handlers(_runner: StrategyRunner) -> None:
    def _handler(signum, _frame):
        global _stop_flag
        logging.getLogger("main").info(f"Signal {signum} received — shutting down…")
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

    kite = _build_kite()

    # 1) create runner with a temporary no-op telegram so ctor won't raise
    runner = StrategyRunner(kite=kite, telegram_controller=_NoopTelegram())
    _install_signal_handlers(runner)

    # 2) now build your real TelegramController and wire providers from runner
    _wire_real_telegram(runner)

    # announce
    try:
        runner.telegram_controller.send_message("🚀 Bot starting (shadow mode by default).")
    except Exception:
        log.warning("Telegram startup message failed (continuing).")

    # If your runner has start(), call it (else we just health-loop)
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
                if hasattr(runner, "health_check"):
                    runner.health_check()
            except Exception as e:
                log.warning("Health check warn: %s", e)
            time.sleep(5)
    finally:
        try:
            if hasattr(runner, "shutdown"):
                runner.shutdown()
        except Exception:
            pass
        try:
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