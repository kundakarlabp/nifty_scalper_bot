from __future__ import annotations

import logging
import signal
import sys
import time
from typing import List, Optional

from src.config import settings
from src.strategies.runner import StrategyRunner

# Broker SDK (only used in live mode)
try:
    from kiteconnect import KiteConnect  # type: ignore
except Exception:
    KiteConnect = None  # type: ignore


def _setup_logging() -> None:
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def _build_kite() -> Optional["KiteConnect"]:
    """Create KiteConnect only in live mode; otherwise return None (paper mode)."""
    if not settings.enable_live_trading:
        logging.getLogger("main").info("Live trading disabled → paper mode.")
        return None

    if KiteConnect is None:
        raise RuntimeError("kiteconnect library not available but ENABLE_LIVE_TRADING=true")

    api_key = settings.zerodha.api_key
    access_token = settings.zerodha.access_token
    if not api_key or not access_token:
        raise RuntimeError("Missing Zerodha credentials while live mode is enabled.")

    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)
    logging.getLogger("main").info("KiteConnect initialized (live mode).")
    return kite


# -----------------------------
# App lifecycle
# -----------------------------

class _NoopTelegram:
    """Temporary placeholder so we can construct the runner before wiring the real Telegram controller."""
    def send_message(self, *_args, **_kwargs) -> None:
        pass


def main() -> int:
    _setup_logging()
    log = logging.getLogger("main")

    # Log important toggles for clarity
    log.info(
        "Boot | live=%s window=%s-%s rr_min=%.2f",
        settings.enable_live_trading,
        settings.data.time_filter_start,
        settings.data.time_filter_end,
        getattr(settings.strategy, "rr_min", 0.0),
    )

    # Build integrations
    kite = _build_kite()

    # 1) Build runner with a temporary Telegram object (your runner requires it non‑None)
    runner = StrategyRunner(kite=kite, telegram_controller=_NoopTelegram())

    # 2) Import your TelegramController and wire all providers from the runner/executor.
    from src.notifications.telegram_controller import TelegramController  # type: ignore

    def logs_provider(n: int) -> List[str]:
        # Minimal stub to satisfy /logs command; adapt if you later persist logs to a file.
        return []

    telegram = TelegramController(
        # providers
        status_provider=runner.get_status_snapshot,
        positions_provider=runner.executor.get_positions_kite if hasattr(runner, "executor") else None,
        actives_provider=runner.executor.get_active_orders if hasattr(runner, "executor") else None,
        diag_provider=lambda: {"ok": True, "checks": []},
        logs_provider=logs_provider,
        last_signal_provider=runner.get_last_signal_debug,
        # controls
        runner_pause=runner.pause,
        runner_resume=runner.resume,
        runner_tick=runner.runner_tick,
        cancel_all=runner.executor.cancel_all_orders if hasattr(runner, "executor") else None,
        # execution/strategy mutators (only what runner exposes)
        set_live_mode=runner.set_live_mode,
        # others are optional and left as None
    )

    # 3) Replace the placeholder with the real Telegram controller
    runner.telegram = telegram  # type: ignore[attr-defined]

    # Signal handlers
    def _handler(signum, _frame):
        log.info(f"Signal {signum} received — shutting down…")
        try:
            runner.shutdown()
        finally:
            try:
                telegram.send_message("🛑 Bot stopped.")
            except Exception:
                pass
            sys.exit(0)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handler)
        except Exception:
            pass

    # Announce boot
    try:
        telegram.send_message("🚀 Bot starting (shadow mode by default).")
    except Exception:
        log.warning("Telegram startup message failed (continuing).")

    # Start the runner (if your runner has start(); otherwise the health loop will still run)
    try:
        if hasattr(runner, "start"):
            runner.start()
    except Exception as e:
        log.exception("Runner start failed: %s", e)
        return 1

    # Lightweight health loop
    try:
        while True:
            try:
                if hasattr(runner, "health_check"):
                    runner.health_check()
            except Exception as e:
                log.warning("Health check warn: %s", e)
            time.sleep(5)
    except KeyboardInterrupt:
        _handler(signal.SIGINT, None)
    except Exception as e:
        log.exception("Fatal error in main: %s", e)
        _handler(signal.SIGTERM, None)
        return 1


if __name__ == "__main__":
    sys.exit(main())