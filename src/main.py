# src/main.py
from __future__ import annotations

import logging, signal, sys, time
from typing import Optional
from src.config import settings
from src.strategies.runner import StrategyRunner
from src.notifications.telegram_controller import TelegramController

try:
    from kiteconnect import KiteConnect  # type: ignore
except Exception:
    KiteConnect = None

def _setup_logging() -> None:
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)

def _build_kite() -> Optional["KiteConnect"]:
    if not settings.enable_live_trading:
        logging.getLogger("main").info("Live trading disabled → paper mode.")
        return None
    if KiteConnect is None:
        raise RuntimeError("kiteconnect not available but ENABLE_LIVE_TRADING=true")
    kite = KiteConnect(api_key=settings.zerodha.api_key)
    kite.set_access_token(settings.zerodha.access_token)
    return kite

_stop_flag = False
def _install_signal_handlers():
    def _handler(signum, _frame):
        global _stop_flag
        logging.getLogger("main").info(f"Signal {signum} received — shutting down…")
        _stop_flag = True
    for sig in (signal.SIGINT, signal.SIGTERM):
        try: signal.signal(sig, _handler)
        except Exception: pass

def main() -> int:
    _setup_logging()
    log = logging.getLogger("main")
    kite = _build_kite()

    # Runner first (has providers we pass to TelegramController)
    runner = StrategyRunner(kite=kite)

    telegram = TelegramController(
        status_provider=runner.get_status_snapshot,
        positions_provider=runner.executor.get_positions_kite,
        actives_provider=runner.executor.get_active_orders,
        diag_provider=runner.get_last_flow_debug,
        logs_provider=None,
        last_signal_provider=runner.get_last_signal_debug,
        runner_pause=runner.pause,
        runner_resume=runner.resume,
        runner_tick=runner.runner_tick,
        cancel_all=runner.executor.cancel_all_orders,
        set_live_mode=runner.set_live_mode,
    )
    runner.telegram = telegram

    _install_signal_handlers()
    telegram.send_message("🚀 Bot starting (shadow mode by default).")

    if hasattr(runner, "start"): runner.start()
    try:
        while not _stop_flag:
            runner.health_check()
            time.sleep(5)
    finally:
        runner.shutdown()
        telegram.send_message("🛑 Bot stopped.")
    return 0

if __name__ == "__main__":
    try: sys.exit(main())
    except Exception as e:
        logging.getLogger("main").exception("Fatal error: %s", e)
        sys.exit(1)