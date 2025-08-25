from __future__ import annotations
import logging, signal, sys, time
from typing import Optional

from src.config import settings
from src.strategies.runner import StrategyRunner
from src.notifications.telegram_controller import TelegramController

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
    if not settings.enable_live_trading:
        logging.getLogger("main").info("Live trading disabled → paper mode.")
        return None
    if KiteConnect is None:
        raise RuntimeError("kiteconnect not available but live trading is enabled")
    kite = KiteConnect(api_key=settings.zerodha.api_key)
    kite.set_access_token(settings.zerodha.access_token)
    logging.getLogger("main").info("KiteConnect initialized.")
    return kite

def main() -> int:
    _setup_logging()
    log = logging.getLogger("main")

    kite = _build_kite()

    # Build StrategyRunner (which wires in TelegramController itself)
    runner = StrategyRunner(kite=kite)
    def _handler(signum, _frame):
        log.info(f"Signal {signum} received — shutting down…")
        runner.shutdown()
        sys.exit(0)
    for sig in (signal.SIGINT, signal.SIGTERM):
        try: signal.signal(sig, _handler)
        except Exception: pass

    try:
        runner.telegram.send_message("🚀 Bot starting (shadow mode by default).")
    except Exception:
        log.warning("Telegram startup message failed.")

    try:
        runner.start()
        while True:
            time.sleep(5)
            runner.health_check()
    except KeyboardInterrupt:
        runner.shutdown()
    except Exception as e:
        log.exception("Fatal error: %s", e)
        runner.shutdown()
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())