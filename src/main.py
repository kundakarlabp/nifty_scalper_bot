# Path: src/main.py
from __future__ import annotations

import logging
import signal
import sys
import time
import os
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Callable, List, Optional, Dict, Any

# Ensure project root in sys.path when executed as a script
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.boot.validate_env import (
    seed_env_from_defaults,
    validate_critical_settings,
    broker_connect_for_data,
)  # noqa: E402

seed_env_from_defaults()
from src.config import settings  # noqa: E402
import src.strategies.patches  # noqa: E402,F401  # activate runtime patches
from src.utils.logging_tools import get_recent_logs  # noqa: E402
from src.diagnostics.metrics import metrics  # noqa: E402
from src.strategies.runner import StrategyRunner  # noqa: E402
from src.server import health  # noqa: E402
from src.diagnostics.file_check import run_file_diagnostics  # noqa: E402
from src.notifications.telegram_commands import TelegramCommands  # noqa: E402
from src.server.logging_utils import _setup_logging, _import_telegram_class  # noqa: E402

# Optional broker SDK
try:
    from kiteconnect import KiteConnect  # type: ignore
except Exception as exc:
    logging.getLogger("main").warning(
        "KiteConnect import failed: %s", exc, exc_info=True
    )
    KiteConnect = None  # type: ignore


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
        if not broker_connect_for_data():
            log.info("Live trading disabled → paper mode.")
            return None
        log.info("Paper mode with broker data enabled.")

    if KiteConnect is None:
        raise RuntimeError("ENABLE_LIVE_TRADING=true but kiteconnect not installed.")

    api_key = settings.zerodha.api_key
    access_token = settings.zerodha.access_token
    if not api_key or not access_token:
        raise RuntimeError("Missing Zerodha credentials for broker data.")

    kite = KiteConnect(api_key=str(api_key))
    kite.set_access_token(str(access_token))
    log.info("✅ KiteConnect session initialized")
    return kite


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
        logging.getLogger("main").error(
            "TelegramController import failed: %s", e, exc_info=True
        )
        return None


# -----------------------------
# Wire Telegram Controller
# -----------------------------
def _wire_real_telegram(runner: StrategyRunner) -> None:
    """Replace the temporary placeholder Telegram controller with the real one."""

    # Clear any placeholders so diagnostics don't see the no-op object
    runner.telegram_controller = None
    runner.telegram = None

    try:
        TelegramController = _import_telegram_class()
    except ImportError:
        runner.telegram_controller = _NoopTelegram()
        runner.telegram = runner.telegram_controller
        return

    tg = TelegramController.create(
        # providers
        status_provider=getattr(runner, "get_status_snapshot", lambda: {"ok": False}),
        positions_provider=getattr(runner.executor, "get_positions_kite", None),
        actives_provider=getattr(runner.executor, "get_active_orders", None),
        diag_provider=getattr(runner, "build_diag", None),
        compact_diag_provider=getattr(runner, "get_compact_diag_summary", None),
        risk_provider=getattr(runner, "risk_snapshot", None),
        limits_provider=lambda: asdict(runner.risk_engine.cfg),
        risk_reset_today=getattr(runner, "risk_reset_today", None),
        logs_provider=get_recent_logs,
        last_signal_provider=getattr(runner, "get_last_signal_debug", None),
        bars_provider=getattr(runner, "get_recent_bars", None),
        quotes_provider=getattr(runner.executor, "quote_diagnostics", None),
        probe_provider=getattr(runner, "get_probe_info", None),
        trace_provider=getattr(runner, "enable_trace", None),
        selftest_provider=getattr(runner.executor, "selftest", None),
        backtest_provider=getattr(runner, "run_backtest", None),
        filecheck_provider=run_file_diagnostics,
        atm_provider=getattr(runner, "get_current_atm", None),
        l1_provider=getattr(runner, "get_current_l1", None),
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
        runner.telegram_controller = _NoopTelegram()
        runner.telegram = runner.telegram_controller
        return

    runner.telegram_controller = tg  # back-compat
    runner.telegram = tg

    try:
        tg.start_polling()
        logging.getLogger("main").info("📡 Telegram polling started")
    except Exception as exc:
        logging.getLogger("main").warning(
            "Telegram polling failed to start: %s", exc, exc_info=True
        )


# -----------------------------
# Telegram command handler
# -----------------------------
def _make_cmd_handler(runner: StrategyRunner) -> Callable[[str, str], None]:
    """Create a handler that maps Telegram commands to runner actions."""

    commands: dict[str, Callable[[], None]] = {
        "/pause": runner.pause,
        "/resume": runner.resume,
    }

    def handle(cmd: str, _arg: str) -> None:
        action = commands.get(cmd)
        if action:
            action()

    return handle


# -----------------------------
# Lifecycle
# -----------------------------
_stop_flag = False


def _install_signal_handlers(_runner: StrategyRunner) -> None:
    def _handler(signum, _frame):
        global _stop_flag
        logging.getLogger("main").info("Signal %s received — shutting down…", signum)
        _stop_flag = True
        # Ensure the main loop exits promptly even if blocked in sleep or I/O
        raise SystemExit(0)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handler)
        except Exception as exc:
            logging.getLogger("main").warning(
                "Failed to set handler for %s: %s", sig, exc, exc_info=True
            )


def main() -> int:
    _setup_logging()
    try:
        validate_critical_settings()
    except Exception as e:
        logging.getLogger("main").error("\u274c Config validation failed: %s", e, exc_info=True)
        return 1
    log = logging.getLogger("main")
    rcfg = settings.risk
    loss_cap = (
        f"{rcfg.max_daily_loss_rupees:.0f}" if rcfg.max_daily_loss_rupees is not None else f"{rcfg.max_daily_drawdown_pct:.2%} equity"
    )
    log.info(
        "Risk guardrails → window %s-%s | loss_cap=%s | max_lots=%d | exposure_cap=%.0f | max_consec_losses=%d",
        rcfg.trading_window_start,
        rcfg.trading_window_end,
        loss_cap,
        rcfg.max_lots_per_symbol,
        rcfg.max_notional_rupees,
        rcfg.consecutive_loss_limit,
    )

    kite = None
    try:
        kite = _build_kite_session()
    except Exception as e:
        log.error("❌ Kite session init failed: %s", e, exc_info=True)

    cfg_path = os.environ.get("STRATEGY_CFG")
    runner = StrategyRunner(
        kite=kite or None,
        telegram_controller=_NoopTelegram(),
        strategy_cfg_path=cfg_path,
    )

    threading.Thread(
        target=health.run,
        kwargs={
            "callback": runner.health_check,
            "host": "0.0.0.0",
            "port": int(os.getenv("HEALTH_PORT", "8000")),
        },
        daemon=True,
    ).start()

    try:
        runner.set_live_mode(settings.enable_live_trading)
    except Exception as e:
        log.error("⚠️ Live mode setup failed: %s", e, exc_info=True)

    _install_signal_handlers(runner)

    # Wire Telegram
    _wire_real_telegram(runner)

    # Telegram commands
    cmd_listener: Optional[TelegramCommands] = None  # pragma: no cover
    if settings.telegram.bot_token and settings.telegram.chat_id:  # pragma: no cover
        cmd_listener = TelegramCommands(  # pragma: no cover
            settings.telegram.bot_token,  # pragma: no cover
            str(settings.telegram.chat_id),  # pragma: no cover
            on_cmd=_make_cmd_handler(runner),  # pragma: no cover
        )
        cmd_listener.start()  # pragma: no cover

    # Announce
    try:
        mode = "LIVE" if settings.enable_live_trading else "DRY"
        runner.telegram_controller.send_message(f"🚀 Bot starting ({mode})")
    except Exception as e:
        log.warning("Telegram startup message failed: %s", e, exc_info=True)

    try:
        if hasattr(runner, "start"):
            runner.start()
    except Exception as e:
        log.exception("Runner start failed: %s", e)
        return 1

    last_hb = time.time()
    try:
        while not _stop_flag:
            start_ts = time.perf_counter()
            try:
                runner.process_tick(tick=None)
            except Exception as e:
                log.exception("Main loop error: %s", e)
                time.sleep(1)
                continue
            latency_ms = (time.perf_counter() - start_ts) * 1000.0
            metrics.observe_latency(latency_ms)
            metrics.inc_ticks()
            qd = sum(len(q) for q in getattr(getattr(runner, "executor", None), "_queues", {}).values())
            metrics.set_queue_depth(qd)
            flow: Dict[str, Any] = getattr(runner, "get_last_flow_debug", lambda: {})()
            if isinstance(flow, dict):
                if flow.get("signal_ok"):
                    log.info("Signal generated: %s", flow.get("signal"))
                else:
                    log.debug("No signal generated: %s", flow.get("reason_block"))
            runner.health_check()
            now = time.time()
            if (now - last_hb) >= 15 * 60:
                snap = metrics.snapshot()
                cb = getattr(getattr(runner, "executor", None), "cb_orders", None)
                cb_state = getattr(cb, "state", "unknown")
                hb = (
                    f"HB last_tick_age={snap['last_tick_age']:.1f}s "
                    f"queue={snap['queue_depth']} cb={cb_state}"
                )
                log.info(hb)
                try:
                    if settings.telegram.enabled:
                        runner.telegram_controller.send_message(hb)
                except Exception as e:
                    log.debug("Heartbeat telegram failed: %s", e, exc_info=True)
                last_hb = now
            time.sleep(5)
    finally:
        try:
            runner.shutdown()
        except Exception as exc:
            logging.getLogger("main").warning(
                "Runner shutdown failed: %s", exc, exc_info=True
            )
        try:
            runner.telegram_controller.send_message("🛑 Bot stopped.")
        except Exception as e:
            log.warning("Failed to send shutdown message to Telegram: %s", e, exc_info=True)
        try:
            runner.telegram_controller.stop_polling()
        except Exception as e:
            log.warning("Failed to stop Telegram polling: %s", e, exc_info=True)
        if cmd_listener:  # pragma: no cover
            try:  # pragma: no cover
                cmd_listener.stop()  # pragma: no cover
            except Exception as e:  # pragma: no cover
                log.warning("Failed to stop Telegram commands: %s", e, exc_info=True)  # pragma: no cover

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        logging.getLogger("main").exception("Fatal error in main: %s", e)
        sys.exit(1)
