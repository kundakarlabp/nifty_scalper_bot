from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from src.config import settings
from src.server import health as health_server
from src.strategies.runner import StrategyRunner
from src.notifications.telegram_controller import TelegramController

# Optional broker SDK
try:
    from kiteconnect import KiteConnect  # type: ignore
    from kiteconnect.exceptions import NetworkException, TokenException, InputException  # type: ignore
except Exception:  # pragma: no cover
    KiteConnect = None  # type: ignore
    NetworkException = TokenException = InputException = Exception  # fallbacks

# Data source
try:
    from src.data.source import LiveKiteSource, DataSource  # type: ignore
except Exception:  # pragma: no cover
    LiveKiteSource = None  # type: ignore
    DataSource = object  # type: ignore

# Strike selector & market gate
try:
    from src.utils.strike_selector import get_instrument_tokens, is_market_open  # type: ignore
except Exception:  # pragma: no cover
    def is_market_open() -> bool:
        return True
    def get_instrument_tokens(*args, **kwargs):
        return None

log = logging.getLogger(__name__)


# ------------ small log ring buffer for /logs ------------
class LogBufferHandler(logging.Handler):
    def __init__(self, capacity: int = 2000) -> None:
        super().__init__()
        self.capacity = int(capacity)
        self._lock = threading.Lock()
        self._buf: List[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
        except Exception:
            return
        with self._lock:
            self._buf.append(msg)
            if len(self._buf) > self.capacity:
                # drop oldest 20% to avoid frequent moves
                drop = max(1, self.capacity // 5)
                self._buf = self._buf[drop:]

    def tail(self, n: int = 100) -> List[str]:
        with self._lock:
            return self._buf[-n:]


# ------------ Logging helper ------------
def _setup_logging(logbuf: LogBufferHandler) -> None:
    root = logging.getLogger()
    root.handlers.clear()
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    sh.setLevel(level)
    logbuf.setFormatter(fmt)
    logbuf.setLevel(level)
    root.addHandler(sh)
    root.addHandler(logbuf)
    root.setLevel(level)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


# ------------ time helper ------------
def _now_ist() -> datetime:
    return datetime.now(timezone(timedelta(hours=5, minutes=30)))


def _now_ist_naive() -> datetime:
    ist = _now_ist()
    return ist.replace(tzinfo=None)


class Application:
    """Main application for the Nifty Scalper Bot with Telegram control."""

    def __init__(self) -> None:
        self._start_ts = time.time()

        # stop flag
        self._stop = threading.Event()
        signal.signal(signal.SIGINT, self._sig)
        signal.signal(signal.SIGTERM, self._sig)

        # log ring buffer
        self._logbuf = LogBufferHandler(capacity=4000)
        _setup_logging(self._logbuf)

        # runner with event_sink -> Telegram notify
        self.runner = StrategyRunner(event_sink=self._on_runner_event)

        # Telegram
        self.tg: Optional[TelegramController] = None

        # heartbeat throttling (log at most every 5 min)
        self._last_hb = 0.0

    # ---- signal handling
    def _sig(self, signum: int, frame: Any) -> None:
        log.info("Signal %s received; stopping…", signum)
        self._stop.set()

    # ---- event sink from runner
    def _on_runner_event(self, evt: Dict[str, Any]) -> None:
        if not self.tg:
            return
        t = evt.get("type")
        if t == "ENTRY_PLACED":
            self.tg.notify_entry(
                symbol=str(evt.get("symbol")),
                side=str(evt.get("side")),
                qty=int(evt.get("qty", 0)),
                price=float(evt.get("price", 0.0)),
                record_id=str(evt.get("record_id", "")),
            )
        elif t == "FILLS":
            fills = evt.get("fills") or []
            self.tg.notify_fills(fills)

    # ---- providers for Telegram
    def _status_payload(self) -> Dict[str, Any]:
        ds = self.runner.data_source
        active = self.runner.executor.get_active_orders() if self.runner.executor else []
        return {
            "time_ist": _now_ist_naive().isoformat(sep=" ", timespec="seconds"),
            "broker": "Kite" if self.runner._kite else "none",
            "data_source": type(ds).__name__ if ds else None,
            "live_trading": bool(getattr(settings, "enable_live_trading", False)),
            "paused": getattr(self.runner, "_paused", False),
            "active_orders": len(active),
        }

    def _positions_provider(self) -> Dict[str, Any]:
        if self.runner.executor:
            return self.runner.executor.get_positions_kite()
        return {}

    def _actives_provider(self) -> List[Any]:
        if self.runner.executor:
            return self.runner.executor.get_active_orders()
        return []

    def _logs_provider(self, n: int) -> List[str]:
        return self._logbuf.tail(n)

    def _force_tick(self) -> Dict[str, Any]:
        # run once and capture whether something happened
        res = self.runner.run_once(self._stop) or {"ok": True, "note": "no signal / or market closed"}
        return res

    def _diag_provider(self) -> Dict[str, Any]:
        """
        On-demand end-to-end pipeline checks.
        """
        checks: List[Dict[str, Any]] = []
        ok_all = True

        # 1) market hours
        open_ok = bool(is_market_open())
        checks.append({"name": "market_open", "ok": open_ok})
        if not open_ok:
            ok_all = False

        # 2) spot LTP
        spot_ltp = None
        try:
            inst = getattr(settings, "instruments", object())
            sym = str(getattr(inst, "spot_symbol", "NSE:NIFTY 50"))
            if self.runner.spot_source and hasattr(self.runner.spot_source, "get_last_price"):
                spot_ltp = self.runner.spot_source.get_last_price(sym)  # type: ignore[attr-defined]
        except Exception:
            spot_ltp = None
        checks.append({"name": "spot_ltp", "ok": spot_ltp is not None, "value": spot_ltp})
        if spot_ltp is None:
            ok_all = False

        # 3) spot OHLC
        spot_rows = 0
        spot_ok = False
        spot_df = None
        try:
            inst = getattr(settings, "instruments", object())
            tf = str(getattr(getattr(settings, "data", object()), "timeframe", "minute"))
            lb = int(getattr(getattr(settings, "data", object()), "lookback_minutes", 60))
            tok = int(getattr(inst, "instrument_token", 256265))
            if self.runner.spot_source:
                end = _now_ist_naive()
                start = end - timedelta(minutes=lb)
                spot_df = self.runner.spot_source.fetch_ohlc(tok, start, end, tf)
                spot_rows = 0 if spot_df is None else len(spot_df)
                spot_ok = bool(spot_rows > 0)
        except Exception:
            spot_ok = False
        checks.append({"name": "spot_ohlc", "ok": spot_ok, "rows": spot_rows})
        if not spot_ok:
            ok_all = False

        # 4) strike selection
        token_info = None
        ss_ok = False
        try:
            token_info = get_instrument_tokens(kite_instance=self.runner._kite)
            ss_ok = bool(token_info and token_info.get("tokens", {}).get("ce"))
        except Exception:
            token_info = None
        checks.append({"name": "strike_selection", "ok": ss_ok, "result": token_info})
        if not ss_ok:
            ok_all = False

        # 5) option OHLC
        opt_rows = 0
        opt_ok = False
        opt_df = None
        if token_info and token_info.get("tokens", {}).get("ce"):
            try:
                tf = str(getattr(getattr(settings, "data", object()), "timeframe", "minute"))
                lb = int(getattr(getattr(settings, "data", object()), "lookback_minutes", 60))
                end = _now_ist_naive()
                start = end - timedelta(minutes=lb)
                token = int(token_info["tokens"]["ce"])
                if self.runner.data_source:
                    opt_df = self.runner.data_source.fetch_ohlc(token, start, end, tf)
                    opt_rows = 0 if opt_df is None else len(opt_df)
                    opt_ok = bool(opt_rows > 0)
            except Exception:
                opt_ok = False
        checks.append({"name": "option_ohlc", "ok": opt_ok, "rows": opt_rows})
        if not opt_ok:
            ok_all = False

        # 6) indicators
        ind_ok = bool(spot_ok)
        checks.append({"name": "indicators", "ok": ind_ok, "error": (None if ind_ok else "spot OHLC empty")})
        if not ind_ok:
            ok_all = False

        # 7) signal (we do not actually compute here to avoid side effects; rely on /tick)
        sig_ok = bool(opt_ok and spot_ok and open_ok)
        checks.append({"name": "signal", "ok": sig_ok, "error": (None if sig_ok else "prereq not met")})
        if not sig_ok:
            ok_all = False

        # 8) sizing will depend on signal: same prereq
        siz_ok = sig_ok
        checks.append({"name": "sizing", "ok": siz_ok, "error": (None if siz_ok else "no signal")})
        if not siz_ok:
            ok_all = False

        # 9) execution ready
        exe_ok = bool(getattr(settings, "enable_live_trading", False) and self.runner._kite and self.runner.executor)
        checks.append({"name": "execution_ready", "ok": exe_ok, "live": bool(getattr(settings, "enable_live_trading", False)), "broker": bool(self.runner._kite), "executor": bool(self.runner.executor)})

        # 10) existing open orders
        oc = len(self.runner.executor.get_active_orders()) if self.runner.executor else 0
        checks.append({"name": "open_orders", "ok": True, "count": oc})

        return {"ok": ok_all, "checks": checks, "tokens": token_info}

    # ---- control hooks for Telegram
    def _pause(self) -> None:
        self.runner.pause()
        log.info("Runner paused by Telegram.")

    def _resume(self) -> None:
        self.runner.resume()
        log.info("Runner resumed by Telegram.")

    def _cancel_all(self) -> None:
        if self.runner.executor:
            self.runner.executor.cancel_all_orders()
        log.info("Cancel-all invoked by Telegram.")

    # ---- Telegram setup
    def _setup_telegram(self) -> None:
        try:
            tg_enabled = bool(getattr(getattr(settings, "telegram", object()), "enabled", True))
            bot_token = getattr(getattr(settings, "telegram", object()), "bot_token", None)
            chat_id = getattr(getattr(settings, "telegram", object()), "chat_id", None)
            if tg_enabled and bot_token and chat_id:
                self.tg = TelegramController(
                    status_provider=self._status_payload,
                    positions_provider=self._positions_provider,
                    actives_provider=self._actives_provider,
                    logs_provider=self._logs_provider,
                    diag_provider=self._diag_provider,
                    tick_fn=self._force_tick,
                    runner_pause=self._pause,
                    runner_resume=self._resume,
                    cancel_all=self._cancel_all,
                )
                self.tg.start_polling()
                self.tg.send_startup_alert()
            else:
                log.info("Telegram disabled or credentials missing.")
        except Exception as e:
            log.error("Telegram controller init failed: %s", e)
            self.tg = None

    # ---- health server
    def _start_health(self) -> None:
        threading.Thread(
            target=health_server.run,
            kwargs={"callback": self._status_payload, "host": settings.server.host, "port": settings.server.port},
            daemon=True,
        ).start()

    # ---- main loop
    def run(self) -> None:
        log.info("Starting Nifty Scalper Bot | live_trading=%s", bool(getattr(settings, "enable_live_trading", False)))
        self._start_health()
        self._setup_telegram()

        # Main cadence
        cadence = 0.5
        while not self._stop.is_set():
            try:
                res = self.runner.run_once(self._stop)
                # Throttled heartbeat log (every 5 minutes)
                now = time.time()
                if now - self._last_hb > 300:
                    src_name = type(self.runner.data_source).__name__ if self.runner.data_source else "None"
                    log.info("⏱ heartbeat | live=%d paused=%s active=%d src=%s",
                             1 if getattr(settings, "enable_live_trading", False) else 0,
                             getattr(self.runner, "_paused", False),
                             len(self.runner.executor.get_active_orders()) if self.runner.executor else 0,
                             src_name)
                    self._last_hb = now
            except (NetworkException, TokenException, InputException) as e:
                log.error("Broker transient error: %s", e)
            except Exception as e:
                log.exception("Main loop error: %s", e)

            if self._stop.wait(timeout=cadence):
                break

        if self.tg:
            try:
                self.tg.stop_polling()
            except Exception:
                pass
        log.info("Bot stopped.")


# ------------ CLI ------------
def _parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="nifty_scalper_bot")
    sub = p.add_subparsers(dest="cmd", required=False)
    sub.add_parser("start", help="Start trading loop (default)")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    cmd = args.cmd or "start"
    if cmd == "start":
        app = Application()
        app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())