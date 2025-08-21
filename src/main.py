from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import deque
from typing import Any, Deque, List, Optional

from flask import Flask, jsonify

from src.config import settings
from src.strategies.runner import StrategyRunner
from src.notifications.telegram_controller import TelegramController

# ---------- logging with in-memory ring for /logs ----------

class RingHandler(logging.Handler):
    def __init__(self, capacity: int = 1000) -> None:
        super().__init__()
        self.buf: Deque[str] = deque(maxlen=capacity)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            self.buf.append(msg)
        except Exception:
            pass

    def tail(self, n: int) -> List[str]:
        n = max(1, min(n, len(self.buf)))
        return list(self.buf)[-n:]


ring = RingHandler(capacity=3000)
_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
ring.setFormatter(_formatter)
root = logging.getLogger()
root.setLevel(getattr(logging, (settings.log_level or "INFO").upper(), logging.INFO))
root.addHandler(ring)

log = logging.getLogger(__name__)

# ---------- app / health ----------

app = Flask("src.server.health")

@app.get("/")
def root_health():
    return jsonify({"ok": True, "time": time.time()})

# ---------- runner & telegram wiring ----------

stop_event = threading.Event()
runner = StrategyRunner(event_sink=lambda evt: log.info("event: %s", json.dumps(evt)))

def _hb_line(prefix: str = "heartbeat") -> str:
    src = type(runner.data_source).__name__ if runner.data_source else "None"
    st = runner.to_status_dict()
    return f"⏱ {prefix} | live={int(st['live_trading'])} paused={st['paused']} active={st['active_orders']} src={src}"

def status_provider() -> dict:
    return runner.to_status_dict()

def summary_provider(n: int) -> list[dict]:
    return runner.last_signals(n)

def diag_provider() -> dict:
    try:
        return runner.diagnose()
    except Exception as e:
        log.exception("diag failed: %s", e)
        return {"ok": False, "error": str(e)}

def logs_provider(n: int) -> list[str]:
    return ring.tail(n)

def positions_provider() -> dict:
    if not getattr(runner, "executor", None):
        return {}
    try:
        return runner.executor.get_positions_kite()  # type: ignore[union-attr]
    except Exception:
        return {}

def actives_provider():
    if not getattr(runner, "executor", None):
        return []
    try:
        return runner.executor.get_active_orders()  # type: ignore[union-attr]
    except Exception:
        return []

def pause_runner(minutes: Optional[int] = None):
    runner.pause(minutes)

def resume_runner():
    runner.resume()

def cancel_all():
    if getattr(runner, "executor", None):
        runner.executor.cancel_all_orders()  # type: ignore[union-attr]

def flatten_all():
    if getattr(runner, "executor", None):
        # cancel exits then market out of any remaining qty
        try:
            for rec in runner.executor.get_active_orders():  # type: ignore[union-attr]
                runner.executor.exit_order(rec.order_id, exit_reason="flatten")  # type: ignore[union-attr]
        except Exception:
            pass

def tick_once():
    return runner.run_once(stop_event)

def set_risk_pct(pct: float):
    settings.risk.risk_per_trade = float(pct) / 100.0

def toggle_trailing(v: bool):
    settings.executor.enable_trailing = bool(v)

def set_trailing_mult(v: float):
    settings.executor.trailing_atr_multiplier = float(v)

def toggle_partial(v: bool):
    settings.executor.partial_tp_enable = bool(v)

def set_tp1_ratio(pct: float):
    settings.executor.tp1_qty_ratio = float(pct) / 100.0

def set_breakeven_ticks(ticks: int):
    settings.executor.breakeven_ticks = int(ticks)

def set_live_mode(v: bool):
    settings.enable_live_trading = bool(v)
    runner.set_live(bool(v))
    log.info("Live mode set to %s.", "True" if v else "False")

def set_quality_mode(mode: str):
    runner.set_quality_mode(mode)

def set_regime_mode(mode: str):
    runner.set_regime_mode(mode)

# ---------- telegram ----------

tg: Optional[TelegramController] = None
if settings.telegram_ready:
    try:
        tg = TelegramController(
            status_provider=status_provider,
            summary_provider=summary_provider,
            diag_provider=diag_provider,
            logs_provider=logs_provider,
            positions_provider=positions_provider,
            actives_provider=actives_provider,
            runner_pause=pause_runner,
            runner_resume=resume_runner,
            cancel_all=cancel_all,
            flatten_all=flatten_all,
            tick_once=tick_once,
            set_risk_pct=set_risk_pct,
            toggle_trailing=toggle_trailing,
            set_trailing_mult=set_trailing_mult,
            toggle_partial=toggle_partial,
            set_tp1_ratio=set_tp1_ratio,
            set_breakeven_ticks=set_breakeven_ticks,
            set_live_mode=set_live_mode,
            set_quality_mode=set_quality_mode,
            set_regime_mode=set_regime_mode,
        )
        tg.start_polling()
        log.info("Telegram polling started.")
        # greet
        tg._send("🤖 Nifty Scalper online.\n" + json.dumps(settings.debug_summary(), indent=2), parse_mode=None)
        tg.send_menu()
    except Exception as e:
        log.error("Telegram not started: %s", e)
else:
    log.info("Telegram disabled or credentials missing.")

# ---------- worker loop with gentle heartbeat ----------

def worker_loop():
    last_hb = 0.0
    while not stop_event.is_set():
        try:
            # run trading tick
            tick_once()
            # heartbeat every 5 minutes
            now = time.time()
            if now - last_hb >= 300:
                log.info(_hb_line())
                last_hb = now
        except Exception as e:
            log.exception("worker loop error: %s", e)
        time.sleep(5)  # 5s cadence — signals are minute-based anyway

th = threading.Thread(target=worker_loop, name="runner-loop", daemon=True)
th.start()

if __name__ == "__main__":
    # Gunicorn/uvicorn not strictly required; built-in is OK for Railway health.
    host = getattr(settings.server, "host", "0.0.0.0")
    port = int(getattr(settings.server, "port", 8000))
    log.info("Starting Nifty Scalper Bot | live_trading=%s", settings.enable_live_trading)
    try:
        from waitress import serve  # if present, use a less noisy server
        serve(app, host=host, port=port)
    except Exception:
        app.run(host=host, port=port, debug=False)