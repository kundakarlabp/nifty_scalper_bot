from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]


class _FakeResponse:
    def __init__(self, payload: dict):
        self.payload = payload
    def __enter__(self):
        return self
    def __exit__(self, *_args):
        return False
    def read(self):
        return json.dumps(self.payload).encode()


class _FakeControl:
    def toggle(self, *_args, value=False, **_kwargs):
        return value
    def selectbox(self, _label, options, index=0, **_kwargs):
        return options[index]
    def text_input(self, *_args, **_kwargs):
        return ""
    def number_input(self, *_args, value=0, **_kwargs):
        return value
    def checkbox(self, *_args, value=False, **_kwargs):
        return value


class _FakeStreamlit:
    def __init__(self):
        self.markdown_values: list[str] = []
    def set_page_config(self, **_kwargs):
        return None
    def markdown(self, value, **_kwargs):
        self.markdown_values.append(str(value))
    def cache_data(self, **_kwargs):
        def decorator(fn):
            return fn
        return decorator
    def columns(self, spec, **_kwargs):
        count = len(spec) if isinstance(spec, list) else int(spec)
        return [_FakeControl() for _ in range(count)]
    def download_button(self, *_args, **_kwargs):
        return None


def _load_helpers(monkeypatch):
    fake_st = _FakeStreamlit()
    monkeypatch.setitem(sys.modules, "streamlit", fake_st)
    monkeypatch.setattr("urllib.request.urlopen", lambda *_a, **_k: _FakeResponse({}))
    monkeypatch.setattr(
        "subprocess.run",
        lambda *_a, **_k: SimpleNamespace(returncode=1, stdout=""),
    )
    return runpy.run_path(str(ROOT / "dashboard" / "superlite_console.py"))


def _base_status(**updates):
    status = {
        "engine_http_responsive": True,
        "engine_loaded": True,
        "operational_ready": True,
        "mode": "SHADOW",
        "live_orders_armed": False,
        "broker_authenticated": "unknown",
        "reconciled": None,
        "blockers": [],
    }
    status.update(updates)
    return status


def test_shadow_operational_is_not_trading_ready(monkeypatch):
    helpers = _load_helpers(monkeypatch)
    assert helpers["engine_display_status"](_base_status(mode="SHADOW")) == (
        "ENGINE UP, OPERATIONAL"
    )


def test_market_closed_operational_is_not_trading_ready(monkeypatch):
    helpers = _load_helpers(monkeypatch)
    status = _base_status(mode="LIVE", blockers=["market_closed"])
    assert helpers["engine_display_status"](status) == "ENGINE UP, OPERATIONAL"


def test_live_blocked_status_remains_blocked(monkeypatch):
    helpers = _load_helpers(monkeypatch)
    status = _base_status(
        operational_ready=False, mode="LIVE", blockers=["risk_gate_blocked"]
    )
    assert helpers["engine_display_status"](status) == "ENGINE UP, TRADING BLOCKED"


def test_live_armed_status_is_trading_ready(monkeypatch):
    helpers = _load_helpers(monkeypatch)
    status = _base_status(
        mode="LIVE",
        live_orders_armed=True,
        broker_authenticated="authenticated",
        reconciled=True,
    )
    assert helpers["engine_display_status"](status) == "ENGINE UP, TRADING READY"
