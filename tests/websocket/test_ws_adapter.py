import sys
from types import SimpleNamespace

import pytest

import nifty_scalper_bot.data.rest.zerodha_ws_adapter as zerodha_ws_adapter
from nifty_scalper_bot.data.rest.zerodha_ws_adapter import build_kite_ticker
from nifty_scalper_bot.utils.errors import ConfigurationError


def test_build_kite_ticker_sanitizes_token_and_wraps_connect(monkeypatch):
    captured: dict[str, object] = {}

    class _DummyTicker:
        def __init__(self, api_key: str, token: str) -> None:
            captured["init"] = (api_key, token)

        def connect(self, *, threaded: bool = True) -> None:
            captured["connect_threaded"] = threaded

    monkeypatch.setattr(
        "nifty_scalper_bot.data.rest.zerodha_ws_adapter.KiteTicker",
        _DummyTicker,
    )

    installs: list[bool] = []

    def _record_install() -> None:
        installs.append(True)

    monkeypatch.setattr(
        zerodha_ws_adapter,
        "_install_ws_header_injector",
        _record_install,
    )

    refreshed: list[bool] = []

    def _refresh() -> None:
        refreshed.append(True)

    ticker = build_kite_ticker("key", "user:token", session_refresher=_refresh)
    assert captured["init"] == ("key", "token")

    ticker._nsb_connect(threaded=False)  # type: ignore[attr-defined]
    assert captured["connect_threaded"] is False
    assert installs

    ticker.refresh_session()  # type: ignore[attr-defined]
    assert refreshed


def test_build_kite_ticker_handles_signature_changes(monkeypatch):
    captured: dict[str, object] = {}

    class _StrictTicker:
        def __init__(self, api_key: str, token: str) -> None:
            captured["init"] = (api_key, token)

        def connect(self, *args, **kwargs):  # noqa: ANN002, ANN003
            if kwargs:
                raise TypeError("unexpected keyword")
            captured["connect_called"] = True

    monkeypatch.setattr(
        "nifty_scalper_bot.data.rest.zerodha_ws_adapter.KiteTicker",
        _StrictTicker,
    )

    ticker = build_kite_ticker("key", "token")
    ticker._nsb_connect()  # type: ignore[attr-defined]
    assert captured["connect_called"] is True


def test_build_kite_ticker_requires_dependency(monkeypatch):
    monkeypatch.setattr(
        "nifty_scalper_bot.data.rest.zerodha_ws_adapter.KiteTicker",
        None,
    )
    with pytest.raises(ConfigurationError):
        build_kite_ticker("key", "token")


def test_install_ws_header_injector_merges_browser_headers(monkeypatch):
    captured: dict[str, list[str]] = {}

    class _DummyWSApp:
        def __init__(self, url, header=None, **kwargs):  # noqa: ANN002, ANN003
            captured.setdefault(url, header or [])

    stub = SimpleNamespace(WebSocketApp=_DummyWSApp)
    monkeypatch.setitem(sys.modules, "websocket", stub)

    zerodha_ws_adapter._install_ws_header_injector()

    stub.WebSocketApp(
        "wss://kite", header=["X-Test: 1", ("Origin", "https://ignore.me")]
    )

    header_lines = captured["wss://kite"]
    assert isinstance(header_lines, list)

    header_map = {}
    for line in header_lines:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        header_map[key.strip()] = value.strip()

    assert header_map["Origin"] == "https://kite.zerodha.com"
    assert header_map["User-Agent"].startswith("Mozilla/5.0")
    assert header_map["Accept-Encoding"] == "gzip, deflate, br"
    assert header_map["Accept-Language"].startswith("en-US")
    assert header_map["Cache-Control"] == "no-cache"
    assert header_map["Pragma"] == "no-cache"
    assert header_map["X-Test"] == "1"
