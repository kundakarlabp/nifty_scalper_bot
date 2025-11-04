from __future__ import annotations

from types import SimpleNamespace

from nifty_scalper_bot.infra.ws_diag import run_diag


class _DummyWS:
    _breaker_open_until = 0.0
    _reconnect_timer = None
    _backoff_s = 0.0

    def connection_state(self) -> SimpleNamespace:
        return SimpleNamespace(name="CONNECTED")

    def last_error(self):  # noqa: ANN204
        return None


def test_run_diag_without_host(monkeypatch):
    monkeypatch.setattr("nifty_scalper_bot.infra.ws_diag.time.time", lambda: 200.0)
    monkeypatch.setattr("nifty_scalper_bot.infra.ws_diag.time.monotonic", lambda: 100.0)

    diag = run_diag(None, None, "abcdef987654", 150.0)

    assert diag.token_hint.startswith("abcd")
    assert diag.token_age_s == 50.0
    assert diag.ws_dns == []
    assert diag.tcp_ok is False
    assert diag.tcp_peer is None
    assert diag.breaker_open is False
    assert diag.next_retry_s is None
    assert (
        diag.last_error == "Websocket disabled; set WEBSOCKET__DISABLED=false to enable"
    )
    assert diag.state == "DISABLED"


class _BreakerWS:
    def __init__(self) -> None:
        self._breaker_open_until = 200.0
        self._reconnect_timer = object()
        self._backoff_s = 12.5

    def connection_state(self) -> SimpleNamespace:
        return SimpleNamespace(name="CONNECTING")

    def last_error(self):  # noqa: ANN204
        return RuntimeError("handshake")


def test_run_diag_with_network_details(monkeypatch):
    fake_infos = [
        (None, None, None, None, ("1.1.1.1", 443)),
        (None, None, None, None, ("2.2.2.2", 443)),
    ]

    def _fake_getaddrinfo(host: str, port: int, *, type: int = 0):  # noqa: ANN001, D401
        assert host == "example.com"
        assert port == 443
        assert type == 1
        return list(fake_infos)

    class _FakeSocket:
        def __enter__(self) -> "_FakeSocket":
            return self

        def __exit__(self, exc_type, exc, tb):  # noqa: ANN001, ANN201
            return False

        def getpeername(self):  # noqa: ANN204
            return ("9.9.9.9", 443)

    def _fake_create_connection(addr, timeout):  # noqa: ANN001, ANN202
        assert addr == ("example.com", 443)
        assert timeout == 3
        return _FakeSocket()

    monkeypatch.setattr(
        "nifty_scalper_bot.infra.ws_diag.socket.getaddrinfo", _fake_getaddrinfo
    )
    monkeypatch.setattr(
        "nifty_scalper_bot.infra.ws_diag.socket.create_connection",
        _fake_create_connection,
    )
    monkeypatch.setattr("nifty_scalper_bot.infra.ws_diag.time.time", lambda: 500.0)
    monkeypatch.setattr("nifty_scalper_bot.infra.ws_diag.time.monotonic", lambda: 100.0)

    diag = run_diag(_BreakerWS(), "example.com", "abcd1234wxyz", 490.0)

    assert diag.ws_dns == ["1.1.1.1", "2.2.2.2"]
    assert diag.tcp_ok is True
    assert diag.tcp_peer == "9.9.9.9"
    assert diag.breaker_open is True
    assert diag.next_retry_s == 12.5
    assert diag.last_error is not None and "RuntimeError" in diag.last_error
    assert diag.state == "CONNECTING"
    assert diag.token_age_s == 10.0
    assert "colon=no" in diag.token_hint
