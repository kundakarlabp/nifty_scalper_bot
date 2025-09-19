from __future__ import annotations

import pytest

from src.data.source import LiveKiteSource


class DummyKite:
    MODE_FULL = "full"

    def __init__(self) -> None:
        self.subscribed: list[int] | None = None
        self.mode_payload: tuple[str, list[int]] | None = None
        self._ltp_calls: list[list[str]] = []

    def subscribe(self, payload: list[int]) -> None:
        self.subscribed = list(payload)

    def set_mode(self, mode: str, payload: list[int]) -> None:
        self.mode_payload = (mode, list(payload))

    def quote(self, tokens: list[int]) -> dict[int, dict[str, float]]:
        token = tokens[0]
        return {token: {}}

    def ltp(self, tokens: list[str]) -> dict[str, dict[str, float]]:
        self._ltp_calls.append(list(tokens))
        key = str(tokens[0])
        return {key: {"last_price": 105.5}}


@pytest.fixture
def live_source(monkeypatch: pytest.MonkeyPatch) -> LiveKiteSource:
    kite = DummyKite()
    source = LiveKiteSource(kite)

    def fake_quote_safe(**_: object) -> tuple[dict[str, float], str]:
        return ({"bid": 0.0, "ask": 0.0, "ltp": 0.0, "mid": 0.0}, "depth")

    monkeypatch.setattr(
        "src.data.source.get_option_quote_safe", fake_quote_safe
    )
    return source


def test_prime_option_quote_falls_back_to_rest_ltp(live_source: LiveKiteSource) -> None:
    price, mode, ts = live_source.prime_option_quote(123456)
    assert price == pytest.approx(105.5)
    assert mode == "rest_ltp"
    assert isinstance(ts, int)
    cached = live_source._option_quote_cache[123456]
    assert cached["ltp"] == pytest.approx(105.5)
    assert cached["mode"] == "rest_ltp"


def test_ensure_token_subscribed_invokes_full_mode(live_source: LiveKiteSource) -> None:
    assert live_source.ensure_token_subscribed(654321) is True
    assert live_source.kite.subscribed == [654321]
    assert live_source.kite.mode_payload == ("full", [654321])


def test_quote_snapshot_enriches_cached_quote(live_source: LiveKiteSource) -> None:
    token = 789012
    live_source._option_quote_cache[token] = {
        "bid": 98.5,
        "ask": 101.5,
        "mid": 100.0,
        "ltp": 100.2,
        "mode": "depth",
        "source": "kite",
        "bid_qty": 150,
        "ask_qty": 175,
        "ts_ms": 1_694_000_000_000,
    }

    live_source.atm_tokens = (token, token + 1)

    snap = live_source.quote_snapshot(token)
    assert snap is not None
    assert snap["bid"] == pytest.approx(98.5)
    assert snap["ask"] == pytest.approx(101.5)
    assert snap["spread"] == pytest.approx(3.0)
    assert snap["spread_pct"] == pytest.approx(3.0 / 100.0 * 100)
    assert snap["bid_qty"] == 150
    assert snap["ask_qty"] == 175
    assert snap["ts_ms"] == 1_694_000_000_000
    assert live_source.current_tokens() == (token, token + 1)
