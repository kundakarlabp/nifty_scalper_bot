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
