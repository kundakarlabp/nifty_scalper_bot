import httpx
import pytest

from nifty_scalper_bot.data.rest.zerodha_client import ZerodhaKiteClient
from nifty_scalper_bot.utils.errors import BrokerError


def test_retry_backoff_honors_caps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BROKER_RETRIES", "4")
    monkeypatch.setenv("BROKER_BASE_DELAY_SEC", "0.5")
    monkeypatch.setenv("BROKER_MAX_DELAY_SEC", "2")
    client = ZerodhaKiteClient(api_key="key", access_token="token")

    request = httpx.Request("GET", f"{client._base_url}/quote")
    monkeypatch.setattr(
        "nifty_scalper_bot.utils.retry.random.uniform",
        lambda _low, _high: 1.0,
    )

    def failing_request(*_args: object, **_kwargs: object) -> httpx.Response:
        raise httpx.ReadTimeout("timeout", request=request)

    monkeypatch.setattr(client._client, "request", failing_request, raising=False)

    sleeps: list[float] = []
    captured: list[float] = []

    def fake_sleep(self: ZerodhaKiteClient, delay: float) -> None:
        sleeps.append(delay)

    def fake_register(
        self: ZerodhaKiteClient,
        *,
        status: int | None,
        error: Exception | None,
        delay: float,
        endpoint: str,
    ) -> None:
        captured.append(delay)

    monkeypatch.setattr(ZerodhaKiteClient, "_sleep", fake_sleep, raising=False)
    monkeypatch.setattr(
        ZerodhaKiteClient, "_register_transient_failure", fake_register, raising=False
    )

    with pytest.raises(BrokerError):
        client._make_request("GET", "/quote", operation_label="quote.test")

    assert sleeps == pytest.approx([0.5, 1.0, 2.0])
    assert captured == pytest.approx([0.5, 1.0, 2.0, 0.0], abs=1e-6)

    client._client.close()


def test_transient_errors_open_breaker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TICK_BACKOFF_BASE_SECONDS", "0.1")
    monkeypatch.setenv("TICK_BACKOFF_CAP_SECONDS", "0.2")
    monkeypatch.setenv("TICK_MAX_CONSECUTIVE_ERRORS", "2")
    client = ZerodhaKiteClient(api_key="key", access_token="token")
    times: dict[str, float] = {"mono": 0.0, "wall": 0.0}
    sleep_calls: list[float] = []

    def fake_monotonic() -> float:
        return times["mono"]

    def fake_sleep(duration: float) -> None:
        sleep_calls.append(duration)
        times["mono"] += duration
        times["wall"] += duration

    monkeypatch.setattr(
        "nifty_scalper_bot.data.rest.zerodha_client.time.monotonic",
        fake_monotonic,
    )
    monkeypatch.setattr(
        "nifty_scalper_bot.data.rest.zerodha_client.time.sleep",
        fake_sleep,
    )
    monkeypatch.setattr(
        "nifty_scalper_bot.utils.retry.random.uniform",
        lambda _low, _high: 1.0,
    )
    client._log_time_fn = lambda: times["wall"]  # type: ignore[assignment]
    client._max_retries = 1

    def fake_request(method: str, url: str, **_kwargs: object) -> httpx.Response:
        request = httpx.Request(method, f"{client._base_url}{url}")
        return httpx.Response(504, request=request)

    monkeypatch.setattr(client._client, "request", fake_request, raising=False)

    with pytest.raises(BrokerError):
        client._make_request("GET", "/quote")
    with pytest.raises(BrokerError):
        client._make_request("GET", "/quote")
    with pytest.raises(BrokerError):
        client._make_request("GET", "/quote")

    assert any(
        pytest.approx(client._breaker_cooldown_sec) == call for call in sleep_calls
    )
    client._client.close()


def test_get_ltp_bulk_depth_maps_symbol_payload_to_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv('POLL_REQUIRE_DEPTH', 'true')
    client = ZerodhaKiteClient(api_key='key', access_token='token')

    class _Resolver:
        @staticmethod
        def format_token_as_symbol(token: int) -> str:
            return {256265: 'NSE:NIFTY 50'}.get(token, '')

    client.attach_resolver(_Resolver())

    monkeypatch.setattr(
        client,
        'get_quote_bulk',
        lambda _tokens: {
            'NSE:NIFTY 50': {'instrument_token': 256265, 'last_price': 25382.4}
        },
    )

    out = client.get_ltp_bulk([256265])

    assert out == {256265: 25382.4}

    client._client.close()
