import pytest

from nifty_scalper_bot.data.rest.zerodha_client import ZerodhaKiteClient


def test_tokens_to_symbols_accepts_canonical_without_resolver() -> None:
    client = ZerodhaKiteClient(api_key="k", access_token="t")
    symbols, symbol_map = client._tokens_to_symbols(["NSE:NIFTY"])
    assert symbols == ["NSE:NIFTY"]
    assert symbol_map == {}
    client.close()


def test_tokens_to_symbols_canonical_reverse_enrichment_is_best_effort() -> None:
    class BrokenResolver:
        @staticmethod
        def get_token_for_symbol(_symbol: str) -> int:
            raise RuntimeError("boom")

    client = ZerodhaKiteClient(api_key="k", access_token="t")
    client.attach_resolver(BrokenResolver())

    symbols, symbol_map = client._tokens_to_symbols(["NSE:NIFTY"])

    assert symbols == ["NSE:NIFTY"]
    assert symbol_map == {}
    client.close()


def test_get_quote_bulk_short_circuits_all_canonical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = ZerodhaKiteClient(api_key="k", access_token="t")

    called = {"acquire": 0}

    monkeypatch.setattr(
        client,
        "_acquire_bucket",
        lambda _bucket: called.__setitem__("acquire", called["acquire"] + 1),
    )
    monkeypatch.setattr(
        client,
        "_make_request",
        lambda *_args, **_kwargs: {
            "data": {"NSE:NIFTY": {"last_price": 25000.0, "instrument_token": 256265}}
        },
    )

    out = client.get_quote_bulk(["NSE:NIFTY"])
    assert called["acquire"] == 1
    assert "NSE:NIFTY" in out
    assert out["NSE:NIFTY"]["instrument_token"] == 256265
    client.close()


def test_get_order_status_returns_latest_history_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = ZerodhaKiteClient(api_key="k", access_token="t")
    monkeypatch.setattr(client, "_acquire_bucket", lambda _bucket: None)
    monkeypatch.setattr(
        client,
        "_make_request",
        lambda *_args, **_kwargs: {
            "data": [
                {"order_id": "123", "status": "OPEN PENDING"},
                {"order_id": "123", "status": "OPEN"},
                {
                    "order_id": "123",
                    "status": "COMPLETE",
                    "average_price": 113.45,
                },
            ]
        },
    )

    status = client.get_order_status("123")

    assert status["status"] == "COMPLETE"
    assert status["average_price"] == 113.45
    client.close()
