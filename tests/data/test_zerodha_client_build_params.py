import pytest

from nifty_scalper_bot.data.rest.zerodha_client import ZerodhaKiteClient
from nifty_scalper_bot.utils.errors import BrokerError


def _client() -> ZerodhaKiteClient:
    return ZerodhaKiteClient(api_key="k", api_secret="s", access_token="t")


def test_futures_rejected() -> None:
    client = _client()
    with pytest.raises(BrokerError):
        client._build_kite_params(  # type: ignore[attr-defined]
            {"symbol": "NFO:NIFTY25APR25000FUT", "quantity": 75, "side": "BUY"}
        )


def test_non_option_rejected() -> None:
    client = _client()
    with pytest.raises(BrokerError):
        client._build_kite_params(  # type: ignore[attr-defined]
            {"symbol": "NFO:NIFTY25APR25000XX", "quantity": 75, "side": "BUY"}
        )


def test_exchange_must_be_nfo() -> None:
    client = _client()
    with pytest.raises(BrokerError):
        client._build_kite_params(  # type: ignore[attr-defined]
            {"symbol": "NSE:NIFTY25APR25000CE", "quantity": 75, "side": "BUY"}
        )


def test_tradingsymbol_preserves_case() -> None:
    client = _client()
    params = client._build_kite_params(  # type: ignore[attr-defined]
        {"symbol": "nfo:nifty25apr25000ce", "quantity": 75, "side": "BUY"}
    )
    assert params["tradingsymbol"] == "nifty25apr25000ce"
    assert params["exchange"] == "NFO"
