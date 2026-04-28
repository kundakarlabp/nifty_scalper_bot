from __future__ import annotations

from typing import Any

from nifty_scalper_bot.data.market_data_policy import MarketDataPolicy
from nifty_scalper_bot.data.rest.zerodha_client import ZerodhaKiteClient


def test_zerodha_quote_bulk_maps_nifty_alias() -> None:
    client = ZerodhaKiteClient.__new__(ZerodhaKiteClient)
    client._md_policy = MarketDataPolicy.from_env()  # noqa: SLF001
    client._quote_api_available = True  # noqa: SLF001
    client._quote_api_error = None  # noqa: SLF001
    client._quote_api_last_checked_at = None  # noqa: SLF001
    client._last_quote_error_log_at = {}  # noqa: SLF001
    client._acquire_bucket = lambda _bucket: None  # noqa: SLF001
    client._ensure_json = lambda payload: payload  # noqa: SLF001
    client._mark_quote_api_status = lambda **_kwargs: None  # noqa: SLF001
    client._is_quote_access_denied = lambda _exc: False  # noqa: SLF001
    calls: list[dict[str, Any]] = []

    def _make_request(_method: str, _path: str, params: dict[str, Any]):  # noqa: ANN001
        calls.append(params)
        return {
            'data': {
                'NSE:NIFTY 50': {
                    'last_price': 24011.60,
                    'instrument_token': 256265,
                }
            }
        }

    client._make_request = _make_request  # type: ignore[attr-defined]  # noqa: SLF001
    out = client.get_quote_bulk(['NSE:NIFTY'])
    assert 'NSE:NIFTY' in out
    assert out['NSE:NIFTY']['last_price'] == 24011.60
    requested = calls[0]['i']
    assert 'NSE:NIFTY 50' in requested
