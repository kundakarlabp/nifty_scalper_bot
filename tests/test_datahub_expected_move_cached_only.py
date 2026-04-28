from __future__ import annotations

from nifty_scalper_bot.data.data_hub import DataHub


class _MdmNoop:
    def get_cached_ltp(
        self,
        _symbol: str,
        *,
        max_age_seconds: float | None = None,
        require_ws: bool = False,
    ) -> None:
        return None


def test_expected_move_uses_cached_ltp_only() -> None:
    hub = DataHub(_MdmNoop())

    def _raise(*_args, **_kwargs):  # noqa: ANN002, ANN003, ANN201
        raise AssertionError('get_latest_price must not be called')

    hub.get_latest_price = _raise  # type: ignore[method-assign]
    assert hub.expected_move(1) is None
