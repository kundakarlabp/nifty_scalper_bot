from __future__ import annotations

from nifty_scalper_bot.data.data_hub import DataHub


class _StubMdm:
    def subscribe(self, _symbol: str, _callback):  # noqa: ANN001, ANN201
        return None


def test_datahub_deferred_subscribe_replay_is_cached_only() -> None:
    hub = DataHub(_StubMdm(), defer_live_symbol_subscriptions=True)

    def _get_quote(symbol: str, allow_pull: bool = True):  # noqa: ANN001, ANN201
        if allow_pull:
            raise AssertionError('allow_pull=True is forbidden for deferred replay')
        return None

    hub.get_quote = _get_quote  # type: ignore[method-assign]
    hub.subscribe_ticks('NSE:NIFTY', callback=lambda _tick: None)
