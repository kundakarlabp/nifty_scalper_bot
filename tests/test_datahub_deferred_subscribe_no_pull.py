from __future__ import annotations

from pathlib import Path

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


def test_datahub_flush_after_token_wiring() -> None:
    text = Path('src/nifty_scalper_bot/core/app.py').read_text(encoding='utf-8')
    req_idx = text.index('await request_token_subscriptions(tokens_to_poll)')
    flush_idx = text.index('ctx.data_hub.flush_pending_live_subscriptions()')
    ready_idx = text.index('set_readiness_requirements(')
    assert req_idx < flush_idx < ready_idx
