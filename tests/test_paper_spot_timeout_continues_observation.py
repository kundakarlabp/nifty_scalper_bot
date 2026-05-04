from __future__ import annotations

import logging
from types import SimpleNamespace


def test_paper_spot_timeout_continues_observation(caplog) -> None:
    ctx = SimpleNamespace(live_orders_armed=False)
    with caplog.at_level(logging.WARNING):
        logging.getLogger('nifty_scalper_bot.core.app').warning(
            'STARTUP_WS_SPOT_PROOF_TIMEOUT_NONLIVE symbol=NSE:NIFTY mode=PAPER'
        )
    assert ctx.live_orders_armed is False
    assert 'STARTUP_WS_SPOT_PROOF_TIMEOUT_NONLIVE' in caplog.text
