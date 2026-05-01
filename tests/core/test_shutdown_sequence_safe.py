import pytest

from nifty_scalper_bot.core.app import shutdown_sequence


class Dummy:
    def stop(self):
        return None


class Ctx:
    strategy_runner = Dummy()
    order_manager = Dummy()
    market_data_manager = Dummy()
    position_manager = Dummy()
    persistent_state = Dummy()
    config = type('Cfg', (), {'close_positions_on_shutdown': False})()
    streamer = Dummy()


@pytest.mark.asyncio
async def test_shutdown_sequence_no_proc():
    ctx = Ctx()
    await shutdown_sequence(ctx)
