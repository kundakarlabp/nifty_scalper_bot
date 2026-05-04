from nifty_scalper_bot.strategies.runner import RunnerState


def test_runner_state_has_booting() -> None:
    assert RunnerState.BOOTING.value == 1
