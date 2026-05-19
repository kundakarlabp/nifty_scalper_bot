from nifty_scalper_bot.execution.bracket_manager import ExitExecutionResult


def test_exit_execution_result_instantiation_safe() -> None:
    result = ExitExecutionResult(submitted=False, confirmed=False, order_id=None, filled_qty=0, reason='test')
    assert result.reason == 'test'
    assert not hasattr(result, 'stop_loss_price')
    assert not hasattr(result, 'target_price')
