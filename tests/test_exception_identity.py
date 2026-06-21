from nifty_scalper_bot.execution import exceptions as execution_exceptions
from nifty_scalper_bot.utils import errors as utils_errors


def test_execution_exceptions_reexport_canonical_order_error():
    assert execution_exceptions.OrderPlacementError is utils_errors.OrderPlacementError
    assert execution_exceptions.BrokerError is utils_errors.BrokerError
