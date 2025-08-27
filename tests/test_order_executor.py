from src.execution.order_executor import OrderExecutor
from src.config import settings


def test_executor_uses_settings_attributes():
    executor = OrderExecutor(kite=None)

    ex = settings.executor
    ins = settings.instruments

    assert executor.exchange == ex.exchange
    assert executor.product == ex.order_product
    assert executor.variety == ex.order_variety
    assert executor.entry_order_type == ex.entry_order_type
    assert executor.tick_size == ex.tick_size
    assert executor.freeze_qty == ex.exchange_freeze_qty
    assert executor.lot_size == ins.nifty_lot_size
    assert executor.partial_enable == ex.partial_tp_enable
    assert executor.tp1_ratio == ex.tp1_qty_ratio
    assert executor.breakeven_ticks == ex.breakeven_ticks
    assert executor.enable_trailing == ex.enable_trailing
    assert executor.trailing_mult == ex.trailing_atr_multiplier
    assert executor.use_slm_exit == ex.use_slm_exit
