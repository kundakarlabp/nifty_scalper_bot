from nifty_scalper_bot.execution.order_manager import OrderManager


class _Broker: pass
class _Pos: pass
class _Limiter: pass
class _IE:
    def get_latest(self, s): return {}


class _Hub:
    def __init__(self):
        self.sub = []
        self.unsub = []
    def subscribe_ticks(self, symbol, cb): self.sub.append(symbol)
    def unsubscribe_ticks(self, symbol, cb): self.unsub.append(symbol)


def test_dynamic_tp_uses_subscribe_ticks_and_unsubscribe():
    om = OrderManager(_Broker(), _Pos(), _Limiter(), indicator_engine=_IE())
    hub = _Hub(); om.attach_data_hub(hub)
    om.modify_order = lambda *a, **k: True
    om.attach_dynamic_tp(tp_order_id='tp1', symbol='NFO:NIFTY26MAY23750CE', side='SELL', initial_price=100.0, parent_order_id='p1')
    assert 'NFO:NIFTY26MAY23750CE' in hub.sub
    om.stop_dynamic_tp('tp1')
    assert 'NFO:NIFTY26MAY23750CE' in hub.unsub


def test_dynamic_tp_no_subscription_does_not_store_controller():
    om = OrderManager(_Broker(), _Pos(), _Limiter(), indicator_engine=_IE())
    om._subscribe_market_callback = lambda symbol, cb: False
    om.modify_order = lambda *a, **k: True
    om.attach_dynamic_tp(tp_order_id='tp2', symbol='NFO:NIFTY26MAY23750CE', side='SELL', initial_price=100.0, parent_order_id='p1')
    assert 'tp2' not in getattr(om, '_tp_controllers', {})
