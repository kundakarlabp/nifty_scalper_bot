"""Linked basket risk must use broker funds and initial-margin authority."""

from nifty_scalper_bot.execution.margin_engine import MarginEngine

LEGS = {
    "long_put": {
        "symbol": "NFO:NIFTY26NOV23800PE",
        "side": "BUY",
        "quantity": 65,
    },
    "long_call": {
        "symbol": "NFO:NIFTY26NOV24300CE",
        "side": "BUY",
        "quantity": 65,
    },
    "short_put": {
        "symbol": "NFO:NIFTY26NOV23900PE",
        "side": "SELL",
        "quantity": 65,
    },
    "short_call": {
        "symbol": "NFO:NIFTY26NOV24200CE",
        "side": "SELL",
        "quantity": 65,
    },
}


class Broker:
    def __init__(self):
        self.margin = {"initial": {"total": 120000}, "final": {"total": 58000}}
        self.available = 140000
        self.requests = []
        self.orders_placed = 0

    def basket_order_margins(self, orders, consider_positions=True):
        self.requests.append((orders, consider_positions))
        return self.margin

    def margins(self):
        return {"equity": {"available": {"live_balance": self.available}}}

    def place_order(self, **kwargs):
        self.orders_placed += 1
        raise AssertionError("basket preflight cannot place orders")


def screen(broker=None, **overrides):
    broker = broker or Broker()
    engine = MarginEngine(
        broker=broker,
        data_hub=None,
        lot_size_resolver=None,
        clock=None,
    )
    kwargs = {
        "legs": LEGS,
        "product": "NRML",
        "gross_loss_bound": 13000,
        "remaining_trade_risk": 16000,
        "remaining_daily_risk": 18000,
        "entry_fee_reserve": 100,
        "exit_fee_reserve": 200,
        "stressed_slippage_reserve": 400,
    }
    kwargs.update(overrides)
    return engine.screen_linked_portfolio(**kwargs)


def test_broker_initial_margin_outranks_hedged_final_margin():
    broker = Broker()
    decision = screen(broker)
    assert decision.ok is True
    assert decision.required_cash_with_reserves == 120700
    assert decision.stressed_loss_with_costs == 13700
    assert decision.live_eligible is False
    assert [row["transaction_type"] for row in broker.requests[0][0]] == [
        "BUY",
        "BUY",
        "SELL",
        "SELL",
    ]
    assert broker.requests[0][1] is True
    assert broker.orders_placed == 0


def test_small_cash_cannot_be_rescued_by_lower_final_spread_margin():
    broker = Broker()
    broker.available = 110000
    decision = screen(broker)
    assert decision.ok is False
    assert decision.reason == "linked_margin_cash_insufficient"


def test_trade_and_daily_risk_independently_block():
    assert screen(remaining_trade_risk=12000).reason == (
        "linked_margin_trade_risk_exceeded"
    )
    assert screen(remaining_daily_risk=13000).reason == (
        "linked_margin_daily_risk_exceeded"
    )


def test_missing_or_partial_broker_margin_fails_closed():
    broker = Broker()
    broker.margin.pop("initial")
    assert screen(broker).reason == "linked_margin_basket_invalid"
    broker.margin = {"initial": {"total": 100}, "final": {"total": 0}}
    broker.available = None
    assert screen(broker).reason == "linked_margin_broker_values_invalid"


def test_broker_opening_balance_is_not_live_available_funds():
    broker = Broker()
    broker.margins = lambda: {
        "equity": {"available": {"opening_balance": 500000}}
    }
    decision = screen(broker)
    assert decision.ok is False
    assert decision.reason == "linked_margin_broker_funds_invalid"


def test_no_fallback_from_missing_broker_to_estimated_margin():
    class NoBroker:
        def place_order(self, **kwargs):
            raise AssertionError("no submission")

    assert screen(NoBroker()).reason == "linked_margin_broker_authority_missing"


def test_invalid_risk_price_or_partial_basket_is_never_accepted():
    assert screen(stressed_slippage_reserve=0).reason == (
        "linked_margin_risk_inputs_invalid"
    )
    assert screen(remaining_daily_risk=float("nan")).ok is False
    assert screen(legs={"long_call": LEGS["long_call"]}).reason == (
        "linked_margin_contract_set_invalid"
    )
