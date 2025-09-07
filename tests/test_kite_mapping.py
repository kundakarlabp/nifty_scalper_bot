from __future__ import annotations

from decimal import Decimal

from src.broker.interface import OrderRequest, OrderType, Side, TimeInForce
from src.broker.instruments import Instrument, InstrumentStore
from src.brokers.kite import KiteBroker


def test_order_param_mapping_with_store() -> None:
    store = InstrumentStore([
        Instrument(token=123, symbol="NIFTY24SEP18000CE", exchange="NFO", product="MIS", variety="regular")
    ])
    kb = KiteBroker(api_key="x", access_token="y", instrument_store=store, enable_ws=False)
    req = OrderRequest(
        instrument_id=123,
        side=Side.BUY,
        qty=50,
        order_type=OrderType.LIMIT,
        price=Decimal("12.5"),
        tif=TimeInForce.IOC,
        client_order_id="T1",
    )
    params = kb._map_order_request(req)
    assert params["exchange"] == "NFO"
    assert params["tradingsymbol"] == "NIFTY24SEP18000CE"
    assert params["transaction_type"] == "BUY"
    assert params["product"] == "MIS"
    assert params["order_type"] == "LIMIT"
    assert params["price"] == 12.5
    assert params["validity"] == "IOC"
