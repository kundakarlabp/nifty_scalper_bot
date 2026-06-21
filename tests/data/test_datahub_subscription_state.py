from nifty_scalper_bot.data.data_hub import DataHub, SubscriptionState


class _Mdm:
    def __init__(self):
        self.subscribed = []

    def subscribe(self, symbol, callback):
        self.subscribed.append((symbol, callback))


def test_datahub_subscription_queued_is_not_live():
    mdm = _Mdm()
    hub = DataHub(mdm)

    hub.subscribe_ticks("NFO:NIFTY26MAY23750CE", lambda tick: None, force_live=True)

    assert hub.get_subscription_state("NFO:NIFTY26MAY23750CE") == SubscriptionState.QUEUED
    assert mdm.subscribed


def test_datahub_first_ws_tick_advances_subscription_live():
    mdm = _Mdm()
    hub = DataHub(mdm)
    symbol = "NFO:NIFTY26MAY23750CE"
    hub.subscribe_ticks(symbol, lambda tick: None, force_live=True)

    hub.ingest_tick_sync(
        {
            "symbol": symbol,
            "instrument_token": 12345,
            "last_price": 100.0,
            "timestamp": "2026-06-21T09:16:00+05:30",
            "source": "ws",
        }
    )

    record = hub.get_subscription_record(symbol)
    assert record is not None
    assert record.state == SubscriptionState.LIVE
    assert record.token == 12345
