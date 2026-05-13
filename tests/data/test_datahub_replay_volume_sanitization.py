from nifty_scalper_bot.data.data_hub import DataHub


class _MDM:
    def attach_tick_bus(self, _):
        return None


def test_datahub_replay_sanitizes_contaminated_option_volume() -> None:
    hub = DataHub(_MDM())
    hub._quotes['NFO:NIFTY26MAY23500CE'] = {"symbol": "NFO:NIFTY26MAY23500CE", "volume": 6008275, "volume_cumulative": 6008275}
    received = []
    hub.subscribe_ticks('NFO:NIFTY26MAY23500CE', lambda q: received.append(q), force_live=True)
    assert received
    assert received[0]['volume'] == 0
    assert received[0]['volume_delta'] == 0
