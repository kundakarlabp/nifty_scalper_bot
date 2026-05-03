from nifty_scalper_bot.data.data_hub import DataHub


class _DummyMDM:
    async def fetch_history(self, symbol: str, interval: str, days: int = 3):
        return []


def test_datahub_normalizes_dict_and_list_history_rows() -> None:
    hub = DataHub(_DummyMDM())
    rows = [
        {
            'timestamp': '2026-05-01T09:15:00+00:00',
            'open': 100,
            'high': 110,
            'low': 95,
            'close': 105,
            'volume': 1200,
        },
        ['2026-05-01T09:16:00+00:00', 105, 112, 101, 111, 1400],
    ]

    normalized = hub._normalize_history_rows('NFO:NIFTY26MAY24000CE', rows)

    assert len(normalized) == 2
    for bar in normalized:
        assert set(['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume', 'source']).issubset(bar.keys())
        assert bar['source'] == 'historical'
