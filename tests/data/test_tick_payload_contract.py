from datetime import datetime, timezone

import numpy as np
import pandas as pd

from nifty_scalper_bot.data.data_hub import DataHub
from nifty_scalper_bot.data.market_data_manager import MarketDataManager
from nifty_scalper_bot.utils.serialization import to_json_safe


def _has_datetime(v):
    if isinstance(v, (datetime, pd.Timestamp)):
        return True
    if isinstance(v, dict):
        return any(_has_datetime(x) for x in v.values())
    if isinstance(v, list):
        return any(_has_datetime(x) for x in v)
    return False


def test_to_json_safe_nested():
    payload = {"a": datetime.now(timezone.utc), "b": [pd.Timestamp.utcnow(), np.float64(1.2)]}
    safe = to_json_safe(payload)
    assert isinstance(safe["a"], str)
    assert not _has_datetime(safe)
