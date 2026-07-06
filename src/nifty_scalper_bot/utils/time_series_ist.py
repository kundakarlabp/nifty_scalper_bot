from __future__ import annotations

from typing import Any

import pandas as pd

from nifty_scalper_bot.utils.ist_clock import timestamp


def as_ist(values: Any) -> pd.Series:
    return pd.Series(values).map(lambda value: timestamp(value, errors="coerce"))
