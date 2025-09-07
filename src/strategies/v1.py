from __future__ import annotations

"""Compatibility wrapper for legacy strategy interface.

This module exposes a ``v1`` function mirroring the old strategy API
while delegating the actual signal generation to
:class:`EnhancedScalpingStrategy`.  The previous implementation defined
local variables like ``short_entry_price`` only inside conditional
branches, which could raise :class:`UnboundLocalError` when a branch was
not taken.  Delegating to the well tested ``EnhancedScalpingStrategy``
removes those fragile code paths without altering the public interface.
"""

from typing import Any, Dict, Optional

import pandas as pd

from .scalping_strategy import EnhancedScalpingStrategy


def v1(
    df: pd.DataFrame,
    current_tick: Optional[Dict[str, Any]] = None,
    current_price: Optional[float] = None,
    spot_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Generate a trade plan using the legacy ``v1`` interface.

    Parameters are passed directly to
    :meth:`EnhancedScalpingStrategy.generate_signal`.  Returning the plan
    dictionary keeps existing callers functional while preventing
    ``UnboundLocalError`` from missing variables such as
    ``short_entry_price``.
    """

    strategy = EnhancedScalpingStrategy()
    return strategy.generate_signal(
        df=df, current_tick=current_tick, current_price=current_price, spot_df=spot_df
    ) or {}
