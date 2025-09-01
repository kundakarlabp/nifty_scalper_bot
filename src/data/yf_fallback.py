"""Utility helpers for Yahoo Finance symbol lookups.

This module implements a small map from common index names used by
exchanges or data providers to the tickers understood by Yahoo Finance.
It is used as a lightweight fallback when the primary data vendor is
unavailable.
"""

from __future__ import annotations

# Known index name -> Yahoo Finance ticker mapping.
# The keys are case sensitive to keep the implementation simple and
# explicit. Additional entries can be added here as required.
_SYMBOL_MAP: dict[str, str] = {
    "NIFTY 50": "^NSEI",
    "NIFTY BANK": "^NSEBANK",
}


def _map_symbol(name: str) -> str:
    """Return the Yahoo Finance ticker for a human readable *name*.

    Parameters
    ----------
    name:
        The instrument name as reported by the upstream data source.

    Returns
    -------
    str
        The corresponding Yahoo Finance ticker. If ``name`` is unknown,
        it is returned unchanged so the caller can decide how to handle
        the missing mapping.
    """

    return _SYMBOL_MAP.get(name, name)


__all__ = ["_map_symbol"]
