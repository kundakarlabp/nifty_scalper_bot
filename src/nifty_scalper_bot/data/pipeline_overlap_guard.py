"""Legacy overlap guard retained for compatibility.

The live candle store now owns timestamp normalization and protection directly.
"""

from __future__ import annotations


def install_candle_store_overlap_guard() -> bool:
    return False
