"""Sequential independent book evidence for a structurally valid SMC setup.

Research opt-in; OrderFlow remains context-only, never an entry trigger.
"""

from __future__ import annotations

import math
import time
from collections import OrderedDict
from collections.abc import Sequence
from typing import Any


class SustainedSMCFlow:
    """Require two distinct fresh, non-adverse temporal OFI observations."""

    def __init__(self) -> None:
        # Tuple: first observation time, quote version, consumed proof.
        # Proof is once per setup identity, not once per quote update.
        self._prior: OrderedDict[str, tuple[float, int, bool]] = OrderedDict()

    def observe(
        self,
        *,
        setup_id: str,
        symbol: str,
        side: str,
        votes: Sequence[tuple[Any, Any]],
        monotonic_now: float,
    ) -> bool:
        if not setup_id or not symbol or side not in {"CE", "PE"}:
            return False
        key = f"{side}:{symbol}:{setup_id}"
        now = time.time()
        eligible: list[int] = []
        for _signal, evidence in votes:
            if str(getattr(evidence, "strategy", "")).lower() != "orderflow":
                continue
            if str(getattr(evidence, "side", "")).upper() != side:
                continue
            md = dict(getattr(evidence, "metadata", None) or {})
            if not (
                md.get("context_quality_eligible") is True
                and md.get("effective_context_alignment") is True
                and md.get("effective_context_conflict") is False
                and md.get("context_alignment_source") == "temporal_ofi"
                and md.get("ofi_ready") is True
                and md.get("ofi_sustained_confirmation") is True
                and md.get("ofi_3s_complete") is True
                and md.get("ofi_reversal_holdoff_active") is False
            ):
                continue
            try:
                stamp = float(md["vote_timestamp"])
                version = int(md["quote_update_version"])
                updates = int(md["ofi_update_count_3s"])
                span = float(md["ofi_3s_span_seconds"])
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
            if (
                math.isfinite(stamp)
                and math.isfinite(span)
                and 0.0 <= now - stamp <= 3.0
                and version > 0
                and updates >= 3
                and span >= 1.5
            ):
                eligible.append(version)

        prior = self._prior.get(key)
        if len(eligible) != 1 or not math.isfinite(monotonic_now):
            # Reset incomplete unproven evidence, but never rearm a setup
            # whose independent two-book proof has already been consumed.
            if prior is None or not prior[2]:
                self._prior.pop(key, None)
            return False

        if prior is not None and prior[2]:
            self._prior.move_to_end(key)
            return False
        version = eligible[0]
        confirmed = bool(
            prior is not None
            and version > prior[1]
            and 4.0 <= monotonic_now - prior[0] <= 12.0
        )
        self._prior[key] = (monotonic_now, version, confirmed)
        self._prior.move_to_end(key)
        while len(self._prior) > 512:
            self._prior.popitem(last=False)
        return confirmed
