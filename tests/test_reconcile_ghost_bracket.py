"""Regression test for ghost/orphan safety-bracket attachment on reconcile.

``reconcile_with_broker`` adopts naked broker positions by calling
``BracketManager.attach_orphan_position(symbol, side, qty, entry_price)``.
The method computes its own ATR-based rescue levels and does NOT accept
``sl``/``tp`` kwargs; passing them raised TypeError, logged
``SAFETY_BRACKET_FAILED`` and left the ghost position unprotected.
"""

from __future__ import annotations

from typing import Any

from nifty_scalper_bot.core.app import reconcile_with_broker


class _RecordingLogger:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def _record(self, msg: str, *args: Any, **kwargs: Any) -> None:
        try:
            self.messages.append(msg % args if args else msg)
        except Exception:
            self.messages.append(msg)

    info = warning = error = debug = exception = _record


class _BracketManager:
    """Mirrors the real attach_orphan_position signature exactly."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def is_symbol_managed(self, symbol: str) -> bool:
        return False

    def attach_orphan_position(
        self, symbol: str, side: str, qty: int, entry_price: float
    ) -> str:
        self.calls.append(
            {"symbol": symbol, "side": side, "qty": qty, "entry_price": entry_price}
        )
        return f"orphan_{symbol}"


class _Broker:
    def get_orders(self) -> list[dict[str, Any]]:
        return []

    def get_positions(self) -> list[dict[str, Any]]:
        return [
            {
                "tradingsymbol": "NIFTY2662324050PE",
                "quantity": 65,
                "average_price": 94.72,
                "product": "MIS",
            }
        ]


class _DummyOM:
    def __getattr__(self, _name: str) -> Any:
        return lambda *a, **k: None


async def test_ghost_position_attaches_bracket_without_sl_tp() -> None:
    bm = _BracketManager()
    logger = _RecordingLogger()

    await reconcile_with_broker(_Broker(), bm, _DummyOM(), logger)

    # Exactly one adoption, with the four supported args and no sl/tp.
    assert len(bm.calls) == 1
    call = bm.calls[0]
    assert call["symbol"] == "NIFTY2662324050PE"
    assert call["side"] == "BUY"  # qty > 0
    assert call["qty"] == 65
    assert call["entry_price"] == 94.72

    joined = "\n".join(logger.messages)
    assert "SAFETY_BRACKET_ATTACHED" in joined
    assert "SAFETY_BRACKET_FAILED" not in joined
