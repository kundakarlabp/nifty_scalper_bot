from __future__ import annotations

from types import SimpleNamespace

import pytest

from nifty_scalper_bot.core.app import _dynamic_symbol_removal_blocker


class _OwnershipManager:
    def __init__(self, owns_symbol: bool) -> None:
        self._owns_symbol = owns_symbol

    def is_symbol_managed(self, _symbol: str) -> bool:
        return self._owns_symbol

    def has_open_position(self, _symbol: str) -> bool:
        return self._owns_symbol


@pytest.mark.parametrize(
    ("bracket_owned", "position_owned", "expected"),
    [
        (True, False, "active_bracket"),
        (False, True, "open_position"),
        (False, False, None),
    ],
)
def test_dynamic_symbol_removal_respects_execution_ownership(
    bracket_owned: bool,
    position_owned: bool,
    expected: str | None,
) -> None:
    ctx = SimpleNamespace(
        bracket_manager=_OwnershipManager(bracket_owned),
        position_manager=_OwnershipManager(position_owned),
    )

    assert _dynamic_symbol_removal_blocker(ctx, "NFO:NIFTY2681824400PE") == expected


def test_dynamic_symbol_removal_fails_safe_when_ownership_is_unknown() -> None:
    class _BrokenBracketManager:
        def is_symbol_managed(self, _symbol: str) -> bool:
            raise RuntimeError("state unavailable")

    ctx = SimpleNamespace(
        bracket_manager=_BrokenBracketManager(),
        position_manager=_OwnershipManager(False),
    )

    assert (
        _dynamic_symbol_removal_blocker(ctx, "NFO:NIFTY2681824400PE")
        == "bracket_state_unknown"
    )
