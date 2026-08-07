from __future__ import annotations

from nifty_scalper_bot.core.universe_controller import UniverseController


def _base_update(controller: UniverseController, symbols: list[str]):
    """Exercise UniverseController semantics independent of runtime market-hour safety."""
    update = getattr(
        UniverseController,
        "_off_market_basket_safety_original_update",
        UniverseController.update,
    )
    return update(controller, symbols)


def test_update_tracks_added_and_removed_members() -> None:
    controller = UniverseController()

    added, removed = _base_update(controller, ["A", "B"])
    assert added == {"A", "B"}
    assert removed == set()

    added, removed = _base_update(controller, ["B", "C"])
    assert added == {"C"}
    assert removed == {"A"}


def test_update_without_changes_keeps_empty_diff() -> None:
    controller = UniverseController()
    _base_update(controller, ["A", "B"])

    added, removed = _base_update(controller, ["A", "B"])
    assert added == set()
    assert removed == set()
