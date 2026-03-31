from __future__ import annotations

from nifty_scalper_bot.core.universe_controller import UniverseController


def test_update_tracks_added_and_removed_members() -> None:
    controller = UniverseController()

    added, removed = controller.update(["A", "B"])
    assert added == {"A", "B"}
    assert removed == set()

    added, removed = controller.update(["B", "C"])
    assert added == {"C"}
    assert removed == {"A"}


def test_update_without_changes_keeps_empty_diff() -> None:
    controller = UniverseController()
    controller.update(["A", "B"])

    added, removed = controller.update(["A", "B"])
    assert added == set()
    assert removed == set()
