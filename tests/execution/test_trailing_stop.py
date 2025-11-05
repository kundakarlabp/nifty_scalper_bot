from __future__ import annotations

from pathlib import Path
import tempfile
import types
from typing import Any, Callable
from uuid import uuid4

import pytest

from nifty_scalper_bot.execution.trailing_stop import (
    TrailingSpec,
    TrailingStopController,
)
from nifty_scalper_bot.storage.journal import AtomicKV


class FakeClock:
    def __init__(self, values: list[float]) -> None:
        self._values = iter(values)
        self._last = values[-1]

    def __call__(self) -> float:
        try:
            self._last = next(self._values)
        except StopIteration:
            pass
        return self._last


class FakeModify:
    def __init__(self) -> None:
        self.calls: list[types.SimpleNamespace] = []

    def __call__(
        self, variety: str, order_id: str, qty: Any, price: Any
    ) -> dict[str, Any]:
        self.calls.append(
            types.SimpleNamespace(
                variety=variety,
                order_id=order_id,
                quantity=qty,
                price=price,
            )
        )
        return {"status": "ok"}


def make_controller(
    *,
    side: str = "LONG",
    entry: float = 100.0,
    journal: AtomicKV | None = None,
    get_ltp: Callable[[str], float | None] | None = None,
) -> tuple[TrailingStopController, FakeModify, AtomicKV]:
    tmp_journal = journal or AtomicKV(
        Path(tempfile.gettempdir()) / f"trailing-test-{uuid4().hex}.json"
    )
    modifier = FakeModify()

    def _get_ltp(_symbol: str) -> float | None:
        return 100.0

    controller = TrailingStopController(
        symbol="NIFTY23",
        side=side,
        entry=entry,
        sl_order_id="SL123",
        variety="regular",
        spec=TrailingSpec(trail_by=5.0, step=0.5, min_gap=1.0, debounce_ms=200),
        get_ltp=get_ltp or _get_ltp,
        modify_order=modifier,
        journal=tmp_journal,
    )
    return controller, modifier, tmp_journal


def test_trailing_debounce_and_monotonic(monkeypatch: pytest.MonkeyPatch) -> None:
    ltp = 100.0

    def _get_ltp(_symbol: str) -> float:
        return ltp

    controller, modify, journal = make_controller(get_ltp=_get_ltp)

    clock = FakeClock([0.0, 0.01, 0.02, 0.5, 0.51, 0.52])
    monkeypatch.setattr(
        "nifty_scalper_bot.execution.trailing_stop.time.monotonic",
        clock,
    )

    controller.on_tick({"ltp": ltp})
    assert modify.calls[-1].price == pytest.approx(95.0)
    assert journal.get("SL123") == pytest.approx(95.0)

    # Within debounce window -> no call
    controller.on_tick({"ltp": ltp + 1})
    assert len(modify.calls) == 1

    # Outside debounce window and price higher -> tighten
    ltp = 108.0
    controller.on_tick({"ltp": ltp})
    assert len(modify.calls) == 2
    assert modify.calls[-1].price > 95.0


def test_trailing_restores_from_journal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    journal = AtomicKV(tmp_path / "journal.json")
    controller, modify, _ = make_controller(journal=journal)
    monkeypatch.setattr(
        "nifty_scalper_bot.execution.trailing_stop.time.monotonic", lambda: 0.0
    )
    controller.on_tick({"ltp": 100.0})
    assert modify.calls[-1].price == pytest.approx(95.0)

    # Re-create controller and ensure trigger restored
    controller2, modify2, _ = make_controller(journal=journal)
    assert journal.get("SL123") == pytest.approx(95.0)
    # Next move should only tighten further
    clock = FakeClock([0.5, 0.7, 0.9, 1.1])
    monkeypatch.setattr(
        "nifty_scalper_bot.execution.trailing_stop.time.monotonic",
        clock,
    )
    controller2.on_tick({"ltp": 110.0})
    assert modify2.calls[-1].price > 95.0


def test_trailing_short_respects_direction(monkeypatch: pytest.MonkeyPatch) -> None:
    ltp = 100.0

    def _get_ltp(_symbol: str) -> float:
        return ltp

    controller, modify, _ = make_controller(side="SHORT", get_ltp=_get_ltp)
    clock = FakeClock([0.0, 0.3, 0.6, 0.9, 1.2])
    monkeypatch.setattr(
        "nifty_scalper_bot.execution.trailing_stop.time.monotonic",
        clock,
    )

    controller.on_tick({"ltp": ltp})
    first_trigger = modify.calls[-1].price
    assert first_trigger == pytest.approx(105.0)

    ltp = 95.0
    controller.on_tick({"ltp": ltp})
    second_trigger = modify.calls[-1].price
    assert second_trigger < first_trigger

    ltp = 96.0
    controller.on_tick({"ltp": ltp})
    # Should not loosen stop-loss beyond the favourable adjustment
    assert modify.calls[-1].price <= second_trigger
