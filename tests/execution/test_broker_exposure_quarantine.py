from __future__ import annotations

import nifty_scalper_bot.execution  # noqa: F401 - applies runtime safety patches
from nifty_scalper_bot.execution.position_manager import PositionManager


SYMBOL = "NFO:NIFTY24JAN100CE"


def test_unresolved_broker_position_is_quarantined(tmp_path):
    manager = PositionManager(str(tmp_path / "positions.json"))

    manager.synchronize_with_broker(
        [
            {
                "tradingsymbol": "NIFTY24JAN100CE",
                "quantity": 65,
                "average_price": 0,
                "last_price": 88.55,
                "product": "MIS",
            }
        ]
    )

    exposures = manager.get_quarantined_broker_exposures()
    assert list(exposures) == [SYMBOL]
    exposure = exposures[SYMBOL]
    assert exposure["status"] == "BROKER_POSITION_QUARANTINED"
    assert exposure["reason"] == "cost_basis_unresolved"
    assert exposure["quantity"] == 65
    assert exposure["managed_position"] is False
    assert exposure["requires_history_recovery"] is True
    assert manager.current_entry_protection_blocker(SYMBOL) == "broker_exposure_quarantined"
    assert manager.get_position(SYMBOL) is None
