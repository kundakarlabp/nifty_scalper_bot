"""Cost repair must price broker units without changing strategy lot counts."""

from dataclasses import dataclass, replace
from unittest import TestCase
from unittest.mock import patch

from nifty_scalper_bot.risk.net_rr_gate import evaluate_final_net_rr
from nifty_scalper_bot.strategies.premium_risk_geometry import (
    apply_cost_aware_risk_floor,
)


@dataclass(frozen=True)
class _Signal:
    symbol: str
    action: str
    quantity: int
    stop_loss: float
    take_profit: float
    metadata: dict


class CostRepairExecutionQuantityTests(TestCase):
    @patch.dict(
        "os.environ",
        {"MIN_NET_REWARD_RISK": "1.5", "MAX_NET_RR_TARGET_UPLIFT_R": "0.35"},
    )
    def test_lots_and_units_produce_identical_bounded_target(self):
        for lots, distance in ((1, 4.50), (2, 3.50)):
            with self.subTest(lots=lots):
                entry = 82.15
                signal = _Signal(
                    "NFO:NIFTY2691523400PE",
                    "BUY",
                    lots,
                    entry - distance,
                    entry + 2 * distance,
                    {"entry_price": entry, "bid": 81.95, "ask": 82.15},
                )
                units = lots * 65
                expected = apply_cost_aware_risk_floor(
                    replace(signal, quantity=units),
                    entry_price=entry,
                    quantity=units,
                    half_spread=0.10,
                )
                result = apply_cost_aware_risk_floor(
                    signal, entry_price=entry, quantity=units, half_spread=0.10
                )
                self.assertTrue(expected.metadata["premium_cost_target_repair_viable"])
                self.assertGreater(expected.take_profit, signal.take_profit)
                self.assertEqual(result.take_profit, expected.take_profit)
                self.assertEqual(result.stop_loss, signal.stop_loss)
                self.assertEqual(result.quantity, lots)
                self.assertLessEqual(result.take_profit, entry + 2.35 * distance)
                economics = evaluate_final_net_rr(replace(result, quantity=units))
                self.assertTrue(economics.allowed)

    @patch.dict(
        "os.environ",
        {"MIN_NET_REWARD_RISK": "1.5", "MAX_NET_RR_TARGET_UPLIFT_R": "0.35"},
    )
    def test_narrow_live_setup_remains_blocked_without_widening_stop(self):
        signal = _Signal(
            "NFO:NIFTY2691523400PE",
            "BUY",
            1,
            79.88093141471029,
            86.83813717057943,
            {"entry_price": 82.2, "bid": 82.1, "ask": 82.3},
        )
        result = apply_cost_aware_risk_floor(
            signal, entry_price=82.2, quantity=65, half_spread=0.1
        )
        self.assertFalse(result.metadata["premium_cost_target_repair_viable"])
        self.assertEqual(result.stop_loss, signal.stop_loss)
        self.assertEqual(result.take_profit, signal.take_profit)
        self.assertEqual(result.quantity, 1)
        self.assertFalse(evaluate_final_net_rr(replace(result, quantity=65)).allowed)
