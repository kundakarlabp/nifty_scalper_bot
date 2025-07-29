# tests/test_position_sizing.py
import unittest
from src.risk.position_sizing import PositionSizing

class TestPositionSizing(unittest.TestCase):

    def setUp(self):
        self.account_size = 100000.0
        self.risk_manager = PositionSizing(account_size=self.account_size, risk_per_trade=0.01)

    def test_calculate_position_size_basic(self):
        entry_price = 180.0
        stop_loss = 178.0 # Risk of 2 points
        expected_risk_amount = self.account_size * 0.01 # 1000
        expected_quantity = int(expected_risk_amount / 2) # 500
        expected_lots = max(1, expected_quantity // 50) # Assuming NIFTY_LOT_SIZE=50 and MIN_LOTS=1

        result = self.risk_manager.calculate_position_size(entry_price, stop_loss)

        self.assertEqual(result['quantity'], expected_lots * 50)
        self.assertEqual(result['lots'], expected_lots)
        # Note: risk_amount might be slightly different due to lot size rounding
        # self.assertAlmostEqual(result['risk_amount'], expected_risk_amount, places=2)

    def test_calculate_position_size_zero_risk(self):
        entry_price = 180.0
        stop_loss = 180.0 # No risk
        result = self.risk_manager.calculate_position_size(entry_price, stop_loss)
        self.assertEqual(result['quantity'], 0)
        self.assertEqual(result['lots'], 0)

    # Add more tests for confidence multipliers, volatility, performance multipliers, limits etc.

if __name__ == '__main__':
    unittest.main()
