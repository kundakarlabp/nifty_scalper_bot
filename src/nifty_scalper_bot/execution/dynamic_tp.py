# src/nifty_scalper_bot/execution/dynamic_tp.py

class DynamicTPController:
    """Expands Take Profit targets during high momentum."""
    
    def __init__(self, tp_order_id, initial_price, ...):
        # ...
    
    def on_tick(self, tick):
        # If RSI > 70 or Momentum is huge:
        # Move TP Price UP by 10 points
        # Update Broker Order
