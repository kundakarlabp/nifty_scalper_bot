# Nifty Scalper Bot — Trading Path

## Authoritative live path

1. Startup:
   src/nifty_scalper_bot/core/app.py

2. Market data:
   src/nifty_scalper_bot/data/market_data_manager.py

3. Data routing:
   src/nifty_scalper_bot/data/data_hub.py
   message bus if used

4. Signal generation:
   src/nifty_scalper_bot/strategies/runner.py
   StrategyRunner receives market data, evaluates strategy, creates TradePlan.

5. Entry execution:
   src/nifty_scalper_bot/execution/order_manager.py
   OrderManager submits entry order through broker/paper executor.

6. Exit execution:
   src/nifty_scalper_bot/execution/bracket_manager.py
   BracketManager owns SL/TP/trailing/EOD virtual bracket exits.

7. Notifications/logs:
   src/nifty_scalper_bot/notifications/telegram_controller.py
   journal/trade logs

## Forbidden removed layers

These must not exist in runtime imports:
- order_execution_hub.py
- execution_router.py
- preflight_validator.py

## Engineering rule

One signal path. One entry manager. One exit manager. No extra routers.
