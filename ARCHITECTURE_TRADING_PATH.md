# Nifty Scalper Bot — Trading Path

## Final Runtime Path

WebSocketManager
  → MarketDataManager
  → MessageBus
  → DataHub
  → StrategyRunner
  → RiskManager / Position sizing
  → OrderManager
  → Broker / Paper
  → BracketManager
  → Journal / Telegram

## File Ownership

### MarketDataManager
File:
src/nifty_scalper_bot/data/market_data_manager.py

### DataHub
File:
src/nifty_scalper_bot/data/data_hub.py

### StrategyRunner
File:
src/nifty_scalper_bot/strategies/runner.py

### OrderManager
File:
src/nifty_scalper_bot/execution/order_manager.py

### BracketManager
File:
src/nifty_scalper_bot/execution/bracket_manager.py

## Removed Runtime Files

Removed:
- src/nifty_scalper_bot/execution/order_execution_hub.py
- src/nifty_scalper_bot/execution/execution_router.py
- src/nifty_scalper_bot/execution/preflight_validator.py
