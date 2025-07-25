AttributeError: 'TechnicalIndicators' object has no attribute 'calculate_rsi'
2025-07-25 04:36:45,873 - INFO - Price: ₹24926.75
2025-07-25 04:36:45,873 - INFO - P&L: ₹0.00
2025-07-25 04:36:45,873 - INFO - Balance: ₹100000.00
2025-07-25 04:36:50,953 - ERROR - Error calculating indicators: 'TechnicalIndicators' object has no attribute 'calculate_rsi'
Traceback (most recent call last):
  File "/app/signal_generator.py", line 49, in calculate_all_indicators
    indicators['rsi'] = self.indicators.calculate_rsi(close_prices, Config.RSI_PERIOD)
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: 'TechnicalIndicators' object has no attribute 'calculate_rsi'
2025-07-25 04:36:50,953 - INFO - Price: ₹24926.35
2025-07-25 04:36:50,953 - INFO - P&L: ₹0.00
2025-07-25 04:36:50,953 - INFO - Balance: ₹100000.00
2025-07-25 04:36:53,611 - INFO - HTTP Request: POST https://api.telegram.org/bot7962917167:AAEgUdBKEznk8LRVm73RhXkTWzJ8CAX8GfI/getUpdates "HTTP/1.1 200 OK"
2025-07-25 04:36:56,027 - ERROR - Error calculating indicators: 'TechnicalIndicators' object has no attribute 'calculate_rsi'
Traceback (most recent call last):
  File "/app/signal_generator.py", line 49, in calculate_all_indicators
    indicators['rsi'] = self.indicators.calculate_rsi(close_prices, Config.RSI_PERIOD)
                        ^^^^^^^^^^^^^^^^^^^