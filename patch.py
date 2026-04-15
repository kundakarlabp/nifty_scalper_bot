import sys
import re

# 1. Patch runner.py
with open("src/nifty_scalper_bot/strategies/runner.py", "r") as f:
    content = f.read()

on_data_code = """
    async def on_data(self, message):
        token = message.data.get("token")
        if not token:
            return

        if not self._data_hub or not self._data_hub.is_ready(token):
            return

        candles, indicators = self._data_hub.get_data(token)
        if candles is None:
            return

        await self._process_token(token, candles, indicators)

    async def _process_token(self, token, candles, indicators):
        symbol = None
        if self._market_data and hasattr(self._market_data, "_symbol_by_token"):
            symbol = self._market_data._symbol_by_token.get(token)

        if not symbol:
            return

        try:
            if candles.empty:
                return
            latest_candle = candles.iloc[-1]
            price = float(latest_candle['close'])

            signal = self._strategy_manager.generate_signal(symbol, price)
            if signal:
                from datetime import datetime, timezone
                now = datetime.now(timezone.utc)
                self._handle_signal(signal, price, now)

            # print(f"STRATEGY TRIGGERED: {token}")
        except Exception as e:
            self._logger.error(f"Error in _process_token for {symbol}: {e}")
"""

if "def on_data" not in content:
    content = content.replace("    def start(self) -> None:", on_data_code + "\n    def start(self) -> None:")

content = content.replace(
    "worker = threading.Thread(target=self._strategy_worker, daemon=True)\n        worker.start()",
    "# worker = threading.Thread(target=self._strategy_worker, daemon=True)\n        # worker.start()"
)

with open("src/nifty_scalper_bot/strategies/runner.py", "w") as f:
    f.write(content)

# 2. Patch app.py
with open("src/nifty_scalper_bot/core/app.py", "r") as f:
    content = f.read()

content = content.replace(
    "                    ctx.strategy_runner.start()",
    "                    # ctx.strategy_runner.start()\n                    if hasattr(ctx, 'message_bus') and getattr(ctx.message_bus, 'subscribe', None):\n                        from nifty_scalper_bot.core.message_bus import MessageType\n                        ctx.message_bus.subscribe(MessageType.DATA_READY, ctx.strategy_runner.on_data)"
)

with open("src/nifty_scalper_bot/core/app.py", "w") as f:
    f.write(content)

# 3. Patch components.py
with open("src/nifty_scalper_bot/bot/components.py", "r") as f:
    content = f.read()

on_data_components = """
    async def on_data(self, message):
        token = message.data.get("token")

        if getattr(self, "datahub", None) is None:
            return

        if not self.datahub.is_ready(token):
            return

        candles, indicators = self.datahub.get_data(token)
        if candles is None:
            return

        self._process_token(token, candles, indicators)

    def _process_token(self, token, candles, indicators):
        try:
            if candles.empty:
                return
            latest = candles.iloc[-1]
            price = float(latest['close'])
            symbol = str(token)

            self._on_tick(symbol, price)
        except Exception as e:
            import logging
            logging.error(f"Error in components._process_token: {e}")
"""

if "def on_data" not in content:
    content = content.replace("    def start(self) -> None:", on_data_components + "\n    def start(self) -> None:")

if "datahub=None" not in content:
    content = content.replace(
        "config: StrategyRunnerConfig | None = None,\n    ) -> None:",
        "config: StrategyRunnerConfig | None = None,\n        datahub=None,\n    ) -> None:"
    )

if "self.datahub = datahub" not in content:
    content = content.replace(
        "self._config = config or StrategyRunnerConfig()",
        "self._config = config or StrategyRunnerConfig()\n        self.datahub = datahub"
    )

content = content.replace(
"""        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="StrategyRunner"
        )
        self._thread.start()""",
"""        # self._thread = threading.Thread(
        #     target=self._run_loop, daemon=True, name="StrategyRunner"
        # )
        # self._thread.start()"""
)

with open("src/nifty_scalper_bot/bot/components.py", "w") as f:
    f.write(content)
