import sys

filepath = "src/nifty_scalper_bot/bot/components.py"
with open(filepath, "r") as f:
    content = f.read()

on_data_code = """
    async def on_data(self, message):
        token = message.data.get("token")

        # In components.py, we might not have a datahub or maybe it was injected
        # The prompt says "ENSURE THIS: strategy_runner = StrategyRunner(datahub=...)"
        # But this happens in app.py. So we need to add `datahub` to `StrategyRunner.__init__`

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
            symbol = str(token)  # simplistic fallback

            # Use internal logic
            self._on_tick(symbol, price)
        except Exception as e:
            import logging
            logging.error(f"Error in components._process_token: {e}")
"""

# Let's add it carefully.
if "def on_data" not in content:
    content = content.replace("    def start(self) -> None:", on_data_code + "\n    def start(self) -> None:")

# add datahub=None
content = content.replace("config: StrategyRunnerConfig | None = None,\n    ) -> None:", "config: StrategyRunnerConfig | None = None,\n        datahub=None,\n    ) -> None:")
content = content.replace("self._config = config or StrategyRunnerConfig()", "self._config = config or StrategyRunnerConfig()\n        self.datahub = datahub")

# remove thread start
content = content.replace("self._thread.start()", "# self._thread.start()")

with open(filepath, "w") as f:
    f.write(content)
