# Just the minimum patch

with open("src/nifty_scalper_bot/core/message_bus.py", "r") as f:
    content = f.read()

content = content.replace('    TICK = "tick"  # Deprecated: disallowed for publish/subscribe', '    TICK = "tick"\n    DATA_READY = "data_ready"')
content = content.replace("        if message.type.value == 'tick':\n            raise RuntimeError('MessageBus does not carry tick events')\n", "")
content = content.replace("        if message_type.value == 'tick':\n            raise RuntimeError('MessageBus does not carry tick events')\n", "")

with open("src/nifty_scalper_bot/core/message_bus.py", "w") as f:
    f.write(content)

with open("src/nifty_scalper_bot/strategies/runner.py", "r") as f:
    content = f.read()

target = "    def on_tick_event(self, tick: dict[str, Any]) -> None:"
replacement = """    async def on_data(self, message: "Message") -> None:
        self.on_tick_event(message.data)

    def on_tick_event(self, tick: dict[str, Any]) -> None:"""

if target in content and "async def on_data" not in content:
    content = content.replace(target, replacement)
    with open("src/nifty_scalper_bot/strategies/runner.py", "w") as f:
        f.write(content)

with open("src/nifty_scalper_bot/data/data_hub.py", "r") as f:
    content = f.read()

if "async def on_tick(" not in content:
    content = content.replace("async def ingest_tick(self, tick: Tick) -> None:",
"""
    async def on_tick(self, message: "Message") -> None:
        await self.ingest_tick(message.data)

    async def ingest_tick(self, tick: Tick) -> None:"""
    )

if "from nifty_scalper_bot.core.message_bus import Message, MessageType" not in content:
    content = content.replace("from nifty_scalper_bot.core.message_bus import MessageBus", "from nifty_scalper_bot.core.message_bus import MessageBus, Message, MessageType")

target = 'self._event_bus.publish("tick", tick)'
replacement = """self._event_bus.publish("tick", tick)
        if getattr(self, "bus", None) is not None:
            try:
                import asyncio
                from nifty_scalper_bot.core.message_bus import Message, MessageType
                msg = Message(
                    type=MessageType.DATA_READY,
                    timestamp=datetime.now(timezone.utc),
                    data=tick,
                    source="data_hub"
                )
                asyncio.create_task(self.bus.publish(msg))
            except Exception as e:
                LOGGER.error("Failed to publish DATA_READY: %s", e)"""
if target in content and "type=MessageType.DATA_READY" not in content:
    content = content.replace(target, replacement)

with open("src/nifty_scalper_bot/data/data_hub.py", "w") as f:
    f.write(content)

with open("src/nifty_scalper_bot/market_data_streamer.py", "r") as f:
    content = f.read()

if "import asyncio" not in content:
    content = content.replace("import time", "import time\nimport asyncio")

if "from nifty_scalper_bot.core.message_bus import Message, MessageType" not in content:
    content = content.replace("from nifty_scalper_bot.data.market_state import MarketState",
                        "from nifty_scalper_bot.data.market_state import MarketState\nfrom nifty_scalper_bot.core.message_bus import Message, MessageType")

if "self.bus = None" not in content:
    content = content.replace("self._last_heartbeat: float | None = None", "self._last_heartbeat: float | None = None\n        self.bus = None")

new_on_ticks = """
    def on_ticks(self, ticks: Sequence[dict[str, Any]]) -> None:
        \"\"\"Update market state from websocket ticks. Args: ticks. Returns: none. Raises: none.\"\"\"

        now = time.time()
        self._last_heartbeat = now
        for tick in ticks:
            try:
                token = int(tick.get("instrument_token"))
                price = float(tick.get("last_price") or tick.get("ltp"))
                if token == 256265:
                    self._market_state.set_spot_ltp(price, source="ws")
                self._market_state.update_tick(token, price, source="ws")

                if self.bus is not None:
                    msg = Message(
                        type=MessageType.TICK,
                        timestamp=now,
                        data={
                            "token": token,
                            "ltp": price,
                            "ts": tick.get("exchange_timestamp", now)
                        },
                        source="market_data_streamer"
                    )
                    asyncio.create_task(self.bus.publish(msg))
            except Exception as e:
                LOGGER.exception("Failure in MarketDataStreamer.on_ticks: %s", e)
"""

if "if self.bus is not None:" not in content:
    import re
    match = re.search(r"    def on_ticks\(self, ticks: Sequence\[dict\[str, Any\]\]\) -> None:[\s\S]*?except Exception as e:[\s]*LOGGER.exception\(\"Failure in MarketDataStreamer.on_ticks: %s\", e\)", content)
    if match:
        content = content.replace(match.group(0), new_on_ticks.strip("\n"))

with open("src/nifty_scalper_bot/market_data_streamer.py", "w") as f:
    f.write(content)

with open("src/nifty_scalper_bot/data/market_data_manager.py", "r") as f:
    content = f.read()

if "self.bus = None" not in content:
    content = content.replace("self._fallback_enabled = False", "self._fallback_enabled = False\n        self.bus = None")

if "from nifty_scalper_bot.core.message_bus import Message, MessageType" not in content:
    lines = content.split('\n')
    for i, l in enumerate(lines):
        if l.startswith("from __future__ import"):
            lines.insert(i+1, "from nifty_scalper_bot.core.message_bus import Message, MessageType")
            break
    content = "\n".join(lines)

target_mdm_ticks = "    def process_ticks(self, ticks: list[dict[str, Any]]) -> None:"
if target_mdm_ticks in content and "Message(" not in content.split("def process_ticks")[1][:500]:
    replacement = """    def process_ticks(self, ticks: list[dict[str, Any]]) -> None:
        if getattr(self, "bus", None) is not None:
            try:
                from nifty_scalper_bot.core.message_bus import Message, MessageType
                import asyncio
                import time
                now = time.time()
                for t in ticks:
                    token = t.get("instrument_token")
                    price = float(t.get("last_price", t.get("ltp", 0.0)))
                    msg = Message(
                        type=MessageType.TICK,
                        timestamp=now,
                        data={
                            "token": token,
                            "ltp": price,
                            "ts": t.get("exchange_timestamp", now)
                        },
                        source="market_data_manager"
                    )
                    asyncio.create_task(self.bus.publish(msg))
            except Exception as e:
                pass"""
    content = content.replace(target_mdm_ticks, replacement)

with open("src/nifty_scalper_bot/data/market_data_manager.py", "w") as f:
    f.write(content)

with open("src/nifty_scalper_bot/core/app.py", "r") as f:
    content = f.read()

content = content.replace("                    ctx.strategy_runner.start()\n", "")

app_target = """                # 🚨 CRITICAL: Start MessageBus AFTER subscribers are registered 🚨
                if ctx.message_bus:
                    LOGGER.info("🚀 Starting MessageBus Dispatchers...")
                    ctx.message_bus.start()"""
app_replacement = """                # 🚨 CRITICAL: Start MessageBus AFTER subscribers are registered 🚨
                if ctx.message_bus and ctx.data_hub and ctx.strategy_runner:
                    ctx.data_hub.bus = ctx.message_bus
                    if getattr(ctx, 'market_data_streamer', None):
                        ctx.market_data_streamer.bus = ctx.message_bus
                    if getattr(ctx, 'market_data_manager', None):
                        ctx.market_data_manager.bus = ctx.message_bus
                    ctx.message_bus.subscribe(MessageType.TICK, ctx.data_hub.on_tick)
                    ctx.message_bus.subscribe(MessageType.DATA_READY, ctx.strategy_runner.on_data)

                if ctx.message_bus:
                    LOGGER.info("🚀 Starting MessageBus Dispatchers...")
                    ctx.message_bus.start()"""
if app_target in content:
    content = content.replace(app_target, app_replacement)

mdm_start_target = """                if ctx.market_data_manager is not None and hasattr(ctx.market_data_manager, "start"):
                    try:
                        ctx.market_data_manager.start()
                        LOGGER.info("✅ MarketDataManager started — tick consumer active")
                    except Exception as _mdm_start_exc:
                        LOGGER.error("MarketDataManager.start() failed: %s", _mdm_start_exc)"""

mdm_start_replacement = """                if ctx.message_bus:
                    LOGGER.info("🚀 Starting MessageBus Dispatchers...")
                    ctx.message_bus.start()
                    LOGGER.info(
                        "✅ MessageBus running with %d active dispatchers",
                        len(ctx.message_bus._tasks),
                    )

                if ctx.market_data_manager is not None and hasattr(ctx.market_data_manager, "start"):
                    try:
                        ctx.market_data_manager.start()
                        LOGGER.info("✅ MarketDataManager started — tick consumer active")
                    except Exception as _mdm_start_exc:
                        LOGGER.error("MarketDataManager.start() failed: %s", _mdm_start_exc)"""

if "if ctx.market_data_manager is not None and hasattr(ctx.market_data_manager, \"start\"):" in content:
    content = content.replace(mdm_start_target, "")
    content = content.replace("""                if ctx.message_bus:
                    LOGGER.info("🚀 Starting MessageBus Dispatchers...")
                    ctx.message_bus.start()
                    LOGGER.info(
                        "✅ MessageBus running with %d active dispatchers",
                        len(ctx.message_bus._tasks),
                    )""", mdm_start_replacement)

with open("src/nifty_scalper_bot/core/app.py", "w") as f:
    f.write(content)
