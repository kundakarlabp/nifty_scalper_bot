async def startup_sequence(ctx: BotContext) -> None:
    """Execute startup sequence with Smart Hydration and Option-Only Trading."""
    LOGGER.info("Starting Nifty Scalper Bot...")

    # ✅ FIX: Create data directory immediately
    import os
    try:
        os.makedirs("data", exist_ok=True)
        LOGGER.info("✅ Verified/Created 'data/' directory.")
    except Exception as e:
        LOGGER.critical(f"❌ Failed to create data directory: {e}")
    
    _validate_config(ctx.config)
    broker_ready = True
    guard = ctx.session_guard

    # [FIX 1] Define _notify helper locally
    async def _notify(event: str, payload: Mapping[str, object] | None = None) -> None:
        notifier = ctx.telegram_notifier
        if notifier is None:
            return
        try:
            await notifier.send_event(event, payload)
        except Exception:
            LOGGER.debug("Startup notifier failed", exc_info=True)

    # 1. Validate Broker & Session
    try:
        broker_proxy = getattr(ctx.broker_client, "_broker", getattr(ctx.broker_client, "broker", ctx.broker_client))
        if hasattr(broker_proxy, "get_profile"):
             broker_proxy.get_profile()
        LOGGER.info("✅ Broker Connection: ACTIVE")
    except Exception as exc:
        LOGGER.error(f"❌ Broker Connection Failed: {exc}")
        broker_ready = False

    # 2. Market Status Check
    try:
        if guard:
            status = guard.evaluate()
            LOGGER.info(f"🕒 Market Status: {'OPEN' if status.market_open else 'CLOSED'} | Session Valid: {status.session_valid}")
            
            if not status.market_open and not ctx.out_of_hours_override:
                LOGGER.warning("⚠️ Market Closed. Live trading will be BLOCKED unless override is active.")
    except Exception:
        pass

    # 3. Smart State Hydration (The "Amnesia" Fix)
    # We must ensure we have a valid state BEFORE we start processing ticks.
    LOGGER.info("🧠 Hydrating State (Positions & Orders)...")
    
    try:
         # 3.1 Load Local State
         if ctx.persistent_state:
             ctx.persistent_state.load_state()
             
         # 3.2 Reconcile with Broker (Source of Truth)
         if broker_ready:
             await reconcile_positions_on_startup(
                 ctx.broker_client, 
                 ctx.position_manager, 
                 ctx.order_manager, 
                 LOGGER
             )
             
         # 3.3 Auto-Guard Activation (Missing Link)
         # Scan all open positions and ensure they have Brackets
         if ctx.position_manager and ctx.bracket_manager:
             open_positions = ctx.position_manager.get_open_positions()
             for pos in open_positions:
                 # Check if bracket exists, if not -> Create Emergency Guard
                 if not ctx.bracket_manager.get_bracket_order(pos.symbol):
                     LOGGER.warning(f"🛡️ Found UNGUARDED Position {pos.symbol}. engaging Auto-Guard...")
                     ctx.bracket_manager.create_emergency_bracket(
                         symbol=pos.symbol,
                         quantity=abs(pos.quantity),
                         entry_price=pos.entry_price,
                         side=pos.side
                     )
    except Exception as exc:
        LOGGER.error(f"❌ State Hydration Failed: {exc}", exc_info=True)

    # 4. Notify Telegram
    await _notify("BOT_STARTED", {
        "mode": "PAPER" if ctx.shadow_mode_enabled else "LIVE",
        "market": "OPEN" if guard and guard._is_market_open(datetime.now(timezone.utc)) else "CLOSED",
        "positions": len(ctx.position_manager.get_open_positions()) if ctx.position_manager else 0
    })

    LOGGER.info("✅ Startup Sequence Complete. Bot is Ready.")