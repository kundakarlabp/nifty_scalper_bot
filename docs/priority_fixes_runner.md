# Priority Fixes for `src/nifty_scalper_bot/strategies/runner.py`

This document details critical fixes required in the `_handle_entry_signal` method of `src/nifty_scalper_bot/strategies/runner.py` to address production-grade issues.

## 1. VWAP Check (The Strategy Killer)

**Issue:** The hardcoded VWAP check blocks strategies like SMC and Gamma that may legitimately trade against the VWAP.
**Fix:** Allow strategies to explicitly bypass this check via metadata.

### Pre-Correction
```python
        # Check VWAP filter
        current_vwap = None
        with self._lock:
            if state := self._symbol_state.get(base_symbol):
                current_vwap = state.vwap
        
        if current_vwap and current_vwap > 0:
            # ... existing VWAP logic ...
```

### Post-Correction
```python
        # ===========================================================
        # ✅ FIX 1: SMART VWAP FILTER (Unshackles Elite Strategies)
        # ===========================================================
        # Allow strategies to explicitly bypass VWAP check via metadata
        should_check_vwap = True
        if signal.metadata and signal.metadata.get("ignore_vwap"):
            should_check_vwap = False
            self._logger.debug(f"ℹ️ VWAP Check Bypassed by Strategy: {signal.strategy_name}")

        current_vwap = None
        with self._lock:
            if state := self._symbol_state.get(base_symbol):
                current_vwap = state.vwap
        
        # Only block if Strategy did NOT opt-out
        if should_check_vwap and current_vwap and current_vwap > 0:
             # ... existing VWAP logic ...
```

## 2. Infinite Loop (The Timer Update)

**Issue:** The signal timer update (`last_signal_at`) is inside the success block. If an order fails, the bot retries instantly, causing an infinite loop.
**Fix:** Move the timer update *outside* the order success block to ensure a cooldown regardless of success/failure.

### Pre-Correction
```python
            if order_id:
                # ...
                with self._lock:
                    state = self._symbol_state.get(base_symbol)
                    if state: 
                        state.last_signal_at = timestamp
```

### Post-Correction
```python
            # Execute Order
            order_id = self._order_manager.place_order(...)

            # ✅ CRITICAL FIX: Always update timer OUTSIDE the success block
            # This stops the infinite retry loop immediately.
            with self._lock:
                state = self._symbol_state.get(base_symbol)
                if state: 
                    state.last_signal_at = timestamp
                    # Also debounce the specific option symbol
                    if trade_symbol != base_symbol:
                        opt_state = self._symbol_state.get(trade_symbol)
                        if opt_state: opt_state.last_signal_at = timestamp

            if order_id:
                # ... (rest of success logic)
```

## 3. Traceability (Unique Tag)

**Issue:** Orders are sent without a unique identifier linking them to the signal, making debugging difficult.
**Fix:** Generate a unique tag and pass it to `place_order`.

### Pre-Correction
```python
            # Execute Order
            order_id = self._order_manager.place_order(
                symbol=trade_symbol,
                side=entry_side,
                # ...
                signal_id=signal.id
            )
```

### Post-Correction
```python
            # Generate Unique ID for Idempotency
            unique_tag = f"{signal.strategy_name[:3]}_{int(timestamp.timestamp())}"

            # Execute Order
            order_id = self._order_manager.place_order(
                symbol=trade_symbol,
                side=entry_side,
                # ...
                signal_id=signal.id,
                tag=unique_tag  # <--- Added unique tag
            )
```
