# src/nifty_scalper_bot/execution/adaptive_trailing.py

class AdaptiveTrailingController:
    """
    World-class ATR-based trailing stop with:
    - Volatility regime detection
    - Data staleness protection
    - Graceful degradation
    """
    
    def __init__(
        self,
        symbol: str,
        side: Literal["LONG", "SHORT"],
        entry: float,
        sl_order_id: str,
        variety: str,
        spec: TrailingSpec,
        get_ltp: Callable[[str], float | None],
        modify_order: Callable,
        atr_provider: SafeATRProvider,  # ✅ NEW: Validated ATR source
        journal: AtomicKV,
    ):
        self.symbol = symbol
        self.side = side
        self.entry_price = entry
        self.current_sl = entry  # Will be updated
        self.spec = spec
        self._get_ltp = get_ltp
        self._modify = modify_order
        self._atr = atr_provider  # ✅ NEW
        self._journal = journal
        self._logger = get_logger(__name__)
        
        # State tracking
        self.trailing_active = False
        self.highest_price = entry if side == "LONG" else entry  # Best price seen
        self.lowest_price = entry if side == "SHORT" else entry
        self.last_update_time = time()
        self.update_count = 0
        self.failed_modifications = 0  # ✅ NEW: Track failures
        
        # Emergency halt flags
        self._halted = False
        self._halt_reason: str | None = None
    
    def on_tick(self, tick: dict | None) -> None:
        """Process price tick and update trailing stop if needed"""
        
        # ✅ 1. SAFETY: Check if halted
        if self._halted:
            return
        
        # ✅ 2. VALIDATE LTP
        ltp = self._get_ltp(self.symbol)
        if ltp is None or ltp <= 0:
            self._logger.warning(
                f"⚠️ Invalid LTP for {self.symbol}, skipping trailing update",
                extra={"event": "trailing_ltp_invalid", "ltp": ltp}
            )
            return
        
        # ✅ 3. CHECK ACTIVATION
        profit_pct = self._calculate_profit_pct(ltp)
        if not self.trailing_active:
            if profit_pct >= self.spec.activation:
                self.trailing_active = True
                self._logger.info(
                    f"🚀 Trailing stop ACTIVATED for {self.symbol} at {profit_pct:.2f}% profit",
                    extra={"event": "trailing_activated", "profit_pct": profit_pct}
                )
            else:
                return  # Not profitable enough yet
        
        # ✅ 4. FETCH ATR WITH VALIDATION
        atr_snapshot = self._atr.get_atr(
            self.symbol, 
            fallback=self.spec.trail_by  # Static fallback
        )
        
        if atr_snapshot is None:
            self._emergency_halt("ATR unavailable and no fallback")
            return
        
        if not atr_snapshot.is_fresh(max_age_sec=60.0):
            self._emergency_halt(f"Stale ATR (age: {atr_snapshot.age_seconds:.1f}s)")
            return
        
        # ✅ 5. CALCULATE DYNAMIC TRAIL DISTANCE
        trail_distance = self._calculate_trail_distance(atr_snapshot, ltp)
        
        # ✅ 6. UPDATE STOP LOSS IF NEEDED
        new_sl = self._calculate_new_sl(ltp, trail_distance)
        
        if self._should_update_sl(new_sl):
            success = self._execute_sl_update(new_sl)
            
            if not success:
                self.failed_modifications += 1
                if self.failed_modifications >= 3:
                    self._emergency_halt("3 consecutive SL modification failures")
    
    def _calculate_trail_distance(self, atr: ATRSnapshot, ltp: float) -> float:
        """
        Calculate dynamic trail distance based on ATR and volatility regime.
        
        Strategy:
        - Low Volatility: Tighter stops (1.5x ATR)
        - Normal: Standard (2.0x ATR)
        - High Volatility: Wider stops (3.0x ATR)
        """
        base_atr = atr.value
        
        # ✅ Detect volatility regime
        atr_pct_of_price = (base_atr / ltp) * 100  # ATR as % of price
        
        if atr_pct_of_price < 1.0:
            # Low volatility - Tighten
            multiplier = 1.5
            regime = "low_vol"
        elif atr_pct_of_price > 3.0:
            # High volatility - Widen
            multiplier = 3.0
            regime = "high_vol"
        else:
            # Normal
            multiplier = 2.0
            regime = "normal"
        
        distance = base_atr * multiplier
        
        self._logger.debug(
            f"Trail calc: ATR={base_atr:.2f}, Regime={regime}, Distance={distance:.2f}",
            extra={
                "event": "trail_calculation",
                "atr": base_atr,
                "regime": regime,
                "distance": distance
            }
        )
        
        return distance
    
    def _calculate_new_sl(self, ltp: float, trail_distance: float) -> float:
        """Calculate new stop loss based on current price and trail distance"""
        if self.side == "LONG":
            return ltp - trail_distance
        else:  # SHORT
            return ltp + trail_distance
    
    def _should_update_sl(self, new_sl: float) -> bool:
        """
        Check if SL should be updated based on:
        1. Direction correctness (only move in favorable direction)
        2. Minimum step requirement
        """
        if self.side == "LONG":
            # For longs, SL should only move UP
            if new_sl <= self.current_sl:
                return False
            
            improvement = new_sl - self.current_sl
        else:  # SHORT
            # For shorts, SL should only move DOWN
            if new_sl >= self.current_sl:
                return False
            
            improvement = self.current_sl - new_sl
        
        # Check minimum step
        if improvement < self.spec.step:
            return False
        
        return True
    
    def _execute_sl_update(self, new_sl: float) -> bool:
        """
        Execute stop loss modification with state persistence.
        
        Returns:
            bool: True if successful, False if failed
        """
        old_sl = self.current_sl
        
        try:
            # ✅ Round to valid tick size
            new_sl_rounded = round(new_sl, 1)  # Adjust based on instrument
            
            # ✅ Attempt modification
            result = self._modify(
                var=self.variety,
                order_id=self.sl_order_id,
                qty=None,  # Don't change quantity
                price=new_sl_rounded
            )
            
            if result:
                # ✅ Update internal state ONLY after confirmation
                self.current_sl = new_sl_rounded
                self.last_update_time = time()
                self.update_count += 1
                self.failed_modifications = 0  # Reset failure counter
                
                # ✅ Persist to journal (for crash recovery)
                self._journal.set(self.sl_order_id, {
                    "current_sl": new_sl_rounded,
                    "last_update": time(),
                    "update_count": self.update_count
                })
                
                self._logger.info(
                    f"✅ Trailing SL updated: {old_sl:.2f} → {new_sl_rounded:.2f}",
                    extra={
                        "event": "trailing_sl_updated",
                        "symbol": self.symbol,
                        "old_sl": old_sl,
                        "new_sl": new_sl_rounded,
                        "update_count": self.update_count
                    }
                )
                return True
            else:
                self._logger.error(
                    f"❌ SL modification returned False/None",
                    extra={"event": "sl_mod_null_response", "symbol": self.symbol}
                )
                return False
                
        except Exception as exc:
            self._logger.error(
                f"❌ SL modification failed: {exc}",
                extra={"event": "sl_mod_exception", "symbol": self.symbol, "error": str(exc)},
                exc_info=True
            )
            return False
    
    def _emergency_halt(self, reason: str) -> None:
        """
        Stop trailing updates and mark for manual intervention.
        """
        if self._halted:
            return  # Already halted
        
        self._halted = True
        self._halt_reason = reason
        
        self._logger.critical(
            f"🚨 EMERGENCY HALT: Trailing stopped for {self.symbol}. Reason: {reason}",
            extra={
                "event": "trailing_emergency_halt",
                "symbol": self.symbol,
                "reason": reason,
                "current_sl": self.current_sl
            }
        )
        
        # ✅ Persist halt state
        self._journal.set(f"{self.sl_order_id}_halted", {
            "halted": True,
            "reason": reason,
            "timestamp": time()
        })
    
    def _calculate_profit_pct(self, ltp: float) -> float:
        """Calculate current profit percentage"""
        if self.side == "LONG":
            return ((ltp - self.entry_price) / self.entry_price) * 100
        else:
            return ((self.entry_price - ltp) / self.entry_price) * 100
