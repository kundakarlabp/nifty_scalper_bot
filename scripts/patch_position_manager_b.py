from __future__ import annotations

from scripts._execution_patch_utils import assert_parses, method_text, replace_method, replace_once

POSITION = "src/nifty_scalper_bot/execution/position_manager.py"
POLICY = "src/nifty_scalper_bot/execution/execution_policy.py"
AUDIT_TEST = "tests/execution/test_execution_safety_audit_fixes.py"

replace_method(
    POSITION,
    "PositionManager",
    "open_position",
    '''
def open_position(
    self,
    symbol: str,
    side: Side,
    quantity: int,
    entry_price: float,
    stop_loss: float | None = None,
    take_profit: float | None = None,
    trailing_stop_distance: float | None = None,
    order_id: str | None = None,
) -> Position:
    """Open a position under the same lock used by broker reconciliation."""
    symbol_key = symbol.upper()
    position = Position(
        symbol=symbol_key,
        side=_normalize_side(str(side)),
        quantity=int(quantity),
        entry_price=float(entry_price),
        entry_time=_now(),
        current_price=float(entry_price),
        stop_loss=stop_loss,
        take_profit=take_profit,
        trailing_stop_distance=trailing_stop_distance,
        order_id=order_id,
    )
    with self._lock:
        if symbol_key in self._positions:
            raise ValueError(f"Position already exists for {symbol_key}")
        self._positions[symbol_key] = position
    self._logger.info("Opened %s position for %s", position.side, symbol_key)
    self.save_state()
    return position
''',
)

replace_method(
    POSITION,
    "PositionManager",
    "close_position",
    '''
def close_position(
    self,
    symbol: str,
    exit_price: float,
    reason: str,
    close_time: datetime | None = None,
) -> Position:
    """Close a position atomically and retain conservative realised P&L."""
    symbol_key = symbol.upper()
    with self._lock:
        position = self._positions.get(symbol_key)
        if position is None:
            raise ValueError(f"No open position for {symbol_key}")
        qty = position.quantity
        realized = self._calculate_realized_pnl(
            position.side, position.entry_price, float(exit_price), qty
        )
        position.realized_pnl += realized
        self._local_realized_pnl += realized
        self._refresh_realized_pnl_locked()
        position.current_price = float(exit_price)
        position.quantity = 0
        del self._positions[symbol_key]
    closed_at = close_time or _now()
    self._logger.info(
        "Closed %s position for %s at %.2f (%s) due to %s [PnL=%.2f]",
        position.side,
        symbol_key,
        exit_price,
        closed_at.isoformat(),
        reason,
        realized,
    )
    self.clear_active_contract_by_symbol(symbol_key)
    self.save_state()
    return position
''',
)

replace_once(
    POSITION,
    "    def get_realized_pnl(self) -> float:\n",
    method_text('''
def _refresh_realized_pnl_locked(self) -> None:
    """Use the most conservative confirmed P&L for capital-protection gates."""
    candidates = [float(self._local_realized_pnl)]
    if self._broker_realized_pnl is not None:
        candidates.append(float(self._broker_realized_pnl))
    self._daily_realized_pnl = min(candidates)
''') + "    def get_realized_pnl(self) -> float:\n",
)
replace_method(
    POSITION,
    "PositionManager",
    "get_realized_pnl",
    '''
def get_realized_pnl(self) -> float:
    """Return conservative realised P&L used by capital-protection gates."""
    with self._lock:
        return float(self._daily_realized_pnl)
''',
)

replace_once(
    POSITION,
    '                "daily_realized_pnl": self._daily_realized_pnl,\n',
    '                "daily_realized_pnl": self._daily_realized_pnl,\n'
    '                "local_realized_pnl": self._local_realized_pnl,\n'
    '                "broker_realized_pnl": self._broker_realized_pnl,\n',
)
replace_once(
    POSITION,
    '        self._daily_realized_pnl = float(payload.get("daily_realized_pnl", 0.0))\n',
    '        legacy_daily = float(payload.get("daily_realized_pnl", 0.0))\n'
    '        self._local_realized_pnl = float(\n'
    '            payload.get("local_realized_pnl", legacy_daily)\n'
    '        )\n'
    '        broker_realized = payload.get("broker_realized_pnl")\n'
    '        self._broker_realized_pnl = (\n'
    '            None if broker_realized is None else float(broker_realized)\n'
    '        )\n'
    '        with self._lock:\n'
    '            self._refresh_realized_pnl_locked()\n',
)

replace_once(
    POLICY,
    "        base = float(self.max_spread_pct)\n        upper = symbol.upper()\n",
    "        base = float(self.max_spread_pct)\n"
    "        if base == 0.0:\n"
    "            return 0.0\n"
    "        upper = symbol.upper()\n",
)

replace_once(
    AUDIT_TEST,
    "from nifty_scalper_bot.execution import bracket_core\nfrom nifty_scalper_bot.execution.bracket_core import BracketManager, BracketState\n",
    "from nifty_scalper_bot.execution import BracketManager\n"
    "from nifty_scalper_bot.execution import bracket_core\n"
    "from nifty_scalper_bot.execution.bracket_core import BracketState\n",
)
replace_once(
    AUDIT_TEST,
    "    restored = BracketManager(order_manager=SimpleNamespace())\n"
    "    _stop(restored)\n"
    "    restored.load_state()\n",
    "    restored = BracketManager(order_manager=SimpleNamespace())\n"
    "    _stop(restored)\n",
)
replace_once(
    AUDIT_TEST,
    '        ExecutionPolicy(_Hub(quote), max_spread_pct=0.0).build_plan("NSE:NIFTY", "BUY")\n'
    '    plan = ExecutionPolicy(_Hub(quote), max_spread_pct=None).build_plan("NSE:NIFTY", "BUY")\n',
    '        ExecutionPolicy(_Hub(quote), max_spread_pct=0.0).build_plan(SYMBOL, "BUY")\n'
    '    plan = ExecutionPolicy(_Hub(quote), max_spread_pct=None).build_plan(SYMBOL, "BUY")\n',
)

assert_parses(POSITION, POLICY, AUDIT_TEST)
print("patched position manager part B")
