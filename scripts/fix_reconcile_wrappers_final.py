from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


position_path = Path("src/nifty_scalper_bot/execution/position_manager.py")
position = position_path.read_text(encoding="utf-8")
position = replace_once(
    position,
    '''        for column in ("unrealised", "unrealized", "m2m"):
            if column not in row:
                continue
            raw_value = row.get(column)
            numeric = _safe_float(raw_value)
''',
    '''        for column in ("unrealised", "unrealized", "m2m"):
            if column not in row:
                continue
            raw_unrealized_value = row.get(column)
            numeric = _safe_float(raw_unrealized_value)
''',
    "unrealized reconciliation temporary",
)
position = replace_once(
    position,
    '''        for column in ("realised", "realized"):
            if column not in row:
                continue
            raw_value = row.get(column)
            numeric = _safe_float(raw_value)
''',
    '''        for column in ("realised", "realized"):
            if column not in row:
                continue
            raw_realized_value = row.get(column)
            numeric = _safe_float(raw_realized_value)
''',
    "realized reconciliation temporary",
)
position_path.write_text(position, encoding="utf-8")

identity_path = Path("src/nifty_scalper_bot/execution/position_identity_extension.py")
identity = identity_path.read_text(encoding="utf-8")
identity = replace_once(
    identity,
    "import inspect\nimport threading\n",
    "import inspect\n",
    "identity threading import",
)
identity = replace_once(
    identity,
    '''        self._single_reconcile_lock = threading.Lock()
        self._single_reconcile_generation = 0
        self._single_reconcile_coalesced = 0
        self._cost_basis_unresolved_symbols = set()
''',
    '''        self._cost_basis_unresolved_symbols = set()
''',
    "identity duplicate reconcile state",
)
identity = replace_once(
    identity,
    '''    def reconcile_now(self: Any) -> bool:
        lock = getattr(self, "_single_reconcile_lock", None)
        if lock is None:
            self._single_reconcile_lock = threading.Lock()
            lock = self._single_reconcile_lock
        if not lock.acquire(False):
            self._single_reconcile_coalesced = int(getattr(self, "_single_reconcile_coalesced", 0)) + 1
            return bool(getattr(self, "_last_reconcile_success_at", None))
        try:
            self._single_reconcile_generation = int(getattr(self, "_single_reconcile_generation", 0)) + 1
            return bool(_ORIGINALS["PositionManager.reconcile_now"](self))
        finally:
            lock.release()
''',
    '''    def reconcile_now(self: Any) -> bool:
        return bool(_ORIGINALS["PositionManager.reconcile_now"](self))
''',
    "identity reconcile wrapper",
)
identity = replace_once(
    identity,
    '''    def synchronize_with_broker(self: Any, broker_positions: Any) -> Any:
        prepared, unresolved = _prepare_broker_positions(self, broker_positions)
        self._cost_basis_unresolved_symbols = set(unresolved)
        if unresolved and isinstance(prepared, list):
            prepared = [row for row in prepared if _prepared_row_symbol(row) not in unresolved]
        result = _ORIGINALS["PositionManager.synchronize_with_broker"](
            self,
            prepared,
        )
        _canonicalize_position_store(self)
        return result
''',
    '''    def synchronize_with_broker(
        self: Any,
        broker_positions: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        prepared, unresolved = _prepare_broker_positions(self, broker_positions)
        self._cost_basis_unresolved_symbols = set(unresolved)
        if unresolved and isinstance(prepared, list):
            prepared = [row for row in prepared if _prepared_row_symbol(row) not in unresolved]
        result = _ORIGINALS["PositionManager.synchronize_with_broker"](
            self,
            prepared,
            *args,
            **kwargs,
        )
        _canonicalize_position_store(self)
        return result
''',
    "identity synchronize wrapper",
)
identity_path.write_text(identity, encoding="utf-8")

quarantine_path = Path(
    "src/nifty_scalper_bot/execution/broker_exposure_quarantine_extension.py"
)
quarantine = quarantine_path.read_text(encoding="utf-8")
quarantine = replace_once(
    quarantine,
    '''    def synchronize_with_broker(self: Any, broker_positions: Any) -> Any:
        prepared, unresolved = _position_identity._prepare_broker_positions(self, broker_positions)
        self._quarantined_broker_exposures = _build_exposures(prepared, set(unresolved))
        return _ORIGINALS["PositionManager.synchronize_with_broker"](self, broker_positions)
''',
    '''    def synchronize_with_broker(
        self: Any,
        broker_positions: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        prepared, unresolved = _position_identity._prepare_broker_positions(
            self, broker_positions
        )
        self._quarantined_broker_exposures = _build_exposures(
            prepared, set(unresolved)
        )
        return _ORIGINALS["PositionManager.synchronize_with_broker"](
            self,
            broker_positions,
            *args,
            **kwargs,
        )
''',
    "quarantine synchronize wrapper",
)
quarantine_path.write_text(quarantine, encoding="utf-8")
