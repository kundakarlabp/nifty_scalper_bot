from __future__ import annotations

import re
from pathlib import Path

ROOT = Path.cwd()


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def write(path: str, text: str) -> None:
    (ROOT / path).write_text(text, encoding="utf-8")


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, found {count}")
    return text.replace(old, new, 1)


def regex_once(
    text: str,
    pattern: str,
    replacement: str,
    *,
    label: str,
    flags: int = 0,
) -> str:
    updated, count = re.subn(pattern, replacement, text, count=1, flags=flags)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one regex match, found {count}")
    return updated


# ---------------------------------------------------------------------------
# PositionManager becomes the single native owner of broker-position identity
# preparation, lifecycle preservation, and position-reconcile single-flight.
# ---------------------------------------------------------------------------
path = "src/nifty_scalper_bot/execution/position_manager.py"
text = read(path)
text = replace_once(
    text,
    """from nifty_scalper_bot.execution.position_snapshot import (\n    BrokerExposureState,\n    PositionSnapshotError,\n    decode_position_snapshot,\n)\n""",
    """from nifty_scalper_bot.execution.position_reconciliation_identity import (\n    _canonicalize_position_store,\n    _prepare_broker_positions,\n    _prepared_row_symbol,\n    _restore_owned_position_lifecycle,\n    _snapshot_owned_position_lifecycle,\n)\nfrom nifty_scalper_bot.execution.position_snapshot import (\n    BrokerExposureState,\n    PositionSnapshotError,\n    decode_position_snapshot,\n)\n""",
    label="native reconciliation helper imports",
)
text = replace_once(
    text,
    """        self._broker_client: Any | None = None\n        self._reconcile_timer: threading.Timer | None = None\n        self._reconcile_interval_s: float = 60.0\n""",
    """        self._broker_client: Any | None = None\n        self._reconcile_timer: threading.Timer | None = None\n        self._single_reconcile_lock = threading.Lock()\n        self._single_reconcile_generation = 0\n        self._single_reconcile_coalesced = 0\n        self._cost_basis_unresolved_symbols: set[str] = set()\n        self._reconcile_interval_s: float = 60.0\n""",
    label="native reconciliation runtime state",
)
text = replace_once(
    text,
    """    def reconcile_now(self) -> bool:\n        \"\"\"Fetch and atomically apply one authoritative broker snapshot.\"\"\"\n        payload_count = 0\n""",
    """    def reconcile_now(self) -> bool:\n        \"\"\"Fetch and apply one authoritative broker-position snapshot, single-flight.\"\"\"\n        lock = getattr(self, \"_single_reconcile_lock\", None)\n        if lock is None:\n            self._single_reconcile_lock = threading.Lock()\n            lock = self._single_reconcile_lock\n        if not lock.acquire(False):\n            self._single_reconcile_coalesced = int(\n                getattr(self, \"_single_reconcile_coalesced\", 0)\n            ) + 1\n            return bool(getattr(self, \"_last_reconcile_success_at\", None))\n        try:\n            self._single_reconcile_generation = int(\n                getattr(self, \"_single_reconcile_generation\", 0)\n            ) + 1\n            return bool(self._reconcile_positions_from_broker())\n        finally:\n            lock.release()\n\n    def _reconcile_positions_from_broker(self) -> bool:\n        \"\"\"Fetch and atomically apply one authoritative broker snapshot.\"\"\"\n        payload_count = 0\n""",
    label="native single-flight reconcile owner",
)
text = replace_once(
    text,
    """    def synchronize_with_broker(\n        self, broker_positions: Sequence[Mapping[str, object]]\n    ) -> None:\n        \"\"\"Validate and atomically replace managed positions from broker truth.\"\"\"\n        try:\n""",
    """    def synchronize_with_broker(self, broker_positions: Any) -> None:\n        \"\"\"Canonicalize broker truth and preserve bot-owned lifecycle identity.\"\"\"\n        lifecycle_snapshot = _snapshot_owned_position_lifecycle(self)\n        prepared, unresolved = _prepare_broker_positions(self, broker_positions)\n        self._cost_basis_unresolved_symbols = set(unresolved)\n        if unresolved and isinstance(prepared, list):\n            prepared = [\n                row\n                for row in prepared\n                if _prepared_row_symbol(row) not in unresolved\n            ]\n        self._synchronize_managed_positions_from_broker(prepared)\n        _canonicalize_position_store(self)\n        restored = _restore_owned_position_lifecycle(self, lifecycle_snapshot)\n        if restored:\n            self.save_state()\n\n    def _synchronize_managed_positions_from_broker(\n        self, broker_positions: Any\n    ) -> None:\n        \"\"\"Validate and atomically replace managed positions from broker truth.\"\"\"\n        try:\n""",
    label="native broker identity sync owner",
)
write(path, text)


# ---------------------------------------------------------------------------
# The identity extension remains a compatibility overlay only for ingress
# methods not yet migrated. Re-export the pure helpers for existing callers.
# ---------------------------------------------------------------------------
path = "src/nifty_scalper_bot/execution/position_identity_extension.py"
text = read(path)
text = replace_once(
    text,
    """\"\"\"Canonical PositionManager ingress patch for broker and pending-order paths.\n\nFollow-up scope: broker reconciliation ownership and orphan protection are handled\nby the loaded runtime guards in this module.\n\"\"\"\n""",
    """\"\"\"Compatibility overlay for remaining PositionManager ingress guards.\n\nBroker-position reconciliation identity, cost-basis preparation, lifecycle\npreservation, and single-flight ownership now live natively in PositionManager.\n\"\"\"\n""",
    label="identity overlay docstring",
)
text = replace_once(text, "import threading\n", "", label="remove threading import")
text = replace_once(
    text,
    """from nifty_scalper_bot.execution import position_manager as _position_manager\nfrom nifty_scalper_bot.execution.position_snapshot import (\n""",
    """from nifty_scalper_bot.execution import position_manager as _position_manager\nfrom nifty_scalper_bot.execution.position_reconciliation_identity import (\n    _canonical_key,\n    _canonicalize_broker_positions,\n    _canonicalize_payload_symbol,\n    _canonicalize_position_store,\n    _prepare_broker_positions,\n    _prepared_row_symbol,\n    _restore_owned_position_lifecycle,\n    _snapshot_owned_position_lifecycle,\n)\nfrom nifty_scalper_bot.execution.position_snapshot import (\n""",
    label="identity pure helper imports",
)
text = replace_once(
    text,
    "from nifty_scalper_bot.utils.symbols import normalize_symbol\n",
    "",
    label="remove symbol helper import",
)
text = regex_once(
    text,
    r"_SYMBOL_FIELDS = .*?\n\n\ndef _restore_persistent_state_methods",
    "_QUARANTINE_INTENTS = {\n    \"\",\n    \"UNKNOWN\",\n    \"BROKER_IMPORTED_ORDER\",\n    \"MANUAL_ORDER_QUARANTINED\",\n}\n\n\ndef _restore_persistent_state_methods",
    label="remove duplicated reconciliation helpers",
    flags=re.S,
)
text = replace_once(
    text,
    """        \"__init__\",\n        \"_symbol_lifecycle_lock_for\",\n        \"reconcile_now\",\n        \"add_pending_order\",\n        \"get_pending_orders\",\n        \"synchronize_with_broker\",\n        \"apply_broker_order_update\",\n""",
    """        \"_symbol_lifecycle_lock_for\",\n        \"add_pending_order\",\n        \"get_pending_orders\",\n        \"apply_broker_order_update\",\n""",
    label="identity captured method ownership",
)
text = regex_once(
    text,
    r"    def __init__\(self: Any, \*args: Any, \*\*kwargs: Any\) -> None:\n.*?\n    def _symbol_lifecycle_lock_for",
    "    def _symbol_lifecycle_lock_for",
    label="remove identity init wrapper",
    flags=re.S,
)
text = regex_once(
    text,
    r"    def reconcile_now\(self: Any\) -> bool:\n.*?\n    def add_pending_order",
    "    def add_pending_order",
    label="remove identity reconcile wrapper",
    flags=re.S,
)
text = regex_once(
    text,
    r"    def synchronize_with_broker\(self: Any, broker_positions: Any\) -> Any:\n.*?\n    def apply_broker_order_update",
    "    def apply_broker_order_update",
    label="remove identity synchronize wrapper",
    flags=re.S,
)
for old, label in (
    (
        """    if \"PositionManager.__init__\" in _ORIGINALS:\n        cls.__init__ = __init__\n""",
        "remove identity init assignment",
    ),
    (
        """    if \"PositionManager.reconcile_now\" in _ORIGINALS:\n        cls.reconcile_now = reconcile_now\n""",
        "remove identity reconcile assignment",
    ),
    (
        """    if \"PositionManager.synchronize_with_broker\" in _ORIGINALS:\n        cls.synchronize_with_broker = synchronize_with_broker\n""",
        "remove identity synchronize assignment",
    ),
):
    text = replace_once(text, old, "", label=label)
write(path, text)


# Ownership guard: the compatibility overlay must not retain hidden definitions
# of the methods migrated above.
overlay = read(path)
for forbidden in (
    "def reconcile_now(self: Any)",
    "def synchronize_with_broker(self: Any, broker_positions: Any)",
    "cls.reconcile_now = reconcile_now",
    "cls.synchronize_with_broker = synchronize_with_broker",
    "_single_reconcile_lock",
):
    if forbidden in overlay:
        raise RuntimeError(f"identity overlay still owns migrated behavior: {forbidden}")
