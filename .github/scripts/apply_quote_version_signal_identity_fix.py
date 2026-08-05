from __future__ import annotations

from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text()
    if old not in text:
        raise SystemExit(f"expected patch anchor missing: {path}: {old[:120]!r}")
    target.write_text(text.replace(old, new, 1))


Path("src/nifty_scalper_bot/strategies/quote_update_identity.py").write_text(
    '''"""Authoritative quote-version and evaluation-snapshot identity helpers."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping

_QUOTE_VERSION_KEYS = (
    "quote_update_version",
    "update_version",
    "tick_version",
)


def coerce_quote_update_version(value: Any) -> int | None:
    """Return a positive integer quote version, otherwise ``None``."""
    try:
        version = int(float(value))
    except (TypeError, ValueError, OverflowError):
        return None
    return version if version > 0 else None


def resolve_quote_update_identity(
    *sources: tuple[str, Mapping[str, Any] | None],
) -> tuple[int | None, str | None]:
    """Resolve the first authoritative quote version and its provenance."""
    for source_name, payload in sources:
        if not isinstance(payload, Mapping):
            continue
        for key in _QUOTE_VERSION_KEYS:
            version = coerce_quote_update_version(payload.get(key))
            if version is not None:
                return version, f"{source_name}:{key}"
    return None, None


def build_evaluation_snapshot_id(
    setup_signal_id: str, quote_update_version: int | None
) -> str | None:
    """Build exact-evaluation identity without changing setup idempotency."""
    version = coerce_quote_update_version(quote_update_version)
    setup_id = str(setup_signal_id or "").strip()
    if not setup_id or version is None:
        return None
    raw = f"{setup_id}:quote_update_version:{version}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]


__all__ = [
    "build_evaluation_snapshot_id",
    "coerce_quote_update_version",
    "resolve_quote_update_identity",
]
'''
)

replace_once(
    "src/nifty_scalper_bot/strategies/runner.py",
    "from nifty_scalper_bot.strategies.signal_generator import Signal\n",
    "from nifty_scalper_bot.strategies.signal_generator import Signal\n"
    "from nifty_scalper_bot.strategies.quote_update_identity import (\n"
    "    resolve_quote_update_identity,\n"
    ")\n",
)

replace_once(
    "src/nifty_scalper_bot/strategies/runner.py",
    '''                        tradable_quote = bool(quote_map.get("tradable_quote"))
                        if (
                            not tradable_quote
                            and bid_f is not None
                            and ask_f is not None
                        ):
                            tradable_quote = ask_f > bid_f
                        runtime_ctx.update(
''',
    '''                        tradable_quote = bool(quote_map.get("tradable_quote"))
                        if (
                            not tradable_quote
                            and bid_f is not None
                            and ask_f is not None
                        ):
                            tradable_quote = ask_f > bid_f
                        quote_update_version, quote_update_version_source = (
                            resolve_quote_update_identity(
                                ("datahub_quote", quote_map),
                                ("runner_tick", tick_map),
                                (
                                    "runner_counter",
                                    {
                                        "quote_update_version": self._quote_update_version_for_eval(
                                            symbol
                                        )
                                    },
                                ),
                            )
                        )
                        runtime_ctx.update(
''',
)

replace_once(
    "src/nifty_scalper_bot/strategies/runner.py",
    '''                                "quote_age_s": quote_map.get("quote_age_s")
                                or quote_map.get("data_age_seconds")
                                or tick_map.get("data_age_seconds"),
                            }
''',
    '''                                "quote_age_s": quote_map.get("quote_age_s")
                                or quote_map.get("data_age_seconds")
                                or tick_map.get("data_age_seconds"),
                                "quote_update_version": quote_update_version,
                                "quote_update_version_source": quote_update_version_source,
                            }
''',
)

replace_once(
    "src/nifty_scalper_bot/strategies/runtime_context_contract.py",
    '''        "quote_update_version",
        "update_version",
''',
    '''        "quote_update_version",
        "quote_update_version_source",
        "update_version",
''',
)

replace_once(
    "src/nifty_scalper_bot/strategies/signal_identity_patch.py",
    '''from typing import Any, Mapping

from nifty_scalper_bot.utils.logging import get_logger
''',
    '''from typing import Any, Mapping

from nifty_scalper_bot.strategies.quote_update_identity import (
    build_evaluation_snapshot_id,
    resolve_quote_update_identity,
)
from nifty_scalper_bot.utils.logging import get_logger
''',
)

replace_once(
    "src/nifty_scalper_bot/strategies/signal_identity_patch.py",
    '''def _deterministic_id(signal: Any) -> str:
    metadata = dict(getattr(signal, "metadata", {}) or {})
    strategy = str(metadata.get("strategy_name") or metadata.get("strategy") or "manual")
    underlying, option_side = _option_thesis(getattr(signal, "symbol", ""), metadata)
    setup_anchor = _anchor(metadata)
    action = str(getattr(signal, "action", ""))
    raw = f"{strategy}:{underlying}:{option_side}:{action}:{setup_anchor}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]


def _install_elite_signal_observability() -> None:
''',
    '''def _deterministic_id(signal: Any) -> str:
    metadata = dict(getattr(signal, "metadata", {}) or {})
    strategy = str(metadata.get("strategy_name") or metadata.get("strategy") or "manual")
    underlying, option_side = _option_thesis(getattr(signal, "symbol", ""), metadata)
    setup_anchor = _anchor(metadata)
    action = str(getattr(signal, "action", ""))
    raw = f"{strategy}:{underlying}:{option_side}:{action}:{setup_anchor}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]


def _stamp_evaluation_identity(
    signal: Any, indicators: Mapping[str, Any]
) -> Any:
    """Attach exact quote-snapshot identity while preserving setup identity."""
    metadata = dict(getattr(signal, "metadata", {}) or {})
    version, resolved_source = resolve_quote_update_identity(
        ("signal_metadata", metadata),
        ("indicator_context", indicators),
    )
    if version is None:
        return signal
    setup_signal_id = _deterministic_id(signal)
    evaluation_snapshot_id = build_evaluation_snapshot_id(setup_signal_id, version)
    source = str(metadata.get("quote_update_version_source") or resolved_source or "")
    updates = {
        "quote_update_version": version,
        "quote_update_version_source": source or None,
        "setup_signal_id": setup_signal_id,
        "evaluation_snapshot_id": evaluation_snapshot_id,
    }
    with_metadata = getattr(signal, "with_metadata", None)
    if callable(with_metadata):
        return with_metadata(**updates)
    mutable = getattr(signal, "metadata", None)
    if isinstance(mutable, dict):
        mutable.update(updates)
    return signal


def _install_elite_signal_observability() -> None:
''',
)

replace_once(
    "src/nifty_scalper_bot/strategies/signal_identity_patch.py",
    '''        signal = current(self, symbol, indicators, current_price, position)
        if signal is None:
            return None
        metadata = dict(getattr(signal, "metadata", {}) or {})
''',
    '''        signal = current(self, symbol, indicators, current_price, position)
        if signal is None:
            return None
        signal = _stamp_evaluation_identity(signal, indicators)
        metadata = dict(getattr(signal, "metadata", {}) or {})
''',
)

replace_once(
    "src/nifty_scalper_bot/strategies/signal_identity_patch.py",
    '''            "ELITE_SIGNAL_GENERATED strategy=%s symbol=%s side=%s raw_setup_score=%s confidence=%s setup_id=%s setup_anchor=%s quote_update_version=%s",
''',
    '''            "ELITE_SIGNAL_GENERATED strategy=%s symbol=%s side=%s raw_setup_score=%s confidence=%s setup_id=%s setup_anchor=%s quote_update_version=%s evaluation_snapshot_id=%s",
''',
)

replace_once(
    "src/nifty_scalper_bot/strategies/signal_identity_patch.py",
    '''            metadata.get("quote_update_version"),
            extra={
''',
    '''            metadata.get("quote_update_version"),
            metadata.get("evaluation_snapshot_id"),
            extra={
''',
)

replace_once(
    "src/nifty_scalper_bot/strategies/signal_identity_patch.py",
    '''                "quote_update_version": metadata.get("quote_update_version"),
                "role": role,
''',
    '''                "quote_update_version": metadata.get("quote_update_version"),
                "quote_update_version_source": metadata.get(
                    "quote_update_version_source"
                ),
                "setup_signal_id": metadata.get("setup_signal_id"),
                "evaluation_snapshot_id": metadata.get("evaluation_snapshot_id"),
                "role": role,
''',
)

replace_once(
    "src/nifty_scalper_bot/strategies/signal_identity_patch.py",
    '''    "_deterministic_id",
    "_option_thesis",
''',
    '''    "_deterministic_id",
    "_option_thesis",
    "_stamp_evaluation_identity",
''',
)

replace_once(
    "tests/strategies/test_signal_observability_contract.py",
    '''                "setup_id": "observable:ce:1",
                "latest_bar_ts": indicators["latest_bar_ts"],
                "quote_update_version": 7,
''',
    '''                "setup_id": "observable:ce:1",
                "latest_bar_ts": indicators["latest_bar_ts"],
''',
)

replace_once(
    "tests/strategies/test_signal_observability_contract.py",
    '''        {"latest_bar_ts": 1_785_000_000.0},
        100.0,
    )

    assert signal is not None
''',
    '''        {"latest_bar_ts": 1_785_000_000.0, "quote_update_version": 7},
        100.0,
    )

    assert signal is not None
    assert signal.metadata["quote_update_version"] == 7
    assert (
        signal.metadata["quote_update_version_source"]
        == "indicator_context:quote_update_version"
    )
    assert signal.metadata["setup_signal_id"] == signal.deterministic_id
    assert signal.metadata["evaluation_snapshot_id"]
''',
)

replace_once(
    "tests/strategies/test_signal_observability_contract.py",
    '''    assert record.setup_id == "observable:ce:1"
    assert record.quote_update_version == 7


def _orderflow_signal(*, bid: float = 99.5) -> SimpleNamespace:
''',
    '''    assert record.setup_id == "observable:ce:1"
    assert record.quote_update_version == 7
    assert record.quote_update_version_source == "indicator_context:quote_update_version"
    assert record.setup_signal_id == signal.deterministic_id
    assert record.evaluation_snapshot_id == signal.metadata["evaluation_snapshot_id"]


def test_elite_evaluation_snapshot_identity_tracks_quote_version() -> None:
    strategy = _ObservableStrategy()
    base = {"latest_bar_ts": 1_785_000_000.0}

    first = strategy.generate_signal(
        "NFO:NIFTY2680724500CE",
        {**base, "quote_update_version": 7},
        100.0,
    )
    repeated = strategy.generate_signal(
        "NFO:NIFTY2680724500CE",
        {**base, "quote_update_version": 7},
        100.0,
    )
    changed = strategy.generate_signal(
        "NFO:NIFTY2680724500CE",
        {**base, "quote_update_version": 8},
        100.0,
    )

    assert first is not None and repeated is not None and changed is not None
    assert first.deterministic_id == repeated.deterministic_id == changed.deterministic_id
    assert (
        first.metadata["evaluation_snapshot_id"]
        == repeated.metadata["evaluation_snapshot_id"]
    )
    assert (
        first.metadata["evaluation_snapshot_id"]
        != changed.metadata["evaluation_snapshot_id"]
    )


def _orderflow_signal(*, bid: float = 99.5) -> SimpleNamespace:
''',
)

Path("tests/strategies/test_quote_update_identity.py").write_text(
    '''from __future__ import annotations

from nifty_scalper_bot.strategies.quote_update_identity import (
    build_evaluation_snapshot_id,
    coerce_quote_update_version,
    resolve_quote_update_identity,
)


def test_quote_identity_prefers_authoritative_source_order() -> None:
    version, source = resolve_quote_update_identity(
        ("datahub_quote", {"quote_update_version": 41}),
        ("runner_tick", {"quote_update_version": 42}),
        ("runner_counter", {"quote_update_version": 43}),
    )

    assert version == 41
    assert source == "datahub_quote:quote_update_version"


def test_quote_identity_uses_alias_and_ignores_invalid_values() -> None:
    version, source = resolve_quote_update_identity(
        ("datahub_quote", {"quote_update_version": 0}),
        ("runner_tick", {"update_version": "17"}),
    )

    assert version == 17
    assert source == "runner_tick:update_version"
    assert coerce_quote_update_version("not-a-version") is None


def test_evaluation_snapshot_identity_is_stable_per_setup_and_quote() -> None:
    first = build_evaluation_snapshot_id("setup-1", 7)
    repeated = build_evaluation_snapshot_id("setup-1", 7)
    changed_quote = build_evaluation_snapshot_id("setup-1", 8)
    changed_setup = build_evaluation_snapshot_id("setup-2", 7)

    assert first == repeated
    assert first != changed_quote
    assert first != changed_setup
    assert build_evaluation_snapshot_id("setup-1", None) is None
'''
)
