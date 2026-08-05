from __future__ import annotations

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


def test_quote_version_coercion_rejects_lossy_or_boolean_values() -> None:
    assert coerce_quote_update_version(True) is None
    assert coerce_quote_update_version(False) is None
    assert coerce_quote_update_version(7.5) is None
    assert coerce_quote_update_version("7.5") is None
    assert coerce_quote_update_version("7.0") == 7
    assert coerce_quote_update_version("1e3") == 1000


def test_evaluation_snapshot_identity_is_stable_per_setup_and_quote() -> None:
    first = build_evaluation_snapshot_id("setup-1", 7)
    repeated = build_evaluation_snapshot_id("setup-1", 7)
    changed_quote = build_evaluation_snapshot_id("setup-1", 8)
    changed_setup = build_evaluation_snapshot_id("setup-2", 7)

    assert first == repeated
    assert first != changed_quote
    assert first != changed_setup
    assert build_evaluation_snapshot_id("setup-1", None) is None
