"""Guard the exact, causal 2025 Breeze re-acquisition contract identities.

No credentials, order clients or historical prices are required by this test.
"""

from pathlib import Path


PLAN = (
    Path(__file__).resolve().parents[2]
    / "deploy"
    / "breeze_five_date_recovery_20261127.csv"
)


def test_historical_backfill_targets_only_exact_missing_contracts() -> None:
    """Never replay the obsolete broad 158-series plan or invent contracts."""
    from scripts.archive_breeze_backfill import load_targets

    targets = load_targets(PLAN)
    identities = {
        (
            row["day"],
            row["kind"],
            row["expiry"],
            row["strike"],
            row["side"],
        )
        for row in targets
    }
    assert len(targets) == len(identities) == 8
    assert identities == {
        ("2025-06-20", "futures", "2025-06-26", "", "FUT"),
        ("2025-06-20", "options", "2025-06-26", "24850", "CE"),
        ("2025-06-20", "options", "2025-06-26", "24850", "PE"),
        ("2025-06-27", "futures", "2025-07-31", "", "FUT"),
        ("2025-06-27", "options", "2025-07-03", "25600", "CE"),
        ("2025-06-27", "options", "2025-07-03", "25600", "PE"),
        ("2025-07-04", "options", "2025-07-10", "25450", "PE"),
        ("2025-07-11", "options", "2025-07-17", "25250", "PE"),
    }
    assert {row["priority"] for row in targets} == {"1"}
    assert {row["reason"] for row in targets} == {
        "provider_or_validation_failed"
    }
