"""The directional research bundle cannot alter LIVE strategy configuration."""

from pathlib import Path

import pytest

from scripts import audit_directional_strategies as audit


def test_bundle_uses_one_archive_and_restores_vwap_experiment_flag(monkeypatch):
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    monkeypatch.setenv("VWAP_RESEARCH_COMPARE_RETEST_REBREAK_ONLY", "false")
    seen = []

    def vwap(directory):
        seen.append(("vwap", directory, audit.os.getenv(
            "VWAP_RESEARCH_COMPARE_RETEST_REBREAK_ONLY"
        )))
        return {"data_sha256": "same-source"}

    def orb(directory):
        seen.append(("orb", directory))
        return {"data_sha256": "same-source"}

    def reversal(directory):
        seen.append(("reversal", directory))
        return {"data_sha256": "same-source"}

    monkeypatch.setattr(audit, "run_vwap_comparison", vwap)
    monkeypatch.setattr(audit, "run_orb_comparison", orb)
    monkeypatch.setattr(audit, "run_failed_breakout_smc_comparison", reversal)
    directory = Path("/read-only-archive")
    result = audit.run_directional_research_bundle(directory)
    assert seen == [
        ("vwap", directory, "true"),
        ("orb", directory),
        ("reversal", directory),
    ]
    assert audit.os.getenv("VWAP_RESEARCH_COMPARE_RETEST_REBREAK_ONLY") == "false"
    assert result["data_sha256"] == "same-source"
    assert result["selected_for_live"] is None
    assert result["promotion_eligible"] is False
    assert result["live_equivalent"] is False


def test_bundle_fails_closed_on_source_drift_and_live_mode(monkeypatch):
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    monkeypatch.delenv("VWAP_RESEARCH_COMPARE_RETEST_REBREAK_ONLY", raising=False)
    monkeypatch.setattr(
        audit, "run_vwap_comparison", lambda _: {"data_sha256": "first"}
    )
    monkeypatch.setattr(
        audit, "run_orb_comparison", lambda _: {"data_sha256": "other"}
    )
    monkeypatch.setattr(
        audit,
        "run_failed_breakout_smc_comparison",
        lambda _: {"data_sha256": "first"},
    )
    with pytest.raises(ValueError, match="directional_research_source_drift"):
        audit.run_directional_research_bundle(Path("/unused"))
    assert audit.os.getenv("VWAP_RESEARCH_COMPARE_RETEST_REBREAK_ONLY") is None

    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    with pytest.raises(ValueError, match="directional_research_bundle_requires_shadow"):
        audit.run_directional_research_bundle(Path("/unused"))
