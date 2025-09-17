import smoke  # noqa: F401

import src.strategies.patches as patches
import src.diagnostics.checks as checks


def test_gate_allows_min_atr(monkeypatch):
    """Ensure atr gate passes when atr_pct equals the configured minimum."""

    def gate(band_low, band_high, atr_pct):
        return "atr_out_of_band" if atr_pct < band_low else "ok"

    monkeypatch.setattr(checks, "gate_atr_band", gate, raising=False)
    monkeypatch.setattr(patches, "_resolve_min_atr_pct", lambda: 0.05)
    patches._patch_atr_band()
    res = checks.gate_atr_band(0.08, 0.9, 0.05)
    assert res != "atr_out_of_band"


def test_gate_raises_to_resolved_min(monkeypatch):
    """Lower raw bands should be raised to the resolved minimum ATR pct."""

    recorder = {}

    def gate(band_low, band_high, atr_pct):
        recorder["band_low"] = band_low
        return "ok"

    monkeypatch.setattr(checks, "gate_atr_band", gate, raising=False)
    monkeypatch.setattr(patches, "_resolve_min_atr_pct", lambda: 0.05)
    patches._patch_atr_band()
    checks.gate_atr_band(0.02, 0.9, 0.10)
    assert recorder["band_low"] == 0.05
