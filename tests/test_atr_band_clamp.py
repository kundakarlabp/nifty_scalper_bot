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
