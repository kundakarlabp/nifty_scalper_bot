"""Contract tests for the read-only historical entry-funnel exporter."""
import copy
import importlib.util
from pathlib import Path

import pytest

SCRIPT = (\n    Path(__file__).resolve().parents[2]\n    / "scripts"\n    / "diagnose_fixed_hypothesis_funnel.py"\n)
spec = importlib.util.spec_from_file_location("funnel_diagnostic", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture_report():
    results = []
    for phase, sessions in module.EXPECTED_PHASES.items():
        for strategy in module.EXPECTED_STRATEGIES:
            for bps in module.EXPECTED_SLIPPAGE:
                results.append({
                    "phase": phase, "candidate": strategy,
                    "slippage_bps_per_side": bps, "session_count": sessions,
                    "entry_funnel": {"evaluations": 100, "signals_emitted": 10,
                                     "geometry_valid": 10, "next_bar_intents": 10,
                                     "accepted_fills": 2},
                    "no_vote_reasons": {"cost_net_rr_rejected": 7},
                    "data_quality": {\n                        "resolved_exit_count": 1,\n                        "unresolved_exit_count": 1,\n                    },
                })
    return {"results": results}


def test_valid_cells_and_denominators():
    rows = module.diagnose(fixture_report())
    assert len(rows) == 45
    assert all(row["cost_rejection_share_of_intents"] == 0.7 for row in rows)
    assert all(row["other_nonfilled_intents"] == 1 for row in rows)


@pytest.mark.parametrize("field,value", [
    ("session_count", 460), ("candidate", "unknown"),
])
def test_invalid_cells_rejected(field, value):
    report = fixture_report()
    report["results"][0][field] = value
    with pytest.raises(ValueError):
        module.diagnose(report)


def test_missing_exit_rejected():
    report = fixture_report()
    report["results"][0]["data_quality"]["unresolved_exit_count"] = 0
    with pytest.raises(ValueError, match="fill/exit mismatch"):
        module.diagnose(report)


def test_duplicate_cell_rejected():
    report = fixture_report()
    report["results"][1] = copy.deepcopy(report["results"][0])
    with pytest.raises(ValueError, match="duplicate result"):
        module.diagnose(report)
