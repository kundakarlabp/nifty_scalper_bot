"""Option microstructure quality has one owner and no invented default.

``option_score`` is the verdict of ``TradeCandidateSelector`` and carries 20%
of the composite execution score. The runner used to seed it with a mid-scale
5.5 before the genuine candidate score was promoted, which both awarded points
for evidence never gathered and displaced the real candidate score. Absent
candidate evidence must leave the component unset so the final-score precheck
rejects the entry.
"""

from __future__ import annotations

import ast
from pathlib import Path

from nifty_scalper_bot.strategies.signal_quality import (
    REQUIRED_SCORE_COMPONENTS,
    missing_score_components,
)

_RUNNER = Path("src/nifty_scalper_bot/strategies/runner.py")


def _setdefault_calls(source: str, key: str) -> list[ast.Call]:
    tree = ast.parse(source)
    found: list[ast.Call] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if getattr(node.func, "attr", None) != "setdefault":
            continue
        if not node.args:
            continue
        first = node.args[0]
        if isinstance(first, ast.Constant) and first.value == key:
            found.append(node)
    return found


def test_runner_never_seeds_option_score_outside_candidate_selection() -> None:
    source = _RUNNER.read_text(encoding="utf-8")
    assert _setdefault_calls(source, "option_score") == []


def test_option_score_is_a_required_component() -> None:
    assert "option_score" in REQUIRED_SCORE_COMPONENTS


def test_absent_option_evidence_is_reported_as_missing() -> None:
    """This is the mechanism the runner's final-score precheck rejects on."""
    metadata = {
        "direction_score": 8.0,
        "strategy_score": 8.0,
        "data_score": 8.0,
        "rr_score": 8.0,
    }

    assert missing_score_components(metadata) == ["option_score"]

    metadata["option_score"] = 0.0
    assert missing_score_components(metadata) == []


def test_promoted_candidate_score_is_the_option_score_source() -> None:
    """The runner must read option_score off the selected candidate."""
    source = _RUNNER.read_text(encoding="utf-8")
    assert "float(candidate.score or 0.0)" in source
    assert 'metadata["option_score"] = max(' in source
