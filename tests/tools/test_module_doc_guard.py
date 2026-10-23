"""Regression tests for the manifest-driven module documentation guard."""

from __future__ import annotations

from pathlib import Path

from scripts import module_doc_guard as guard


def test_manifest_coverage_uses_canonical_owners_not_duplicate_table() -> None:
    manifest = {
        "runtime_path": ["src/nifty_scalper_bot/strategies/runner.py"],
        "owners": [
            {
                "owner": "src/nifty_scalper_bot/core/app.py",
                "read_facade": "src/nifty_scalper_bot/data/data_hub.py",
                "collaborators": ["src/nifty_scalper_bot/risk/risk_manager.py"],
            }
        ],
        "module_documentation": ["src/nifty_scalper_bot/main.py"],
    }
    assert guard.covered_modules(manifest) == {
        "src/nifty_scalper_bot/strategies/runner.py",
        "src/nifty_scalper_bot/core/app.py",
        "src/nifty_scalper_bot/data/data_hub.py",
        "src/nifty_scalper_bot/risk/risk_manager.py",
        "src/nifty_scalper_bot/main.py",
    }


def test_short_bare_module_header_is_rejected() -> None:
    assert guard.header_issue("module.py", '"""Short."""\n') is not None


def test_header_with_purpose_and_boundary_is_accepted() -> None:
    source = (
        '"""Purpose: Normalize already selected option quotes for downstream '
        "consumers.\n"
        "\nResponsibilities: Preserve price and source quality markers.\n"
        "Boundaries: Contract discovery and order submission belong elsewhere.\n"
        '"""\n'
    )
    assert guard.header_issue("module.py", source) is None


def test_method_or_signature_change_is_detectable() -> None:
    before = (
        '"""Purpose: Example.\nSafety: none.\n"""\n'
        "class Owner:\n"
        "    def go(self, price):\n"
        "        return price\n"
    )
    after = before.replace("def go(self, price)", "def go(self, price, quantity=1)")
    assert guard.api_inventory(before) != guard.api_inventory(after)


def test_method_body_change_does_not_require_unnecessary_header_edit() -> None:
    before = "class Owner:\n    def go(self):\n        return 1\n"
    after = before.replace("return 1", "return 2")
    assert guard.api_inventory(before) == guard.api_inventory(after)


def test_owner_contract_change_detected_but_unrelated_manifest_change_ignored() -> None:
    old = {"runtime_path": ["src/a.py"], "owners": [{"owner": "src/a.py"}]}
    same = {**old, "module_documentation": ["src/b.py"]}
    changed = {**old, "runtime_path": ["src/a.py", "src/b.py"]}
    assert guard.runtime_contract(old) == guard.runtime_contract(same)
    assert guard.runtime_contract(old) != guard.runtime_contract(changed)


def test_all_real_critical_modules_have_informative_headers() -> None:
    root = Path(__file__).resolve().parents[2]
    issues = guard.check(root)
    assert not issues, "\n".join(issues)
