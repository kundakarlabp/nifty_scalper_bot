"""Regression tests for the manifest-driven module documentation guard."""

from __future__ import annotations

import json
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


def _temporary_manifest(root: Path) -> None:
    path = root / "docs/architecture/agent_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"runtime_path": [], "owners": []}), encoding="utf-8")


def test_existing_untracked_runtime_edit_requires_header(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _temporary_manifest(tmp_path)
    path = "src/nifty_scalper_bot/risk/new_policy.py"
    file = tmp_path / path
    file.parent.mkdir(parents=True)
    file.write_text('"""Short."""\n\nvalue = 1\n', encoding="utf-8")
    monkeypatch.setattr(guard, "git_changed", lambda *_: {path})
    assert any(path in issue for issue in guard.check(tmp_path, base="main"))


def test_changed_dashboard_file_with_header_passes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _temporary_manifest(tmp_path)
    path = "dashboard/example.py"
    file = tmp_path / path
    file.parent.mkdir(parents=True)
    file.write_text(
        '"""Purpose: Describe operator-only dashboard observability here.\n'
        "\nResponsibilities: Render read-only runtime state summaries.\n"
        'Boundaries: No broker submissions or trading decision authority.\n"""\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(guard, "git_changed", lambda *_: {path})
    assert guard.check(tmp_path, base="main") == []


def test_changed_deployment_asset_requires_opening_labels(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _temporary_manifest(tmp_path)
    path = "deploy/scripts/run_release.sh"
    file = tmp_path / path
    file.parent.mkdir(parents=True)
    file.write_text("#!/usr/bin/env bash\n# release script\ntrue\n", encoding="utf-8")
    monkeypatch.setattr(guard, "git_changed", lambda *_: {path})
    assert any("File purpose:" in issue for issue in guard.check(tmp_path, base="main"))
    file.write_text(
        "#!/usr/bin/env bash\n# File purpose: Start a release.\n"
        "# Key responsibilities: Validate and restart.\n"
        "# Operational constraints: Do not bypass broker-safe checks.\n"
        "true\n",
        encoding="utf-8",
    )
    assert guard.check(tmp_path, base="main") == []


def test_coverage_expands_to_dashboard_and_tools() -> None:
    payload = {
        "module_documentation": [
            "dashboard/superlite_console.py",
            "scripts/module_doc_guard.py",
        ]
    }
    assert guard.covered_modules(payload) == {
        "dashboard/superlite_console.py",
        "scripts/module_doc_guard.py",
    }


def test_existing_untracked_callable_change_requires_updated_header(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Unlisted source still gets full API/header drift enforcement."""
    _temporary_manifest(tmp_path)
    path = "src/nifty_scalper_bot/execution/new_gate.py"
    file = tmp_path / path
    file.parent.mkdir(parents=True)
    old_header = (
        '"""Purpose: Authorize an isolated test signal with safe boundaries.\n'
        "\nResponsibilities: Check a prepared signal and return a decision.\n"
        'Boundaries: No direct broker action or position mutation.\n"""\n'
    )
    before = old_header + "def authorize(symbol):\n    return bool(symbol)\n"
    after = old_header + "def authorize(symbol, live=False):\n    return bool(symbol)\n"
    file.write_text(after, encoding="utf-8")
    monkeypatch.setattr(guard, "git_changed", lambda *_: {path})
    monkeypatch.setattr(guard, "git_read", lambda *_: before)
    issues = guard.check(tmp_path, base="main")
    assert any("callable interface changed" in issue for issue in issues)
    revised_header = old_header.replace(
        "Check a prepared signal", "Check a prepared signal and a live-mode flag"
    )
    file.write_text(after.replace(old_header, revised_header), encoding="utf-8")
    assert guard.check(tmp_path, base="main") == []
