from __future__ import annotations

from nifty_scalper_bot.admin_install_proof_display import install_proof_display


def test_install_proof_display_reports_all_installed() -> None:
    proof = {
        "market_data_manager_hardened": True,
        "websocket_hardened": True,
        "datahub_synthetic_guard_installed": True,
        "polling_failover_runtime_patch_installed": True,
        "core_app_import_hook_installed": True,
        "datahub_import_hook_installed": True,
        "all_required_installed": True,
        "import_hook_counts": {"core_app": 1, "datahub": 1},
    }

    display = install_proof_display(proof)

    assert display["label"] == "ALL HARDENING INSTALLED"
    assert display["css"] == "ok"
    assert display["all_installed"] is True
    assert display["missing"] == []
    assert display["hook_counts"] == {"core_app": 1, "datahub": 1}


def test_install_proof_display_lists_missing_items() -> None:
    proof = {
        "market_data_manager_hardened": True,
        "websocket_hardened": False,
        "datahub_synthetic_guard_installed": True,
        "polling_failover_runtime_patch_installed": False,
        "core_app_import_hook_installed": True,
        "datahub_import_hook_installed": False,
        "all_required_installed": False,
        "import_hook_counts": {"core_app": 1, "datahub": 2},
    }

    display = install_proof_display(proof)

    assert display["label"] == "HARDENING INCOMPLETE"
    assert display["css"] == "bad"
    assert display["all_installed"] is False
    assert display["missing"] == ["WebSocket", "Polling failover", "DataHub hook"]


def test_install_proof_display_unknown_when_missing() -> None:
    display = install_proof_display(None)

    assert display["label"] == "UNKNOWN"
    assert display["css"] == "warn"
    assert display["all_installed"] is None
