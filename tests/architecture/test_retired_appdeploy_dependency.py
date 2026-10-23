"""Protect the single Lightsail/Supabase LIVE production architecture.

AppDeploy was a temporary read-only probe, not an order, auth, or deploy
dependency. Keep retired provider endpoints out of production source.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

_PRODUCTION_ROOTS = (
    ROOT / "src" / "nifty_scalper_bot",
    ROOT / "deploy",
    ROOT / "dashboard",
    ROOT / ".github" / "workflows",
)
_DEPLOY_EXTENSIONS = frozenset(
    {
        ".py",
        ".sh",
        ".ts",
        ".js",
        ".json",
        ".yml",
        ".yaml",
        ".toml",
        ".service",
        ".timer",
    }
)
_PROHIBITED = (
    "api-v2.appdeploy.ai",
    ".v2.appdeploy.ai",
    "@appdeploy/sdk",
    "nifty-endpoint-probe-qspiq0",
    "nifty-deploy-recovery-relay-u7c57n",
    "nifty-live-diagnostic-relay-09rb32",
    "nifty-live-readonly-relay-kb0r4e",
    "nifty-cloud-probe-capture-w3mrp1",
)


def test_live_runtime_has_no_retired_appdeploy_network_dependencies() -> None:
    failures: list[str] = []
    for root in _PRODUCTION_ROOTS:
        assert root.exists()
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix not in _DEPLOY_EXTENSIONS:
                continue
            source = path.read_text(encoding="utf-8").lower()
            for term in _PROHIBITED:
                if term.lower() in source:
                    failures.append(f"{path.relative_to(ROOT)}: {term}")
    assert not failures, "Retired AppDeploy dependency reintroduced: " + ", ".join(
        failures
    )


def test_one_production_release_authority_and_no_github_aws_deploy() -> None:
    """Only validated Lightsail release script can restart trading engine."""
    release = (ROOT / "deploy" / "lightsail_release.sh").read_text(encoding="utf-8")
    setup = (ROOT / "deploy" / "lightsail_setup.sh").read_text(encoding="utf-8")
    assert "verify_release_ci.py" in release
    assert "niftybot-autodeploy.timer" in setup
    assert "lightsail_release.sh" in setup
    for workflow in (ROOT / ".github" / "workflows").glob("*.yml"):
        source = workflow.read_text(encoding="utf-8").lower()
        assert "aws-actions/configure-aws-credentials" not in source
        assert "appleboy/ssh-action" not in source


def test_supabase_is_independent_and_read_only_for_operator_monitoring() -> None:
    heartbeat = (
        ROOT / "src" / "nifty_scalper_bot" / "infra" / "runtime_heartbeat.py"
    ).read_text(encoding="utf-8")
    assert "nifty-runtime-heartbeat-ingest" in heartbeat
    assert "127.0.0.1" in heartbeat
    assert "token_sha256" not in heartbeat
    assert "trade/order" not in heartbeat
