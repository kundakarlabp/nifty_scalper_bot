"""Protect AWS Lightsail release ownership and independent Supabase monitoring.

Operations observe the existing AWS runtime and private Supabase telemetry;
only the canonical Lightsail release process can change the live engine.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


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
    defaults = (
        ROOT / "src" / "nifty_scalper_bot" / "config" / "defaults.py"
    ).read_text(encoding="utf-8")
    assert "nifty-runtime-heartbeat-ingest" in defaults
    assert "DEFAULT_SUPABASE_HEARTBEAT_URL" in heartbeat
    assert "127.0.0.1" in heartbeat
    assert "token_sha256" not in heartbeat
    assert "trade/order" not in heartbeat
