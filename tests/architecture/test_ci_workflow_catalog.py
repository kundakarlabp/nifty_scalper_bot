from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"
EXPECTED_WORKFLOWS = {
    "agent-context.yml",
    "ci.yml",
    "failure-memory-candidates.yml",
    "slow-suite-weekly.yml",
}


def test_workflow_catalog_has_only_canonical_workflows() -> None:
    discovered = {path.name for path in WORKFLOWS.glob("*.yml")}
    assert discovered == EXPECTED_WORKFLOWS


def test_ci_is_read_only_and_deduplicates_e2e_markers() -> None:
    text = (WORKFLOWS / "ci.yml").read_text(encoding="utf-8")

    assert "contents: read" in text
    assert "contents: write" not in text
    assert "branches: [main]" in text
    assert "changed-code-quality:" in text
    assert "normal-test-shard:" in text
    assert "tests:" in text
    assert "e2e-simulation:" in text
    assert "needs: normal-test-shard" in text
    assert "NORMAL_SHARDS_RESULT" in text
    assert "scripts/ci_test_shard.py" in text
    assert "fail-fast: false" in text
    assert "shard: [0, 1, 2, 3]" in text

    normal_suite = (
        "not slow and not simulation_component and not live_runtime_e2e "
        "and not e2e_live_sim"
    )
    e2e_suite = "simulation_component or live_runtime_e2e or e2e_live_sim"
    assert normal_suite in text
    assert e2e_suite in text


def test_no_one_off_patch_or_branch_specific_ci_remains() -> None:
    combined = "\n".join(
        path.read_text(encoding="utf-8") for path in WORKFLOWS.glob("*.yml")
    )

    assert "p0-p1-runtime-hardening" not in combined
    assert "strategy-stall-production" not in combined
    assert "feat/market-aware-execution-optimisations" not in combined
    assert "git push origin HEAD:fix/" not in combined


def test_failure_memory_workflow_is_read_only() -> None:
    text = (WORKFLOWS / "failure-memory-candidates.yml").read_text(encoding="utf-8")

    assert "actions: read" in text
    assert "contents: read" in text
    assert "contents: write" not in text
    assert "pull-requests: write" not in text
    assert "issues: write" not in text
    assert "agent_failure_learn.py" in text


def test_ci_normal_test_shards_preserve_one_stable_aggregate_gate() -> None:
    text = (WORKFLOWS / "ci.yml").read_text(encoding="utf-8")

    assert "name: normal-tests-${{ matrix.shard }}" in text
    assert "name: tests" in text
    assert 'test "$NORMAL_SHARDS_RESULT" = "success"' in text
    assert "${{ strategy.job-index }}" in text
    assert "${{ strategy.job-total }}" in text
