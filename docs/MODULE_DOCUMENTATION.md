# Module headers and documentation synchronization

## Documentation layers (one home for each fact)

1. **README.md:** concise purpose, flowchart, boundaries and links. Do not turn it into an environment-variable catalogue or incident log.
2. **docs/TRADING_LIFECYCLE.md:** understandable chronological trading, exit and release flows; links to exact owners.
3. **docs/architecture/agent_manifest.json:** authoritative path ownership, validation routing and `module_documentation` coverage. Do not duplicate its mapping in another manifest.
4. **ARCHITECTURE_TRADING_PATH.md:** operational implementation and handoff details; update when owners or lifecycle responsibilities change.
5. **Opening module docstrings / script comments:** single-screen summary of *that file*; the nearest authoritative explanation of purpose, entrypoints and boundaries.
6. **Operational documentation:** `deploy/LIGHTSAIL_SETUP.md`, `RUNBOOK.md` and specific incident/research docs own concrete run procedures, failure modes and evidence.

## Standard source header

Use a Python **module docstring** immediately after permitted `# fmt:` / `# ruff:` pragmas, before imports and `from __future__`. Keep it roughly **5–14 readable lines** (existing detailed legacy notes may remain where useful):

```python
"""Purpose: One sentence describing this module.

Responsibilities:
- What state, processing and lifecycle this module owns.
- Most important callable entrypoints (Class.method or function_name).

Boundaries: What it delegates; which neighbouring owner must not be duplicated.
Safety: An invariant or failure behavior where relevant.
"""
```

For Bash/YAML files use top-of-file `#` comments after any required shebang; preserve valid YAML/Bash syntax. Do not list every private helper, annotate every function again, paste secrets, or claim that a header replaces reading code in a safety-critical change.

## Mandatory edit review

When a source file changes, inspect and update its **opening header as needed**. Changes in top-level classes, functions, signatures or methods **require a header update**. When contracts/owners or critical behavior changes, also update the lifecycle/architecture document and manifest where appropriate. If changes affect deployment, update its runbook. Record the reviewed documentation impact in the PR template, including an explicit `no change necessary` rationale when appropriate.

CI and `python scripts/agent_task.py check` invoke `scripts/module_doc_guard.py`. It:
- checks short, meaningful headers on all key modules listed by the architecture manifest;
- checks documentation on newly added production Python modules;
- requires a header edit when **any changed production/dashboard/scripts/ops Python module** changes its callable API inventory against the PR base;
- requires an architecture-path document review when the manifest's runtime-path/owner contract changes.

It intentionally **does not generate plausible but unverified explanations**. Semantics cannot be proven by a syntax check; developer/reviewer accountability remains necessary.

Commands:

```bash
python scripts/module_doc_guard.py --all
python scripts/module_doc_guard.py --base-ref origin/main
python scripts/agent_task.py check --files src/nifty_scalper_bot/strategies/runner.py
```

Avoid timestamped copies of whole README/architecture documents, auto-rewriting production source with a formatter, and runtime-generated docs on the live AWS host. Updates are reviewed, tested and merged in GitHub; AWS only consumes validated releases.

## Extended coverage (phase 2)

The manifest now includes admission/risk policy, recovery, configuration, release,
operator surfaces and replication modules. This is a **documentation coverage list**,
not a duplicate runtime ownership map: `runtime_path` and `owners` retain authority.

The PR-base check requires a meaningful module docstring on **every modified
production, dashboard, scripts or ops Python file**, including existing files not
yet in the major-module manifest (except package `__init__.py`). Covered major
modules and unlisted Python modules alike require a header edit when their callable interface changes.

Changed deployment Bash scripts, systemd service units, Dockerfile, Railway TOML,
and GitHub workflows require `File purpose:`, `Key responsibilities:`, and
`Operational constraints:` as opening comments. The older
[File Header Standard](FILE_HEADER_STANDARD.md) specifies this stricter labeled
form for an existing tested subset. Other Python files can use descriptive
narrative docstrings as shown above.

Code reviewers still need to assess behavior and responsibility changes that
AST signatures cannot detect, including whether README, architecture, lifecycle
and operations documentation needs updating. The guard never modifies live code.

## Phase 3: trading-safety handoff coverage

The documented major-module set also includes startup readiness, broker REST,
market-data policy and persistence, release entry holds, state machine, bounded
entry recovery, broker position identity, fill-ledger brackets, net reward/risk,
SQLite trade journal, runtime heartbeat, service controls and NSE session gates.
These are **documentation references only**; ownership and business logic remain
in their original modules.

Interface drift is checked for **all changed, non-package Python modules**
in the production/dashboard/scripts/ops trees, whether or not those modules are
explicitly named in the architecture manifest. The opening summary must be
edited when top-level functions, class methods or signatures change.
An unchanged public interface does **not** prove unchanged behavior; reviewers
must still revise header descriptions whenever responsibilities, gates,
control flow or recovery semantics materially change.
