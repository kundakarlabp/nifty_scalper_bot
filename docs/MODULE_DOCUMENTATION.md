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
- requires a header edit when a covered module's callable API inventory changes against the PR base;
- requires an architecture-path document review when the manifest's runtime-path/owner contract changes.

It intentionally **does not generate plausible but unverified explanations**. Semantics cannot be proven by a syntax check; developer/reviewer accountability remains necessary.

Commands:

```bash
python scripts/module_doc_guard.py --all
python scripts/module_doc_guard.py --base-ref origin/main
python scripts/agent_task.py check --files src/nifty_scalper_bot/strategies/runner.py
```

Avoid timestamped copies of whole README/architecture documents, auto-rewriting production source with a formatter, and runtime-generated docs on the live AWS host. Updates are reviewed, tested and merged in GitHub; AWS only consumes validated releases.
