# Agent tooling design

## Objective

Reduce repeated GitHub searches, file-by-file retrieval, duplicated architecture discovery, and unnecessary model context while retaining independent CI validation and trading-safety controls.

## Evidence adapted

### Aider repository maps

Aider describes the central problem in larger codebases as finding the correct code and understanding how it relates to the rest of the repository. Its repository map uses parsed symbols and signatures to provide concise whole-repository context and ranks the most relevant portions to fit a token budget.

Adaptation in this repository:

- `scripts/agent_context.py` parses Python syntax without importing runtime modules.
- It ranks matching paths and symbol signatures.
- It links likely regression tests.
- It emits a compact reading order rather than packing complete source files.

Source: `Aider-AI/aider`, repository-map documentation.

### SWE-agent Agent-Computer Interface

SWE-agent reports that simple, model-oriented commands and feedback formats make it easier for an agent to browse, view, edit, and execute repository work. The important principle is a constrained, repeatable interface rather than ad hoc browsing.

Adaptation in this repository:

- one issue title triggers one standardized context report
- one repository workflow defines investigation, editing, validation, PR, and merge stages
- `scripts/agent_check.py` provides a deterministic changed-file-to-test plan

Source: `SWE-agent/SWE-agent`, architecture documentation.

### Repomix code compression

Repomix uses syntax parsing to preserve imports, function signatures, class structures, and interfaces while removing implementation detail, reducing context size for architecture and code analysis.

Adaptation in this repository:

- context reports show paths and signatures, not full implementations
- implementation files are retrieved only after ranking
- runtime-data and environment paths are excluded

Source: `yamadashy/repomix`, code-compression documentation.

### Serena semantic retrieval

Serena emphasizes symbol-oriented code navigation using language-server-style operations rather than treating the repository as undifferentiated text.

Adaptation in this repository:

- classes, methods, functions, line numbers, and related tests are surfaced directly
- exact symbols and log event names are preferred in task requests
- broad repository reads are a fallback, not the first action

Source: `oraios/serena`.

## Architecture for the available environment

The user currently has ordinary ChatGPT chats and the GitHub connector, but no persistent local clone or dedicated coding VM. The implemented path therefore uses GitHub itself as the durable workspace:

```text
ChatGPT request
→ GitHub connector authorization
→ compact repository contract and map
→ optional owner-created Agent Context issue
→ GitHub Actions context report
→ targeted GitHub file retrieval
→ branch and narrow code changes
→ pull request
→ independent GitHub Actions validation
→ squash merge after final-head evidence
```

## Why a personal access token is not the primary optimization

A token changes authorization, not code understanding. It does not automatically provide:

- repository structure
- symbol relationships
- test ownership
- root-cause isolation
- executable validation
- durable context between chats

The connected GitHub application already supports authorized repository reads and writes in ChatGPT. The principal efficiency gains come from reducing discovery and context volume, not from changing the credential mechanism.

## Executable guardrails

- `scripts/agent_check.py` classifies change risk and selects the smallest safe validation ring.
- `scripts/architecture_lint.py` enforces mechanically provable ownership boundaries.
- `scripts/agent_merge_guard.py` rejects stale validated base/head combinations before merge.
- `scripts/agent_failure_learn.py` classifies repeated CI failures without mutating repository memory.
- `.github/workflows/failure-memory-candidates.yml` surfaces learning candidates after failed CI.

These tools turn recurring reasoning into deterministic repository feedback.

## Validation assets

- `docs/contracts/contract_manifest.json` indexes machine-readable boundary contracts and executable samples.
- `tests/fixtures/replay/golden_market_path.csv` is the canonical sanitized market-path replay fixture.
- `tests/properties/` uses generated inputs to test deterministic invariants rather than only hand-picked examples.
- `benchmarks/agent/historical_regressions.json` is the historical coding-agent regression set.
- `scripts/agent_benchmark.py` validates or runs selected historical regression cases without starting the live runtime.

These assets extend existing runtime owners and replay infrastructure; they are not parallel production systems.

## Safety boundaries

- Generated context never imports or executes bot source.
- Known environment, database, log, runtime-data, backup, and key paths are excluded.
- Live execution is never needed for context generation or automated tests.
- Context ranking does not override `AGENTS.md` ownership or trading invariants.
- Focused validation is advisory; the complete CI suite remains mandatory before merge.

## Institutional engineering memory

Recurring coding-agent failures are captured in `docs/ENGINEERING_FAILURE_PATTERNS.md`. Add a pattern only after the root mechanism is understood and the failure is repeated, costly, safety-relevant, or easy to reintroduce.

Prefer prevention in this order:

```text
formatter/linter/type checker
→ focused regression or architecture test
→ agent tooling
→ concise failure-pattern note
```

The document should shrink when automation makes a prose warning unnecessary. It is not a chronological incident log.

## Maintenance

When architecture changes:

1. Update `docs/REPO_MAP.md` and `AGENTS.md` in the same PR.
2. Extend the area-to-test mapping in `scripts/agent_check.py`.
3. Add a tooling regression test if ranking, classification, or quality-preflight behavior changes.
4. Record a durable failure pattern only when it meets the institutional-memory criteria above.
5. Keep the global repository contract compact; detailed procedures belong in skill files and these tooling documents.
