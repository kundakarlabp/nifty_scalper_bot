---
name: live-runtime-diagnosis
description: Diagnose the deployed NIFTY scalper safely from health, status, snapshots, and filtered logs. Use for requests to check the live bot, runner, signals, market-data freshness, orders, brackets, deployment state, or a suspected production incident.
---

# Live Runtime Diagnosis

Production diagnosis is read-only by default. Do not redeploy, restart, change configuration, enable live execution, or alter risk controls merely to obtain diagnostic information.

## Access rule

Use the configured direct production health/status endpoints first when available.

If direct network access fails:

- do **not** conclude that the bot is down;
- use the repository/operator-approved relay or observability integration available in the current private environment;
- never add private relay URLs, credentials, tokens, or operator-only endpoints to this public repository;
- do not redeploy a healthy relay merely to read logs.

A TCP/connectivity failure from the current client is evidence about that path, not proof of engine failure.

## Diagnostic order

Use the cheapest, most structured evidence first:

```text
liveness
→ health/readiness
→ trading/runtime status
→ structured snapshot
→ narrowly filtered logs
→ broader logs only when needed
```

Confirm the deployed commit/SHA when the question could depend on whether a fix is actually running.

## Runtime layers

Classify findings by the earliest affected layer:

1. process/liveness
2. broker/session/connectivity
3. WebSocket/subscription/market-data freshness
4. OHLC hydration/readiness
5. strategy evaluation and blockers
6. risk/capital/cooldown/position gates
7. order submission and broker acknowledgement
8. fill/position reconciliation
9. bracket/exit management
10. operator/dashboard/Telegram presentation

Do not label a later-layer symptom as root cause while an earlier layer is degraded.

## Evidence to capture

Prefer structured fields over log prose:

- timestamp and timezone
- deployed SHA
- market session state
- execution mode
- broker/WebSocket connectivity
- active basket symbols/tokens
- quote source/freshness/depth quality
- hydration/readiness blockers
- runner last evaluation/bar identity
- candidate/signal/blocker reason
- risk decision
- open/pending order state
- fills/position quantity
- bracket state
- recent error/warning counters

Never expose credentials, session tokens, or sensitive account data in the report.

## Classification

Every material finding should be one of:

- `HEALTHY` — expected and supported by current evidence
- `EXPECTED` — intentional state such as market closed or a valid blocker
- `WARNING` — degraded but not proven defective
- `DEFECT` — reproducible incorrect behavior with an identified owner/path
- `UNKNOWN` — evidence is insufficient or contradictory

Separate **runtime state**, **environment/data issue**, **code defect**, and **deployment mismatch**.

## If a code defect is found

Do not hot-edit production.

```text
production evidence
→ reproduce safely
→ identify owner/root cause
→ branch from authoritative main
→ regression test
→ smallest fix
→ focused + full validation
→ reviewed merge
→ controlled deployment
→ post-deploy verification
```

Define a rollback trigger for any production-affecting correction.

## Output

```markdown
Overall state:
Deployed SHA:
Earliest affected layer:
Evidence:
Classification:
Root cause or leading hypothesis:
Immediate safe action:
Code change required: YES/NO/UNKNOWN
Residual uncertainty:
```
