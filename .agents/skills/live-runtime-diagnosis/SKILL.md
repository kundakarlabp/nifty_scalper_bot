---
name: live-runtime-diagnosis
description: Diagnose the deployed NIFTY scalper safely from health, status, snapshots, and filtered logs. Use for requests to check the live bot, runner, signals, market-data freshness, orders, brackets, deployment state, or a suspected production incident.
---

# Live Runtime Diagnosis

Production diagnosis is read-only by default. Do not redeploy, restart, change configuration, enable live execution, or alter risk controls merely to obtain diagnostic information.

## Access rule

For **live logs, status, and trading investigations**, use existing authenticated Supabase tables first: `public.nifty_runtime_heartbeat` (age, host-local HTTP, SHA, release and session state) and, when needed, `public.nifty_daily_market_logs` (archive timestamp and bounded runner lines). See `docs/ops/PRODUCTION_HTTP_ACCESS.md` and `docs/runbooks/nifty-diagnostics-relay-fallback.md`; do not ask repeatedly for direct access when telemetry answers the question.

For **explicit public HTTP reachability**, make one short direct external probe when this client's network supports it. After a tool-level refusal/timeout, do not repeat the same blocked path within the investigation; report public ingress as unverified from that client, then use Supabase for independent local-health evidence. Never equate public access, host-local liveness, trading readiness, or broker-authoritative state.

- Do **not** conclude that the bot is down from a blocked client request.
- AppDeploy `Backend not found` indicates a relay routing/backend fault, not an engine fault; no relay is required for logs. Preserve v15 and do not redeploy or add a substitute relay.
- Never use SSH merely to recover diagnostics; use existing authenticated admin for immediate operational detail.
- Do not expose private operator URLs, credentials, tokens, or account-only endpoints in this public repository.

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
