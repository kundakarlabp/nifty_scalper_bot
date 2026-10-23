# October 23: narrowly bounded entry-admission hypotheses

## Scope and attribution
Observed production: the engine was healthy and armed, but most evaluations
produced no trigger. Repeated OrderFlow observations are not distinct
opportunities; distinct SMC setup IDs and option quote timestamps must be
deduplicated. A post-trade profitable counterfactual has **not** been proved.

## Single owners and experiment switches

| Question | Canonical owner | LIVE profile default (host-neutral; explicit false opts out) | Unchanged safety gates |
| --- | --- | --- | --- |
| Qualified SMC with two separately timed supportive books | `SustainedSMCFlow` called by `StrategyManager` | `SMC_SUSTAINED_FLOW_CONFIRMATION_ENABLED` (default true in canonical LIVE) | SMC structural setup ID, direction, opposing-context veto, capacity-retry consensus, broker/runner/risk |
| OFI reversal persistence versus latency | `OrderFlowStrategy` | `ORDERFLOW_OFI_ADAPTIVE_HOLDOFF_ENABLED` (default true in canonical LIVE) | Base holdoff 15s; minimum 10s when opted in and 3s sustained OFI and strong same-side book support; adverse veto |
| Mid-trend premium pullback and rebreak beyond ordinary ATR VWAP band | `VWAPProStrategy` | `VWAP_TREND_RETEST_EXTENSION_ENABLED` (default true in canonical LIVE) | Session VWAP, strong fresh underlying direction, TREND regime, finalized current and previous adjacent bars, bearish prior retracement, rebreak, option RVOL, futures slope and risk |

No alternate entry arbiter, second quote freshness authority, second stop/sizing
system, or unbounded global VWAP distance increase is introduced.
The SMC proof engine may observe evidence with its admission flag off:
`SMC_SUSTAINED_FLOW_PROOF` reports `is_counterfactual=true` and
**does not authorize a live order**. Proof is consumed and logged only once
per unique setup identity; subsequent quote versions or adverse transitions
cannot manufacture additional missed opportunities.

The OrderFlow knob `ORDERFLOW_OFI_REVERSAL_HOLDOFF_SECONDS` retains its
existing default of 15 seconds. Adaptive adjustment is limited to 10 seconds
only after **complete** and **sustained** supportive 3s OFI plus strong depth.
The experimental path enforces a hard 10-second minimum even if the legacy
holdoff environment setting is shorter; with the experiment off, explicit
legacy settings remain authoritative.
Missing, neutral, stale or adverse flow retains the original holdoff.

VWAPPro's `VWAP_TREND_RETEST_EXTENSION_MAX_ATR` defaults to 3.5;
the opt-in alternative path still uses the existing session- and
option-contract-specific thesis and setup ID, preserving stop rearm semantics.

## Required promotion evidence
1. Pass complete exact-head GitHub CI, safety and temporal regression tests.
2. Replay baseline and one-switch-at-a-time ablations on capture-complete
   matching option contracts and underlying futures, with unique setup IDs.
3. Verify executable bid/ask at signal timestamps, slippage, all costs,
   stop/fill behavior, capital/risk affordability and unresolved outcomes.
4. Compare candidate counts, rejection stages, net expectancy, drawdown and
   false-positive rate over development, separate validation and current
   weekly-options out-of-sample sessions. No profitable-miss inference from
   no-trade counters alone.
5. The operator explicitly requested limited LIVE activation of all three
   existing bounded paths. This approval does not prove net profitability; retain
   independent rollback switches and evaluate the actual outcomes.

Deployment during the live session must not interrupt a broker order or
position. A merge alone is not proof production has deployed that SHA.

## October 23 historical Lightsail-only SMC rollout (superseded)

- The existing SMC two-book admission was enabled first in the normal
  `aws_lightsail` LIVE startup environment. VWAP retest and adaptive OFI were
  still default-disabled at that stage; no new trade authority or risk change existed.
- An explicit `SMC_SUSTAINED_FLOW_CONFIRMATION_ENABLED=false` remains a
  production rollback switch after a broker-flat controlled restart.
- Startup logs `SMC_SUSTAINED_FLOW_ADMISSION_CONFIG enabled=True/False` as
  observed runtime proof of the flag; deployment SHA by itself is insufficient.
- A qualified SMC setup still requires two fresh aligned, complete three-second
  OFI observations 4–12 seconds apart with independent quote versions. An
  adverse context, capacity retry, missing setup ID, or short-lived flow
  cannot be promoted by this flag. Sizing, exchange SL and broker safeguards
  remain mandatory.
- This is an operator-requested live-small experiment, not a verified improvement
  in out-of-sample trading expectancy. Measure actual setups, admission,
  affordability, net after-cost P&L and adverse outcomes before further tuning.

## October 23 historical Lightsail-only VWAP and OrderFlow rollout (superseded)

- On `aws_lightsail` when canonical execution mode requests LIVE, the existing
  configuration normalizer now defaults
  `VWAP_TREND_RETEST_EXTENSION_ENABLED=true` and
  `ORDERFLOW_OFI_ADAPTIVE_HOLDOFF_ENABLED=true` *only if the respective
  environment setting was absent*. Each explicitly set `false` independently
  overrides that default; paper, shadow and non-Lightsail default unchanged.
- The release logs
  `LIVE_ENTRY_QUALITY_CONFIG vwap_retest=True adaptive_ofi=True` along with
  mode and platform. A deployed Git SHA alone does not certify the effective
  flags. The existing SMC startup event remains separate.
- Strategy code, 3.5 ATR cap, strict two-bar finalized trend retest, sustained
  3-second OFI, 10-second adaptive minimum, adverse microstructure veto,
  risk/capital, exchange protective stops and broker reconciliation are
  unchanged. This activates already-implemented guarded alternatives rather
  than widening global entry or removing any protections.
- Deploy through the canonical exact-post-merge-push-CI and broker-flat release
  barrier. Recheck the exact SHA, `release_state=current`, engine readiness,
  and both effective startup values before claiming live activation.
- Operator approval to test small live exposure is not evidence of positive
  economic expectancy. Analyze unique admissions, fills and actual net costs,
  stopped positions, drawdown, and chronological out-of-sample replay before
  further raising trade frequency.

## Canonical GitHub LIVE profile (current)

The repository's `config/env_utils.py:normalise_live_env_defaults` now owns
all three default-on admission flags. This policy is independent of the host:
AWS Lightsail, any other deployment, and local LIVE mode share the same
strategy configuration. Each `setdefault_env` respects an explicit `false`
for independent operational rollback.

The profile is applied **only** when normalized `EXECUTION_MODE=LIVE` and
`ENABLE_LIVE=true` agree. PAPER, SHADOW, and mismatched mode/enable state
do not acquire any new implicit flags. Logs retain the canonical events
`SMC_SUSTAINED_FLOW_ADMISSION_CONFIG` and `LIVE_ENTRY_QUALITY_CONFIG` to
verify actual startup values. The AWS-specific first-boot legacy LIVE migration
and guarded deployment belong to infrastructure operations, not strategy
selection; they remain unchanged.

The strategy owners, two-book confirmation proof, max 3.5 ATR trend-retest
extension, conditional 10s OFI holdoff, adverse context vetoes, economic
admission, exchange protection and broker-risk checks remain intact. Neither
GitHub merge nor deployment SHA proves that an external explicit override is
not present; verify actual effective flags from the runtime logs.
