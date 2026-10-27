# October 22, 2026: admission and affordability audit

**Evidence grade:** production events, not verified executable price-path P&L. Times in IST.

## Denominators
- Distinct identified structural trigger IDs: six in full-day STRATEGY_TRIGGER_VOTE log (one VWAPPro CE plus five SMC PE/CE).
- Repeated trigger votes (151 log events) are **not** 151 independent opportunities.
- VWAPPro emitted three downstream STRUCTURAL_CANDIDATE_QUALIFIED events at 09:49:48, 10:18:39 and 10:20:22, all CE 22650. All passed runner readiness but **none** passed cost-inclusive one-lot risk/cash candidate capacity.
- Broker attempts for the logged 09:49 candidate: zero. No recorded broker fill is implied.
- Bot process-local candidate counts reset on restart; `candidate_generated_since_start=0` is not a full-session funnel denominator.

## Manager-qualified but unaffordable (rupees per 65-unit lot)

| Local time | Preferred CE 22650 risk | Lowest other viable quoted CE risk | Actual risk budget | Result |
| --- | ---: | ---: | ---: | --- |
| 09:49:49 | 948.82 | 861.51 (22700 CE) | 760.32 | All four options unaffordable |
| 10:18:39 | 856.45 | 851.56 (22700 CE) | 760.32 | All eligible options unaffordable; 22550 quote was stale |
| 10:20:22 | 847.74 | 886.39 (22700 CE) | 760.32 | All four options unaffordable |

Values are estimated cost-inclusive stop losses in production `CANDIDATE_CAPACITY_DECISION`, not subsequently realized losses. Options 22550 and 22600 had still higher risk or exceeded available cash. Do not bypass cost-inclusive loss budgeting.

## SMC setup-level conservative alternative (not approved live)

A retrospective *log-based proxy* for an opt-in variant requires two consecutive sustained three-second OFI observations 4–12 seconds apart, distinct increasing quote versions, no adverse context, and a structurally qualified SMC trigger. This uses timestamped log metadata, not independently integrity-verified full-depth replay.

| Local trigger | Side | Variants qualified in logs? | Earliest pair completion |
| --- | --- | --- | --- |
| 10:03:05 | 22600 PE | Not assessed (flow log tail begins 11:45) | N/A |
| 12:05:05 | 22650 PE | No | N/A |
| 12:39:06 | 22650 CE | Yes | 12:39:38 |
| 12:49:11 | 22650 CE | No: intervening adverse flow | N/A |
| 14:33:10 | 22650 CE | Yes | 14:34:05 |

The early VWAPPro CE event was eligible for independent OrderFlow and produced three manager-qualified outputs, but cash/risk rejected each downstream. An OFI acceptance improvement alone does not address this separate constraint.

**Not verified:** bid/ask entry/exit, quote-depth market impact, original stop/target path, broker fees, live-equivalent replay, forward opportunity P&L, out-of-sample expectancy or drawdown. Capture counters (214,691+ events) do not establish archival fidelity. Previous runtime replay job processed zero of 25 attempted partitions. Therefore variants remain opt-in disabled by default.

## Code decisions
1. Keep broker readiness, authenticated quote freshness, underlying direction and microstructure conflict vetoes.
2. Allow the narrowly defined **experimental**, setup-identity-keyed SMC double-book confirmation, disabled unless `SMC_SUSTAINED_FLOW_CONFIRMATION_ENABLED=true`. Keep previous admission unchanged by default.
3. Log actual VWAP futures slope bps and threshold, underlying side, and option/futures volume ratios for every rejected structural event. No slopes or signal rules are changed.
4. Do not make a second order-sizing authority or narrow a technically necessary stop merely to fit a fixed daily risk budget. Continue to evaluate available strikes with canonical cash, quote and stop-risk gates.
5. Retest original and modified policies using actual capture integrity checks, exact instrument+bar identities, executable bid/ask and canonical brokerage+spread costs, and an out-of-sample holdout before promoting new live eligibility.

Related replay tools: `src/nifty_scalper_bot/backtesting/executable_counterfactual.py` and `docs/research/on_demand_runtime_replay.md`.
