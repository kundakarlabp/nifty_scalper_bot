# SMC entry-quality audit — 16 October 2026

**Status:** Engineering-correctness work; no profitability or live-entry-policy promotion.

## Evidence and scope

- Authoritative SMC setup: `smc_liquidity.py`, NIFTY futures first, spot context fallback; only NIFTY option contracts executable.
- Structure sequence: completed pivot -> sweep/reclaim -> *later* underlying displacement; option BOS/CHOCH and premium reclaim are additional observations, not independent futures sweeps.
- StrategyManager retains the direction, independent-trigger/context, executable option and risk controls. OrderFlow is context-only; SMC and VWAP share the canonical `reclaim_structure` family. Do not relabel either to manufacture independent votes.
- Latest component research (15 Sep–14 Oct 2026) had **zero resolved SMC trades**. The historical option archive uses the current contract basket and supplies just 19 source option sessions, not a date-correct expired weekly ATM universe. The research cannot validate net SMC profitability.
- Historical evaluation rejections are **per option/bar**, not unique sweep events. No-liquidity-sweep, awaiting confirmation and too-deep sweeps dominate; their raw counts must not be interpreted as independent trades.

## Implemented correctness work

1. PR #1664: reject completed NIFTY futures/spot bars *ahead* of the option observation; preserve within-budget backward lag. RED/GREEN tests include future sweep and future displacement with same-time recovery.
2. PR #1665: label BOS/CHOCH support derived from option OHLC separately from futures-led structure; preserve external-feature provenance instead of assuming manager enrichment owns every supplied flag. No change to setup, entry, stop, target or OrderFlow gates.

## Priority-ordered remaining work (do not loosen LIVE yet)

| Priority | Research or engineering gate | Minimum acceptance |
|---|---|---|
| P0 | **Date-causal expired weekly options** — issue #1663 | Verified instrument expiry/strike and minute bars by historical date, realistic bid/ask or explicit slippage limits, eligibility denominators |
| P0 | **Unique SMC setup/admission funnel** | `setup_id` counts at sweep, reclaim, displacement, StrategyManager independent confirmation, quote/economics and broker admission; distinct per-option observations from underlying events |
| P0 | **Underlying structural invalidation and exit ownership** | Trace who consumes `underlying_invalidation_level`; paper-only paired exit ablation before changing reconciled protective orders or broker execution |
| P1 | **Reversal versus continuation** | Pre-registered regime/direction classification and chronological holdout; retain current live continuation restriction until a distinct reversal model earns evidence |
| P1 | **Confirmation latency** | Paired comparison at completion, next tradable quote, 1/2/3-bar delays; no future-candle leakage |
| P1 | **Liquidity significance and ATR geometry** | Ablate prior-day/session levels, multi-touch pivots, 0.08–0.75 ATR depth, 0.03 ATR reclaim and 0.25 ATR displacement without multiple-testing selection |
| P2 | **Time-of-day normalised futures participation** | Contiguous prior-session same-clock minute baselines; fallback to historical existing RVOL when the sample is insufficient |
| P2 | **OFI and premium confirmation** | Test persistent OFI against option premium and *actual* futures outcome, distinguish correlated observations, avoid treating a single OFI tick as independent trigger |
| P2 | **Options net economics** | Realistic expiry/contract rotation, spread/depth, fees/taxes/slippage and executable entry and exit quotes at the final stop/target geometry |

**Promotion gate:** Sufficient unique resolved development trades (current research minimum 30), untouched chronological holdout, adverse slippage/fee sensitivity, live-equivalent recorded feed, and subsequent SHADOW/paper support. No code edit implies a profitable strategy.

## Sources and limitations

- Cont, Kukanov and Stoikov, *The Price Impact of Order Book Events*, Journal of Financial Econometrics, 2014. OFI explains contemporaneous price impact; not proof of a NIFTY SMC entry edge. https://doi.org/10.1093/jjfinec/nbt003
- Bailey and López de Prado, *The Deflated Sharpe Ratio*: guard against data snooping and multiple-testing selection. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551
- Repository contracts: `AGENTS.md`, `.agents/skills/strategy-research-validation/SKILL.md`, `src/nifty_scalper_bot/backtesting/strategy_research.py` and issue #1663.
