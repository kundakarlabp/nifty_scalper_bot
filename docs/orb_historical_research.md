# Expanded ORB monthly-options research

The `ORB historical research` GitHub Actions workflow downloads public expired
NIFTY options and matching spot/futures minute data, verifies the publisher's
checksums, prepares historical daily baskets and completes settings selection
and later-year tests without broker credentials or AWS commands. It runs on
relevant changes merged into `main` and through **Actions → ORB historical
research → Run workflow**. Results and the frozen protocol are downloadable
artifacts; the job summary contains the comparison table.

This is conditional component research, not a replay of live execution. It
never writes live configuration, instantiates a broker client, places orders,
or promotes a setting.

## Data and eligibility

- Source: Aparna Bhat, *Nifty spot, futures and options one-minute data from
  2017 to 2020*, DOI [10.5281/zenodo.10899828](https://doi.org/10.5281/zenodo.10899828).
- Both ZIP checksums are verified on every preparation, including cache hits.
- Select the exact ATM CE/PE pair using the opening minute's spot close and
  only options already observed by that minute. Hold that basket for the day.
- Use the nearest monthly expiry represented by the calendar month. A missing
  expiry or missing opening pair does not authorize a later expiry or a better
  hindsight strike. January 2018 is absent from the publisher's options ZIP.
- Expiry labels without dates in their tickers are inferred from the last
  traded day across the month folder. Continuous `NIFTY_F1` is context only;
  its original futures-token and roll identities are unavailable.
- Context timestamps and option labels before March 2018 are assumed to be
  minute ends. Later option labels are assumed to be minute starts, reflecting
  the format transition and observed 09:15 rows. The publisher has not verified
  these conventions. This is a material evidence limitation.
- Invalid option candles and zero-volume placeholders are missing observations,
  never repaired prices. Their counts and excluded sessions are recorded.

## Registered comparison

The protocol is written before any development run and fingerprinted against
the replay code and data manifest. Resumption requires an identical protocol.
There are 12 configurations: the raw current component reference, a 3×3 grid
of option ATR stop multipliers `0.75 / 1.0 / 1.25` and target multiples
`1.8 / 2.2 / 2.5`, plus retest-only and a 60-minute entry window. All grid and
entry candidates retain the research net reward/risk threshold of 1.5. The
current premium-stop cap remains in force.

| Period | Role |
|---|---|
| 2017–2018 | Compare every candidate at 10, 25 and 50 bps per side |
| 2019 | Validate only the frozen candidate and both references |
| 2020 | Final held-out evaluation of those same frozen configurations |

Choose the highest worst-slippage development expectancy among candidates
with at least 100 trades at **each** slippage level. This is a preliminary
sample screen, not a statistical proof. If none qualifies, the frozen candidate
is null and later periods evaluate the references only. Never lower the sample
threshold or choose another candidate after inspecting the held-out results.
Negative results remain negative; abstention is not demonstrated alpha.

## Execution assumptions

The production ORB strategy and canonical option ATR are used. The compact
ORB-only research context has an actual-strategy regression comparing its trades
against full indicator context. Other components retain the full context path.
Signals start at 09:30; the opening range remains 15 minutes. Expired entry
windows stop evaluations but pending fills and open-position exits continue.

Fill at the next available minute's opening price with adverse slippage. Require
a positive-volume option candle, reject invalid gapped geometry, include the
canonical modeled fees, and resolve a candle hitting both stop and target as a
stop. A missing/zero-volume exit observation is charged a full-premium loss as
an explicit worst-case penalty. It is not a claim that such a fill occurred.
Scheduled session exits occur at the 14:59 bar close.

Quantity is standardized to 75 units per trade and fees use the code's current
cost model. Results are rupee P&L for that standardized experiment, not historical
account returns or annual return percentages. Wider stops change rupee risk;
the study does not reproduce dynamic live sizing or a portfolio capital budget.

Minute OHLC cannot establish executable quotes, bid/ask spread, depth, queue
position, rejected orders or partial fills. The study also omits live arbitration,
direction authorization, bounded live target repair, trailing brackets and actual
broker-confirmed fills. Timestamp/expiry/futures identity verification, recent
weekly-contract data and prospective paper validation remain necessary before
considering a live change, even if modeled held-out P&L is positive.

## Local reproduction

Use an isolated Python environment with the repository installed. Replace the
two directories with local research paths:

```bash
python scripts/research_orb_multiyear.py prepare --source /tmp/orb-source --output /tmp/orb-study
python scripts/research_orb_multiyear.py development --source /tmp/orb-source --output /tmp/orb-study --workers 4
python scripts/research_orb_multiyear.py validation --source /tmp/orb-source --output /tmp/orb-study --workers 4
python scripts/research_orb_multiyear.py final --source /tmp/orb-source --output /tmp/orb-study --workers 4
```

Each phase saves atomic candidate results, trades, yearly metrics, rejection
reasons and exit reasons. Completed tasks resume without repeating the entire
study. Review `data_manifest.json`, `protocol.json`, `selection.json` and the
three `*_results.json` files together.
