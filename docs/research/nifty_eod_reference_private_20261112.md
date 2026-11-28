# NIFTY historical EOD reference — operational handoff (12 Nov 2026)

## Observed, independently built dataset

The original, unchanged private Drive source is `NIFTY_Historical_20241106_20261105.tar.gz` (83,238,083 bytes), with its SHA-256 sidecar. `NIFTY_Special_Session_Addendum_20250201.tar.gz` (1,216,756 bytes) adds the separately verified Budget-session EOD report. Both source sidecars and every inner `SHA256SUMS` entry were checked before conversion.

An isolated private workspace successfully constructed `nifty_eod_reference.sqlite` (191,860,736 bytes, SHA-256 `78672c93ae7e4da59feec2fee6f2fc862ff641417c640e543435e376cad9ce7c`). Integrity was `ok`; queryable rows were:

| Data | Count |
| --- | ---: |
| Observed F&O sessions | 494 |
| Dated NIFTY options EOD contract rows | 829,045 |
| Dated NIFTY futures EOD contract rows | 1,482 |
| Valid separate vendor NIFTY index daily bars | 493 |

Budget session `2025-02-01`: 1,583 options + 3 futures rows. No same-day EOD reference availability is permitted. The compressed SQLite deliverable is `NIFTY_EOD_Research_SQLite_20261112.sqlite.gz` (46,152,472 bytes), held privately and not committed to GitHub.

### Reproduce offline (no broker account or paid feed)

Use a separate trusted research workstation. Original source archives and the sidecars must be available from the user's private Drive, unchanged. Run:

```sh
python -m scripts.build_eod_contract_db \
  --main /private/NIFTY_Historical_20241106_20261105.tar.gz \
  --main-sha /private/NIFTY_Historical_20241106_20261105.tar.gz.sha256 \
  --special /private/NIFTY_Special_Session_Addendum_20250201.tar.gz \
  --special-sha /private/NIFTY_Special_Session_Addendum_20250201.tar.gz.sha256 \
  --output /private/nifty_eod_reference.sqlite

python -m scripts.eod_research_reader \
  --database /private/nifty_eod_reference.sqlite \
  --asof 2025-02-04T09:30:00+05:30
```

The reader enforces ex-post availability and refuses historical bid/ask depth or fills. For causal daily context:

```python
from datetime import datetime
from pathlib import Path
from scripts.eod_research_reader import EodReferenceReader

reader = EodReferenceReader(Path('/private/nifty_eod_reference.sqlite'))
bars = reader.daily_index_bars(
    asof=datetime.fromisoformat('2025-02-04T09:30:00+05:30'),
    limit=30,
)
assert not reader.readiness()['execution_grade_ready']
```

**No 1-minute strategy P&L or execution price follows from EOD records.** These rows are date-stamped *ex-post references* from original NSE F&O reports and a mirror; their presence does not establish historically eligible broker tokens or contemporaneous listing at 09:15 on the same day. Never feed their `LastPric`/`ClsPric` into executable-quote or risk promotion paths.

## Breeze 25-window intraday conversion blocker

All 25 original encrypted ZIPs (107,821,151 bytes) were downloaded into an isolated workspace on 12 Nov 2026 and their outer CMS SHA-256 sidecars passed independently, **25/25**. The matching CMS **private key is not available** in the connected research environment. Therefore no claim of decrypted source rows, 1-minute SQLite completeness, runner historical option fills, or end-to-end intraday replay is justified.

The offline unsealer and strict 1/5/15-minute research importer are already implemented and full-CI tested on research PR #2024. Run them on the trusted machine containing the original matching private key; do not upload the key, or attempt to replace it. After deriving a private intraday SQLite, reconcile expected 1,454,867 source minutes minus explicit quarantine, complete timeframe windows, missing futures and option contract-days, and source clock conventions.

No paid NSE data, live bot changes, broker orders or automatic strategy promotion have been authorized by this research work.
