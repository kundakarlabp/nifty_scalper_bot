# Finish Breeze intraday conversion and offline research replay — 12 Nov 2026

**This is the supported end-to-end *private-workstation* route, not a live bot deploy.**
The original private CMS key has not been found in the currently connected environment.
A newly generated key cannot decrypt the archives. This repository neither stores
nor needs access to the key.

## Verified existing inputs
- 25 original encrypted Breeze ZIPs; 107,821,151 total bytes; 25/25 CMS outer SHA256 checks passed again in the isolated research workspace on 12 November 2026.
- Official recipient **public** certificate at `deploy/breeze_transport_recipient.pem`; CMS recipient common name `nifty-breeze-offline-research-transfer-20261109`.
- Prior offline inner audit identified 1,454,867 source 1m observations. This total has **not** been independently reproduced by newly decrypting the archives in this session.
- Existing complete offline EOD research reference SQLite is independently built and queryable (494 F&O sessions, 829,045 option contract-days, 1,482 futures contract-days, 493 index daily bars). The compressed dataset is private and must be extracted before replay.
- No NSE purchase is needed for current OHLCV research; historical bid/ask/depth remains unavailable.

## One command, on the original owner-held trusted machine

Use Python 3.11+, OpenSSL, an encrypted private disk and a trusted clone of the
repository with offline dependencies installed. Place the **already downloaded**
25 ZIPs in a private folder. Locate the existing matching private key using only
its *public-key fingerprint*; do not upload or print the key.

```sh
umask 077
python -m scripts.finish_breeze_private_replay \
  --archives /private/breeze/source-zips \
  --certificate deploy/breeze_transport_recipient.pem \
  --key-search-dir "$HOME" \
  --output-root /private/breeze/converted-20261112
```

For a known location, supply `--key /private/keys/original.pem` instead of
the search directory. The command does **not** print key bytes or key paths.
A preflight-only `--preflight-only` verifies the original 25 transport ZIPs
without needing any key or creating unencrypted files.

It stops before decryption if the matching original key cannot be located.
If a key matches, it decrypts all 25 windows, revalidates inner SHA256SUMS and
per-contract digests, quarantines the known special-session clock anomaly,
deduplicates identical overlaps, and builds:
- `/private/breeze/converted-20261112/nifty_intraday_research.sqlite`
- `/private/breeze/converted-20261112/coverage_summary.json`
- `/private/breeze/converted-20261112/pipeline_readiness.json`

Reconcile admitted plus explicitly quarantined minutes against the previous
1,454,867-source observation audit. A mismatch does not silently pass as
certified coverage; all gaps remain visible.

## Offline strategy-engine adapter

Once the minute SQLite exists, provide the independently built EOD reference
SQLite *as a separate local input*. Select a named trading day for research:

```sh
ENABLE_LIVE=false EXECUTION_MODE=SHADOW \
python -m scripts.replay_breeze_archive_sqlite \
  --minute-db /private/breeze/converted-20261112/nifty_intraday_research.sqlite \
  --eod-db /private/nifty_eod_reference.sqlite \
  --day 2025-11-04 \
  --output /private/breeze/replays/2025-11-04
```

The specified day is an example, **not a claim that exact historical CE/PE
minute archives are available**. The adapter chooses the previous published
EOD snapshot, the earliest listed expiry and exact ATM CE/PE pair by previous
index close, and the earliest previous-day listed futures contract. If any
required instrument lacks exact same-day Breeze minute bars, it **abstains**
instead of opportunistically substituting a profitable looking strike.

It stages four read-only candlestick series into a **private** research directory
and executes the existing SHADOW component research with 10/25/50bps stress
scenarios and next-bar assumptions. Use `--prepare-only` to inspect identity and
coverage without running the component engine.

## Evidence and production safeguards

- These are *bar-based research proxies*, NOT verified broker-executable fills,
  tick depth, contemporaneous best bid/ask, exchange queue or live replay parity.
- Breeze source timestamp semantics are not independently verified. The 2025-10-21
  index clock anomaly remains quarantined; 2025-02-01 Budget is special.
- The EOD contract snapshot is used only from a prior, published session. Do not
  use same-day EOD to select strikes/expiry/lot size at the 09:15 open.
- The research replay rejects ENABLE_LIVE=true and any EXECUTION_MODE other
  than SHADOW; it never imports broker entrypoints or places orders.
- Never use any reported trades to alter the live bot automatically.
- The historical source and ZIPs remain immutable; all derived SQLite/JSON
  files stay on private storage and are never pushed into GitHub.

## Validation

Run:

```sh
python -m pytest -q \
  tests/scripts/test_finish_breeze_private_replay.py \
  tests/scripts/test_replay_breeze_archive_sqlite.py \
  tests/scripts/test_prepare_breeze_research_dataset.py \
  tests/scripts/test_unseal_breeze_private_windows.py \
  tests/backtests/test_strategy_bar_research.py
```

CI with synthetic keys tests code correctness, not private-key recovery or actual
25-window decryption. The indispensable remaining blocker, if the key is not
available on the trusted workstation, is retrieval of the **original** matching
private key. Encryption cannot be bypassed with a different key.
